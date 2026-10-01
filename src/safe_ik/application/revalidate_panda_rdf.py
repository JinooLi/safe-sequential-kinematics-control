"""Revalidate pinned saved holdouts into a NEW candidate; never fit or deploy."""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from ..adapters.panda_kinematics import PandaKinematics
from ..adapters.panda_rdf_geometry import extract_visual_geometry
from ..adapters.rdf_checkpoint import file_sha256
from .panda_configuration import load_panda_config
from .panda_rdf_artifact import load_bundle, read_candidate_for_validation, save_bundle
from .panda_rdf_training import validate_joint_gradients, validate_saved_model
from .panda_rdf_validation import accuracy_masks, validation_policy


def pinned(record):
    path = Path(record["path"]).resolve()
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"source SHA mismatch: {path}")
    return path


def check_coverage(manifest, bundle):
    evidence = manifest["evidence"]
    if set(evidence) != set(bundle.models):
        raise ValueError("saved evidence must cover every model")
    previous = bundle.metadata.get("validation_set_requirements", {})
    for name, entry in evidence.items():
        if not isinstance(entry, dict) or "regression" not in entry:
            raise ValueError(f"{name}: regression holdout missing")
        required = set(previous.get(name, ["regression"]))
        if "fresh_and_regression_passed" in bundle.metadata["validation"][name]:
            required.add("archived_purpose102")
        if not required.issubset(entry):
            raise ValueError(f"{name}: archived holdout must not be dropped")
        if not set(entry).issubset({"regression", "archived_purpose102"}):
            raise ValueError("unknown saved holdout identity")


def revalidate(config, manifest, output, *, threads=2):
    output = Path(output).resolve()
    source = pinned(manifest["candidate"])
    runtime = config.safety.rdf.artifact.resolve()
    runtime_before = file_sha256(runtime)
    source_before = file_sha256(source)
    if runtime_before != manifest["runtime_sha256"]:
        raise ValueError("runtime model differs from pinned baseline")
    if output.exists() or output == runtime.parent or output in source.parents:
        raise ValueError("choose a new, non-runtime output directory")
    options = config.safety.rdf
    if options.underestimation_limit is None:
        raise ValueError("explicit asymmetric underestimation_limit required")
    geometry = extract_visual_geometry(config.model_path)
    bundle = read_candidate_for_validation(source, geometry, options)
    check_coverage(manifest, bundle)
    evidence = {
        name: {label: pinned(record) for label, record in entry.items()}
        for name, entry in manifest["evidence"].items()
    }
    old = copy.deepcopy(bundle.metadata)
    # Never edit a historical report in place or erase a failed audit.
    meta = bundle.metadata
    meta.setdefault("validation_history", []).append(
        {
            "source": str(source),
            "source_sha256": source_before,
            "validation_policy": old.get(
                "validation_policy", "legacy_symmetric_absolute_error"
            ),
            "validation": old["validation"],
            "joint_validation": old["joint_validation"],
        }
    )
    meta.update(
        complete=False,
        validation={},
        joint_validation={"passed": False},
        validation_policy=validation_policy(options),
        validation_set_requirements={n: list(v) for n, v in evidence.items()},
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = perf_counter()
    summary = {
        "complete": False,
        "source_sha256": source_before,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "new_holdout_sampling": False,
        "runtime_installed": False,
        "continuous_error_certificate": False,
        "validation_policy": validation_policy(options),
        "checks": {},
    }
    try:
        with threadpool_limits(limits=threads):
            for index, name in enumerate(bundle.models):
                directory = output / name
                directory.mkdir()
                reports, checks = {}, {}
                for set_index, (label, path) in enumerate(evidence[name].items()):
                    print(f"VALIDATE {name}/{label}", flush=True)
                    with np.load(path, allow_pickle=False) as archive:
                        original = {
                            k: archive[k].copy()
                            for k in (
                                "points",
                                "truth",
                                "prediction",
                                "exterior",
                                "near",
                            )
                        }
                    destination = directory / f"{label}.npz"
                    t0 = perf_counter()
                    report = validate_saved_model(
                        bundle.reconstructed[name],
                        bundle.models[name],
                        original["points"],
                        np.random.default_rng(
                            np.random.SeedSequence(
                                20260930, spawn_key=(index, set_index, 201)
                            )
                        ),
                        options.error_margin,
                        underestimation_limit=options.underestimation_limit,
                        evidence_path=destination,
                    )
                    with np.load(destination, allow_pickle=False) as fresh:
                        if not np.array_equal(original["points"], fresh["points"]):
                            raise ValueError("holdout points changed")
                        dt = float(np.abs(original["truth"] - fresh["truth"]).max())
                        dp = float(
                            np.abs(original["prediction"] - fresh["prediction"]).max()
                        )
                        if dt > 1e-12 or dp > 1e-10:
                            raise ValueError(
                                f"{name}/{label}: archived truth/prediction mismatch: {dt}, {dp}"
                            )
                        outside, near = accuracy_masks(fresh["truth"])
                        if not np.array_equal(
                            outside, original["exterior"]
                        ) or not np.array_equal(near, original["near"]):
                            raise ValueError("reference-based evaluation masks changed")
                    report["reference_mesh_digest"] = meta["reconstruction"][name][
                        "mesh_digest"
                    ]
                    report["evidence_sha256"] = file_sha256(destination)
                    report["evidence_source_sha256"] = file_sha256(path)
                    reports[label] = report
                    checks[label] = {
                        "points_unchanged": True,
                        "masks_unchanged": True,
                        "truth_max_difference_m": dt,
                        "prediction_max_difference_m": dp,
                        "seconds": perf_counter() - t0,
                    }
                    print(
                        f"COMPLETE {name}/{label}: under={1000*report['near_max_underestimate']:.6f}mm over={1000*report['near_max_overestimate']:.6f}mm passed={report['passed']}",
                        flush=True,
                    )
                combined = copy.deepcopy(reports["regression"])
                combined["validation_sets"] = reports
                for key in ("passed", "distance_passed", "derivatives_passed"):
                    combined[key] = all(r[key] for r in reports.values())
                if "archived_purpose102" in reports:
                    combined["fresh_and_regression_passed"] = combined["passed"]
                meta["validation"][name] = combined
                summary["checks"][name] = checks
                (directory / "validation.json").write_text(
                    json.dumps(combined, indent=2, allow_nan=False) + "\n"
                )
            kinematics = PandaKinematics(
                config.model_path, config.ee_frame, geometry.vertices
            )
            meta["joint_validation"] = validate_joint_gradients(
                kinematics,
                bundle,
                options,
                np.random.default_rng(
                    np.random.SeedSequence(options.seed, spawn_key=(0, 3))
                ),
            )
            expected = meta["joint_validation"]["passed"] and all(
                r["passed"] for r in meta["validation"].values()
            )
            meta["complete"] = True
            meta["revalidation"] = {
                "source_sha256": source_before,
                "policy_only_change": True,
                "all_model_arrays_preserved": True,
                "holdouts_previously_observed": True,
                "spatial_check_seed": "SeedSequence(20260930, spawn_key=(link_index, set_index, 201))",
            }
            candidate = output / "candidate.npz"
            identifier = save_bundle(bundle, candidate)
            with (
                np.load(source, allow_pickle=False) as before,
                np.load(candidate, allow_pickle=False) as after,
            ):
                if set(before.files) != set(after.files) or any(
                    not np.array_equal(before[k], after[k])
                    for k in before.files
                    if k != "metadata"
                ):
                    raise ValueError("revalidation changed model/geometry/query arrays")
            if old["array_hashes"] != meta["array_hashes"]:
                raise ValueError("model array hash mismatch")
            if any(
                meta[k] != old[k]
                for k in (
                    "geometry",
                    "geometry_fingerprint",
                    "settings",
                    "spheres",
                    "reconstruction",
                    "maxima",
                )
            ):
                raise ValueError("geometry/training settings changed")
            try:
                load_bundle(candidate, geometry, options)
                approved, reason = True, None
            except ValueError as exc:
                approved, reason = False, str(exc)
            if approved != expected:
                raise ValueError(f"strict loader/revalidation disagreement: {reason}")
            summary.update(
                complete=True,
                candidate_sha256=identifier,
                all_model_arrays_unchanged=True,
                geometry_radii_queries_unchanged=True,
                joint_validation=meta["joint_validation"],
                loader_approved=approved,
                loader_rejection=reason,
            )
    except Exception as exc:
        meta["complete"] = False
        meta["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        summary["failure"] = meta["failure"]
        save_bundle(bundle, output / "candidate.npz")
        raise
    finally:
        summary["seconds"] = perf_counter() - started
        summary["runtime_sha256"] = file_sha256(runtime)
        summary["runtime_model_unchanged"] = summary["runtime_sha256"] == runtime_before
        summary["source_unchanged"] = file_sha256(source) == source_before
        summary["evidence_sources_unchanged"] = all(
            file_sha256(path) == manifest["evidence"][n][label]["sha256"]
            for n, entry in evidence.items()
            for label, path in entry.items()
        )
        if not all(
            summary[k]
            for k in (
                "runtime_model_unchanged",
                "source_unchanged",
                "evidence_sources_unchanged",
            )
        ):
            summary["complete"] = False
            summary["loader_approved"] = False
            summary["failure"] = {
                "type": "IntegrityError",
                "message": "read-only source/runtime integrity check failed",
            }
            meta["complete"] = False
            meta["failure"] = summary["failure"]
            save_bundle(bundle, output / "candidate.npz")
        summary["ended_at"] = datetime.now(timezone.utc).isoformat()
        (output / "candidate.report.json").write_text(
            json.dumps(meta, indent=2, allow_nan=False) + "\n"
        )
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
    if not all(
        summary[k]
        for k in (
            "runtime_model_unchanged",
            "source_unchanged",
            "evidence_sources_unchanged",
        )
    ):
        raise ValueError("read-only source/runtime integrity check failed")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    summary = revalidate(
        load_panda_config(args.config),
        json.loads(args.manifest.read_text()),
        args.output,
        threads=args.threads,
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["loader_approved"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

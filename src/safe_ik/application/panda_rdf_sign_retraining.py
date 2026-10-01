"""One approved fresh link6 fit with pinned deterministic labels; no resume."""

import argparse
import copy
import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from ..adapters.mesh_distance import SIGN_ORACLE
from ..adapters.panda_kinematics import PandaKinematics
from ..adapters.panda_rdf_geometry import extract_visual_geometry
from ..adapters.rdf_checkpoint import file_sha256
from ..adapters.rdf_torch_training import torch_training_environment
from .panda_configuration import load_panda_config
from .panda_rdf_artifact import load_bundle, read_candidate_for_validation, save_bundle
from .panda_rdf_enriched_training import write_json
from .panda_rdf_failure_study import POOL_KEYS, read_inputs, require, rng_for, run_link
from .panda_rdf_training import (
    TrainingOptions,
    benchmark_candidate,
    validate_joint_gradients,
)


def validate_protocol(protocol):
    require(protocol["schema"] == 1 and protocol["links"] == ["link6"], "link6 only")
    require(protocol["purpose"] == "deterministic_sign_fresh_refit", "purpose")
    require(protocol["sign_oracle"] == SIGN_ORACLE, "sign oracle version")
    require(protocol["basis_count"] == 24 and protocol["updates"] == 800, "fixed fit")
    require(
        protocol["regularization"] == 1e-4 and protocol["dtype"] == "float64",
        "fit precision",
    )
    require(protocol["initial_state"] == "w=0,B=I/lambda", "fresh state only")
    require(
        protocol["batch_sizes"] == [512, 128, 640] and protocol["pool_size"] == 500000,
        "fixed batches/pools",
    )
    require(protocol["checkpoint_steps"] == [400, 800], "fixed checkpoints")
    require(
        protocol["error_margin"] == 0.010 and protocol["seed"] == 20260929,
        "fixed validation",
    )
    require(
        protocol["relabel"] == "all_pools_mesh_truth_preserve_label_dtype",
        "label precision",
    )
    require(
        type(protocol["data_workers"]) is int and 1 <= protocol["data_workers"] <= 4,
        "workers",
    )
    hashes = protocol["expected_mesh_label_array_sha256"]
    require(set(hashes) == set(POOL_KEYS), "independent label coverage")
    require(
        all(
            len(v) == 64 and set(v) <= set("0123456789abcdef") for v in hashes.values()
        ),
        "label SHA-256",
    )
    require(set(protocol["input_sha256"]) == {"link6"}, "input coverage")


def verify_preservation(baseline, candidate):
    preserved = [name for name in baseline.models if name != "link6"]
    require(len(preserved) == 8, "eight preserved links")
    for name in baseline.models:
        a, b = baseline.models[name], candidate.models[name]
        require(
            np.array_equal(a.rdf.center, b.rdf.center)
            and a.rdf.radius == b.rdf.radius
            and a.inner_radius == b.inner_radius,
            "geometry/radius changed",
        )
        require(
            np.array_equal(baseline.queries[name], candidate.queries[name]),
            "queries changed",
        )
        require(
            np.array_equal(
                baseline.reconstructed[name].vertices,
                candidate.reconstructed[name].vertices,
            )
            and np.array_equal(
                baseline.reconstructed[name].faces, candidate.reconstructed[name].faces
            ),
            "reference mesh changed",
        )
        if name in preserved:
            require(
                np.array_equal(a.rdf.weights, b.rdf.weights)
                and a.surface_maximum == b.surface_maximum,
                "preserved model changed",
            )
            for key in ["validation", "maxima", "link_training"]:
                require(
                    baseline.metadata[key][name] == candidate.metadata[key][name],
                    "preserved report changed",
                )
    return preserved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["source", "inputs", "output", "protocol", "config", "runtime-model"]:
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    require(
        file_sha256(args.source / "candidate.npz")
        == protocol["source_candidate_sha256"],
        "source candidate SHA",
    )
    require(
        file_sha256(args.source / "candidate.report.json")
        == protocol["source_report_sha256"],
        "source report SHA",
    )
    require(
        file_sha256(args.inputs / "candidate.npz")
        == protocol["input_candidate_sha256"],
        "original inputs candidate SHA",
    )
    require(
        file_sha256(args.runtime_model) == protocol["runtime_model_sha256"],
        "runtime model before SHA",
    )
    config = load_panda_config(args.config)
    options = config.safety.rdf
    require(
        [
            options.error_margin,
            options.physical_margin,
            options.effective_margin,
            options.softmin_error,
        ]
        == [0.010, 0.030, 0.040, 0.005],
        "fixed margins",
    )
    geometry = extract_visual_geometry(config.model_path)
    baseline = read_candidate_for_validation(
        args.source / "candidate.npz", geometry, options
    )
    require(baseline.metadata["complete"], "complete baseline")
    training = TrainingOptions(**baseline.metadata["training"]["baseline_options"])
    require(
        [training.basis_count, training.updates, training.regularization]
        == [24, 800, 1e-4],
        "fixed training options",
    )
    read_inputs(
        args.inputs,
        "link6",
        baseline.models["link6"].rdf,
        protocol["input_sha256"]["link6"],
    )
    compute = torch_training_environment("cuda")
    require(compute["gpu"] == "NVIDIA GeForce RTX 4070", "approved GPU")
    require(not args.output.exists(), "output already exists; never restart")
    require(shutil.disk_usage(args.output.parent).free > 8 * 1024**3, "need 8 GiB free")
    frames = set(baseline.models) | {
        f for c in config.safety.capsules for f in [c.start_frame, c.end_frame]
    }
    kin = PandaKinematics(config.model_path, config.ee_frame, frames)
    if args.preflight:
        print(
            "PREFLIGHT PASS: one fresh link6 fit; eight models preserved; labels pinned; no training",
            flush=True,
        )
        return
    before = file_sha256(args.runtime_model)
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    status, failure = 1, None
    write_json(
        args.output / "job.json",
        dict(
            started_at=datetime.now(timezone.utc).isoformat(),
            protocol=protocol,
            source=str(args.source.resolve()),
            inputs=str(args.inputs.resolve()),
            runtime_model_sha256_before=before,
        ),
    )
    for source, destination in [
        (args.protocol, "protocol.json"),
        (args.source / "candidate.npz", "baseline.npz"),
        (args.source / "candidate.report.json", "baseline.report.json"),
    ]:
        shutil.copyfile(source, args.output / destination)
    bundle = copy.deepcopy(baseline)
    bundle.metadata["complete"] = False
    bundle.metadata["sign_retraining"] = dict(
        protocol=protocol,
        parent=str(args.source),
        preserved_links=[n for n in bundle.models if n != "link6"],
        original_audit_failure_preserved=True,
    )
    try:
        with threadpool_limits(limits=1):
            print("LINK START link6", flush=True)
            comparison = run_link(
                args.inputs, args.output, "link6", baseline, bundle, protocol, training
            )
            preserved = verify_preservation(baseline, bundle)
            bundle.metadata["joint_validation"] = validate_joint_gradients(
                kin, bundle, options, rng_for(options.seed, 0, 3)
            )
            bundle.metadata["runtime_benchmark"] = benchmark_candidate(
                config, kin, bundle
            )
            bundle.metadata["complete"] = True
            bundle.metadata["wall_seconds"] = perf_counter() - started
            save_bundle(bundle, args.output / "candidate.npz")
            write_json(args.output / "candidate.report.json", bundle.metadata)
            expected = (
                all(v["passed"] for v in bundle.metadata["validation"].values())
                and bundle.metadata["joint_validation"]["passed"]
            )
            try:
                load_bundle(args.output / "candidate.npz", geometry, options)
                approved, reason = True, None
            except ValueError as exc:
                approved, reason = False, str(exc)
            require(approved == expected, "strict loader/gate disagreement")
            write_json(
                args.output / "summary.json",
                dict(
                    complete=True,
                    comparison=comparison,
                    passed=expected,
                    link6_passed=comparison["passed"],
                    runtime_loader_approved=approved,
                    runtime_rejection=reason,
                    preserved_links=preserved,
                    joint_validation=bundle.metadata["joint_validation"],
                    runtime_benchmark=bundle.metadata["runtime_benchmark"],
                    wall_seconds=perf_counter() - started,
                    original_audit_failure_preserved=True,
                ),
            )
            status = 0 if expected else 2
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc))
        traceback.print_exc()
    finally:
        after = file_sha256(args.runtime_model)
        if after != before:
            status, failure = 1, dict(
                type="RuntimeModelChanged", message="runtime model SHA changed"
            )
        write_json(
            args.output / "exit.json",
            dict(
                returncode=status,
                failure=failure,
                finished_at=datetime.now(timezone.utc).isoformat(),
                wall_seconds=perf_counter() - started,
                runtime_model_sha256_before=before,
                runtime_model_sha256_after=after,
                runtime_model_unchanged=before == after,
            ),
        )
        with (args.output / "remote.sha256").open("x") as stream:
            for path in sorted(args.output.rglob("*")):
                if path.is_file() and path.name != "remote.sha256":
                    stream.write(
                        f"{file_sha256(path)}  {path.relative_to(args.output)}\n"
                    )
    raise SystemExit(status)


if __name__ == "__main__":
    main()

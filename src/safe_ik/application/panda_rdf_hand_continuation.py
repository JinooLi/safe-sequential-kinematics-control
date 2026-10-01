"""One bounded append-only hand experiment from a verified weighted state."""

import argparse
import copy
import gc
import json
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from ..adapters.mesh_distance import SIGN_ORACLE
from ..adapters.panda_kinematics import PandaKinematics
from ..adapters.panda_rdf_geometry import extract_visual_geometry
from ..adapters.rdf_checkpoint import (
    RidgeState,
    array_sha256,
    file_sha256,
    load_checkpoint,
    model_signature,
    save_checkpoint,
)
from ..adapters.rdf_torch_training import (
    recursive_ridge_torch,
    torch_training_environment,
)
from ..domain.panda_rdf import ExtendedRDF
from .panda_configuration import load_panda_config
from .panda_rdf_artifact import load_bundle, read_candidate_for_validation, save_bundle
from .panda_rdf_enriched_training import write_json
from .panda_rdf_failure_study import (
    POOL_KEYS,
    accuracy_on_saved,
    draw_mining_points,
    query_truth,
    require,
    rng_for,
    save_arrays,
    summarize_labels,
)
from .panda_rdf_training import (
    TrainingOptions,
    benchmark_candidate,
    estimate_maximum,
    uniform_ball,
    validate_joint_gradients,
    validate_model,
)


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {key: a[key] for key in a.files}


REWEIGHT_PURPOSE = "targeted_reweight_saved5000_from_weighted_800"
REUSE_FILES = (
    "protocol.json",
    "candidate.npz",
    "candidate.report.json",
    "hand/training.npz",
    "hand/batch_indices.npz",
    "hand/mining.npz",
    "hand/proposals.npz",
    "hand/scan_cloud.npz",
    "hand/data.report.json",
    "hand/fit.npz",
    "hand/comparison.json",
    "hand/regression.npz",
    "hand/fresh.npz",
)


def validate_protocol(p):
    reweight = p.get("purpose") == REWEIGHT_PURPOSE
    expected = dict(
        schema=1,
        link="hand",
        purpose=REWEIGHT_PURPOSE if reweight else "targeted_append_from_weighted_800",
        basis_count=24,
        regularization=1e-4,
        dtype="float64",
        parent_step=800,
        parent_samples=1024000,
        new_count=5000,
        batch_sizes=[1280, 1280, 1280, 1160],
        new_weight=4.0 if reweight else 1.0,
        passes=1,
        seed=20260930,
        index=8,
        mining_purpose=201,
        proposal_purpose=202,
        mining_count=20000,
        proposal_count=50000,
        error_threshold=0.008,
        local_radius=0.010,
        near_distance=0.060,
        error_margin=0.010,
        sign_oracle=SIGN_ORACLE,
        label_method="scan_normal",
        data_workers=4,
    )
    for key, value in expected.items():
        require(p[key] == value, "fixed protocol " + key)
    if reweight:
        require(
            set(p["reuse_files"]) == set(REUSE_FILES), "fixed protocol reuse coverage"
        )
        require(
            all(len(v) == 64 for v in p["reuse_files"].values()),
            "fixed protocol reuse SHA",
        )


def verify_reuse(directory, p, model, dtype):
    """Require the already audited 1x input; never generate or relabel it."""
    require(directory is not None, "saved 1x data is required")
    for name, digest in p["reuse_files"].items():
        require(file_sha256(directory / name) == digest, "saved 1x SHA " + name)
    original = read(directory / "protocol.json")
    validate_protocol(original)
    require(
        original["purpose"] == "targeted_append_from_weighted_800", "1x source purpose"
    )
    for key, value in original.items():
        if key not in ["purpose", "new_weight"]:
            require(p[key] == value, "1x fixed setting " + key)
    new = arrays(directory / "hand/training.npz")
    require(
        set(new) == {"points", "targets", "truth", "weights"}, "saved training fields"
    )
    require(new["points"].shape == (5000, 3), "saved point count")
    require(
        all(new[k].shape == (5000,) for k in ["targets", "truth", "weights"]),
        "saved target count",
    )
    require(all(np.isfinite(v).all() for v in new.values()), "finite saved training")
    require(new["targets"].dtype == dtype, "saved target dtype")
    require(
        new["weights"].dtype == np.float64
        and np.array_equal(new["weights"], np.ones(5000)),
        "saved unit weights",
    )
    require(
        np.all((new["truth"] >= 0) & (new["truth"] <= 0.060)), "saved exterior points"
    )
    require(
        np.all(np.linalg.norm(new["points"] - model.center, axis=1) <= model.radius),
        "saved domain",
    )
    require(len(np.unique(new["points"], axis=0)) == 5000, "saved unique points")
    batches = arrays(directory / "hand/batch_indices.npz")
    require(
        np.array_equal(batches["indices"], np.arange(5000))
        and np.array_equal(batches["offsets"], [0, 1280, 2560, 3840, 5000]),
        "saved one-pass batch order",
    )
    require(
        read(directory / "hand/comparison.json")["complete"], "1x complete comparison"
    )
    return new, batches


def copy_reweighted_data(source, directory, p, model, dtype):
    new, batches = verify_reuse(source, p, model, dtype)
    for name in ["mining.npz", "proposals.npz", "scan_cloud.npz", "batch_indices.npz"]:
        shutil.copyfile(source / "hand" / name, directory / name)
    new["weights"] = np.full(5000, 4.0, dtype=np.float64)
    save_arrays(directory / "training.npz", **new)
    data = read(source / "hand/data.report.json")
    data.update(
        new_weight=4.0,
        data_reused=True,
        generated_points_this_run=0,
        source_training_sha256=p["reuse_files"]["hand/training.npz"],
        effective_new_weight_sum=20000.0,
    )
    write_json(directory / "data.report.json", data)
    return new["points"], new["targets"], new["weights"], batches["offsets"], data


def select_anchors(points, truth, prediction, center, radius, threshold=0.008):
    require(
        points.shape == (len(truth), 3) and truth.shape == prediction.shape,
        "mining shapes",
    )
    require(
        all(np.isfinite(a).all() for a in [points, truth, prediction]), "finite mining"
    )
    return (
        (truth >= 0)
        & (truth <= 0.060)
        & (np.abs(prediction - truth) >= threshold)
        & (np.linalg.norm(points - center, axis=1) <= radius)
    )


def draw_local_proposals(anchors, rng, count, radius):
    require(len(anchors) > 0 and anchors.shape[1:] == (3,), "nonempty anchors")
    owners = rng.integers(0, len(anchors), size=count)
    return anchors[owners] + uniform_ball(rng, count, np.zeros(3), radius), owners


def select_new_points(points, truth, center, radius, forbidden, count):
    require(
        points.shape == (len(truth), 3)
        and np.isfinite(points).all()
        and np.isfinite(truth).all(),
        "finite proposal shapes",
    )
    nearest = cKDTree(forbidden).query(points, workers=1)[0]
    eligible = (
        (truth >= 0)
        & (truth <= 0.060)
        & (np.linalg.norm(points - center, axis=1) <= radius)
        & (nearest > 1e-12)
    )
    # Keep deterministic first occurrence only; never duplicate to fill the quota.
    _, first = np.unique(points, axis=0, return_index=True)
    unique = np.zeros(len(points), dtype=bool)
    unique[first] = True
    eligible &= unique
    chosen = np.flatnonzero(eligible)[:count]
    require(
        len(chosen) == count, "insufficient new points in fixed candidate budget; stop"
    )
    return chosen, eligible, nearest


def verify_parent(study, inputs, baseline, p):
    parent = study / "hand"
    for name, digest in p["parent_files"].items():
        require(file_sha256(parent / name) == digest, "parent SHA " + name)
    source = inputs / "candidate.training/hand"
    for name, digest in p["original_input_files"].items():
        require(file_sha256(source / name) == digest, "original input SHA " + name)
    context = read(parent / "context.json")
    indices = arrays(parent / "source_batch_indices.npz")
    weights = arrays(parent / "sample_weights.npz")
    targets = arrays(parent / "targets.npz")
    require(
        file_sha256(parent / "targets.npz") == context["targets_sha256"],
        "parent targets",
    )
    require(
        file_sha256(parent / "sample_weights.npz") == context["sample_weights_sha256"],
        "parent weights",
    )
    prefix = lambda n: {k: array_sha256(indices[k][:n]) for k in POOL_KEYS}
    state = load_checkpoint(
        parent / "checkpoints/step_000800",
        baseline.models["hand"].rdf,
        p["regularization"],
        context,
        prefix,
    )
    require(
        state.metadata["objective"] == "weighted"
        and state.metadata["completed_steps"] == 800
        and state.metadata["consumed_samples"] == 1024000,
        "parent weighted counters",
    )
    require(
        state.metadata["sample_weight_prefix_sha256"]
        == {k: array_sha256(weights[k][indices[k]]) for k in POOL_KEYS},
        "parent actual weights prefix",
    )
    for k in POOL_KEYS:
        require(
            np.isfinite(weights[k]).all() and np.all(weights[k] > 0),
            "positive parent weights",
        )
    pools = arrays(source / "training_pool.npz")
    ext = arrays(source / "exterior_pool.npz")
    require(
        np.array_equal(targets["near"], pools["near_labels"])
        and np.array_equal(targets["random"], pools["random_labels"])
        and np.array_equal(targets["exterior"], ext["labels"]),
        "parent scan targets unchanged",
    )
    model = baseline.models["hand"].rdf
    require(
        np.array_equal(
            (state.weights * model.radius).reshape(model.weights.shape), model.weights
        )
        and np.array_equal(arrays(parent / "fit.npz")["weights"], model.weights),
        "parent800=baseline",
    )
    require(
        [model.count, model.weights.dtype.name] == [24, "float64"],
        "parent basis/precision",
    )
    forbidden = np.vstack(
        [
            pools["near_points"],
            pools["random_points"],
            ext["points"],
            arrays(parent / "regression.npz")["points"],
            arrays(parent / "fresh.npz")["points"],
            arrays(parent / "mining.npz")["points"],
        ]
    )
    return state, forbidden, ext["labels"].dtype


def scan_targets(mesh, model, points, training, directory, dtype):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from mesh_to_sdf import get_surface_point_cloud

    normalized = trimesh.Trimesh(
        (mesh.vertices - model.center) / model.radius, mesh.faces, process=False
    )
    cloud = get_surface_point_cloud(
        normalized,
        surface_point_method="scan",
        bounding_radius=1.0,
        scan_count=training.scan_count,
        scan_resolution=training.scan_resolution,
        calculate_normals=True,
    )
    # Persist ordered cloud arrays, including KD-tree tie order inputs, for a
    # local query-only reimplementation of the scan/normal label calculation.
    save_arrays(
        directory / "scan_cloud.npz", points=cloud.points, normals=cloud.normals
    )
    values = []
    for start in range(0, len(points), 2048):
        values.append(
            cloud.get_sdf(
                (points[start : start + 2048] - model.center) / model.radius,
                sample_count=training.normal_count,
            )
            * model.radius
        )
    target = np.concatenate(values).astype(dtype)
    require(
        target.shape == (len(points),) and np.isfinite(target).all(), "new scan labels"
    )
    del cloud
    gc.collect()
    return target


def preserve(baseline, candidate):
    names = [n for n in baseline.models if n != "hand"]
    require(len(names) == 8, "eight other models")
    for n in baseline.models:
        a, b = baseline.models[n], candidate.models[n]
        require(
            np.array_equal(a.rdf.center, b.rdf.center)
            and a.rdf.radius == b.rdf.radius
            and a.inner_radius == b.inner_radius,
            "normalization unchanged",
        )
        require(
            np.array_equal(baseline.queries[n], candidate.queries[n]),
            "queries unchanged",
        )
        for key in ["vertices", "faces"]:
            require(
                np.array_equal(
                    getattr(baseline.reconstructed[n], key),
                    getattr(candidate.reconstructed[n], key),
                ),
                "mesh unchanged",
            )
        if n in names:
            require(
                np.array_equal(a.rdf.weights, b.rdf.weights)
                and a.surface_maximum == b.surface_maximum,
                "other coefficients unchanged",
            )
            for key in ["validation", "link_training", "maxima"]:
                require(
                    baseline.metadata[key][n] == candidate.metadata[key][n],
                    "other reports unchanged",
                )
    return names


def run_hand(
    study,
    output,
    baseline,
    bundle,
    protocol,
    training,
    parent,
    forbidden,
    dtype,
    reuse_data=None,
    weight1=None,
):
    p = protocol
    directory = output / "hand"
    directory.mkdir()
    mesh = bundle.reconstructed["hand"]
    model = baseline.models["hand"]
    began = perf_counter()
    if reuse_data is not None:
        points, targets, weights, offsets, data = copy_reweighted_data(
            reuse_data, directory, p, model.rdf, dtype
        )
        print("DATA hand: reuse exact saved 5000; weights 4; no sampling", flush=True)
    else:
        mining = draw_mining_points(
            mesh, model.rdf, rng_for(p["seed"], 8, 201), p["mining_count"]
        )
        truth = query_truth(mesh, mining, p["data_workers"], "hand/mining")
        prediction = model.rdf.query(mining, False)[0]
        selected = select_anchors(
            mining,
            truth,
            prediction,
            model.rdf.center,
            model.rdf.radius,
            p["error_threshold"],
        )
        anchors = mining[selected]
        require(len(anchors) > 0, "no anchors; no automatic threshold/seed change")
        save_arrays(
            directory / "mining.npz",
            points=mining,
            truth=truth,
            baseline_prediction=prediction,
            selected=selected,
            anchors=anchors,
        )
        print(f"DATA hand: {len(anchors)} independent anchors", flush=True)
        proposals, owners = draw_local_proposals(
            anchors, rng_for(p["seed"], 8, 202), p["proposal_count"], p["local_radius"]
        )
        proposal_truth = query_truth(
            mesh, proposals, p["data_workers"], "hand/proposals"
        )
        chosen, eligible, nearest = select_new_points(
            proposals,
            proposal_truth,
            model.rdf.center,
            model.rdf.radius,
            np.vstack([forbidden, mining]),
            p["new_count"],
        )
        points = proposals[chosen]
        targets = scan_targets(mesh, model.rdf, points, training, directory, dtype)
        weights = np.ones(len(points), dtype=np.float64)
        save_arrays(
            directory / "proposals.npz",
            points=proposals,
            truth=proposal_truth,
            anchor_indices=owners,
            eligible=eligible,
            chosen=chosen,
            nearest_forbidden=nearest,
        )
        save_arrays(
            directory / "training.npz",
            points=points,
            targets=targets,
            weights=weights,
            truth=proposal_truth[chosen],
        )
        offsets = np.r_[0, np.cumsum(p["batch_sizes"])]
        save_arrays(
            directory / "batch_indices.npz",
            indices=np.arange(5000, dtype=np.int64),
            offsets=offsets,
        )
        data = dict(
            anchor_count=len(anchors),
            eligible_proposals=int(eligible.sum()),
            new_count=5000,
            minimum_distance_from_old_points=float(nearest[chosen].min()),
            label_audit=summarize_labels(targets, proposal_truth[chosen]),
            target_dtype=str(dtype),
            scan_count=training.scan_count,
            scan_resolution=training.scan_resolution,
            normal_count=training.normal_count,
            label_method="mesh_to_sdf scan/normal",
            mesh_labels_used_for_training=False,
            new_weight=1.0,
            passes=1,
            old_training_points_replayed=0,
        )
        write_json(directory / "data.report.json", data)
    data_seconds = perf_counter() - began
    context = dict(
        link="hand",
        objective="weighted",
        protocol=p,
        parent_context=parent.metadata["context"],
        parent_metadata_sha256=p["parent_files"][
            "checkpoints/step_000800/metadata.json"
        ],
        training_sha256=file_sha256(directory / "training.npz"),
        sample_weights_sha256=file_sha256(directory / "training.npz"),
        batch_plan_sha256=file_sha256(directory / "batch_indices.npz"),
    )
    write_json(directory / "context.json", context)
    shutil.copyfile(
        study / "hand/checkpoints/step_000800/metadata.json",
        directory / "parent_metadata.json",
    )
    rdf = copy.deepcopy(model.rdf)
    compute = torch_training_environment("cuda")
    records = []

    def prefix(n):
        end = int(offsets[n - 800])
        return {
            **{
                "parent_" + k: v
                for k, v in parent.metadata["batch_prefix_sha256"].items()
            },
            "new_points": array_sha256(points[:end]),
            "new_targets": array_sha256(targets[:end]),
        }

    def weight_prefix(n):
        return {
            **{
                "parent_" + k: v
                for k, v in parent.metadata["sample_weight_prefix_sha256"].items()
            },
            "new": array_sha256(weights[: offsets[n - 800]]),
        }

    def checkpoint(n, seen, w, b):
        start = perf_counter()
        metadata = save_checkpoint(
            directory / "checkpoints" / f"step_{n:06d}",
            RidgeState(
                w.cpu().numpy(),
                b.cpu().numpy(),
                dict(
                    schema_version=1,
                    model=model_signature(rdf, p["regularization"]),
                    completed_steps=n,
                    consumed_samples=seen,
                    context=context,
                    compute=compute,
                    objective="weighted",
                    batch_prefix_sha256=prefix(n),
                    sample_weight_prefix_sha256=weight_prefix(n),
                ),
            ),
            rdf,
            p["regularization"],
        )
        records.append(
            dict(step=n, seconds=perf_counter() - start, sha256=metadata["sha256"])
        )
        print(f"CHECKPOINT hand {n}: samples={seen}", flush=True)

    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    print("FIT START hand: weighted parent800; append5000 once; no reset", flush=True)
    recursive_ridge_torch(
        rdf,
        (
            (points[a:b], targets[a:b], weights[a:b])
            for a, b in zip(offsets[:-1], offsets[1:])
        ),
        p["regularization"],
        weighted=True,
        allow_weighted_resume=True,
        resume=parent,
        checkpoint=checkpoint,
        checkpoint_interval=10000,
        progress=lambda n: print(f"FIT hand {n}/804", flush=True),
    )
    torch.cuda.synchronize()
    wall = perf_counter() - start
    cp_seconds = sum(r["seconds"] for r in records)
    peak = torch.cuda.max_memory_allocated()
    save_arrays(
        directory / "fit.npz",
        weights=rdf.weights,
        center=rdf.center,
        radius=np.array(rdf.radius),
    )
    final = load_checkpoint(
        directory / "checkpoints/step_000804", rdf, p["regularization"], context, prefix
    )
    require(final.metadata["consumed_samples"] == 1029000, "final samples")
    require(
        final.metadata["sample_weight_prefix_sha256"] == weight_prefix(804),
        "final weight lineage",
    )
    require(
        np.array_equal(
            (final.weights * rdf.radius).reshape(rdf.weights.shape), rdf.weights
        ),
        "804=fit",
    )
    del final
    fit = dict(
        compute=compute,
        data_seconds=data_seconds,
        fit_seconds=wall - cp_seconds,
        checkpoint_seconds=cp_seconds,
        checkpoints=records,
        peak_cuda_allocated_bytes=peak,
        context=context,
        fit_sha256=file_sha256(directory / "fit.npz"),
        parent_step=800,
        completed_steps=804,
        consumed_samples=1029000,
        new_observations=5000,
        new_sample_weight=p["new_weight"],
        effective_new_weight_sum=float(weights.sum()),
        data_reused=reuse_data is not None,
    )
    write_json(directory / "fit.report.json", fit)
    maximum, maximum_report = estimate_maximum(rdf, model.inner_radius, training)
    candidate = ExtendedRDF(rdf, model.inner_radius, maximum)
    results = {}
    for label, seed, purpose in [
        ("regression", baseline.metadata["settings"]["seed"], 1),
        ("fresh", 20260929, 102),
    ]:
        evidence = directory / (label + ".npz")
        report = validate_model(
            mesh, candidate, rng_for(seed, 8, purpose), 0.010, evidence_path=evidence
        )
        new, old = arrays(evidence), arrays(study / "hand" / (label + ".npz"))
        require(
            np.array_equal(new["points"], old["points"])
            and np.array_equal(new["truth"], old["truth"]),
            "archived validation changed",
        )
        report.update(accuracy_on_saved(rdf, evidence, 0.010))
        results[label] = dict(
            candidate=report,
            baseline=accuracy_on_saved(model.rdf, evidence, 0.010),
            seed=[seed, 8, purpose],
        )
    if weight1 is not None:
        for label in results:
            results[label]["weight1"] = accuracy_on_saved(
                weight1.models["hand"].rdf, directory / (label + ".npz"), 0.010
            )
        truth = arrays(directory / "training.npz")["truth"]
        training_comparison = {}
        for name, fit_model in [
            ("parent800", model.rdf),
            ("weight1", weight1.models["hand"].rdf),
            ("weight4", rdf),
        ]:
            prediction = fit_model.query(points, False)[0]
            error = prediction - truth
            training_comparison[name] = dict(
                max_abs_error=float(np.abs(error).max()),
                mean_abs_error=float(np.abs(error).mean()),
                max_overestimate=float(np.maximum(error, 0).max()),
                over_10mm_count=int((np.abs(error) > 0.010).sum()),
                mean_abs_label_residual=float(np.abs(prediction - targets).mean()),
            )
        write_json(directory / "training_comparison.json", training_comparison)
    final_report = copy.deepcopy(results["fresh"]["candidate"])
    final_report["fresh_and_regression_passed"] = all(
        r["candidate"]["passed"] for r in results.values()
    )
    final_report["passed"] = final_report["fresh_and_regression_passed"]
    final_report["reference_mesh_digest"] = bundle.metadata["reconstruction"]["hand"][
        "mesh_digest"
    ]
    bundle.models["hand"] = candidate
    bundle.metadata["validation"]["hand"] = final_report
    bundle.metadata["maxima"]["hand"] = maximum_report
    bundle.metadata["link_training"]["hand"] = fit
    result = dict(
        link="hand",
        complete=True,
        comparisons=results,
        data=data,
        fit=fit,
        maximum=maximum_report,
        passed=final_report["passed"],
        historical_baseline_compute=baseline.metadata["link_training"]["hand"][
            "compute"
        ],
    )
    write_json(directory / "comparison.json", result)
    print(f"LINK COMPLETE hand: passed={result['passed']}", flush=True)
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in [
        "source",
        "study",
        "inputs",
        "output",
        "protocol",
        "config",
        "runtime-model",
    ]:
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--reuse-data", type=Path)
    args = parser.parse_args()
    p = read(args.protocol)
    validate_protocol(p)
    for name in ["candidate.npz", "candidate.report.json"]:
        require(
            file_sha256(args.source / name) == p["baseline_files"][name],
            "latest baseline SHA",
        )
    before = file_sha256(args.runtime_model)
    require(before == p["runtime_model_sha256"], "runtime SHA")
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
    training = TrainingOptions(**baseline.metadata["training"]["baseline_options"])
    require(
        training.basis_count == 24 and training.regularization == 1e-4,
        "basis and regularization",
    )
    parent, forbidden, dtype = verify_parent(args.study, args.inputs, baseline, p)
    reweight = p["purpose"] == REWEIGHT_PURPOSE
    require(
        reweight == (args.reuse_data is not None),
        "reuse-data requires exact 4x protocol",
    )
    weight1 = None
    if reweight:
        verify_reuse(args.reuse_data, p, baseline.models["hand"].rdf, dtype)
        weight1 = read_candidate_for_validation(
            args.reuse_data / "candidate.npz", geometry, options
        )
        preserve(baseline, weight1)
    require(
        torch_training_environment("cuda")["gpu"] == "NVIDIA GeForce RTX 4070", "GPU"
    )
    require(not args.output.exists(), "output exists; never overwrite/restart")
    require(shutil.disk_usage(args.output.parent).free > 8 * 1024**3, "disk space")
    if args.preflight:
        print("PREFLIGHT PASS: weighted parent verified; no training", flush=True)
        return
    args.output.mkdir()
    started = perf_counter()
    status, failure = 1, None
    write_json(
        args.output / "job.json",
        dict(
            started_at=datetime.now(timezone.utc).isoformat(),
            protocol=p,
            source=str(args.source),
            study=str(args.study),
            inputs=str(args.inputs),
            runtime_model_sha256_before=before,
        ),
    )
    for src, dst in [
        (args.protocol, "protocol.json"),
        (args.source / "candidate.npz", "baseline.npz"),
        (args.source / "candidate.report.json", "baseline.report.json"),
    ]:
        shutil.copyfile(src, args.output / dst)
    bundle = copy.deepcopy(baseline)
    bundle.metadata["complete"] = False
    bundle.metadata["hand_continuation"] = dict(
        protocol=p,
        parent=str(args.study),
        preserved_links=[n for n in bundle.models if n != "hand"],
        original_audit_failure_preserved=True,
    )
    try:
        with threadpool_limits(limits=1):
            print("LINK START hand", flush=True)
            comparison = run_hand(
                args.study,
                args.output,
                baseline,
                bundle,
                p,
                training,
                parent,
                forbidden,
                dtype,
                args.reuse_data,
                weight1,
            )
            preserved = preserve(baseline, bundle)
            frames = set(bundle.models) | {
                f for c in config.safety.capsules for f in [c.start_frame, c.end_frame]
            }
            kin = PandaKinematics(config.model_path, config.ee_frame, frames)
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
            require(expected == approved, "loader verdict mismatch")
            for name, digest in p["parent_files"].items():
                require(
                    file_sha256(args.study / "hand" / name) == digest,
                    "parent was modified",
                )
            if reweight:
                for name, digest in p["reuse_files"].items():
                    require(
                        file_sha256(args.reuse_data / name) == digest,
                        "1x source was modified",
                    )
                preserve(weight1, bundle)
            write_json(
                args.output / "summary.json",
                dict(
                    complete=True,
                    comparison=comparison,
                    hand_passed=comparison["passed"],
                    passed=expected,
                    runtime_loader_approved=approved,
                    runtime_rejection=reason,
                    preserved_links=preserved,
                    joint_validation=bundle.metadata["joint_validation"],
                    wall_seconds=perf_counter() - started,
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
                type="RuntimeModelChanged", message="runtime SHA changed"
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
        with (args.output / "remote.sha256").open("x") as f:
            for path in sorted(args.output.rglob("*")):
                if path.is_file() and path.name != "remote.sha256":
                    f.write(f"{file_sha256(path)}  {path.relative_to(args.output)}\n")
    raise SystemExit(status)


if __name__ == "__main__":
    main()

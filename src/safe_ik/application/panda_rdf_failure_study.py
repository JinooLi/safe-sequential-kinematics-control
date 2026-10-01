"""Bounded, paired offline experiments on the three failed 10 mm RDF links.

No search loop, checkpoint resume, runtime installation or simulation. Original
training coordinates/batches and all six other models are immutable controls.
"""

import argparse
import copy
import gc
import json
import shutil
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

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
from ..domain.panda_rdf import BernsteinRDF, ExtendedRDF
from .panda_configuration import load_panda_config
from .panda_rdf_artifact import load_bundle, read_candidate_for_validation, save_bundle
from .panda_rdf_enriched_training import write_json
from .panda_rdf_training import (
    TrainingOptions,
    benchmark_candidate,
    estimate_maximum,
    uniform_ball,
    validate_joint_gradients,
    validate_model,
)
from .panda_rdf_validation import accuracy_masks, distance_accuracy_report
from .rdf_data_workers import data_execution, mesh_distance_evaluator

LINKS = ("link6", "link0", "hand")
POOL_KEYS = ("near", "random", "exterior")
SIZES = (512, 128, 640)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_arrays(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def rng_for(seed, index, purpose):
    return np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(index, purpose))
    )


def draw_mining_points(mesh, model, rng, count):
    require(type(count) is int and count >= 5 and count % 5 == 0, "mining count")
    global_points = uniform_ball(rng, count * 3 // 5, model.center, model.radius)
    surface, ids = trimesh.sample.sample_surface(mesh, count * 2 // 5, seed=rng)
    near = (
        surface + rng.uniform(-0.060, 0.060, (len(surface), 1)) * mesh.face_normals[ids]
    )
    return np.vstack((global_points, near))


def local_weights(points, labels, truth, anchors, radius, maximum, label_tolerance):
    """Compact spatial taper; uncertain labels stay at weight one, not removed."""
    points, anchors = np.asarray(points), np.asarray(anchors)
    labels, truth = np.asarray(labels), np.asarray(truth)
    require(points.ndim == 2 and points.shape[1] == 3, "weight points shape")
    require(anchors.ndim == 2 and anchors.shape[1] == 3, "anchor shape")
    require(labels.shape == truth.shape == (len(points),), "weight label shape")
    require(
        all(np.isfinite(a).all() for a in (points, anchors, labels, truth)),
        "weight finite",
    )
    require(
        np.isfinite([radius, maximum, label_tolerance]).all()
        and radius > 0
        and maximum > 1
        and label_tolerance >= 0,
        "weight parameters",
    )
    result = np.ones(len(points), dtype=np.float64)
    if len(anchors):
        distance = cKDTree(anchors).query(points, workers=1)[0]
        reliable = (
            (truth >= 0)
            & (truth <= 0.060)
            & (np.abs(labels - truth) <= label_tolerance)
        )
        result[reliable] += (maximum - 1) * np.maximum(
            1 - distance[reliable] / radius, 0
        ) ** 2
    return result


def query_truth(mesh, points, workers, description):
    result = []
    done, next_report = 0, 50000
    with mesh_distance_evaluator(mesh, workers) as query:
        for values in query.chunks(points, 64):
            result.append(values)
            done += len(values)
            if done >= next_report:
                print(f"DATA {description}: {done}/{len(points)}", flush=True)
                next_report += 50000
    return np.concatenate(result) if result else np.empty(0)


def summarize_labels(labels, truth):
    if not len(truth):
        return {"count": 0}
    error = np.abs(labels - truth)
    return dict(
        count=len(truth),
        mean_error=float(error.mean()),
        max_error=float(error.max()),
        above_1mm=int((error > 0.001).sum()),
        above_10mm=int((error > 0.010).sum()),
        exterior_negative_labels=int(((truth >= 0) & (labels < 0)).sum()),
    )


def accuracy_on_saved(model, path, margin):
    with np.load(path, allow_pickle=False) as a:
        points, truth = a["points"], a["truth"]
    predicted = model.query(points, False)[0]
    report = distance_accuracy_report(points, truth, predicted, margin)
    near = accuracy_masks(truth)[1]
    report["near_max_overestimate"] = float((predicted - truth)[near].max())
    report["near_above_margin_count"] = int(
        (np.abs(predicted - truth)[near] > margin).sum()
    )
    return report


def validate_protocol(protocol):
    require(
        protocol["schema"] == 1 and protocol["links"] == list(LINKS), "study coverage"
    )
    require(protocol["basis_count"] == 24 and protocol["updates"] == 800, "fixed fit")
    require(protocol["batch_sizes"] == list(SIZES), "fixed batches")
    require(protocol["checkpoint_steps"] == [400, 800], "fixed checkpoints")
    require(
        protocol["error_margin"] == 0.010 and protocol["pool_size"] == 500000,
        "fixed data/gate",
    )
    require(
        type(protocol["data_workers"]) is int and 1 <= protocol["data_workers"] <= 4,
        "workers",
    )
    weight = protocol["local_weighting"]
    require(
        weight
        == dict(
            mining_count=20000,
            threshold=0.008,
            radius=0.010,
            maximum=4.0,
            label_tolerance=0.001,
            pool="exterior",
        ),
        "weight protocol",
    )
    require(protocol["seed"] == 20260929, "new experiment seed")
    require(
        protocol["relabel"] == "all_pools_mesh_truth_preserve_label_dtype",
        "relabel protocol",
    )


def read_inputs(source, name, model, expected):
    root = source / "candidate.training" / name
    for filename, digest in expected.items():
        require(
            file_sha256(root / filename) == digest, f"{name}: input hash {filename}"
        )
    with np.load(root / "training_pool.npz", allow_pickle=False) as a:
        points = {key: a[f"{key}_points"] for key in ("near", "random")}
        labels = {key: a[f"{key}_labels"] for key in ("near", "random")}
    with np.load(root / "exterior_pool.npz", allow_pickle=False) as a:
        points["exterior"], labels["exterior"], truth = (
            a["points"],
            a["labels"],
            a["selection_truth"],
        )
    with np.load(root / "batch_indices.npz", allow_pickle=False) as a:
        indices = {key: a[key] for key in POOL_KEYS}
    plan = json.loads((root / "batch_plan.json").read_text())
    for key, size in zip(POOL_KEYS, SIZES):
        require(
            points[key].shape == (500000, 3) and labels[key].shape == (500000,),
            "pool shape",
        )
        require(
            np.isfinite(points[key]).all() and np.isfinite(labels[key]).all(),
            "pool finite",
        )
        require(
            np.all(
                np.linalg.norm(points[key] - model.center, axis=1)
                <= model.radius + 1e-12
            ),
            "pool domain",
        )
        ids = indices[key]
        require(
            ids.shape == (800, size)
            and ids.dtype.kind in "iu"
            and ids.min() >= 0
            and ids.max() < 500000,
            "batch shape/range",
        )
        require(array_sha256(ids) == plan["array_sha256"][key], "batch hash")
    require(
        np.isfinite(truth).all() and np.all((truth >= 0) & (truth <= 0.060)),
        "selection truth",
    )
    return root, points, labels, truth, indices


def run_link(source, output, name, baseline, bundle, protocol, training):
    index = bundle.metadata["links"].index(name)
    mesh = bundle.reconstructed[name]
    model = baseline.models[name]
    directory = output / name
    directory.mkdir(exist_ok=False)
    began = perf_counter()
    source_root, points, labels, truth, indices = read_inputs(
        source, name, model.rdf, protocol["input_sha256"][name]
    )
    for filename in protocol["input_sha256"][name]:
        shutil.copyfile(source_root / filename, directory / ("source_" + filename))
    targets = {key: value.copy() for key, value in labels.items()}
    weights = {key: np.ones(len(value)) for key, value in labels.items()}
    data_report = {"execution": data_execution(protocol["data_workers"])}
    weighted = name != "link6"
    if not weighted:
        # All points retained; only the label oracle changes. Keep label storage
        # precision equal to the original scan targets to avoid a second factor.
        exact = {}
        for key in POOL_KEYS:
            exact[key] = query_truth(
                mesh, points[key], protocol["data_workers"], f"{name}/{key}"
            )
            if "expected_mesh_label_array_sha256" in protocol:
                require(
                    array_sha256(exact[key])
                    == protocol["expected_mesh_label_array_sha256"][key],
                    f"{key}: deterministic mesh labels differ from independent audit",
                )
            if key == "exterior":
                require(
                    np.allclose(exact[key], truth, rtol=0, atol=1e-10),
                    "exterior reference changed",
                )
            data_report[key] = summarize_labels(labels[key], exact[key])
            targets[key] = exact[key].astype(labels[key].dtype)
        save_arrays(directory / "mesh_labels.npz", **exact)
    else:
        cfg = protocol["local_weighting"]
        mining = draw_mining_points(
            mesh, model.rdf, rng_for(protocol["seed"], index, 101), cfg["mining_count"]
        )
        mining_truth = query_truth(
            mesh, mining, protocol["data_workers"], f"{name}/mining"
        )
        prediction = model.rdf.query(mining, False)[0]
        selected = (
            (mining_truth >= 0)
            & (mining_truth <= 0.060)
            & (np.abs(prediction - mining_truth) >= cfg["threshold"])
            & (np.linalg.norm(mining - model.rdf.center, axis=1) <= model.rdf.radius)
        )
        anchors = mining[selected]
        require(
            len(anchors) > 0,
            f"{name}: no independent mining anchors; no automatic retuning",
        )
        save_arrays(
            directory / "mining.npz",
            points=mining,
            truth=mining_truth,
            baseline_prediction=prediction,
            selected=selected,
            anchors=anchors,
        )
        weights["exterior"] = local_weights(
            points["exterior"],
            labels["exterior"],
            truth,
            anchors,
            cfg["radius"],
            cfg["maximum"],
            cfg["label_tolerance"],
        )
        require(np.any(weights["exterior"] > 1), "no effective local weights")
        data_report.update(
            mining_seed=[protocol["seed"], index, 101],
            anchor_count=len(anchors),
            boosted_exterior_count=int((weights["exterior"] > 1).sum()),
            exterior_weight_sum=float(weights["exterior"].sum()),
            maximum_weight=float(weights["exterior"].max()),
            configuration=cfg,
        )
        # Audit potentially conflicting near/random labels in the same regions,
        # but do not change their targets or weights in this single-factor trial.
        tree = cKDTree(anchors)
        for key in POOL_KEYS:
            selected_ids = np.flatnonzero(
                tree.query(points[key], workers=1)[0] < cfg["radius"]
            )
            audit_truth = (
                truth[selected_ids]
                if key == "exterior"
                else query_truth(
                    mesh,
                    points[key][selected_ids],
                    protocol["data_workers"],
                    f"{name}/{key}_local_audit",
                )
            )
            save_arrays(
                directory / f"{key}_local_label_audit.npz",
                indices=selected_ids,
                truth=audit_truth,
            )
            data_report[key] = summarize_labels(labels[key][selected_ids], audit_truth)
    save_arrays(directory / "targets.npz", **targets)
    save_arrays(directory / "sample_weights.npz", **weights)
    write_json(directory / "data.report.json", data_report)
    data_seconds = perf_counter() - began
    objective = "weighted" if weighted else "unweighted"
    context = dict(
        link=name,
        source_candidate_sha256=protocol["source_candidate_sha256"],
        source_files=protocol["input_sha256"][name],
        protocol=protocol,
        reference_mesh_digest=bundle.metadata["reconstruction"][name]["mesh_digest"],
        training_options=asdict(training),
        objective=objective,
        targets_sha256=file_sha256(directory / "targets.npz"),
        sample_weights_sha256=file_sha256(directory / "sample_weights.npz"),
    )
    write_json(directory / "context.json", context)
    rdf = BernsteinRDF(
        model.rdf.center.copy(), model.rdf.radius, np.zeros_like(model.rdf.weights)
    )
    compute = torch_training_environment("cuda")
    import torch

    records = []

    def batches():
        for step in range(800):
            x = np.vstack([points[k][indices[k][step]] for k in POOL_KEYS])
            y = np.concatenate([targets[k][indices[k][step]] for k in POOL_KEYS])
            if weighted:
                yield x, y, np.concatenate(
                    [weights[k][indices[k][step]] for k in POOL_KEYS]
                )
            else:
                yield x, y

    prefix = lambda step: {k: array_sha256(ids[:step]) for k, ids in indices.items()}

    def checkpoint(step, seen, w, b):
        started = perf_counter()
        state = RidgeState(
            w.cpu().numpy(),
            b.cpu().numpy(),
            dict(
                schema_version=1,
                model=model_signature(rdf, training.regularization),
                completed_steps=step,
                consumed_samples=seen,
                context=context,
                compute=compute,
                objective=objective,
                batch_prefix_sha256=prefix(step),
                sample_weight_prefix_sha256={
                    k: array_sha256(weights[k][indices[k][:step]]) for k in POOL_KEYS
                },
            ),
        )
        meta = save_checkpoint(
            directory / "checkpoints" / f"step_{step:06d}",
            state,
            rdf,
            training.regularization,
        )
        records.append(
            dict(step=step, seconds=perf_counter() - started, sha256=meta["sha256"])
        )
        print(f"CHECKPOINT {name} {step}/800", flush=True)

    print(f"FIT START {name}: {objective}; fresh w=0, B=I/lambda", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    recursive_ridge_torch(
        rdf,
        batches(),
        training.regularization,
        weighted=weighted,
        checkpoint=checkpoint,
        checkpoint_interval=400,
        progress=lambda n: (
            print(f"FIT {name} {n}/800", flush=True) if n % 20 == 0 else None
        ),
    )
    torch.cuda.synchronize()
    wall = perf_counter() - started
    checkpoint_seconds = sum(r["seconds"] for r in records)
    save_arrays(
        directory / "fit.npz",
        weights=rdf.weights,
        center=rdf.center,
        radius=np.array(rdf.radius),
    )
    fit = dict(
        compute=compute,
        data_seconds=data_seconds,
        fit_seconds=wall - checkpoint_seconds,
        checkpoint_seconds=checkpoint_seconds,
        checkpoints=records,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        context=context,
        fit_sha256=file_sha256(directory / "fit.npz"),
    )
    write_json(directory / "fit.report.json", fit)
    # Verify stored states using read-only mmap before publishing completion.
    for step in (400, 800):
        state = load_checkpoint(
            directory / "checkpoints" / f"step_{step:06d}",
            rdf,
            training.regularization,
            context,
            prefix,
        )
        require(state.metadata["consumed_samples"] == step * 1280, "sample count")
        require(state.metadata["objective"] == objective, "objective state")
        require(
            state.metadata["sample_weight_prefix_sha256"]
            == {k: array_sha256(weights[k][indices[k][:step]]) for k in POOL_KEYS},
            "weight prefix",
        )
        if step == 800:
            require(
                np.array_equal(
                    (state.weights * rdf.radius).reshape(rdf.weights.shape), rdf.weights
                ),
                "final weights",
            )
        del state
    maximum, maximum_report = estimate_maximum(rdf, model.inner_radius, training)
    candidate = ExtendedRDF(rdf, model.inner_radius, maximum)
    results = {}
    for label, seed, purpose in (
        ("regression", baseline.metadata["settings"]["seed"], 1),
        ("fresh", protocol["seed"], 102),
    ):
        evidence = directory / f"{label}.npz"
        report = validate_model(
            mesh,
            candidate,
            rng_for(seed, index, purpose),
            0.010,
            evidence_path=evidence,
        )
        if label == "regression":
            with (
                np.load(evidence, allow_pickle=False) as a,
                np.load(source_root / "validation.npz", allow_pickle=False) as b,
            ):
                require(
                    np.array_equal(a["points"], b["points"]),
                    "original holdout regenerated differently",
                )
                require(
                    np.allclose(a["truth"], b["truth"], rtol=0, atol=1e-10),
                    "original truth changed",
                )
        report.update(accuracy_on_saved(rdf, evidence, 0.010))
        results[label] = dict(
            candidate=report,
            baseline=accuracy_on_saved(model.rdf, evidence, 0.010),
            seed=[seed, index, purpose],
        )
    final_report = copy.deepcopy(results["fresh"]["candidate"])
    final_report["fresh_and_regression_passed"] = all(
        r["candidate"]["passed"] for r in results.values()
    )
    final_report["passed"] = final_report["fresh_and_regression_passed"]
    final_report["reference_mesh_digest"] = bundle.metadata["reconstruction"][name][
        "mesh_digest"
    ]
    bundle.models[name] = candidate
    bundle.metadata["validation"][name] = final_report
    bundle.metadata["maxima"][name] = maximum_report
    bundle.metadata["link_training"][name] = fit
    result = dict(
        link=name,
        complete=True,
        comparisons=results,
        data=data_report,
        fit=fit,
        maximum=maximum_report,
        passed=final_report["passed"],
        historical_baseline_compute=baseline.metadata["link_training"][name]["compute"],
    )
    write_json(directory / "comparison.json", result)
    print(
        f"LINK COMPLETE {name}: old={results['regression']['candidate']['near_max_error']*1000:.6f}mm "
        f"fresh={results['fresh']['candidate']['near_max_error']*1000:.6f}mm passed={result['passed']}",
        flush=True,
    )
    del points, labels, weights, targets
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-model", type=Path, required=True)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="read-only checks; never create results or fit",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    require(
        file_sha256(args.source / "candidate.npz")
        == protocol["source_candidate_sha256"],
        "source model hash",
    )
    require(
        file_sha256(args.source / "candidate.report.json")
        == protocol["source_report_sha256"],
        "source report hash",
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
        "fixed safety margins",
    )
    geometry = extract_visual_geometry(config.model_path)
    baseline = read_candidate_for_validation(
        args.source / "candidate.npz", geometry, options
    )
    require(baseline.metadata["complete"] is True, "baseline incomplete")
    training = TrainingOptions(**baseline.metadata["training"]["baseline_options"])
    require(training.basis_count == 24 and training.updates == 800, "baseline training")
    torch_training_environment("cuda")
    require(
        shutil.disk_usage(args.output.parent).free > 15 * 1024**3, "need 15 GiB free"
    )
    frames = set(baseline.models) | {
        frame
        for capsule in config.safety.capsules
        for frame in (capsule.start_frame, capsule.end_frame)
    }
    kinematics = PandaKinematics(config.model_path, config.ee_frame, frames)
    require(
        not args.output.exists(), "output already exists; do not restart or overwrite"
    )
    if args.preflight:
        for name in LINKS:
            read_inputs(
                args.source,
                name,
                baseline.models[name].rdf,
                protocol["input_sha256"][name],
            )
        print(
            "PREFLIGHT PASS: fixed inputs, geometry, frames, GPU and space; no training",
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
            runtime_model_sha256_before=before,
        ),
    )
    shutil.copyfile(args.protocol, args.output / "protocol.json")
    shutil.copyfile(args.source / "candidate.npz", args.output / "baseline.npz")
    shutil.copyfile(
        args.source / "candidate.report.json", args.output / "baseline.report.json"
    )
    bundle = copy.deepcopy(baseline)
    bundle.metadata["complete"] = False
    bundle.metadata["failure_study"] = dict(
        protocol=protocol,
        parent=str(args.source),
        preserved_links=[n for n in bundle.models if n not in LINKS],
        purpose="offline paired comparison; no installation",
    )
    comparisons = {}
    try:
        with threadpool_limits(limits=1):
            for name in LINKS:
                print(f"LINK START {name}", flush=True)
                comparisons[name] = run_link(
                    args.source, args.output, name, baseline, bundle, protocol, training
                )
            for name in bundle.models:
                if name not in LINKS:
                    require(
                        np.array_equal(
                            bundle.models[name].rdf.weights,
                            baseline.models[name].rdf.weights,
                        ),
                        "preserved weights",
                    )
                    require(
                        bundle.models[name].surface_maximum
                        == baseline.models[name].surface_maximum,
                        "preserved maximum",
                    )
                require(
                    np.array_equal(bundle.queries[name], baseline.queries[name]),
                    "query points changed",
                )
            bundle.metadata["joint_validation"] = validate_joint_gradients(
                kinematics, bundle, options, rng_for(options.seed, 0, 3)
            )
            bundle.metadata["runtime_benchmark"] = benchmark_candidate(
                config, kinematics, bundle
            )
            bundle.metadata["complete"] = True
            bundle.metadata["wall_seconds"] = perf_counter() - started
            save_bundle(bundle, args.output / "candidate.npz")
            write_json(args.output / "candidate.report.json", bundle.metadata)
            expected = (
                all(r["passed"] for r in bundle.metadata["validation"].values())
                and bundle.metadata["joint_validation"]["passed"]
            )
            try:
                load_bundle(args.output / "candidate.npz", geometry, options)
                approved, reason = True, None
            except ValueError as exc:
                approved, reason = False, str(exc)
            require(approved == expected, "runtime loader/gate disagreement")
            write_json(
                args.output / "summary.json",
                dict(
                    complete=True,
                    comparisons=comparisons,
                    passed=expected,
                    runtime_loader_approved=approved,
                    runtime_rejection=reason,
                    joint_validation=bundle.metadata["joint_validation"],
                    preserved_links=bundle.metadata["failure_study"]["preserved_links"],
                    runtime_benchmark=bundle.metadata["runtime_benchmark"],
                    wall_seconds=perf_counter() - started,
                ),
            )
            status = 0 if expected else 2
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc))
        traceback.print_exc()
    finally:
        after = file_sha256(args.runtime_model)
        if before != after:
            failure = dict(
                type="RuntimeModelChanged", message="runtime model SHA changed"
            )
            status = 1
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
        # Immutable final inventory; monitoring logs are kept outside this tree.
        with (args.output / "remote.sha256").open("x") as stream:
            for path in sorted(args.output.rglob("*")):
                if path.is_file() and path.name != "remote.sha256":
                    stream.write(
                        f"{file_sha256(path)}  {path.relative_to(args.output)}\n"
                    )
    raise SystemExit(status)


if __name__ == "__main__":
    main()

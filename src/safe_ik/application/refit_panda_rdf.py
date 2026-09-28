"""Refit a single RDF link on a GPU using hashed, fixed CPU-experiment inputs.

All inputs are explicit and non-pickle. This diagnostic never approves a robot
artifact; CPU geometry validation and runtime failure policy are unchanged.
"""

import argparse
import copy
import json
import platform
import subprocess
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from ..adapters.rdf_torch_training import (
    recursive_ridge_torch,
    torch_training_environment,
)
from ..domain.panda_rdf import BernsteinRDF, ExtendedRDF
from .panda_rdf_artifact import RDFBundle, read_reconstructed_meshes, save_bundle
from .panda_rdf_training import (
    TrainingOptions,
    estimate_maximum,
    training_provenance,
    validate_model,
)
from .panda_rdf_validation import accuracy_masks, distance_accuracy_report
from .paths import default_resource
from .rdf_exterior_sampling import (
    ExteriorSamplingOptions,
    generate_exterior_pool,
    mix_exterior_batches,
)


def prepare_fixed_batches(
    near_ids,
    random_ids,
    reference_options,
    options,
    near_count,
    random_count,
    extra_batch_seed=0,
):
    """Keep the original prefix; append reproducible batches from the same pools.

    A separate random stream deliberately avoids pretending to restore the
    original RNG state after point-cloud sampling. No labels are regenerated.
    """
    previous = asdict(reference_options)
    current = asdict(options)
    variable = {"basis_count", "updates"}
    if {k: v for k, v in current.items() if k not in variable} != {
        k: v for k, v in previous.items() if k not in variable
    }:
        raise ValueError("only basis_count and updates may differ from fixed inputs")
    if options.updates < reference_options.updates:
        raise ValueError("updates must preserve the full original batch prefix")
    if type(extra_batch_seed) is not int or extra_batch_seed < 0:
        raise ValueError("extra_batch_seed must be a nonnegative integer")
    arrays = []
    for ids, width, count in (
        (near_ids, options.near_batch, near_count),
        (random_ids, options.random_batch, random_count),
    ):
        ids = np.asarray(ids)
        if (
            count != options.pool_size
            or ids.shape != (reference_options.updates, width)
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids >= count)
            or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0)
        ):
            raise ValueError("invalid fixed batch indices or pool size")
        arrays.append(ids.astype(np.int64, copy=True))
    added = options.updates - reference_options.updates
    if added:
        rng = np.random.default_rng(
            np.random.SeedSequence(extra_batch_seed, spawn_key=(4, 6))
        )
        extra_near, extra_random = [], []
        for _ in range(added):
            extra_near.append(rng.choice(near_count, options.near_batch, replace=False))
            extra_random.append(
                rng.choice(random_count, options.random_batch, replace=False)
            )
        arrays = [
            np.vstack((old, extra))
            for old, extra in zip(arrays, (extra_near, extra_random))
        ]
    plan = {
        "original_updates": reference_options.updates,
        "updates": options.updates,
        "appended_updates": added,
        "original_prefix_preserved": True,
        "extra_batch_seed": extra_batch_seed if added else None,
        "extra_batch_spawn_key": [4, 6] if added else None,
        "consumed_samples": options.updates
        * (options.near_batch + options.random_batch),
        "array_sha256": {
            name: sha256(np.ascontiguousarray(ids).tobytes()).hexdigest()
            for name, ids in zip(("near", "random"), arrays)
        },
        "unique_samples": {
            name: int(np.unique(ids).size)
            for name, ids in zip(("near", "random"), arrays)
        },
    }
    return *arrays, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--training-config", default=default_resource("configs/panda_rdf_training.yaml")
    )
    parser.add_argument("--extra-batch-seed", type=int, default=0)
    parser.add_argument("--exterior-sampling-config", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; select a new experiment directory")
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    required = (
        "shared_training_pool.npz",
        "shared_validation.npz",
        "link4_bp10.npz",
        "batch_indices.npz",
    )
    if set(manifest["sha256"]) != set(required) or manifest["link"] != "link4":
        raise ValueError("unexpected input manifest")
    for name in required:
        if (
            sha256((args.inputs / name).read_bytes()).hexdigest()
            != manifest["sha256"][name]
        ):
            raise ValueError(f"input hash mismatch: {name}")
    source = args.inputs / "link4_bp10.npz"
    reconstructed, source_meta = read_reconstructed_meshes(source)
    options = TrainingOptions(**yaml.safe_load(Path(args.training_config).read_text()))
    previous = TrainingOptions(**source_meta["training"]["options"])
    with np.load(source, allow_pickle=False) as data:
        for key, digest in source_meta["array_hashes"].items():
            if sha256(np.ascontiguousarray(data[key]).tobytes()).hexdigest() != digest:
                raise ValueError(f"corrupt reference artifact: {key}")
        center, queries = data["link4_center"].copy(), data["link4_queries"].copy()
        r0, r1, _ = data["link4_radii_max"]
        cpu_reference = BernsteinRDF(center, r1, data["link4_weights"].copy())
    with np.load(args.inputs / "shared_training_pool.npz", allow_pickle=False) as data:
        near, near_d, random, random_d = (
            data[key].copy()
            for key in ("near_points", "near_labels", "random_points", "random_labels")
        )
    with np.load(args.inputs / "batch_indices.npz", allow_pickle=False) as data:
        a, b = data["near"].copy(), data["random"].copy()
    a, b, batch_plan = prepare_fixed_batches(
        a, b, previous, options, len(near), len(random), args.extra_batch_seed
    )

    compute = torch_training_environment("cuda")
    import torch

    args.output.mkdir(parents=True, exist_ok=False)
    indices = {"near": a, "random": b}
    exterior_report = None
    effective_options = options
    if args.exterior_sampling_config is not None:
        exterior_options = ExteriorSamplingOptions(
            **yaml.safe_load(args.exterior_sampling_config.read_text())
        )
        indices, mixed_plan = mix_exterior_batches(a, b, exterior_options)
        batch_plan = {**mixed_plan, "baseline_plan": batch_plan}
        exterior_points, exterior_labels, selection_truth, exterior_report = (
            generate_exterior_pool(
                reconstructed["link4"],
                center,
                r1,
                options,
                exterior_options,
                progress=lambda message: print(message, flush=True),
            )
        )
        pool_path = args.output / "exterior_pool.npz"
        np.savez_compressed(
            pool_path,
            points=exterior_points,
            labels=exterior_labels,
            selection_truth=selection_truth,
        )
        exterior_report["pool_sha256"] = sha256(pool_path.read_bytes()).hexdigest()
        exterior_report["source_reference_mesh_digest"] = source_meta["reconstruction"][
            "link4"
        ]["mesh_digest"]
        (args.output / "exterior_pool.report.json").write_text(
            json.dumps(exterior_report, indent=2, allow_nan=False) + "\n"
        )
        effective_options = replace(
            options,
            near_batch=exterior_options.near_batch,
            random_batch=exterior_options.random_batch,
        )
        print("STAGE COMPLETE: exterior data ready; starting GPU fit", flush=True)

    def batches():
        for step, (ia, ib) in enumerate(zip(indices["near"], indices["random"])):
            blocks, labels = [near[ia], random[ib]], [near_d[ia], random_d[ib]]
            if exterior_report is not None:
                ic = indices["exterior"][step]
                blocks.append(exterior_points[ic])
                labels.append(exterior_labels[ic])
            yield np.vstack(blocks), np.concatenate(labels)

    # Warm up both matrix multiplication and Cholesky before timing the fit.
    warm = torch.eye(32, dtype=torch.float64, device="cuda")
    torch.linalg.cholesky(warm @ warm.T)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps(compute), flush=True)
    np.savez_compressed(args.output / "batch_indices.npz", **indices)
    (args.output / "batch_plan.json").write_text(
        json.dumps(batch_plan, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(batch_plan), flush=True)
    start = perf_counter()
    rdf = recursive_ridge_torch(
        BernsteinRDF(center, r1, np.zeros((options.basis_count,) * 3)),
        batches(),
        options.regularization,
        device="cuda",
        progress=lambda step: (
            print(f"GPU update {step}/{options.updates}", flush=True)
            if step % 20 == 0
            else None
        ),
    )
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    peak_bytes = torch.cuda.max_memory_allocated()
    maximum, maxima = estimate_maximum(rdf, r0, options)
    model = ExtendedRDF(rdf, r0, maximum)
    print("CPU triangle-distance and derivative validation", flush=True)
    report = validate_model(
        reconstructed["link4"],
        model,
        np.random.default_rng(
            np.random.SeedSequence(source_meta["training_seed"], spawn_key=(4, 1))
        ),
        source_meta["settings"]["error_margin"],
    )
    report["reference_mesh_digest"] = source_meta["reconstruction"]["link4"][
        "mesh_digest"
    ]
    with np.load(args.inputs / "shared_validation.npz", allow_pickle=False) as data:
        points, truth = (
            data["points"].copy(),
            data["truth"].copy(),
        )
    # Cached comparisons may contain the obsolete abs(truth) <= 60 mm mask.
    # Always reclassify from the independent mesh sign under the current policy.
    exterior, near_mask = accuracy_masks(truth)
    prediction, _ = rdf.query(points)
    cpu_prediction, _ = cpu_reference.query(points)
    accuracy = distance_accuracy_report(
        points, truth, prediction, source_meta["settings"]["error_margin"]
    )
    np.testing.assert_allclose(
        accuracy["near_max_error"],
        report["near_max_error"],
        atol=1e-10,
        rtol=0,
    )
    summary = {
        **accuracy,
        "link": "link4",
        "basis_count": options.basis_count,
        "degree": options.basis_count - 1,
        "updates": options.updates,
        "batch_plan": batch_plan,
        "compute": compute,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "fit_seconds": elapsed,
        "peak_cuda_allocated_bytes": peak_bytes,
        "derivatives_passed": report["derivatives_passed"],
        "passed": report["passed"],
        "cpu_reference_basis_count": cpu_reference.count,
        "prediction_max_difference_from_cpu": float(
            np.max(np.abs(prediction - cpu_prediction)[exterior])
        ),
        "runtime_approved": False,
        "reason": "single-link offline diagnostic",
        "inputs": manifest,
    }
    if exterior_report is not None:
        summary["exterior_sampling"] = exterior_report
    meta = copy.deepcopy(source_meta)
    for key in ("failure", "array_hashes", "wall_seconds", "comparison"):
        meta.pop(key, None)
    meta.update(
        {
            "complete": False,
            "training": training_provenance(effective_options),
            "training_compute": compute,
            "validation": {"link4": report},
            "maxima": {"link4": maxima},
            "joint_validation": {"passed": False, "reason": "single-link diagnostic"},
            "gpu_experiment": summary,
        }
    )
    if exterior_report is not None:
        meta["training"]["exterior_sampling"] = exterior_report
        meta["training"]["baseline_options"] = asdict(options)
    save_bundle(
        RDFBundle(
            {"link4": model},
            {"link4": queries},
            tuple(map(tuple, meta["pairs"])),
            meta,
            reconstructed={"link4": reconstructed["link4"]},
        ),
        args.output / "link4_gpu.npz",
    )
    (args.output / "link4_gpu.report.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        args.output / "validation.npz",
        points=points,
        truth=truth,
        exterior=exterior,
        near=near_mask,
        prediction=prediction,
        cpu_reference_prediction=cpu_prediction,
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

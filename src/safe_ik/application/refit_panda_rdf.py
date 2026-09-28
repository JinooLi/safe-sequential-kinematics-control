"""Refit a single RDF link on a GPU using hashed, fixed CPU-experiment inputs.

All inputs are explicit and non-pickle. This diagnostic never approves a robot
artifact; CPU geometry validation and runtime failure policy are unchanged.
"""

import argparse
import copy
import json
import platform
import subprocess
from dataclasses import asdict
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
from .paths import default_resource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--training-config", default=default_resource("configs/panda_rdf_training.yaml")
    )
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
    previous = source_meta["training"]["options"]
    if {k: v for k, v in asdict(options).items() if k != "basis_count"} != {
        k: v for k, v in previous.items() if k != "basis_count"
    }:
        raise ValueError("only basis_count may differ from the fixed-input experiment")
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
    for ids, width, count in (
        (a, options.near_batch, len(near)),
        (b, options.random_batch, len(random)),
    ):
        if (
            ids.shape != (options.updates, width)
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids >= count)
        ):
            raise ValueError("invalid fixed batch indices")

    def batches():
        for ia, ib in zip(a, b):
            yield np.vstack((near[ia], random[ib])), np.r_[near_d[ia], random_d[ib]]

    compute = torch_training_environment("cuda")
    import torch

    # Warm up both matrix multiplication and Cholesky before timing the fit.
    warm = torch.eye(32, dtype=torch.float64, device="cuda")
    torch.linalg.cholesky(warm @ warm.T)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps(compute), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
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
        points, truth, near_mask = (
            data["points"].copy(),
            data["truth"].copy(),
            data["near"].copy(),
        )
    prediction, _ = rdf.query(points)
    cpu_prediction, _ = cpu_reference.query(points)
    np.testing.assert_allclose(
        np.abs(prediction - truth)[near_mask].max(),
        report["near_max_error"],
        atol=1e-10,
        rtol=0,
    )
    summary = {
        "link": "link4",
        "basis_count": options.basis_count,
        "degree": options.basis_count - 1,
        "compute": compute,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "fit_seconds": elapsed,
        "peak_cuda_allocated_bytes": peak_bytes,
        "near_max_error": report["near_max_error"],
        "near_mean_error": float(np.abs(prediction - truth)[near_mask].mean()),
        "derivatives_passed": report["derivatives_passed"],
        "passed": report["passed"],
        "cpu_reference_basis_count": cpu_reference.count,
        "prediction_max_difference_from_cpu": float(
            np.max(np.abs(prediction - cpu_prediction))
        ),
        "runtime_approved": False,
        "reason": "single-link offline diagnostic",
        "inputs": manifest,
    }
    meta = copy.deepcopy(source_meta)
    for key in ("failure", "array_hashes", "wall_seconds", "comparison"):
        meta.pop(key, None)
    meta.update(
        {
            "complete": False,
            "training": training_provenance(options),
            "training_compute": compute,
            "validation": {"link4": report},
            "maxima": {"link4": maxima},
            "joint_validation": {"passed": False, "reason": "single-link diagnostic"},
            "gpu_experiment": summary,
        }
    )
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

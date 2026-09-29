"""Per-link exterior-enriched GPU fitting with reproducible inputs and RLS state.

No runtime artifact is approved here. The caller still performs complete mesh,
distance, extension and joint-gradient validation with the standard builder.
"""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from ..adapters.rdf_checkpoint import (
    RidgeState,
    array_sha256,
    file_sha256,
    model_signature,
    save_checkpoint,
)
from ..adapters.rdf_torch_training import (
    recursive_ridge_torch,
    torch_training_environment,
)
from ..domain.panda_rdf import BernsteinRDF
from .panda_rdf_training import training_pools
from .rdf_data_workers import data_execution, validate_data_workers
from .rdf_exterior_sampling import generate_exterior_pool, mix_exterior_batches


def write_json(path, value):
    """Publish new evidence only; previous experiments are never overwritten."""
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_batches(rng, training, exterior, link_index):
    near, random = [], []
    for _ in range(training.updates):
        near.append(rng.choice(training.pool_size, training.near_batch, replace=False))
        random.append(
            rng.choice(training.pool_size, training.random_batch, replace=False)
        )
    return mix_exterior_batches(near, random, exterior, link_index=link_index)


def train_enriched_link(
    mesh,
    center,
    radius,
    rng,
    training,
    exterior,
    *,
    link_index,
    directory,
    context,
    checkpoint_interval=400,
    progress=print,
    device="cuda",
    data_workers=1,
):
    """Create new pools at the new radius; do not relabel or resume older models.

    Explicit torch CPU is supported for tiny parity tests only. The CLI requires
    CUDA for this mode, and never silently falls back to CPU.
    """
    validate_data_workers(data_workers)
    if type(checkpoint_interval) is not int or checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    # Reject incompatible batch sizes before spending time generating labels.
    if (
        exterior.near_batch > training.near_batch
        or exterior.random_batch > training.random_batch
        or exterior.near_batch + exterior.random_batch + exterior.exterior_batch
        != training.near_batch + training.random_batch
    ):
        raise ValueError("exterior mixing must preserve baseline batch size")
    compute = torch_training_environment(device)
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    progress("generating near/random scan pools")
    near, near_d, random, random_d = training_pools(
        mesh, center, radius, rng, training, data_workers=data_workers
    )
    baseline_seconds = perf_counter() - started
    np.savez_compressed(
        directory / "training_pool.npz",
        near_points=near,
        near_labels=near_d,
        random_points=random,
        random_labels=random_d,
    )
    exterior_points, exterior_labels, truth, exterior_report = generate_exterior_pool(
        mesh,
        center,
        radius,
        training,
        exterior,
        progress,
        link_index=link_index,
        data_workers=data_workers,
    )
    np.savez_compressed(
        directory / "exterior_pool.npz",
        points=exterior_points,
        labels=exterior_labels,
        selection_truth=truth,
    )
    exterior_report["pool_sha256"] = file_sha256(directory / "exterior_pool.npz")
    write_json(directory / "exterior_pool.report.json", exterior_report)
    indices, plan = prepare_batches(rng, training, exterior, link_index)
    np.savez_compressed(directory / "batch_indices.npz", **indices)
    write_json(directory / "batch_plan.json", plan)
    context = {
        **context,
        "link_index": link_index,
        "training_options": {
            k: v for k, v in asdict(training).items() if k != "updates"
        },
        "exterior_options": asdict(exterior),
        "pool_sha256": {
            name: file_sha256(directory / name)
            for name in ("training_pool.npz", "exterior_pool.npz")
        },
    }
    write_json(directory / "context.json", context)
    progress("STAGE COMPLETE: training data ready")
    rdf = BernsteinRDF(center, radius, np.zeros((training.basis_count,) * 3))
    records = []

    def batches():
        for step in range(training.updates):
            a, b, c = (indices[key][step] for key in ("near", "random", "exterior"))
            yield (
                np.vstack((near[a], random[b], exterior_points[c])),
                np.concatenate((near_d[a], random_d[b], exterior_labels[c])),
            )

    def checkpoint(step, seen, weights, covariance):
        began = perf_counter()
        path = directory / "checkpoints" / f"step_{step:06d}"
        state = RidgeState(
            weights.cpu().numpy(),
            covariance.cpu().numpy(),
            {
                "schema_version": 1,
                "model": model_signature(rdf, training.regularization),
                "completed_steps": step,
                "consumed_samples": seen,
                "context": context,
                "compute": compute,
                "batch_prefix_sha256": {
                    key: array_sha256(ids[:step]) for key, ids in indices.items()
                },
            },
        )
        meta = save_checkpoint(path, state, rdf, training.regularization)
        records.append(
            {
                "step": step,
                "path": str(path.relative_to(directory)),
                "sha256": meta["sha256"],
                "seconds": perf_counter() - began,
            }
        )
        progress(f"Checkpoint complete: {step}/{training.updates}")

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    began = perf_counter()
    rdf = recursive_ridge_torch(
        rdf,
        batches(),
        training.regularization,
        device=device,
        checkpoint=checkpoint,
        checkpoint_interval=checkpoint_interval,
        progress=lambda step: (
            progress(f"GPU update {step}/{training.updates}")
            if step % 20 == 0
            else None
        ),
    )
    if device == "cuda":
        torch.cuda.synchronize()
    wall = perf_counter() - began
    checkpoint_seconds = sum(record["seconds"] for record in records)
    # Preserve coefficients before any lengthy CPU maximum/validation step.
    np.savez_compressed(
        directory / "fit.npz",
        weights=rdf.weights,
        center=rdf.center,
        radius=np.array(rdf.radius),
    )
    summary = {
        "training": asdict(training),
        "exterior_sampling": exterior_report,
        "compute": compute,
        "data_generation_execution": data_execution(data_workers),
        "batch_plan": plan,
        "context": context,
        "baseline_pool_seconds": baseline_seconds,
        "exterior_pool_seconds": exterior_report["generation_seconds"],
        "fit_seconds": wall - checkpoint_seconds,
        "fit_wall_seconds": wall,
        "checkpoint_seconds": checkpoint_seconds,
        "checkpoints": records,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device == "cuda" else None
        ),
        "fit_sha256": file_sha256(directory / "fit.npz"),
    }
    write_json(directory / "fit.report.json", summary)
    progress("STAGE COMPLETE: GPU fit and checkpoints saved; CPU validation next")
    return rdf, summary

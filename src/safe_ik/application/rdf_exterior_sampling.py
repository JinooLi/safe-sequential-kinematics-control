"""Offline exterior-band sampling ablation; no changes to RDF regression or CBFs."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import trimesh

from ..adapters.rdf_checkpoint import file_sha256
from .rdf_data_workers import (
    data_execution,
    mesh_distance_evaluator,
    scan_label_evaluator,
    validate_data_workers,
)


@dataclass(frozen=True)
class ExteriorSamplingOptions:
    pool_size: int = 500000
    near_batch: int = 512
    random_batch: int = 128
    exterior_batch: int = 640
    distance_max: float = 0.060
    seed: int = 0
    candidate_batch: int = 1024
    truth_batch: int = 64
    max_candidate_multiplier: int = 10

    def __post_init__(self):
        for key in (
            "pool_size",
            "near_batch",
            "random_batch",
            "exterior_batch",
            "candidate_batch",
            "truth_batch",
            "max_candidate_multiplier",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"exterior.{key} must be a positive integer")
        if self.exterior_batch > self.pool_size:
            raise ValueError("exterior pool must contain a full batch")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("exterior.seed must be a nonnegative integer")
        if not np.isfinite(self.distance_max) or self.distance_max <= 0:
            raise ValueError("exterior.distance_max must be finite and positive")


def sample_exterior_points(
    mesh, center, radius, options, progress=None, *, link_index=4, data_workers=1
):
    """Area-weighted faces + uniform outward offset, filtered by exact mesh sign.

    Uniform offset is not uniform true distance in concave regions. Rejections
    and the accepted true-distance histogram are recorded rather than hidden.
    """
    validate_data_workers(data_workers)
    with mesh_distance_evaluator(mesh, data_workers) as evaluator:
        return _sample_exterior_points(
            mesh, center, radius, options, progress, link_index, evaluator
        )


def _sample_exterior_points(
    mesh, center, radius, options, progress, link_index, evaluator
):
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("exterior sampling requires a closed, outward-oriented mesh")
    center = np.asarray(center, dtype=float)
    if (
        center.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(radius)
        or radius <= 0
    ):
        raise ValueError("invalid exterior training sphere")
    if type(link_index) is not int or link_index < 0:
        raise ValueError("link_index must be a nonnegative integer")
    rng = np.random.default_rng(
        np.random.SeedSequence(options.seed, spawn_key=(link_index, 7))
    )
    samples, distances = [], []
    accepted = drawn = outside_ball = outside_band = 0
    limit = options.pool_size * options.max_candidate_multiplier
    last_report = 0
    while accepted < options.pool_size:
        if drawn >= limit:
            raise ValueError("insufficient exterior acceptance within candidate budget")
        count = min(
            options.candidate_batch, options.pool_size - accepted, limit - drawn
        )
        surface, faces = trimesh.sample.sample_surface(mesh, count, seed=rng)
        points = (
            surface
            + rng.uniform(0, options.distance_max, (count, 1))
            * mesh.face_normals[faces]
        )
        drawn += count
        inside = np.linalg.norm(points - center, axis=1) <= radius
        outside_ball += int((~inside).sum())
        points = points[inside]
        if not len(points):
            continue
        truth = np.concatenate(list(evaluator.chunks(points, options.truth_batch)))
        if not np.isfinite(truth).all():
            raise ValueError("nonfinite exterior selection truth")
        valid = (truth >= 0) & (truth <= options.distance_max)
        outside_band += int((~valid).sum())
        if valid.any():
            samples.append(points[valid])
            distances.append(truth[valid])
            accepted += int(valid.sum())
        if progress is not None and (
            accepted - last_report >= 10000 or accepted == options.pool_size
        ):
            progress(
                f"Exterior selection {accepted}/{options.pool_size}; candidates={drawn}"
            )
            last_report = accepted
    points, truth = np.vstack(samples), np.concatenate(distances)
    edges = np.linspace(0, options.distance_max, 7)
    histogram, _ = np.histogram(truth, bins=edges)
    return (
        points,
        truth,
        {
            "generated_candidates": drawn,
            "accepted_count": accepted,
            "rejected_outside_sphere": outside_ball,
            "rejected_outside_band": outside_band,
            "truth_min": float(truth.min()),
            "truth_max": float(truth.max()),
            "distance_bin_edges": edges.tolist(),
            "distance_bin_counts": histogram.tolist(),
        },
    )


def generate_exterior_pool(
    mesh,
    center,
    radius,
    training,
    options,
    progress=print,
    *,
    link_index=4,
    data_workers=1,
):
    """Reuse the official scan/normal label method; mesh truth only selects points."""
    # Must be set before importing the renderer. This remains an offline-only dependency.
    import os

    validate_data_workers(data_workers)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from mesh_to_sdf import get_surface_point_cloud

    start = perf_counter()
    progress("Exterior data: generating and checking new points against mesh triangles")
    points, truth, report = sample_exterior_points(
        mesh,
        center,
        radius,
        options,
        progress,
        link_index=link_index,
        data_workers=data_workers,
    )
    progress("Exterior data: rendering the existing scan/normal label oracle")
    normalized = trimesh.Trimesh(
        (mesh.vertices - center) / radius, mesh.faces, process=False
    )
    cloud = get_surface_point_cloud(
        normalized,
        surface_point_method="scan",
        bounding_radius=1.0,
        scan_count=training.scan_count,
        scan_resolution=training.scan_resolution,
        calculate_normals=True,
    )
    labels = []
    last_report = 0
    with scan_label_evaluator(cloud, data_workers) as evaluator:
        for index, values in enumerate(
            evaluator.chunks((points - center) / radius, 2048, training.normal_count)
        ):
            labels.append(values * radius)
            count = min((index + 1) * 2048, len(points))
            if count - last_report >= 20000 or count == len(points):
                progress(f"Exterior labels {count}/{len(points)}")
                last_report = count
    labels = np.concatenate(labels)
    if labels.shape != truth.shape or not np.isfinite(labels).all():
        raise ValueError("invalid exterior scan labels")
    errors = np.abs(labels - truth)
    report.update(
        {
            "options": asdict(options),
            "point_seed_spawn_key": [link_index, 7],
            "proposal": "mesh-area sampling plus uniform outward normal offset",
            "selection_oracle": "negative trimesh exact triangle signed_distance",
            "training_label_oracle": "mesh_to_sdf scan/normal; same parameters and outer-sphere normalization",
            "reference_truth_used_as_training_labels": False,
            "scan_count": training.scan_count,
            "scan_resolution": training.scan_resolution,
            "normal_count": training.normal_count,
            "negative_label_count": int((labels < 0).sum()),
            "label_mean_absolute_error": float(errors.mean()),
            "label_max_absolute_error": float(errors.max()),
            "label_p95_absolute_error": float(np.quantile(errors, 0.95)),
            "generation_seconds": perf_counter() - start,
            "data_generation_execution": data_execution(data_workers),
        }
    )
    return points, labels, truth, report


def mix_exterior_batches(near, random, options, *, link_index=4):
    """Retain paired baseline subsets and replace the other half with exterior data."""
    near, random = np.asarray(near), np.asarray(random)
    if (
        near.ndim != 2
        or random.ndim != 2
        or len(near) != len(random)
        or not len(near)
        or near.dtype.kind not in "iu"
        or random.dtype.kind not in "iu"
        or np.any(near < 0)
        or np.any(random < 0)
        or options.near_batch > near.shape[1]
        or options.random_batch > random.shape[1]
        or options.near_batch + options.random_batch + options.exterior_batch
        != near.shape[1] + random.shape[1]
    ):
        raise ValueError("exterior mixing must preserve baseline batch size")
    if type(link_index) is not int or link_index < 0:
        raise ValueError("link_index must be a nonnegative integer")
    rng = np.random.default_rng(
        np.random.SeedSequence(options.seed, spawn_key=(link_index, 8))
    )
    indices = {
        "near": near[:, : options.near_batch].copy(),
        "random": random[:, : options.random_batch].copy(),
        "exterior": np.asarray(
            [
                rng.choice(options.pool_size, options.exterior_batch, replace=False)
                for _ in range(len(near))
            ]
        ),
    }
    report = {
        "updates": len(near),
        "full_baseline_batches_preserved": False,
        "retained_baseline_columns": {
            "near": options.near_batch,
            "random": options.random_batch,
        },
        "batch_sizes": {name: a.shape[1] for name, a in indices.items()},
        "exterior_batch_seed": options.seed,
        "exterior_batch_spawn_key": [link_index, 8],
        "consumed_samples": sum(a.size for a in indices.values()),
        "unique_samples": {name: int(np.unique(a).size) for name, a in indices.items()},
        "array_sha256": {
            name: sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
            for name, a in indices.items()
        },
    }
    return indices, report


def load_exterior_pool(root, manifest, mesh_digest, center, radius, training, options):
    """Reuse the exact pool bytes, never regenerate scan labels during a paired run."""
    root = Path(root)
    report = json.loads((root / "exterior_pool.report.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (
        summary.get("inputs") != manifest
        or report != summary.get("exterior_sampling")
        or report.get("source_reference_mesh_digest") != mesh_digest
        or report.get("options") != asdict(options)
        or report.get("reference_truth_used_as_training_labels") is not False
        or any(
            report.get(key) != getattr(training, key)
            for key in ("scan_count", "scan_resolution", "normal_count")
        )
    ):
        raise ValueError("reused exterior pool provenance mismatch")
    if file_sha256(root / "exterior_pool.npz") != report.get("pool_sha256"):
        raise ValueError("reused exterior pool checksum mismatch")
    with np.load(root / "exterior_pool.npz", allow_pickle=False) as data:
        points, labels, truth = (
            data[k].copy() for k in ("points", "labels", "selection_truth")
        )
    if (
        points.shape != (options.pool_size, 3)
        or labels.shape != (options.pool_size,)
        or truth.shape != labels.shape
        or not all(np.isfinite(x).all() for x in (points, labels, truth))
        or np.any((truth < 0) | (truth > options.distance_max))
        or np.any(np.linalg.norm(points - center, axis=1) > radius + 1e-12)
    ):
        raise ValueError("invalid reused exterior pool")
    return points, labels, truth, report

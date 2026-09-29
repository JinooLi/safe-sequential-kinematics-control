"""Offline continuation at completed link boundaries; never grants runtime approval."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..adapters.rdf_checkpoint import array_sha256, file_sha256, load_checkpoint
from ..domain.panda_rdf import BernsteinRDF, ExtendedRDF
from .panda_rdf_artifact import (
    SCHEMA,
    RDFBundle,
    geometry_settings,
    read_reconstructed_meshes,
)
from .panda_rdf_validation import accuracy_masks, distance_accuracy_report


def verify_link_evidence(directory, model, metadata, name, training, exterior):
    """Check saved inputs, full batch prefix, both states, and CPU predictions."""
    directory = Path(directory)
    fit = json.loads((directory / "fit.report.json").read_text())
    if fit != metadata["link_training"][name]:
        raise ValueError(f"{name}: fit report mismatch")
    context = json.loads((directory / "context.json").read_text())
    expected = {
        "link": name,
        "geometry_fingerprint": metadata["geometry_fingerprint"],
        "reference_mesh_digest": metadata["reconstruction"][name]["mesh_digest"],
        "settings": metadata["settings"],
        "sphere": metadata["spheres"][name],
        "training_options": {
            k: v for k, v in asdict(training).items() if k != "updates"
        },
        "exterior_options": asdict(exterior),
    }
    if (
        any(context.get(k) != v for k, v in expected.items())
        or context != fit["context"]
    ):
        raise ValueError(f"{name}: training context mismatch")
    for filename in ("training_pool.npz", "exterior_pool.npz"):
        if file_sha256(directory / filename) != context["pool_sha256"][filename]:
            raise ValueError(f"{name}: pool checksum mismatch")
    if file_sha256(directory / "fit.npz") != fit["fit_sha256"]:
        raise ValueError(f"{name}: fit checksum mismatch")
    plan = json.loads((directory / "batch_plan.json").read_text())
    if plan != fit["batch_plan"]:
        raise ValueError(f"{name}: batch plan mismatch")
    sizes = dict(
        near=exterior.near_batch,
        random=exterior.random_batch,
        exterior=exterior.exterior_batch,
    )
    with np.load(directory / "batch_indices.npz", allow_pickle=False) as archive:
        indices = {k: archive[k].copy() for k in archive.files}
    if set(indices) != set(sizes):
        raise ValueError(f"{name}: batch keys mismatch")
    for key, value in indices.items():
        count = exterior.pool_size if key == "exterior" else training.pool_size
        if (
            value.shape != (training.updates, sizes[key])
            or value.dtype.kind not in "iu"
            or value.min() < 0
            or value.max() >= count
            or array_sha256(value) != plan["array_sha256"][key]
        ):
            raise ValueError(f"{name}: invalid {key} batches")
    prefix = lambda step: {k: array_sha256(v[:step]) for k, v in indices.items()}
    checkpoints = fit["checkpoints"]
    steps = [r["step"] for r in checkpoints]
    if not steps or steps != sorted(set(steps)) or steps[-1] != training.updates:
        raise ValueError(f"{name}: incomplete checkpoint series")
    for record in checkpoints:
        path = directory / "checkpoints" / f"step_{record['step']:06d}"
        state = load_checkpoint(
            path, model.rdf, training.regularization, context, prefix
        )
        if (
            state.metadata["completed_steps"] != record["step"]
            or state.metadata["consumed_samples"]
            != record["step"] * sum(sizes.values())
            or state.metadata["sha256"] != record["sha256"]
        ):
            raise ValueError(f"{name}: checkpoint counter/manifest mismatch")
        if record["step"] == training.updates and not np.array_equal(
            (state.weights * model.rdf.radius).reshape(model.rdf.weights.shape),
            model.rdf.weights,
        ):
            raise ValueError(f"{name}: final checkpoint differs from candidate")
    with np.load(directory / "validation.npz", allow_pickle=False) as archive:
        points, truth, predicted = (
            archive[k] for k in ("points", "truth", "prediction")
        )
        if (
            points.shape != (5000, 3)
            or float(archive["error_margin"]) != metadata["settings"]["error_margin"]
        ):
            raise ValueError(f"{name}: validation size or margin mismatch")
        outside, near = accuracy_masks(truth)
        if not np.array_equal(archive["exterior"], outside) or not np.array_equal(
            archive["near"], near
        ):
            raise ValueError(f"{name}: validation mask mismatch")
        cpu = np.concatenate(
            [model.rdf.query(chunk)[0] for chunk in np.array_split(points, 50)]
        )
        if not np.allclose(cpu, predicted, rtol=0, atol=1e-10):
            raise ValueError(f"{name}: CPU predictions differ")
        accuracy = distance_accuracy_report(
            points, truth, predicted, metadata["settings"]["error_margin"]
        )
        report = metadata["validation"][name]
        if any(report.get(k) != v for k, v in accuracy.items()):
            raise ValueError(f"{name}: saved accuracy report differs")
    return dict(
        checkpoints=steps,
        validation_points=5000,
        cpu_max_difference=float(np.max(np.abs(cpu - predicted))),
        distance_passed=accuracy["distance_passed"],
    )


def read_completed_prefix(source, geometry, options, training, exterior):
    """Read only a nonempty, proper, completed prefix of the full link order.

    Distance failures are preserved, not changed into passes. Runtime load_bundle
    is intentionally NOT used or weakened by this offline-only recovery path.
    """
    source = Path(source)
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        names = tuple(metadata.get("links", ()))
        full = tuple(geometry.vertices)
        if not 0 < len(names) < len(full) or names != full[: len(names)]:
            raise ValueError("handoff requires a proper completed link prefix")
        if (
            metadata.get("schema") != SCHEMA
            or metadata.get("complete") is not False
            or "failure" in metadata
            or metadata.get("geometry_fingerprint") != geometry.fingerprint
            or tuple(map(tuple, metadata["pairs"])) != geometry.pairs
        ):
            raise ValueError("handoff schema, geometry, failure or completion mismatch")
        if metadata["settings"] != geometry_settings(options):
            raise ValueError("handoff settings mismatch")
        if metadata["training"].get("baseline_options") != asdict(training) or metadata[
            "training"
        ].get("exterior_sampling") != asdict(exterior):
            raise ValueError("handoff training options mismatch")
        if set(archive.files) != set(metadata["array_hashes"]) | {"metadata"}:
            raise ValueError("handoff array manifest incomplete")
        for key, digest in metadata["array_hashes"].items():
            value = archive[key]
            if not np.isfinite(value).all() or array_sha256(value) != digest:
                raise ValueError(f"corrupt handoff array {key}")
        models, queries = {}, {}
        for name in names:
            report = metadata["validation"][name]
            if (
                not isinstance(report.get("distance_passed"), bool)
                or not isinstance(report.get("derivatives_passed"), bool)
                or report.get("reference_mesh_digest")
                != metadata["reconstruction"][name]["mesh_digest"]
            ):
                raise ValueError(f"{name}: incomplete link validation")
            r0, r1, maximum = archive[f"{name}_radii_max"]
            sphere = metadata["spheres"][name]
            expected_r0 = sphere["mesh_radius"] + options.effective_margin
            expected_r1 = max(
                (1 + options.transition_ratio) * expected_r0,
                sphere["mesh_radius"] + 0.060,
            )
            if not np.allclose(
                [r0, r1], [expected_r0, expected_r1], rtol=0, atol=1e-12
            ):
                raise ValueError(f"{name}: handoff radii mismatch")
            rdf = BernsteinRDF(
                archive[f"{name}_center"], r1, archive[f"{name}_weights"]
            )
            if rdf.count != training.basis_count:
                raise ValueError(f"{name}: basis count mismatch")
            models[name] = ExtendedRDF(rdf, r0, maximum)
            queries[name] = archive[f"{name}_queries"].copy()
            if queries[name].shape != (
                int(np.ceil(options.query_density * 4 * np.pi * r0**2)),
                3,
            ):
                raise ValueError(f"{name}: handoff queries mismatch")
    reconstructed, _ = read_reconstructed_meshes(source)
    audit = {}
    for index, name in enumerate(names):
        if metadata["link_training"][name]["context"]["link_index"] != index:
            raise ValueError("handoff link random-stream index mismatch")
        audit[name] = verify_link_evidence(
            source.with_suffix(".training") / name,
            models[name],
            metadata,
            name,
            training,
            exterior,
        )
    for key in ("spheres", "maxima", "validation", "reconstruction", "link_training"):
        metadata[key] = {name: metadata[key][name] for name in names}
    metadata.pop("runtime_benchmark", None)
    metadata["joint_validation"] = {"passed": False}
    return (
        RDFBundle(
            models,
            queries,
            geometry.pairs,
            metadata,
            reconstructed={name: reconstructed[name] for name in names},
        ),
        audit,
    )


def copy_prefix_evidence(source, destination, names):
    """Keep all prior pools and matrices; never overwrite an existing link."""
    for name in names:
        shutil.copytree(
            Path(source).with_suffix(".training") / name, Path(destination) / name
        )

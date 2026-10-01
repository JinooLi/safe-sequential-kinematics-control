"""Append surface queries to an approved RDF without fitting or deployment."""

import argparse
import copy
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from ..adapters.panda_kinematics import PandaKinematics
from ..adapters.panda_rdf_geometry import extract_visual_geometry
from ..adapters.rdf_checkpoint import file_sha256
from .panda_configuration import load_panda_config
from .panda_rdf_artifact import load_bundle, read_reconstructed_meshes, save_bundle
from .panda_rdf_training import validate_joint_gradients

QUERY_PURPOSE = 202
SURFACE_TOLERANCE = 1e-9


def append_surface_queries(mesh, original, count, rng):
    """Greedy farthest-point additions, seeded by ALL original queries.

    Candidates are area-weighted samples of the visual mesh, not the enclosing
    sphere. Density still uses the existing ceil(density * 4*pi*r0**2) contract.
    This is finite sampled coverage, not a continuous collision certificate.
    """
    original = np.asarray(original)
    if (
        original.ndim != 2
        or original.shape[1] != 3
        or len(original) == 0
        or not np.isfinite(original).all()
    ):
        raise ValueError("original queries must be finite nonempty (N, 3) points")
    if type(count) is not int or count < len(original):
        raise ValueError("query count must preserve all original points")
    if count == len(original):
        return original.copy()
    candidates, _ = trimesh.sample.sample_surface(mesh, max(30 * count, 1000), seed=rng)
    distances = cKDTree(original).query(candidates)[0] ** 2
    selected = []
    for _ in range(count - len(original)):
        index = int(np.argmax(distances))
        if not np.isfinite(distances[index]) or distances[index] <= 1e-24:
            raise ValueError("surface candidate pool cannot supply distinct points")
        selected.append(index)
        delta = candidates - candidates[index]
        distances = np.minimum(distances, np.einsum("ij,ij->i", delta, delta))
    return np.vstack((original, candidates[selected]))


def surface_residual(mesh, points):
    """Certify membership using triangle barycentrics, not a signed oracle.

    A closest-point routine's absolute edge-region tolerances can move an
    exactly sampled point to an edge on tiny visual triangles. Check actual
    containing triangles with a scale-normalized least-squares projection;
    neither the mesh, the queries nor the learned distance oracle is altered.
    """
    maximum = 0.0
    for point in points:
        ids = mesh.triangles_tree.intersection(
            np.r_[point - SURFACE_TOLERANCE, point + SURFACE_TOLERANCE]
        )
        residual = np.inf
        for index in ids:
            triangle = mesh.triangles[index]
            edges = (triangle[1:] - triangle[0]).T
            scale = np.linalg.norm(edges, axis=0).max()
            if scale == 0:
                continue
            uv, _, rank, _ = np.linalg.lstsq(
                edges / scale, (point - triangle[0]) / scale, rcond=None
            )
            barycentric = np.r_[1 - uv.sum(), uv]
            if rank != 2 or np.any(barycentric < -1e-10):
                continue
            distance = np.linalg.norm(triangle[0] + edges @ uv - point)
            residual = min(residual, float(distance))
        if not np.isfinite(residual) or residual > SURFACE_TOLERANCE:
            raise ValueError(f"queries are not on the visual surface: {residual}")
        maximum = max(maximum, residual)
    return maximum


def verify_preserved_arrays(source, candidate):
    """Only query arrays may change, and their complete prefixes must survive."""
    with (
        np.load(source, allow_pickle=False) as old,
        np.load(candidate, allow_pickle=False) as new,
    ):
        if set(old.files) != set(new.files):
            raise ValueError("artifact array coverage changed")
        for key in old.files:
            if key == "metadata":
                continue
            before, after = old[key], new[key]
            if key.endswith("_queries"):
                after = after[: len(before)]
            if before.dtype != after.dtype or not np.array_equal(before, after):
                raise ValueError(f"original model/geometry/query prefix changed: {key}")


def augment(config, source, output, *, density=None, threads=1):
    """Publish three NEW files only after source and updated loader checks pass."""
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.suffix != ".npz":
        raise ValueError("output must be a new .npz file")
    report = output.with_suffix(".report.json")
    audit = output.with_suffix(".augmentation.json")
    if any(p.exists() or p.is_symlink() for p in (output, report, audit)):
        raise ValueError("output model/report/audit already exists; never overwrite")
    source_report = source.with_suffix(".report.json")
    tracked = {source: file_sha256(source)}
    if source_report.exists():
        tracked[source_report] = file_sha256(source_report)
    runtime = config.safety.rdf.artifact.resolve()
    if runtime.exists():
        tracked[runtime] = file_sha256(runtime)
    options = replace(
        config.safety.rdf,
        query_density=config.safety.rdf.query_density if density is None else density,
    )
    with np.load(source, allow_pickle=False) as archive:
        original_meta = json.loads(str(archive["metadata"]))
    old_density = original_meta["settings"]["query_density"]
    if options.query_density <= old_density:
        raise ValueError("new density must exceed the source density")
    geometry = extract_visual_geometry(config.model_path)
    # Permit ONLY the density difference. Runtime margin reductions remain
    # runtime-only; never rewrite the trained radii/settings using that margin.
    bundle = load_bundle(source, geometry, replace(options, query_density=old_density))
    if (
        source_report.exists()
        and json.loads(source_report.read_text()) != original_meta
    ):
        raise ValueError("source sidecar does not match embedded metadata")
    bundle.reconstructed, _ = read_reconstructed_meshes(source)
    meta = bundle.metadata
    old_joint_validation = copy.deepcopy(meta["joint_validation"])
    meta["complete"] = False
    meta["settings"]["query_density"] = options.query_density
    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": tracked[source],
        "source_density_m_minus_2": old_density,
        "query_density_m_minus_2": options.query_density,
        "count_rule": "ceil(query_density * 4*pi*stored_inner_radius**2)",
        "algorithm": "visual_surface_seeded_farthest_point_append_v1",
        "candidate_count_rule": "max(30*target_count,1000)",
        "seed_stream": f"SeedSequence({options.seed}, spawn_key=(link_index,{QUERY_PURPOSE}))",
        "training_performed": False,
        "new_holdout_sampling": False,
        "distance_validation": "unchanged approved fields and stored holdout reports",
        "runtime_installed": False,
        "continuous_error_certificate": False,
        "runtime_physical_margin": options.physical_margin,
        "links": {},
    }
    started = perf_counter()
    with threadpool_limits(limits=threads):
        for index, (name, model) in enumerate(bundle.models.items()):
            t0 = perf_counter()
            original = bundle.queries[name]
            count = int(
                np.ceil(options.query_density * 4 * np.pi * model.inner_radius**2)
            )
            mesh = trimesh.Trimesh(
                geometry.vertices[name], geometry.faces[name], process=False
            )
            points = append_surface_queries(
                mesh,
                original,
                count,
                np.random.default_rng(
                    np.random.SeedSequence(
                        options.seed, spawn_key=(index, QUERY_PURPOSE)
                    )
                ),
            )
            if not np.array_equal(points[: len(original)], original):
                raise ValueError(f"{name}: old query prefix changed")
            residual = surface_residual(mesh, points)
            bundle.queries[name] = points
            summary["links"][name] = {
                "old_count": len(original),
                "new_count": len(points),
                "added_count": len(points) - len(original),
                "original_prefix_preserved": True,
                "visual_surface_max_residual_m": residual,
                "generation_and_surface_check_seconds": perf_counter() - t0,
            }
            print(
                f"QUERIES {name}: {len(original)} -> {len(points)}; original prefix preserved",
                flush=True,
            )
        print(
            "CHECK joint gradients with the new queries (no IK search or simulation)",
            flush=True,
        )
        kin = PandaKinematics(config.model_path, config.ee_frame, geometry.vertices)
        meta["joint_validation"] = validate_joint_gradients(
            kin,
            bundle,
            options,
            np.random.default_rng(
                np.random.SeedSequence(options.seed, spawn_key=(0, 3))
            ),
        )
    if meta["joint_validation"]["passed"] is not True:
        raise ValueError(
            f"new query joint validation failed: {meta['joint_validation']}"
        )
    meta.setdefault("query_augmentation_history", []).append(
        {
            **copy.deepcopy(summary),
            "previous_joint_validation": old_joint_validation,
            "previous_settings": original_meta["settings"],
        }
    )
    meta["complete"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=output.parent, prefix=".query-augmentation-"
    ) as staging:
        staged = Path(staging) / "candidate.npz"
        digest = save_bundle(bundle, staged)
        verify_preserved_arrays(source, staged)
        # Compare the serialized representations (writer pairs are tuples;
        # artifact JSON pairs are lists), including the actual saved hashes.
        with np.load(staged, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
        # Preserve every pre-existing metadata field, except the documented
        # density/query-hash/joint-check changes and append-only provenance.
        for key, value in original_meta.items():
            if (
                key
                not in {
                    "settings",
                    "array_hashes",
                    "joint_validation",
                    "query_augmentation_history",
                }
                and meta[key] != value
            ):
                raise ValueError(f"preserved metadata changed: {key}")
        for key, value in original_meta["settings"].items():
            if key != "query_density" and meta["settings"][key] != value:
                raise ValueError(f"training setting changed: {key}")
        for key, value in original_meta["array_hashes"].items():
            if not key.endswith("_queries") and meta["array_hashes"][key] != value:
                raise ValueError(f"non-query array hash changed: {key}")
        load_bundle(staged, geometry, options)
        trained_options = replace(
            options, physical_margin=meta["settings"]["physical_margin"]
        )
        load_bundle(staged, geometry, trained_options)
        if any(file_sha256(p) != digest_before for p, digest_before in tracked.items()):
            raise ValueError("source/report/runtime changed during augmentation")
        summary.update(
            complete=True,
            output=str(output),
            output_sha256=digest,
            original_total=sum(v["old_count"] for v in summary["links"].values()),
            new_total=sum(v["new_count"] for v in summary["links"].values()),
            source_and_runtime_unchanged=True,
            protected_files={str(p): h for p, h in tracked.items()},
            all_non_query_arrays_unchanged=True,
            distance_validation_reports_unchanged=True,
            joint_validation=meta["joint_validation"],
            loader_approved=True,
            seconds=perf_counter() - started,
        )
        staged_report = Path(staging) / "candidate.report.json"
        staged_report.write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
        staged_audit = Path(staging) / "augmentation.json"
        staged_audit.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        # Exclusive publication: link() fails if a competing output exists.
        # The model is published last so partial publication is never runnable.
        os.link(staged_report, report)
        os.link(staged_audit, audit)
        os.link(staged, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--density", type=float)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    summary = augment(
        load_panda_config(args.config),
        args.source,
        args.output,
        density=args.density,
        threads=args.threads,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

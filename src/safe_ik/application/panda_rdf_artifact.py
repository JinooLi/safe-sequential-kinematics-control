"""Versioned, non-pickle RDF bundles with fail-closed compatibility checks."""

import json
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import numpy as np

from ..domain.panda_rdf import BernsteinRDF, ExtendedRDF

SCHEMA = 2
REFERENCE = "voxel_reconstructed_mesh"
REGENERATE = "Regenerate with: safe-ik-train-panda-rdf --config configs/panda.yaml"


@dataclass
class RDFBundle:
    models: dict
    queries: dict
    pairs: tuple
    metadata: dict
    identifier: str = ""
    reconstructed: dict = field(default_factory=dict)


def geometry_settings(options):
    # top-k, soft-min and CBF gain affect runtime only; never require retraining.
    return {
        key: getattr(options, key)
        for key in (
            "physical_margin",
            "error_margin",
            "transition_ratio",
            "query_density",
            "seed",
        )
    }


def save_bundle(bundle, path):
    arrays = {}
    for name, mesh in bundle.reconstructed.items():
        arrays[f"{name}_mesh_vertices"] = np.asarray(mesh.vertices, dtype="<f8")
        arrays[f"{name}_mesh_faces"] = np.asarray(mesh.faces, dtype="<i8")
    for name, model in bundle.models.items():
        arrays[f"{name}_weights"] = model.rdf.weights
        arrays[f"{name}_center"] = model.rdf.center
        arrays[f"{name}_radii_max"] = np.array(
            [model.inner_radius, model.rdf.radius, model.surface_maximum]
        )
        arrays[f"{name}_queries"] = bundle.queries[name]
    metadata = dict(
        bundle.metadata, schema=SCHEMA, links=list(bundle.models), pairs=bundle.pairs
    )
    metadata["array_hashes"] = {
        k: sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
        for k, v in arrays.items()
    }
    bundle.metadata.update(metadata)
    arrays["metadata"] = np.array(json.dumps(metadata, sort_keys=True, allow_nan=False))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A failed build is still inspectable, but the loader will reject it.
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)
    return sha256(path.read_bytes()).hexdigest()


def _check_reference(meta, names):
    if meta.get("validation_reference") != REFERENCE:
        raise ValueError(
            "reconstructed validation reference missing; retrain explicitly"
        )
    for name in names:
        reconstruction = meta["reconstruction"][name]
        if (
            reconstruction.get("reference") != REFERENCE
            or reconstruction.get("integrity_passed") is not True
            or not reconstruction.get("mesh_digest")
            or meta["validation"][name].get("reference_mesh_digest")
            != reconstruction["mesh_digest"]
        ):
            raise ValueError(f"{name}: reconstructed validation reference mismatch")


def load_bundle(path, geometry, options):
    try:
        path = Path(path)
        payload = path.read_bytes()
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            if meta["schema"] != SCHEMA:
                raise ValueError("artifact schema")
            if meta.get("complete") is not True or "failure" in meta:
                raise ValueError("artifact incomplete or build failed")
            if meta["geometry_fingerprint"] != geometry.fingerprint:
                raise ValueError(
                    "visual geometry, closed fingers or kinematics mismatch"
                )
            settings = geometry_settings(options)
            mismatch = [k for k, v in settings.items() if meta["settings"].get(k) != v]
            if mismatch:
                raise ValueError("settings mismatch: " + ", ".join(mismatch))
            names = tuple(geometry.vertices)
            _check_reference(meta, names)
            if (
                tuple(meta["links"]) != names
                or tuple(map(tuple, meta["pairs"])) != geometry.pairs
            ):
                raise ValueError("link coverage or adjacency pairs mismatch")
            joint_check = meta["joint_validation"]
            if joint_check.get("passed") is not True:
                raise ValueError(
                    f"relative Jacobian / joint gradient validation: {joint_check}"
                )
            models, queries = {}, {}
            for name in names:
                report = meta["validation"][name]
                if (
                    report.get("passed") is not True
                    or report["sample_counts"] != [3000, 2000]
                    or not np.isfinite(report["near_max_error"])
                    or report["near_max_error"] > options.error_margin
                    or report.get("derivatives_passed") is not True
                ):
                    raise ValueError(f"{name}: validation failed: {report}")
                for suffix in (
                    "weights",
                    "center",
                    "radii_max",
                    "queries",
                    "mesh_vertices",
                    "mesh_faces",
                ):
                    key = f"{name}_{suffix}"
                    value = archive[key]
                    if (
                        not np.isfinite(value).all()
                        or sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                        != meta["array_hashes"][key]
                    ):
                        raise ValueError(f"{name}: corrupt {suffix}")
                r0, r1, maximum = archive[f"{name}_radii_max"]
                mesh_radius = meta["spheres"][name]["mesh_radius"]
                expected_r0 = mesh_radius + options.effective_margin
                expected_r1 = max(
                    (1 + options.transition_ratio) * expected_r0, mesh_radius + 0.060
                )
                if not np.allclose(
                    [r0, r1], [expected_r0, expected_r1], atol=1e-12, rtol=0
                ):
                    raise ValueError(f"{name}: radii mismatch")
                rdf = BernsteinRDF(
                    archive[f"{name}_center"], r1, archive[f"{name}_weights"]
                )
                models[name] = ExtendedRDF(rdf, r0, maximum)
                queries[name] = archive[f"{name}_queries"].copy()
                expected_count = int(np.ceil(options.query_density * 4 * np.pi * r0**2))
                if queries[name].shape != (expected_count, 3):
                    raise ValueError(f"{name}: query count mismatch")
        return RDFBundle(
            models, queries, geometry.pairs, meta, sha256(payload).hexdigest()
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        EOFError,
        BadZipFile,
    ) as exc:
        raise ValueError(f"Panda RDF unavailable: {exc}. {REGENERATE}") from exc


def read_candidate_for_validation(path, geometry, options, *, refresh_queries=False):
    """Offline-only read of a failed candidate; does NOT authorize control use."""
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        if (
            meta["schema"] != SCHEMA
            or meta["geometry_fingerprint"] != geometry.fingerprint
        ):
            raise ValueError(
                "candidate schema or geometry mismatch; retrain explicitly"
            )
        settings = geometry_settings(options)
        ignored = {"query_density", "seed"} if refresh_queries else set()
        if any(
            meta["settings"].get(k) != v
            for k, v in settings.items()
            if k not in ignored
        ):
            raise ValueError("candidate settings mismatch; regenerate explicitly")
        if (
            tuple(meta["links"]) != tuple(geometry.vertices)
            or tuple(map(tuple, meta["pairs"])) != geometry.pairs
        ):
            raise ValueError("revalidation requires a complete set of link models")
        _check_reference(meta, meta["links"])
        for key, digest in meta["array_hashes"].items():
            if (
                sha256(np.ascontiguousarray(archive[key]).tobytes()).hexdigest()
                != digest
            ):
                raise ValueError(f"corrupt candidate array {key}")
        models, queries = {}, {}
        for name in meta["links"]:
            r0, r1, maximum = archive[f"{name}_radii_max"]
            models[name] = ExtendedRDF(
                BernsteinRDF(archive[f"{name}_center"], r1, archive[f"{name}_weights"]),
                r0,
                maximum,
            )
            queries[name] = archive[f"{name}_queries"].copy()
    meta.pop("failure", None)
    meta.setdefault("training_seed", meta["settings"]["seed"])
    meta.setdefault("training_wall_seconds", meta.get("wall_seconds"))
    if refresh_queries:
        meta["settings"] = settings
    reconstructed, _ = read_reconstructed_meshes(path)
    return RDFBundle(models, queries, geometry.pairs, meta, reconstructed=reconstructed)


def read_reconstructed_meshes(path):
    """Offline inspection/revalidation, including incomplete or failed candidates.

    Never used as permission to run the robot. Recover the exact reference arrays,
    not a fresh reconstruction or a lossy STL round trip.
    """
    import trimesh

    from .panda_mesh_reconstruction import mesh_digest, require_valid_reconstruction

    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        if (
            meta.get("schema") != SCHEMA
            or meta.get("validation_reference") != REFERENCE
        ):
            raise ValueError(
                "artifact has no reconstructed reference; retrain explicitly"
            )
        meshes = {}
        for name, report in meta["reconstruction"].items():
            arrays = []
            for suffix in ("mesh_vertices", "mesh_faces"):
                key = f"{name}_{suffix}"
                array = archive[key]
                if (
                    sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                    != meta["array_hashes"][key]
                ):
                    raise ValueError(f"{name}: corrupt reconstructed mesh")
                arrays.append(array)
            mesh = trimesh.Trimesh(*arrays, process=False)
            require_valid_reconstruction(mesh)
            if mesh_digest(mesh) != report["mesh_digest"]:
                raise ValueError(f"{name}: reconstructed geometry digest mismatch")
            meshes[name] = mesh
    return meshes, meta

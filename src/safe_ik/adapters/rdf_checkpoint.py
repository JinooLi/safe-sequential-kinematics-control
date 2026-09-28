"""Versioned, non-pickle RLS snapshots; published only after all files are durable.

The normalized weights and inverse-information matrix B are both necessary to
resume Algorithm 1. Runtime RDF artifacts intentionally remain separate.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    array = np.ascontiguousarray(array)
    view = memoryview(array).cast("B")
    digest = sha256()
    for start in range(0, len(view), 8 * 1024 * 1024):
        digest.update(view[start : start + 8 * 1024 * 1024])
    return digest.hexdigest()


def model_signature(model, regularization):
    return {
        "basis_count": model.count,
        "center": model.center.tolist(),
        "radius": float(model.radius),
        "regularization": float(regularization),
        "dtype": "float64",
        "weights_units": "distance_divided_by_radius",
        "algorithm": "recursive_ridge_symmetric_B_v1",
    }


@dataclass(frozen=True)
class RidgeState:
    weights: np.ndarray
    covariance: np.ndarray
    metadata: dict

    def validate(self, model, regularization):
        meta = self.metadata
        size = model.weights.size
        if meta.get("schema_version") != 1 or meta.get("model") != model_signature(
            model, regularization
        ):
            raise ValueError("checkpoint model or regularization mismatch")
        if (
            any(
                type(meta.get(key)) is not int or meta[key] < 1
                for key in ("completed_steps", "consumed_samples")
            )
            or meta["consumed_samples"] < meta["completed_steps"]
        ):
            raise ValueError("invalid checkpoint counters")
        if (
            self.weights.dtype != np.dtype("float64")
            or self.covariance.dtype != np.dtype("float64")
            or self.weights.shape != (size,)
            or self.covariance.shape != (size, size)
            or not np.isfinite(self.weights).all()
        ):
            raise ValueError("invalid checkpoint arrays")
        # Blockwise validation avoids another full matrix allocation for BP_24.
        for start in range(0, size, 128):
            block = self.covariance[start : start + 128]
            if not np.isfinite(block).all() or not np.allclose(
                block, self.covariance[:, start : start + 128].T, rtol=0, atol=1e-12
            ):
                raise ValueError("nonfinite or asymmetric checkpoint matrix")


def save_checkpoint(path, state, model, regularization):
    """Never overwrite snapshots; incomplete temporary directories are not loadable."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    state.validate(model, regularization)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    hashes = {}
    for name, array in (
        ("normalized_weights.npy", state.weights),
        ("inverse_information.npy", state.covariance),
    ):
        with (temporary / name).open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        hashes[name] = file_sha256(temporary / name)
    metadata = {**state.metadata, "sha256": hashes}
    with (temporary / "metadata.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)
    return metadata


def load_checkpoint(path, model, regularization, context, prefix_hashes):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("context") != context:
        raise ValueError("checkpoint training context mismatch")
    steps = meta.get("completed_steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("invalid checkpoint step")
    if meta.get("batch_prefix_sha256") != prefix_hashes(steps):
        raise ValueError("checkpoint batch prefix mismatch")
    names = {"normalized_weights.npy", "inverse_information.npy"}
    if set(meta.get("sha256", {})) != names:
        raise ValueError("invalid checkpoint file manifest")
    for name in names:
        if file_sha256(path / name) != meta["sha256"][name]:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    state = RidgeState(
        np.load(path / "normalized_weights.npy", mmap_mode="r", allow_pickle=False),
        np.load(path / "inverse_information.npy", mmap_mode="r", allow_pickle=False),
        meta,
    )
    state.validate(model, regularization)
    return state

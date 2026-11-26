"""Portable FK inputs for link-relative obstacle attention.

The packed state contains only the configuration available before execution and
the selected IK configuration. Capsule endpoints are physical geometry, without
the controller's safety margin. No IK, collision solver or trajectory is run.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

RELATIVE_LINK_FRAMES = tuple(f"link{i}" for i in range(2, 7))
_CAPSULE_KEYS = {
    "name",
    "start_frame",
    "end_frame",
    "start_offset",
    "end_offset",
    "radius",
}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _capsule_hash(capsules):
    content = json.dumps(capsules, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def _normalize_capsules(capsules):
    if not isinstance(capsules, list) or not capsules:
        raise ValueError("relative geometry requires a nonempty capsule list")
    result = []
    for capsule in capsules:
        if not isinstance(capsule, dict) or set(capsule) != _CAPSULE_KEYS:
            raise ValueError("invalid relative geometry capsule definition")
        normalized = {}
        for key in ("name", "start_frame", "end_frame"):
            if not isinstance(capsule[key], str) or not capsule[key]:
                raise ValueError(f"capsule {key} must be a nonempty string")
            normalized[key] = capsule[key]
        for key in ("start_offset", "end_offset"):
            offset = np.asarray(capsule[key], dtype=float)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError(f"capsule {key} must have three finite coordinates")
            normalized[key] = offset.tolist()
        radius = capsule["radius"]
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not np.isfinite(radius)
            or radius <= 0
        ):
            raise ValueError("capsule radius must be finite and positive")
        normalized["radius"] = float(radius)
        result.append(normalized)
    names = [capsule["name"] for capsule in result]
    if len(names) != len(set(names)):
        raise ValueError("ambiguous duplicate capsule names")
    if any(frame not in names for frame in RELATIVE_LINK_FRAMES):
        raise ValueError("missing named capsule for a link2..link6 frame")
    return result


def relative_geometry_spec(metadata):
    """Return fresh semantic metadata, independent of asset paths and hashes."""
    if not isinstance(metadata, dict):
        raise ValueError("invalid relative geometry metadata")
    capsules = _normalize_capsules(metadata.get("capsules"))
    names = [capsule["name"] for capsule in capsules]
    count = len(capsules)
    return {
        "link_frames": list(RELATIVE_LINK_FRAMES),
        "pose_layout": ["position_xyz", "rotation_column_0", "rotation_column_1"],
        "coordinate_frame": "world",
        "feature_layout": [
            "initial_q_7_selected_ik_7_target_position_3_target_rotation6",
            "current_link_poses_5x9",
            "target_link_poses_5x9",
            "current_ee_pose_9",
            "current_capsule_endpoints_Jx6",
            "target_capsule_endpoints_Jx6",
        ],
        "capsule_endpoint_layout": ["start_xyz", "end_xyz"],
        "capsule_names": names,
        "capsule_radii": [capsule["radius"] for capsule in capsules],
        "link_capsule_indices": [names.index(frame) for frame in RELATIVE_LINK_FRAMES],
        "physical_distance_safety_margin": 0.0,
        "appendix_dim": 99 + 12 * count,
        "state_dim": 122 + 12 * count,
    }


def relative_geometry_state_dim(metadata):
    return relative_geometry_spec(metadata)["state_dim"]


def make_relative_geometry(
    model_path, capsule_config_path, *, checkpoint_directory=None
):
    """Record exact assets and embed capsule definitions for portable inference.

    The caller may snapshot both assets first. The original configuration is
    provenance only: inference needs the MJCF and these embedded definitions,
    not the configuration or any external RDF artifact.
    """
    model_path = Path(model_path).resolve(strict=True)
    config_path = Path(capsule_config_path).resolve(strict=True)
    config = yaml.safe_load(config_path.read_text())
    capsules = _normalize_capsules(config["safety"]["capsules"])
    ee_frame = config["robot"]["ee_frame"]
    if not isinstance(ee_frame, str) or not ee_frame:
        raise ValueError("relative geometry EE frame must be a nonempty string")

    def recorded(path):
        if checkpoint_directory is None:
            return str(path)
        return os.path.relpath(path, Path(checkpoint_directory).resolve())

    metadata = {
        "schema_version": 1,
        "model_path": recorded(model_path),
        "model_sha256": _hash(model_path),
        "capsule_config_path": recorded(config_path),
        "capsule_config_sha256": _hash(config_path),
        "capsules": capsules,
        "capsules_sha256": _capsule_hash(capsules),
        "ee_frame": ee_frame,
        **relative_geometry_spec({"capsules": capsules}),
    }
    validate_relative_geometry(metadata, model_path=model_path)
    return metadata


def validate_relative_geometry(metadata, *, checkpoint_directory=None, model_path=None):
    """Validate stored conventions, capsule identity and exact MJCF bytes."""
    specification = relative_geometry_spec(metadata)
    expected_keys = {
        "schema_version",
        "model_path",
        "model_sha256",
        "capsule_config_path",
        "capsule_config_sha256",
        "capsules",
        "capsules_sha256",
        "ee_frame",
        *specification,
    }
    if (
        set(metadata) != expected_keys
        or metadata.get("schema_version") != 1
        or any(metadata.get(key) != value for key, value in specification.items())
        or any(
            not isinstance(metadata.get(key), str) or not metadata[key]
            for key in ("model_path", "capsule_config_path", "ee_frame")
        )
        or any(
            not isinstance(metadata.get(key), str)
            or len(metadata[key]) != 64
            or any(char not in "0123456789abcdef" for char in metadata[key])
            for key in ("model_sha256", "capsule_config_sha256", "capsules_sha256")
        )
        or metadata["capsules_sha256"] != _capsule_hash(metadata["capsules"])
    ):
        raise ValueError("invalid relative geometry metadata or capsule checksum")
    path = Path(metadata["model_path"] if model_path is None else model_path)
    if not path.is_absolute() and model_path is None:
        if checkpoint_directory is None:
            raise ValueError(
                "relative geometry model path requires a checkpoint directory"
            )
        path = Path(checkpoint_directory) / path
    try:
        actual_hash = _hash(path)
    except OSError as exc:
        raise ValueError(f"relative geometry model is unavailable: {path}") from exc
    if actual_hash != metadata["model_sha256"]:
        raise ValueError("relative geometry model checksum mismatch")
    return path.resolve()


def compute_relative_geometry_features(initial_q, selected_ik, model_path, metadata):
    """Return finite float32 [N,99+12J] FK appendix in the recorded order."""
    model_path = validate_relative_geometry(metadata, model_path=model_path)
    q = np.asarray(initial_q, dtype=np.float64)
    ik = np.asarray(selected_ik, dtype=np.float64)
    if (
        q.ndim != 2
        or q.shape[1] != 7
        or ik.shape != q.shape
        or not np.isfinite(q).all()
        or not np.isfinite(ik).all()
    ):
        raise ValueError(
            "initial and selected IK angles must be finite matching [N,7] arrays"
        )
    from ..adapters.panda_kinematics import PandaKinematics

    capsules = metadata["capsules"]
    frame_names = {
        *RELATIVE_LINK_FRAMES,
        *(capsule[key] for capsule in capsules for key in ("start_frame", "end_frame")),
    }
    kinematics = PandaKinematics(model_path, metadata["ee_frame"], frame_names)

    def pose(frame, configuration):
        rotation, position = kinematics.frame_pose(frame, configuration)
        return np.r_[position, rotation[:, 0], rotation[:, 1]]

    def endpoints(configuration):
        return np.concatenate(
            [
                kinematics.frame_point(
                    capsule[f"{end}_frame"], configuration, capsule[f"{end}_offset"]
                )
                for capsule in capsules
                for end in ("start", "end")
            ]
        )

    features = np.empty((len(q), metadata["appendix_dim"]), dtype=np.float32)
    for row, (current, target) in enumerate(zip(q, ik)):
        # Group current-state FK and target-state FK to avoid redundant updates.
        current_links = np.concatenate(
            [pose(frame, current) for frame in RELATIVE_LINK_FRAMES]
        )
        current_ee = pose(metadata["ee_frame"], current)
        current_endpoints = endpoints(current)
        target_links = np.concatenate(
            [pose(frame, target) for frame in RELATIVE_LINK_FRAMES]
        )
        target_endpoints = endpoints(target)
        features[row] = np.concatenate(
            (
                current_links,
                target_links,
                current_ee,
                current_endpoints,
                target_endpoints,
            )
        )
    if not np.isfinite(features).all():
        raise ValueError("nonfinite relative geometry FK features")
    return features


def make_relative_state(
    initial_q,
    selected_ik,
    target_position,
    target_quaternion,
    *,
    geometry,
    model_path=None,
    checkpoint_directory=None,
):
    """Prepare the same packed state used by training and online selection."""
    from .link_selection import make_state

    resolved = validate_relative_geometry(
        geometry, checkpoint_directory=checkpoint_directory, model_path=model_path
    )
    original = make_state(initial_q, selected_ik, target_position, target_quaternion)
    appendix = compute_relative_geometry_features(
        initial_q, selected_ik, resolved, geometry
    )
    return np.concatenate((original, appendix), axis=1)

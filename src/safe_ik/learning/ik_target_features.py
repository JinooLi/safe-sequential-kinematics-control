"""Fixed-order, unstandardized Panda link poses at the selected IK target.

These are frame origins and local-to-world rotation columns, matching the
intermediate pose targets used by the controller. Only forward kinematics is
evaluated; no IK search, collision model, or rollout is involved.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

IK_TARGET_FRAMES = tuple(f"link{i}" for i in range(2, 7))
IK_TARGET_FEATURE_DIM = 45


def ik_target_feature_spec():
    """Return fresh metadata so callers cannot mutate the feature convention."""
    return {
        "link_frames": list(IK_TARGET_FRAMES),
        "feature_layout": ["position_xyz", "rotation_column_0", "rotation_column_1"],
        "coordinate_frame": "world",
        "features": IK_TARGET_FEATURE_DIM,
    }


def make_fk_geometry(model_path, *, checkpoint_directory=None):
    """Record the exact MJCF; checkpoint-relative paths survive moving a run."""
    path = Path(model_path).resolve(strict=True)
    recorded_path = (
        os.path.relpath(path, Path(checkpoint_directory).resolve())
        if checkpoint_directory is not None
        else str(path)
    )
    return {
        "schema_version": 1,
        "model_path": recorded_path,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **ik_target_feature_spec(),
    }


def validate_fk_geometry(metadata, *, checkpoint_directory=None, model_path=None):
    """Validate feature semantics and MJCF bytes before preparing an input.

    An explicit model path permits deployment with an independently copied
    model, but that file must have exactly the checkpoint's recorded checksum.
    """
    expected_keys = {
        "schema_version",
        "model_path",
        "model_sha256",
        *ik_target_feature_spec(),
    }
    if (
        not isinstance(metadata, dict)
        or set(metadata) != expected_keys
        or metadata.get("schema_version") != 1
        or not isinstance(metadata.get("model_path"), str)
        or not metadata["model_path"]
        or not isinstance(metadata.get("model_sha256"), str)
        or len(metadata["model_sha256"]) != 64
        or any(
            metadata.get(key) != value
            for key, value in ik_target_feature_spec().items()
        )
    ):
        raise ValueError("invalid IK target FK geometry metadata")
    path = Path(metadata["model_path"] if model_path is None else model_path)
    if not path.is_absolute() and model_path is None:
        if checkpoint_directory is None:
            raise ValueError("relative FK model path requires a checkpoint directory")
        path = Path(checkpoint_directory) / path
    try:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"IK target FK model is unavailable: {path}") from exc
    if actual_hash != metadata["model_sha256"]:
        raise ValueError("IK target FK model checksum mismatch")
    return path.resolve()


def compute_ik_target_features(selected_ik, model_path):
    """Return [N,45]: each link's xyz, rotation column 0, rotation column 1."""
    selected_ik = np.asarray(selected_ik, dtype=np.float64)
    if (
        selected_ik.ndim != 2
        or selected_ik.shape[1] != 7
        or not np.isfinite(selected_ik).all()
    ):
        raise ValueError("selected IK must contain finite [batch, 7] joint angles")
    # Keep baseline inference and precomputed-feature training independent of
    # the optional kinematics import until FK features are actually requested.
    from ..adapters.panda_kinematics import PandaKinematics

    kinematics = PandaKinematics(model_path, "gripper_tip", IK_TARGET_FRAMES)
    features = np.empty((len(selected_ik), IK_TARGET_FEATURE_DIM), dtype=np.float32)
    for row, q in enumerate(selected_ik):
        for index, frame in enumerate(IK_TARGET_FRAMES):
            rotation, position = kinematics.frame_pose(frame, q)
            features[row, index * 9 : (index + 1) * 9] = np.concatenate(
                (position, rotation[:, 0], rotation[:, 1])
            )
    if not np.isfinite(features).all():
        raise ValueError("nonfinite IK target FK features")
    return features

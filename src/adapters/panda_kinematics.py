"""Pinocchio kinematics for the fixed-gripper 7-DoF Panda model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation


def rotation_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized scalar-first quaternion."""
    xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=float)).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def quaternion_orientation_error_components(
    desired_wxyz: np.ndarray, current_wxyz: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return scalar/vector parts of ``Q_desired * inverse(Q_current)``.

    The desired quaternion is flipped to the current hemisphere so the error
    is invariant to the two quaternion representations of the same rotation.
    """
    desired = np.asarray(desired_wxyz, dtype=float).copy()
    current = np.asarray(current_wxyz, dtype=float)
    if desired.shape != (4,) or current.shape != (4,):
        raise ValueError("quaternions must have shape (4,) in [w, x, y, z] order")
    desired /= np.linalg.norm(desired)
    current /= np.linalg.norm(current)
    if desired @ current < 0.0:
        desired = -desired
    eta_d, eta_e = desired[0], current[0]
    eps_d, eps_e = desired[1:], current[1:]
    scalar = float(eta_d * eta_e + eps_d @ eps_e)
    vector = eta_e * eps_d - eta_d * eps_e - np.cross(eps_d, eps_e)
    return scalar, vector


def quaternion_orientation_error(
    desired_wxyz: np.ndarray, current_wxyz: np.ndarray
) -> np.ndarray:
    """Return the vector part of ``Q_desired * inverse(Q_current)``."""
    return quaternion_orientation_error_components(desired_wxyz, current_wxyz)[1]


class PandaKinematics:
    """Shared pose and frame-point kinematics backed by one Pinocchio model."""

    def __init__(self, model_path: str | Path, ee_frame: str, frame_names=()):
        self.model_path = Path(model_path)
        self.model = pin.buildModelFromMJCF(str(self.model_path))
        self.data = self.model.createData()
        if self.model.nq != 7 or self.model.nv != 7:
            raise ValueError("fixed-gripper Panda model must have nq=nv=7")
        requested = {str(ee_frame), *(map(str, frame_names))}
        self.frame_ids = {}
        for name in requested:
            frame_id = self.model.getFrameId(name)
            if frame_id >= self.model.nframes:
                raise ValueError(f"Panda frame was not found: {name}")
            self.frame_ids[name] = frame_id
        self.ee_frame = str(ee_frame)
        self._cached_q: np.ndarray | None = None

    @property
    def lower_limits(self) -> np.ndarray:
        return np.asarray(self.model.lowerPositionLimit, dtype=float).copy()

    @property
    def upper_limits(self) -> np.ndarray:
        return np.asarray(self.model.upperPositionLimit, dtype=float).copy()

    def update(self, q) -> None:
        q = np.asarray(q, dtype=float)
        if q.shape != (7,):
            raise ValueError("Panda q must have shape (7,)")
        if self._cached_q is not None and np.array_equal(q, self._cached_q):
            return
        pin.forwardKinematics(self.model, self.data, q)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        self._cached_q = q.copy()

    def pose_and_jacobian(self, q):
        """Return gripper-tip position, quaternion, and world-aligned 6D J."""
        self.update(q)
        frame_id = self.frame_ids[self.ee_frame]
        placement = self.data.oMf[frame_id]
        jacobian = pin.getFrameJacobian(
            self.model,
            self.data,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return (
            np.asarray(placement.translation, dtype=float).copy(),
            rotation_to_wxyz(placement.rotation),
            np.asarray(jacobian, dtype=float).copy(),
        )

    def frame_point_and_jacobian(self, frame_name: str, q, local_offset=None):
        """Return a frame-fixed point and its world linear Jacobian."""
        self.update(q)
        frame_id = self.frame_ids[frame_name]
        placement = self.data.oMf[frame_id]
        jacobian = pin.getFrameJacobian(
            self.model,
            self.data,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        offset = (
            np.zeros(3)
            if local_offset is None
            else np.asarray(local_offset, dtype=float)
        )
        world_offset = np.asarray(placement.rotation, dtype=float) @ offset
        # v_point = v_frame + omega x r = v_frame - skew(r) omega.
        skew = np.array(
            [
                [0.0, -world_offset[2], world_offset[1]],
                [world_offset[2], 0.0, -world_offset[0]],
                [-world_offset[1], world_offset[0], 0.0],
            ]
        )
        point_jacobian = jacobian[:3] - skew @ jacobian[3:]
        return np.asarray(placement.translation) + world_offset, point_jacobian

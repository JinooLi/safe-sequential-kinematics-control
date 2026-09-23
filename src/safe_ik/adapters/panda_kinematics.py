"""Pinocchio kinematics for the fixed-gripper 7-DoF Panda model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin

from ..domain.rotations import rotation_to_wxyz


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

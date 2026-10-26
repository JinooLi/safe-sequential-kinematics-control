"""Experiment controller: fixed EE waypoints, sharing the existing CLF/CBF QP."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .panda_controller import PandaTrajectoryController


def interpolate_ee_targets(initial_position, initial_quaternion, target, stages):
    """Equal fractions of a fixed line / shortest rotation; final goal included."""
    if type(stages) is not int or stages < 1:
        raise ValueError("stages must be a positive integer including the final goal")
    fractions = np.arange(1, stages + 1, dtype=float) / stages
    initial_position = np.asarray(initial_position, dtype=float)
    initial_quaternion = np.asarray(initial_quaternion, dtype=float)
    positions = initial_position + fractions[:, None] * (
        target.position - initial_position
    )
    rotations = Rotation.from_quat(
        np.array([initial_quaternion, target.quaternion])[:, [1, 2, 3, 0]]
    )
    xyzw = Slerp([0.0, 1.0], rotations)(fractions).as_quat()
    quaternions = xyzw[:, [3, 0, 1, 2]]
    # Preserve the exact original endpoint; antipodal quaternion signs are equal.
    positions[-1] = target.position
    quaternions[-1] = target.quaternion
    return positions, quaternions


class PandaEEWaypointController(PandaTrajectoryController):
    """Direct EE tracking through N-1 intermediate poses, with no IK candidates.

    The existing stage machinery handles tolerance, progress, and history reset.
    Acceleration history, solver, barriers, slack and the final target are shared
    unchanged with ordinary Direct. A rollout owns the single total time budget.
    """

    def __init__(self, config, kinematics, safety, candidates, solver, *, stages):
        if config.trajectory.mode != "direct" or candidates:
            raise ValueError("EE waypoints require Direct mode without IK candidates")
        if config.trajectory.stage_task != "pose":
            raise ValueError("EE waypoints require pose intermediate targets")
        if (
            config.trajectory.stage_orientation_weight
            != config.trajectory.orientation_weight
        ):
            raise ValueError("intermediate and final orientation weights must match")
        super().__init__(config, kinematics, safety, (), solver)
        position, quaternion, _ = kinematics.pose_and_jacobian(config.initial_q)
        self.waypoint_positions, self.waypoint_quaternions = interpolate_ee_targets(
            position, quaternion, config.target, stages
        )
        self.stage_frames = (config.ee_frame,) * (stages - 1)
        # A fixed world-pose target table uses the base stage evaluator. No IK is
        # created or switched: candidate_index stays zero and candidates is empty.
        self._candidate_stage_targets = (tuple(self.waypoint_positions[:-1]),)
        self._candidate_stage_orientations = (tuple(self.waypoint_quaternions[:-1]),)

    def _stage_name(self, stage_index):
        if stage_index < len(self.stage_frames):
            return f"ee_waypoint_{stage_index + 1:02d}_pose"
        return "end_effector_pose"

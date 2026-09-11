"""Closed-form goal configurations for a three-link planar arm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class GoalConfiguration:
    """One elbow branch of a pose-correct three-link IK solution."""

    name: str
    q: np.ndarray
    joint_positions: np.ndarray
    colliding: bool


def joint_positions(q, link_lengths):
    """Return the world positions of every link endpoint for ``q``."""
    q = np.asarray(q, dtype=float)
    lengths = np.asarray(link_lengths, dtype=float)
    angles = np.cumsum(q)
    return np.cumsum(
        np.column_stack((lengths * np.cos(angles), lengths * np.sin(angles))), axis=0
    )


def link1_link3_crosses(positions, tolerance: float = 1e-10) -> bool:
    """Return whether the non-adjacent link1 and link3 centerlines cross.

    Adjacent links intentionally meet at a joint, but link1 and link3 must not
    intersect in their interiors.  This is a binary geometric goal-screening
    test, not a distance-field or differentiable CBF constraint.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (3, 2):
        raise ValueError("positions must have shape (3, 2)")
    base = np.zeros(2)
    link1_end, link3_start, link3_end = positions

    def orientation(start, end, point):
        edge = end - start
        offset = point - start
        return float(edge[0] * offset[1] - edge[1] * offset[0])

    first_start = orientation(base, link1_end, link3_start)
    first_end = orientation(base, link1_end, link3_end)
    third_start = orientation(link3_start, link3_end, base)
    third_end = orientation(link3_start, link3_end, link1_end)
    return (
        first_start * first_end < -tolerance
        and third_start * third_end < -tolerance
    )


def solve_goal_configurations(
    target_position,
    target_yaw: float,
    link_lengths,
    obstacles=(),
    *,
    link_radius: float = 0.0,
    safe_distance: float = 0.0,
    self_collision_distance: float = 0.0,
    joint_limits=(),
):
    """Solve both elbow branches and retain whether each endpoint pose is safe."""
    lengths = np.asarray(link_lengths, dtype=float)
    if lengths.shape != (3,) or np.any(lengths <= 0.0):
        raise ValueError("goal configuration solver requires three positive link lengths")
    target = np.asarray(target_position, dtype=float)
    if target.shape != (2,):
        raise ValueError("target_position must have shape (2,)")
    if self_collision_distance < 0.0:
        raise ValueError("self_collision_distance must be non-negative")

    wrist = target - lengths[2] * np.array([np.cos(target_yaw), np.sin(target_yaw)])
    wrist_distance = float(np.linalg.norm(wrist))
    lower, upper = abs(lengths[0] - lengths[1]), lengths[0] + lengths[1]
    if wrist_distance < lower - 1e-10 or wrist_distance > upper + 1e-10:
        return ()
    if wrist_distance < 1e-12:
        return ()

    base_angle = float(np.arctan2(wrist[1], wrist[0]))
    cosine = np.clip(
        (wrist_distance**2 + lengths[0] ** 2 - lengths[1] ** 2)
        / (2.0 * wrist_distance * lengths[0]),
        -1.0,
        1.0,
    )
    candidates = []
    for name, sign in (("elbow_up", 1.0), ("elbow_down", -1.0)):
        theta_1 = base_angle + sign * float(np.arccos(cosine))
        elbow = lengths[0] * np.array([np.cos(theta_1), np.sin(theta_1)])
        theta_2 = float(np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]))
        q = np.array(
            [wrap_to_pi(theta_1), wrap_to_pi(theta_2 - theta_1), wrap_to_pi(target_yaw - theta_2)]
        )
        positions = joint_positions(q, lengths)
        violates_limit = any(
            limit is not None and not (limit[0] <= angle <= limit[1])
            for angle, limit in zip(q, joint_limits or (None,) * 3)
        )
        # link1/link3 are non-adjacent.  Keep the binary crossing test as an
        # additional screen, then apply the same self-clearance barriers that
        # the runtime QP enforces.
        colliding = violates_limit or link1_link3_crosses(positions)
        if obstacles or self_collision_distance > 0.0:
            from .safety import PlanarCapsuleRobot

            safety_robot = PlanarCapsuleRobot(lengths, link_radius)
        if obstacles:
            values = safety_robot.cbf_constraints(
                q, obstacles, safe_distance, alpha=1.0
            )[2]
            colliding = colliding or bool(np.any(values < 0.0))
        if self_collision_distance > 0.0:
            ee_link1_value = safety_robot.self_collision_cbf_constraint(
                q, self_collision_distance, alpha=1.0
            )[2]
            base_link3_value = safety_robot.base_to_link_cbf_constraint(
                q,
                link_index=len(lengths) - 1,
                minimum_distance=self_collision_distance,
                alpha=1.0,
            )[2]
            colliding = colliding or ee_link1_value < 0.0 or base_link3_value < 0.0
        candidates.append(GoalConfiguration(name, q, positions, colliding))
    return tuple(candidates)

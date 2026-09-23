"""Seeded scenario generation and feasibility filtering."""

from __future__ import annotations

import numpy as np

from ..domain.goal_configuration import solve_goal_configurations
from ..domain.planar_config import (
    ControllerConfig,
    ObstacleConfig,
    SequentialExperimentConfig,
)
from ..domain.safety import obstacles_from_config
from .models import ExperimentCase


def _distance_to_segment(point, start, end):
    segment = end - start
    length_squared = float(segment @ segment)
    if length_squared == 0.0:
        return float(np.linalg.norm(point - start))
    ratio = np.clip(((point - start) @ segment) / length_squared, 0.0, 1.0)
    return float(np.linalg.norm(point - (start + ratio * segment)))


def _initial_robot_clearance(point, initial_q, link_lengths):
    """Distance from a point to the configured initial serial-robot center line."""
    start = np.array([0.0, 0.0])
    clearance = np.inf
    for angle, length in zip(np.cumsum(initial_q), link_lengths):
        end = start + length * np.array([np.cos(angle), np.sin(angle)])
        clearance = min(clearance, _distance_to_segment(point, start, end))
        start = end
    return float(clearance)


def has_feasible_goal_configuration(
    target,
    target_yaw: float,
    obstacles: tuple[ObstacleConfig, ...],
    config: ControllerConfig,
) -> bool:
    """Return whether at least one safe IK branch exists for a sampled case."""
    candidates = solve_goal_configurations(
        target,
        target_yaw,
        config.safety.link_lengths,
        obstacles_from_config(
            [
                {"pos": obstacle.position, "radius": obstacle.radius}
                for obstacle in obstacles
            ]
        ),
        link_radius=config.safety.link_radius,
        safe_distance=config.safety.safe_distance,
        self_collision_distance=config.safety.self_collision_distance,
        joint_limits=config.safety.joint_limits,
    )
    return any(not candidate.colliding for candidate in candidates)


def sample_case(
    case_id: int,
    rng: np.random.Generator,
    base_config: ControllerConfig,
    experiment: SequentialExperimentConfig,
) -> ExperimentCase:
    """Sample a meaningful, initially safe target and two-obstacle placement."""
    obstacle_count = len(base_config.obstacles)
    if obstacle_count != 2:
        raise ValueError("random comparison currently requires exactly two obstacles")
    lengths = np.asarray(base_config.safety.link_lengths, dtype=float)
    robot_exclusion = (
        base_config.safety.link_radius
        + base_config.safety.safe_distance
        + experiment.initial_clearance_margin
    )

    ik_rejections = 0
    for _ in range(experiment.max_case_attempts):
        target_angle = rng.uniform(*experiment.target_angle_range)
        target_radius = rng.uniform(*experiment.target_radius_range)
        target = target_radius * np.array([np.cos(target_angle), np.sin(target_angle)])
        target_yaw = float(rng.uniform(*experiment.target_orientation_range))
        obstacles = []
        for _ in range(obstacle_count):
            for _ in range(experiment.max_obstacle_attempts):
                radius = float(rng.uniform(*experiment.obstacle_radius_range))
                position = rng.uniform(*experiment.obstacle_position_range, size=2)
                if (
                    _initial_robot_clearance(position, experiment.initial_q, lengths)
                    <= radius + robot_exclusion
                ):
                    continue
                if np.linalg.norm(position - target) <= (
                    radius
                    + base_config.safety.link_radius
                    + base_config.safety.safe_distance
                    + experiment.target_clearance_margin
                ):
                    continue
                if any(
                    np.linalg.norm(position - np.asarray(existing.position))
                    <= radius + existing.radius + experiment.target_clearance_margin
                    for existing in obstacles
                ):
                    continue
                obstacles.append(
                    ObstacleConfig(
                        (float(position[0]), float(position[1])), radius, True
                    )
                )
                break
            else:
                break
        if len(obstacles) == obstacle_count:
            if not has_feasible_goal_configuration(
                target, target_yaw, tuple(obstacles), base_config
            ):
                ik_rejections += 1
                continue
            return ExperimentCase(
                case_id,
                (float(target[0]), float(target[1])),
                target_yaw,
                tuple(obstacles),
                ik_rejections,
            )
    raise RuntimeError("unable to sample a valid experiment case")

"""Validated configuration values; no file parsing or backend imports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObstacleConfig:
    position: tuple[float, float]
    radius: float
    enabled: bool = True


@dataclass(frozen=True)
class SafetyConfig:
    enabled: bool
    representation: str
    rdf_model_path: str
    link_lengths: tuple[float, ...]
    link_radius: float
    safe_distance: float
    self_collision_distance: float
    alpha: float
    joint_limits: tuple[tuple[float, float] | None, ...]
    joint_limit_alpha: float
    velocity_limit: float
    clf_rate: float
    clf_weight: float


@dataclass(frozen=True)
class QPConfig:
    R: np.ndarray
    Q: np.ndarray


@dataclass(frozen=True)
class SimulationConfig:
    """Execution model shared by interactive and batch simulations."""

    kinematic: bool


@dataclass(frozen=True)
class SequentialEndEffectorConfig:
    """Settings for progressing the task point from link 1 to the final tip."""

    enabled: bool
    frame_names: tuple[str, ...]
    goal_tolerance: float
    velocity_tolerance: float
    barrier_tolerance: float
    final_yaw_weight: float


@dataclass(frozen=True)
class SequentialExperimentConfig:
    """All randomized sequential-control comparison settings."""

    cases: int
    workers: int
    seed: int
    duration: float
    goal_tolerance: float
    yaw_tolerance: float
    barrier_tolerance: float
    slack_active_threshold: float
    control_mode: str | None
    initial_q: tuple[float, ...]
    output_dir: str
    target_radius_range: tuple[float, float]
    target_angle_range: tuple[float, float]
    target_orientation_range: tuple[float, float]
    obstacle_position_range: tuple[float, float]
    obstacle_radius_range: tuple[float, float]
    initial_clearance_margin: float
    target_clearance_margin: float
    max_case_attempts: int
    max_obstacle_attempts: int


@dataclass(frozen=True)
class ControllerConfig:
    dof: int
    ee_target: np.ndarray
    ee_orientation: float
    control_mode: str
    obstacles: tuple[ObstacleConfig, ...]
    safety: SafetyConfig
    qp: QPConfig
    simulation: SimulationConfig
    sequential_end_effector: SequentialEndEffectorConfig
    sequential_experiment: SequentialExperimentConfig

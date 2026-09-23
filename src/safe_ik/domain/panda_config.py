"""Validated configuration values; no file parsing or backend imports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PandaTargetConfig:
    position: np.ndarray
    approach: np.ndarray  # desired world direction of gripper_tip local +z
    up: np.ndarray  # desired world direction of gripper_tip local +y
    quaternion: np.ndarray  # derived, scalar-first [w, x, y, z]
    position_tolerance: float
    orientation_tolerance: float


@dataclass(frozen=True)
class SafeIKConfig:
    integration_period: float
    duration_per_seed: float
    convergence_hold_duration: float
    max_restarts: int
    random_seed: int
    damping: float
    pose_gain: float
    softmin_temperature: float
    safety_gradient_gain: float
    workers: int = 1

    @property
    def iterations_per_seed(self) -> int:
        return int(np.ceil(self.duration_per_seed / self.integration_period))

    @property
    def convergence_hold_steps(self) -> int:
        return int(np.ceil(self.convergence_hold_duration / self.integration_period))


@dataclass(frozen=True)
class CapsuleConfig:
    name: str
    start_frame: str
    end_frame: str
    start_offset: np.ndarray
    end_offset: np.ndarray
    radius: float


@dataclass(frozen=True)
class SphereObstacleConfig:
    name: str
    position: np.ndarray
    radius: float
    enabled: bool


@dataclass(frozen=True)
class PandaSafetyConfig:
    margin: float
    alpha: float
    capsules: tuple[CapsuleConfig, ...]
    self_collision_exclusions: frozenset[frozenset[str]]


@dataclass(frozen=True)
class PandaTrajectoryConfig:
    duration: float
    integration_period: float
    mode: str
    stage_frames: tuple[str, ...]
    stage_position_tolerance: float
    orientation_weight: float
    stop_window: float
    stop_velocity_tolerance: float
    clf_rate: float
    clf_slack_weight: float
    velocity_limits: np.ndarray
    acceleration_limits: np.ndarray

    @property
    def control_frequency(self) -> float:
        return 1.0 / self.integration_period


@dataclass(frozen=True)
class PandaQPConfig:
    joint_velocity_weight: float
    max_iterations: int
    absolute_tolerance: float
    relative_tolerance: float


@dataclass(frozen=True)
class PandaControllerConfig:
    model_path: Path
    ee_frame: str
    dof: int
    initial_q: np.ndarray
    target: PandaTargetConfig
    safe_ik: SafeIKConfig
    safety: PandaSafetyConfig
    obstacles: tuple[SphereObstacleConfig, ...]
    trajectory: PandaTrajectoryConfig
    qp: PandaQPConfig

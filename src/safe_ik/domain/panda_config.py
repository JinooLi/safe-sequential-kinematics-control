"""Validated configuration values; no file parsing or backend imports."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np

# Numerical IK step in virtual seconds, not a wall-clock or trajectory period.
# Keep the existing update scale independent of the per-seed iteration budget.
SAFE_IK_STEP_SIZE = 0.01


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
    iteration_number: int
    convergence_hold_duration: float
    max_restarts: int
    random_seed: int
    damping: float
    pose_gain: float
    softmin_temperature: float
    safety_gradient_gain: float
    workers: int = 1
    # None inherits the trajectory backend for backward-compatible YAML files.
    self_collision_model: str | None = None

    def __post_init__(self):
        if type(self.iteration_number) is not int or self.iteration_number < 1:
            raise ValueError("safe_ik.iteration_number must be a positive integer")
        if self.self_collision_model not in (None, "rdf", "capsule"):
            raise ValueError(
                "safe_ik.self_collision_model must be rdf, capsule or null"
            )

    @property
    def convergence_hold_steps(self) -> int:
        return int(np.ceil(self.convergence_hold_duration / SAFE_IK_STEP_SIZE))


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
class PandaRDFConfig:
    artifact: Path = Path("models/panda_rdf.npz")
    query_density: float = 500.0
    top_k: int | None = 10
    softmin_error: float = 0.005
    # Sets radii at training; runtime may use <= the artifact's training value.
    # Runtime reductions affect CBF clearance only, not the stored learned field.
    physical_margin: float = 0.030
    # Upper bound on overestimation AND CBF uncertainty margin. Also sets radii.
    error_margin: float = 0.010
    transition_ratio: float = 0.5
    cbf_gain: float = 1.0
    seed: int = 0
    # Validation/performance limit ONLY; None preserves legacy symmetric checks.
    underestimation_limit: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "artifact", Path(self.artifact))
        for name in (
            "query_density",
            "softmin_error",
            "error_margin",
            "transition_ratio",
            "cbf_gain",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"safety.rdf.{name} must be finite and positive")
        if (
            isinstance(self.physical_margin, bool)
            or not isinstance(self.physical_margin, Real)
            or not np.isfinite(self.physical_margin)
            or self.physical_margin < 0
        ):
            raise ValueError("safety.rdf.physical_margin must be nonnegative")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError(
                "safety.rdf.top_k must be a positive integer or null (all)"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("safety.rdf.seed must be a nonnegative integer")
        if self.underestimation_limit is not None and (
            isinstance(self.underestimation_limit, bool)
            or not isinstance(self.underestimation_limit, Real)
            or not np.isfinite(self.underestimation_limit)
            or self.underestimation_limit <= 0
        ):
            raise ValueError(
                "safety.rdf.underestimation_limit must be finite and positive"
            )

    @property
    def effective_margin(self):
        # Never add underestimation_limit: it is not an uncertainty compensation.
        return self.physical_margin + self.error_margin


@dataclass(frozen=True)
class PandaSafetyConfig:
    margin: float
    alpha: float
    capsules: tuple[CapsuleConfig, ...]
    self_collision_exclusions: frozenset[frozenset[str]]
    self_collision_model: str = "rdf"
    rdf: PandaRDFConfig = PandaRDFConfig()

    def __post_init__(self):
        if self.self_collision_model not in {"rdf", "capsule"}:
            raise ValueError("self_collision_model must be rdf or capsule")


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

    @property
    def ik_self_collision_model(self) -> str:
        return self.safe_ik.self_collision_model or self.safety.self_collision_model

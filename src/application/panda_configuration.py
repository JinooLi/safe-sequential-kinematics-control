"""Typed configuration for Panda safe-IK search and trajectory control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


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


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_integer(name: str, value: int) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def orientation_vectors_to_wxyz(approach, up) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a tip rotation from non-normalized world-frame approach/up axes.

    The local +z axis follows ``approach``. ``up`` is projected onto the plane
    normal to it and becomes local +y, fixing the otherwise-free roll.
    """
    approach = np.asarray(approach, dtype=float)
    up = np.asarray(up, dtype=float)
    for name, vector in (
        ("target.orientation.approach", approach),
        ("target.orientation.up", up),
    ):
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain three finite values")
    approach_norm = np.linalg.norm(approach)
    up_norm = np.linalg.norm(up)
    if approach_norm < 1e-12 or up_norm < 1e-12:
        raise ValueError("target orientation approach and up must be nonzero")

    z_axis = approach / approach_norm
    y_projected = up - (up @ z_axis) * z_axis
    y_norm = np.linalg.norm(y_projected)
    if y_norm < 1e-8 * up_norm:
        raise ValueError("target orientation approach and up must not be parallel")
    y_axis = y_projected / y_norm
    x_axis = np.cross(y_axis, z_axis)
    xyzw = Rotation.from_matrix(np.column_stack((x_axis, y_axis, z_axis))).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]]), z_axis, y_axis


def load_panda_config(path: str | Path) -> PandaControllerConfig:
    """Load and validate the dedicated Panda controller configuration."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    robot = data["robot"]
    target = data["target"]
    ik = data["safe_ik"]
    safety = data["safety"]
    trajectory = data["trajectory"]
    trajectory_mode = str(trajectory.get("mode", "sequential")).lower()
    if trajectory_mode not in {"sequential", "direct"}:
        raise ValueError("trajectory.mode must be 'sequential' or 'direct'")
    stage_frames = tuple(map(str, trajectory.get("stage_frames", ())))
    if trajectory_mode == "sequential" and not stage_frames:
        raise ValueError("trajectory.stage_frames must not be empty in sequential mode")
    if len(set(stage_frames)) != len(stage_frames):
        raise ValueError("trajectory.stage_frames must be unique")
    dof = int(robot.get("dof", 7))
    if dof != 7:
        raise ValueError("Panda arm controller requires exactly 7 DoF")

    orientation = target["orientation"]
    quaternion, approach, up = orientation_vectors_to_wxyz(
        orientation["approach"], orientation["up"]
    )

    initial_q = np.asarray(robot["initial_q"], dtype=float)
    velocity_limits = np.asarray(trajectory["velocity_limits"], dtype=float)
    acceleration_limits = np.asarray(trajectory["acceleration_limits"], dtype=float)
    for name, values in (
        ("robot.initial_q", initial_q),
        ("trajectory.velocity_limits", velocity_limits),
        ("trajectory.acceleration_limits", acceleration_limits),
    ):
        if values.shape != (dof,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain {dof} finite values")
    if np.any(velocity_limits <= 0.0) or np.any(acceleration_limits <= 0.0):
        raise ValueError("trajectory velocity/acceleration limits must be positive")

    capsules = tuple(
        CapsuleConfig(
            name=str(entry["name"]),
            start_frame=str(entry["start_frame"]),
            end_frame=str(entry["end_frame"]),
            start_offset=np.asarray(
                entry.get("start_offset", [0.0, 0.0, 0.0]), dtype=float
            ),
            end_offset=np.asarray(
                entry.get("end_offset", [0.0, 0.0, 0.0]), dtype=float
            ),
            radius=_positive("capsule radius", entry["radius"]),
        )
        for entry in safety["capsules"]
    )
    if not capsules or len({item.name for item in capsules}) != len(capsules):
        raise ValueError("safety.capsules must have unique names")
    if any(
        capsule.start_offset.shape != (3,) or capsule.end_offset.shape != (3,)
        for capsule in capsules
    ):
        raise ValueError("capsule offsets must contain [x, y, z]")
    if any(
        not np.all(np.isfinite(capsule.start_offset))
        or not np.all(np.isfinite(capsule.end_offset))
        for capsule in capsules
    ):
        raise ValueError("capsule offsets must be finite")
    exclusions = frozenset(
        frozenset(map(str, pair))
        for pair in safety.get("self_collision_exclusions", ())
    )
    capsule_names = {capsule.name for capsule in capsules}
    if any(len(pair) != 2 or not pair <= capsule_names for pair in exclusions):
        raise ValueError("self-collision exclusions must name two known capsules")
    obstacles = tuple(
        SphereObstacleConfig(
            name=str(entry["name"]),
            position=np.asarray(entry["position"], dtype=float),
            radius=_positive("obstacle radius", entry["radius"]),
            enabled=bool(entry.get("enabled", True)),
        )
        for entry in data.get("obstacles", ())
    )
    if any(item.position.shape != (3,) for item in obstacles):
        raise ValueError("obstacle positions must contain [x, y, z]")

    config = PandaControllerConfig(
        model_path=(path.parent.parent / robot["model_path"]).resolve(),
        ee_frame=str(robot["ee_frame"]),
        dof=dof,
        initial_q=initial_q,
        target=PandaTargetConfig(
            position=np.asarray(target["position"], dtype=float),
            approach=approach,
            up=up,
            quaternion=quaternion,
            position_tolerance=_positive(
                "target.position_tolerance", target["position_tolerance"]
            ),
            orientation_tolerance=_positive(
                "target.orientation_tolerance", target["orientation_tolerance"]
            ),
        ),
        safe_ik=SafeIKConfig(
            integration_period=_positive(
                "safe_ik.integration_period", ik["integration_period"]
            ),
            duration_per_seed=_positive(
                "safe_ik.duration_per_seed", ik["duration_per_seed"]
            ),
            convergence_hold_duration=_positive(
                "safe_ik.convergence_hold_duration",
                ik["convergence_hold_duration"],
            ),
            max_restarts=int(ik["max_restarts"]),
            random_seed=int(ik.get("random_seed", 0)),
            damping=_positive("safe_ik.damping", ik["damping"]),
            pose_gain=_positive("safe_ik.pose_gain", ik["pose_gain"]),
            softmin_temperature=_positive(
                "safe_ik.softmin_temperature", ik["softmin_temperature"]
            ),
            safety_gradient_gain=_positive(
                "safe_ik.safety_gradient_gain", ik["safety_gradient_gain"]
            ),
        ),
        safety=PandaSafetyConfig(
            margin=float(safety["margin"]),
            alpha=_positive("safety.alpha", safety["alpha"]),
            capsules=capsules,
            self_collision_exclusions=exclusions,
        ),
        obstacles=obstacles,
        trajectory=PandaTrajectoryConfig(
            duration=_positive("trajectory.duration", trajectory["duration"]),
            integration_period=_positive(
                "trajectory.integration_period", trajectory["integration_period"]
            ),
            mode=trajectory_mode,
            stage_frames=stage_frames,
            stage_position_tolerance=_positive(
                "trajectory.stage_position_tolerance",
                trajectory["stage_position_tolerance"],
            ),
            orientation_weight=_positive(
                "trajectory.orientation_weight", trajectory["orientation_weight"]
            ),
            stop_window=_positive("trajectory.stop_window", trajectory["stop_window"]),
            stop_velocity_tolerance=float(trajectory["stop_velocity_tolerance"]),
            clf_rate=_positive("trajectory.clf_rate", trajectory["clf_rate"]),
            clf_slack_weight=_positive(
                "trajectory.clf_slack_weight", trajectory["clf_slack_weight"]
            ),
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
        ),
        qp=PandaQPConfig(
            joint_velocity_weight=_positive(
                "qp.joint_velocity_weight", data["qp"]["joint_velocity_weight"]
            ),
            max_iterations=_positive_integer(
                "qp.max_iterations", data["qp"].get("max_iterations", 20_000)
            ),
            absolute_tolerance=_positive(
                "qp.absolute_tolerance", data["qp"].get("absolute_tolerance", 1e-3)
            ),
            relative_tolerance=_positive(
                "qp.relative_tolerance", data["qp"].get("relative_tolerance", 1e-3)
            ),
        ),
    )
    if config.target.position.shape != (3,) or not np.all(
        np.isfinite(config.target.position)
    ):
        raise ValueError("target.position must contain finite [x, y, z]")
    if config.safe_ik.max_restarts < 1:
        raise ValueError("safe_ik.max_restarts must be at least 1")
    if config.safety.margin < 0.0:
        raise ValueError("safety.margin must be non-negative")
    if config.trajectory.stop_velocity_tolerance < 0.0:
        raise ValueError("trajectory.stop_velocity_tolerance must be non-negative")
    if not config.model_path.exists():
        raise ValueError(f"Panda model does not exist: {config.model_path}")
    return config

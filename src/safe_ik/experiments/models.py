"""Planar comparison records and stable method identifiers."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.planar_config import ControllerConfig, ObstacleConfig


@dataclass(frozen=True)
class ExperimentCase:
    """One target/obstacle placement shared by direct and sequential methods."""

    case_id: int
    target: tuple[float, float]
    target_yaw: float
    obstacles: tuple[ObstacleConfig, ...]
    ik_rejections_before_acceptance: int = 0


@dataclass(frozen=True)
class RunResult:
    success: bool
    final_position: tuple[float, float]
    final_yaw: float
    position_error: float
    yaw_error: float
    min_barrier: float
    active_link: int
    switch_reason: str | None
    failure_reason: str | None
    completion_time: float | None
    control_effort: float
    max_joint_velocity: float
    mean_clf_slack: float | None
    max_clf_slack: float | None
    nonzero_clf_slack_ratio_to_goal: float | None
    total_qp_solve_time: float
    mean_qp_solve_time: float | None
    qp_solver_failure_count: int
    qp_infeasible_count: int
    stage_durations: tuple[float, ...]
    stage_switch_reasons: tuple[str, ...]
    trajectory_file: str


@dataclass(frozen=True)
class ComparisonWorkerInput:
    """Pickle-safe payload for one case's direct/sequential pair."""

    base_config: ControllerConfig
    case: ExperimentCase
    mujoco_path: str
    urdf_path: str
    duration: float
    goal_tolerance: float
    yaw_tolerance: float
    barrier_tolerance: float
    slack_active_threshold: float
    initial_q: tuple[float, ...]
    output_dir: str


OUTCOME_LABELS = {
    (False, False): "both_failed",
    (False, True): "candidate_only_success",
    (True, False): "baseline_only_success",
    (True, True): "both_succeeded",
}

METHODS = (
    "direct",
    "sequential",
)

COMPARISONS = {
    "task_decomposition": ("direct", "sequential"),
}

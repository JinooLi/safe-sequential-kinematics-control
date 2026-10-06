"""Panda control status and rollout data, independent of execution and storage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ik_contracts import SafeIKResult


@dataclass(frozen=True)
class PandaControlStatus:
    success: bool
    stalled: bool
    candidates_exhausted: bool
    qp_infeasible: bool
    qp_solver_failed: bool
    stage_index: int
    stage_name: str
    stage_position_error: float
    joint_error: float | None
    position_error: float
    orientation_error: float
    minimum_barrier: float
    stage_orientation_error: float | None = None


@dataclass(frozen=True)
class PandaRunResult:
    ik: SafeIKResult
    ik_skipped: bool
    status: PandaControlStatus | None
    elapsed: float
    steps: int
    final_q: np.ndarray
    safe_ik_wall_time: float
    control_effort: float
    minimum_barrier: float
    total_qp_solve_time: float
    maximum_clf_slack: float
    qp_infeasible_count: int
    qp_solver_failure_count: int
    candidate_switch_count: int
    stage_transition_count: int
    trace: "PandaControlTrace | None"
    safety_metadata: dict | None = None
    failure: dict | None = None


@dataclass(frozen=True)
class PandaControlTrace:
    """Sampled CLF-CBF QP diagnostics for one trajectory rollout."""

    time: np.ndarray
    clf_value: np.ndarray
    clf_slack: np.ndarray
    stage_index: np.ndarray
    stage_names: tuple[str, ...]
    barrier_labels: tuple[str, ...]
    barrier_values: np.ndarray

    q: np.ndarray | None = None
    command: np.ndarray | None = None
    rdf_timings: np.ndarray | None = None
    qp_time: np.ndarray | None = None
    qp_status: tuple[str, ...] = ()
    safety_metadata: dict | None = None
    minimum_self_barrier: np.ndarray | None = None
    minimum_self_pair: tuple[str, ...] = ()
    stage_position_error: np.ndarray | None = None
    stage_orientation_error: np.ndarray | None = None

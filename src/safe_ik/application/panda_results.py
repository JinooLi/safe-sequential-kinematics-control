"""Panda rollout values and serialization, independent of its simulation backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..control.panda_controller import PandaControlStatus
from ..domain.safe_pose_ik import SafeIKResult


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


def save_panda_control_trace(trace: PandaControlTrace | None, path: str | Path) -> bool:
    """Save numerical histories and stable CBF labels for post-processing."""
    if trace is None or trace.time.size == 0:
        return False
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trace_path,
        time=trace.time,
        clf_value=trace.clf_value,
        clf_slack=trace.clf_slack,
        stage_index=trace.stage_index,
        stage_names=np.asarray(trace.stage_names, dtype=str),
        barrier_labels=np.asarray(trace.barrier_labels, dtype=str),
        barrier_values=trace.barrier_values,
        **({"q": trace.q} if trace.q is not None else {}),
        **({"command": trace.command} if trace.command is not None else {}),
    )
    return True


def result_payload(result: PandaRunResult) -> dict:
    """Return a JSON-safe run summary for later experiment aggregation."""
    selected = result.ik.selected
    return {
        "safe_ik": {
            "skipped": result.ik_skipped,
            "success": None if result.ik_skipped else selected is not None,
            "failure_reason": result.ik.failure_reason,
            "candidate_count": len(result.ik.candidates),
            "wall_time": result.safe_ik_wall_time,
            "selected_seed": None if selected is None else selected.seed_index,
            "selected_iterations": None if selected is None else selected.iterations,
            "selected_position_error": (
                None if selected is None else selected.position_error
            ),
            "selected_orientation_error": (
                None if selected is None else selected.orientation_error
            ),
            "selected_minimum_barrier": (
                None if selected is None else selected.minimum_barrier
            ),
        },
        "trajectory": {
            "success": bool(result.status and result.status.success),
            "time": result.elapsed,
            "control_steps": result.steps,
            "control_effort": result.control_effort,
            "minimum_barrier": result.minimum_barrier,
            "total_qp_solve_time": result.total_qp_solve_time,
            "maximum_clf_slack": result.maximum_clf_slack,
            "qp_infeasible_count": result.qp_infeasible_count,
            "qp_solver_failure_count": result.qp_solver_failure_count,
            "candidate_switch_count": result.candidate_switch_count,
            "stage_transition_count": result.stage_transition_count,
            "final_stage": (
                None if result.status is None else result.status.stage_name
            ),
            "final_q": result.final_q.tolist(),
            "final_joint_error": (
                None if result.status is None else result.status.joint_error
            ),
            "final_position_error": (
                None if result.status is None else result.status.position_error
            ),
            "final_orientation_error": (
                None if result.status is None else result.status.orientation_error
            ),
        },
    }

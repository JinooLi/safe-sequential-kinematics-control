"""Serialize Panda rollout results and traces without importing their producers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..domain.panda_types import PandaControlTrace, PandaRunResult


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
        **(
            {"stage_position_error": trace.stage_position_error}
            if trace.stage_position_error is not None
            else {}
        ),
        **(
            {"stage_orientation_error": trace.stage_orientation_error}
            if trace.stage_orientation_error is not None
            else {}
        ),
        barrier_labels=np.asarray(trace.barrier_labels, dtype=str),
        barrier_values=trace.barrier_values,
        **({"q": trace.q} if trace.q is not None else {}),
        **({"command": trace.command} if trace.command is not None else {}),
        **(
            {"applied_velocity": trace.applied_velocity}
            if trace.applied_velocity is not None
            else {}
        ),
        **(
            {"command_applied": trace.command_applied}
            if trace.command_applied is not None
            else {}
        ),
        **(
            {
                "rdf_timings": trace.rdf_timings,
                "rdf_timing_columns": np.array(["distance", "selection", "gradient"]),
            }
            if trace.rdf_timings is not None
            else {}
        ),
        **({"qp_time": trace.qp_time} if trace.qp_time is not None else {}),
        qp_status=np.asarray(trace.qp_status, dtype=str),
        qp_message=np.asarray(trace.qp_message, dtype=str),
        safety_metadata=np.array(json.dumps(trace.safety_metadata)),
        **(
            {
                "minimum_self_barrier": trace.minimum_self_barrier,
                "minimum_self_pair": np.asarray(trace.minimum_self_pair, dtype=str),
            }
            if trace.minimum_self_barrier is not None
            else {}
        ),
    )
    return True


def result_payload(result: PandaRunResult) -> dict:
    """Return a JSON-safe run summary for later experiment aggregation."""
    selected = result.ik.selected
    return {
        "safety": result.safety_metadata,
        "failure": result.failure,
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
            "success": bool(
                not result.failure and result.status and result.status.success
            ),
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
            "final_stage_position_error": (
                None if result.status is None else result.status.stage_position_error
            ),
            "final_stage_orientation_error": (
                None if result.status is None else result.status.stage_orientation_error
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

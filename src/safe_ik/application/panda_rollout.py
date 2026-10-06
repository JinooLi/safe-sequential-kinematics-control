"""Backend-independent safe IK and trajectory orchestration."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from time import perf_counter
from typing import Callable

import numpy as np

from ..control.panda_controller import PandaTrajectoryController
from ..domain.ik_contracts import SafeIKResult
from ..domain.panda_config import PandaControllerConfig
from ..domain.panda_types import PandaControlTrace, PandaRunResult
from ..domain.safe_pose_ik import SafePoseIKSolver
from ..domain.safety_contracts import SafetyEvaluator
from .panda_progress import PandaProgress
from .panda_runtime import PandaRuntime


def _validate_ik_candidates(
    result: SafeIKResult,
    safety: SafetyEvaluator,
    config: PandaControllerConfig,
    progress: PandaProgress | None = None,
) -> tuple[SafeIKResult, dict]:
    """Gate a different IK backend's candidates using the actual control barriers.

    This checks endpoints, not the connecting path. Trajectory CBF constraints
    remain active during motion. Keep each candidate's IK minimum unchanged and
    record the separate trajectory-model minimum in the handoff diagnostics.
    """
    start = perf_counter()
    accepted, checks = [], []
    if progress is not None:
        progress.start(
            "IK trajectory-safety check", len(result.candidates), "candidate"
        )
    for candidate in result.candidates:
        evaluation = safety.evaluate(candidate.q)
        finite = bool(
            np.all(np.isfinite(evaluation.values))
            and np.all(np.isfinite(evaluation.gradients))
        )
        passed = finite and bool(np.all(evaluation.values >= 0.0))
        checks.append(
            {
                "seed_index": candidate.seed_index,
                "passed": passed,
                "finite": finite,
                "minimum_barrier": evaluation.minimum if finite else None,
                "violations": {
                    label: float(value)
                    for label, value in zip(evaluation.labels, evaluation.values)
                    if np.isfinite(value) and value < 0.0
                },
            }
        )
        if passed:
            accepted.append(candidate)
        if progress is not None:
            progress.update(len(checks), f"accepted={len(accepted)}")
    selected = (
        min(accepted, key=lambda item: np.linalg.norm(item.q - config.initial_q))
        if accepted
        else None
    )
    if progress is not None:
        progress.finish("passed" if selected is not None else "rejected")
    return (
        SafeIKResult(
            tuple(accepted),
            selected,
            None if selected is not None else "ik_candidates_unsafe_for_trajectory",
        ),
        {
            "model": config.safety.self_collision_model,
            "checked": len(checks),
            "accepted": len(accepted),
            "candidates": checks,
            "wall_time": perf_counter() - start,
        },
    )


def run_panda(
    config: PandaControllerConfig,
    runtime: PandaRuntime,
    *,
    observe: Callable[[], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
    on_transition: Callable[[float, str, str, str], None] | None = None,
    progress: PandaProgress | None = None,
) -> PandaRunResult:
    """Find safe pose IK, then drive with task-space CLFs and hard CBFs."""
    kinematics, safety, plant = runtime.kinematics, runtime.safety, runtime.plant
    start_time = plant.time
    ik_skipped = config.trajectory.mode == "direct"
    rdf_enabled = config.safety.self_collision_model == "rdf"
    ik_model = config.ik_self_collision_model
    different_ik_model = ik_model != config.safety.self_collision_model
    if not ik_skipped and different_ik_model and runtime.ik_safety is None:
        raise ValueError("a separate IK backend requires PandaRuntime.ik_safety")
    ik_safety = runtime.ik_safety if runtime.ik_safety is not None else safety
    ik_rdf_enabled = not ik_skipped and ik_model == "rdf"
    metadata = {
        "model": config.safety.self_collision_model,
        "safe_ik_model": None if ik_skipped else ik_model,
        "safe_ik_artifact_id": (
            getattr(ik_safety, "artifact_id", None) if not ik_skipped else None
        ),
        "artifact_id": getattr(safety, "artifact_id", None),
        "rdf_runtime": (
            getattr(safety, "runtime_metadata", None) if rdf_enabled else None
        ),
        "safe_ik_rdf_runtime": (
            getattr(ik_safety, "runtime_metadata", None) if ik_rdf_enabled else None
        ),
        "rdf_query_policy": (
            getattr(safety, "query_policy", None) if rdf_enabled else None
        ),
        "safe_ik_rdf_query_policy": (
            getattr(ik_safety, "query_policy", None) if ik_rdf_enabled else None
        ),
        "rdf_sphere_gate": (
            getattr(safety, "gate_metadata", None) if rdf_enabled else None
        ),
        "safe_ik_rdf_sphere_gate": (
            getattr(ik_safety, "gate_metadata", None) if ik_rdf_enabled else None
        ),
        "rdf_config": (
            dict(asdict(config.safety.rdf), artifact=str(config.safety.rdf.artifact))
            if rdf_enabled or ik_rdf_enabled
            else None
        ),
        "safe_ik_qp_time": None,
        "ik_candidate_validation": None,
    }
    initial_barrier = safety.evaluate(plant.q)
    metadata["initial_rdf_timings"] = (
        dict(getattr(safety, "last_timings", {})) if rdf_enabled else None
    )

    def self_minimum(evaluation):
        indices = [
            i for i, label in enumerate(evaluation.labels) if label.startswith("self_")
        ]
        if not indices:
            return None, None
        index = min(indices, key=lambda i: evaluation.values[i])
        return float(evaluation.values[index]), evaluation.labels[index]

    metadata["minimum_self_barrier"], metadata["minimum_self_pair"] = self_minimum(
        initial_barrier
    )
    if rdf_enabled and np.any(initial_barrier.values < 0):
        failure = {
            "reason": "unsafe_initial_state",
            "time": float(plant.time),
            "q": plant.q.tolist(),
            "solver_status": "not_run",
            "barriers": dict(
                zip(initial_barrier.labels, initial_barrier.values.tolist())
            ),
            "violations": {
                label: float(value)
                for label, value in zip(initial_barrier.labels, initial_barrier.values)
                if value < 0
            },
        }
        return PandaRunResult(
            SafeIKResult((), None, "unsafe_initial_state"),
            ik_skipped,
            None,
            0.0,
            0,
            plant.q,
            0.0,
            0.0,
            initial_barrier.minimum,
            0.0,
            0.0,
            0,
            0,
            0,
            0,
            None,
            metadata,
            failure,
        )
    if ik_skipped:
        ik_result = SafeIKResult((), None, None)
        safe_ik_wall_time = 0.0
        candidates = ()
    else:

        def failed_ik_result():
            return PandaRunResult(
                ik_result,
                False,
                None,
                0.0,
                0,
                config.initial_q.copy(),
                safe_ik_wall_time,
                0.0,
                safety.evaluate(config.initial_q).minimum,
                0.0,
                0.0,
                0,
                0,
                0,
                0,
                None,
                safety_metadata=metadata,
            )

        search = (
            runtime.ik_search_factory()
            if runtime.ik_search_factory is not None
            else nullcontext(SafePoseIKSolver(config, kinematics, ik_safety))
        )
        with search as ik_solver:

            def solve_ik(require_safe):
                kwargs = {}
                if progress is not None:
                    progress.start_ik(
                        config.safe_ik.max_restarts,
                        config.safe_ik.iteration_number,
                        require_safe,
                    )
                    kwargs["on_progress"] = progress.ik_update
                try:
                    outcome = ik_solver.solve(
                        config.initial_q, require_safe=require_safe, **kwargs
                    )
                except BaseException:
                    if progress is not None:
                        progress.finish("interrupted / error")
                    raise
                if progress is not None:
                    progress.finish(
                        "candidates found"
                        if outcome.selected is not None
                        else "no candidate"
                    )
                return outcome

            timing_before = dict(getattr(ik_safety, "total_timings", {}))
            ik_start = perf_counter()
            ik_result = solve_ik(True)
            safe_ik_wall_time = perf_counter() - ik_start

            def record_ik_metrics():
                if ik_rdf_enabled:
                    metadata["safe_ik_rdf"] = getattr(
                        ik_solver, "rdf_metrics", None
                    ) or {
                        "timings": {
                            k: v - timing_before.get(k, 0.0)
                            for k, v in getattr(ik_safety, "total_timings", {}).items()
                        },
                        "minimum_self_barrier": getattr(
                            ik_safety, "minimum_self_barrier", None
                        ),
                        "minimum_self_pair": getattr(
                            ik_safety, "minimum_self_pair", None
                        ),
                    }

            record_ik_metrics()
            if ik_result.selected is None:
                diagnostic = solve_ik(False)
                record_ik_metrics()
                failure = (
                    "pose_ik_unreachable"
                    if diagnostic.selected is None
                    else "safe_pose_ik_not_found"
                )
                ik_result = SafeIKResult((), None, failure)
                return failed_ik_result()

        if different_ik_model:
            ik_result, validation = _validate_ik_candidates(
                ik_result, safety, config, progress
            )
            metadata["ik_candidate_validation"] = validation
            safe_ik_wall_time += validation["wall_time"]
            if ik_result.selected is None:
                return failed_ik_result()
        candidates = tuple(
            sorted(
                ik_result.candidates,
                key=lambda item: np.linalg.norm(item.q - config.initial_q),
            )
        )
    controller = PandaTrajectoryController(
        config,
        kinematics,
        safety,
        candidates,
        runtime.solver,
    )
    period = config.trajectory.integration_period
    maximum_steps = int(np.ceil(config.trajectory.duration / period))
    control_effort = 0.0
    minimum_barrier = safety.evaluate(config.initial_q).minimum
    total_qp_solve_time = 0.0
    maximum_clf_slack = 0.0
    trace_q: list[np.ndarray] = []
    trace_command: list[np.ndarray] = []
    trace_time: list[float] = []
    trace_clf_value: list[float] = []
    trace_clf_slack: list[float] = []
    trace_stage_index: list[int] = []
    trace_stage_position_error, trace_stage_orientation_error = [], []
    trace_barrier_values: list[np.ndarray] = []
    trace_rdf_times, trace_qp_times, trace_qp_status = [], [], []
    trace_self_minimum, trace_self_pair = [], []
    failure = None

    def step():
        nonlocal control_effort, minimum_barrier, total_qp_solve_time, maximum_clf_slack
        nonlocal failure
        q = plant.q
        try:
            velocity = controller.get_velocity(q)
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            failure = {
                "reason": "numerical_failure",
                "time": float(plant.time),
                "q": q.tolist(),
                "solver_status": controller.last_qp_status,
                "detail": str(exc),
                "barrier_q": controller.last_barrier_q.tolist(),
                "barriers": dict(
                    zip(
                        controller.last_barrier.labels,
                        controller.last_barrier.values.tolist(),
                    )
                ),
            }
            return False
        for (
            previous_stage,
            next_stage,
            reason,
        ) in controller.consume_stage_transition_events():
            if on_transition is not None:
                on_transition(plant.time, previous_stage, next_stage, reason)
        diagnostics = controller.safety_diagnostics(q)
        trace_q.append(q.copy())
        trace_command.append(velocity.copy())
        trace_time.append(float(plant.time))
        trace_clf_value.append(float(diagnostics.clf_value))
        trace_clf_slack.append(float(controller.last_clf_slack))
        trace_stage_index.append(controller.stage_index)
        trace_stage_position_error.append(controller.last_status.stage_position_error)
        trace_stage_orientation_error.append(
            np.nan
            if controller.last_status.stage_orientation_error is None
            else controller.last_status.stage_orientation_error
        )
        trace_barrier_values.append(diagnostics.barrier_values)
        trace_rdf_times.append(
            [
                getattr(safety, "last_timings", {}).get(k, 0.0)
                for k in ("distance", "selection", "gradient")
            ]
        )
        trace_qp_times.append(controller.last_qp_solve_time)
        trace_qp_status.append(controller.last_qp_status)
        self_value, self_pair = self_minimum(controller.last_barrier)
        trace_self_minimum.append(float("nan") if self_value is None else self_value)
        trace_self_pair.append(self_pair or "")
        if self_value is not None and (
            metadata["minimum_self_barrier"] is None
            or self_value < metadata["minimum_self_barrier"]
        ):
            metadata["minimum_self_barrier"], metadata["minimum_self_pair"] = (
                self_value,
                self_pair,
            )
        control_effort += float(np.linalg.norm(velocity)) * period
        if diagnostics.barrier_values.size:
            minimum_barrier = min(
                minimum_barrier, float(np.min(diagnostics.barrier_values))
            )
        total_qp_solve_time += controller.last_qp_solve_time
        if np.isfinite(controller.last_clf_slack):
            maximum_clf_slack = max(maximum_clf_slack, controller.last_clf_slack)
        if controller.last_status is not None and (
            controller.last_status.qp_infeasible
            or controller.last_status.qp_solver_failed
        ):
            failure = {
                "reason": "qp_failure",
                "time": float(plant.time),
                "q": q.tolist(),
                "solver_status": controller.last_qp_status,
                "barriers": dict(
                    zip(diagnostics.barrier_labels, diagnostics.barrier_values.tolist())
                ),
            }
            return False
        plant.advance(velocity, period)
        status = controller.last_status
        return not (
            status is not None and (status.success or status.candidates_exhausted)
        )

    steps = 0
    if progress is not None:
        progress.start("Trajectory", maximum_steps, "step")
    while steps < maximum_steps and (should_continue is None or should_continue()):
        keep_running = step()
        steps += 1
        if progress is not None:
            progress.update(
                steps,
                f"simulation={plant.time - start_time:.3f}s/{config.trajectory.duration:g}s",
            )
        if observe is not None:
            observe()
        if not keep_running:
            break
    last_status = controller.last_status
    final_status = controller.status(
        plant.q,
        qp_infeasible=bool(last_status and last_status.qp_infeasible),
        qp_solver_failed=bool(last_status and last_status.qp_solver_failed),
    )
    if progress is not None:
        progress.finish(
            "failed"
            if failure
            else (
                "converged"
                if final_status.success
                else (
                    "candidates exhausted"
                    if final_status.candidates_exhausted
                    else "budget exhausted" if steps == maximum_steps else "stopped"
                )
            )
        )
    minimum_barrier = min(minimum_barrier, final_status.minimum_barrier)
    trace = PandaControlTrace(
        q=np.asarray(trace_q, dtype=float).reshape(-1, config.dof),
        command=np.asarray(trace_command, dtype=float).reshape(-1, config.dof),
        time=np.asarray(trace_time, dtype=float),
        clf_value=np.asarray(trace_clf_value, dtype=float),
        clf_slack=np.asarray(trace_clf_slack, dtype=float),
        stage_index=np.asarray(trace_stage_index, dtype=int),
        stage_names=controller.stage_names,
        stage_position_error=np.asarray(trace_stage_position_error),
        stage_orientation_error=np.asarray(trace_stage_orientation_error),
        barrier_labels=controller.barrier_labels,
        barrier_values=np.asarray(trace_barrier_values, dtype=float),
        rdf_timings=(
            np.asarray(trace_rdf_times, dtype=float).reshape(-1, 3)
            if rdf_enabled
            else None
        ),
        qp_time=np.asarray(trace_qp_times),
        qp_status=tuple(trace_qp_status),
        safety_metadata=metadata,
        minimum_self_barrier=np.asarray(trace_self_minimum),
        minimum_self_pair=tuple(trace_self_pair),
    )
    return PandaRunResult(
        ik_result,
        ik_skipped,
        final_status,
        plant.time - start_time,
        steps,
        plant.q,
        safe_ik_wall_time,
        control_effort,
        minimum_barrier,
        total_qp_solve_time,
        maximum_clf_slack,
        controller.qp_infeasible_count,
        controller.qp_solver_failure_count,
        controller.candidate_switch_count,
        controller.stage_transition_count,
        trace,
        metadata,
        failure,
    )

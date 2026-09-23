"""Backend-independent safe IK and trajectory orchestration."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from time import perf_counter
from typing import Callable

import numpy as np

from ..control.panda_controller import PandaTrajectoryController
from ..domain.panda_config import PandaControllerConfig
from ..domain.safe_pose_ik import SafeIKResult, SafePoseIKSolver
from .panda_results import PandaControlTrace, PandaRunResult
from .panda_runtime import PandaRuntime


def run_panda(
    config: PandaControllerConfig,
    runtime: PandaRuntime,
    *,
    observe: Callable[[], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
    on_transition: Callable[[float, str, str, str], None] | None = None,
) -> PandaRunResult:
    """Find safe pose IK, then drive with task-space CLFs and hard CBFs."""
    kinematics, safety, plant = runtime.kinematics, runtime.safety, runtime.plant
    start_time = plant.time
    ik_skipped = config.trajectory.mode == "direct"
    rdf_enabled = config.safety.self_collision_model == "rdf"
    metadata = {
        "model": config.safety.self_collision_model,
        "artifact_id": getattr(safety, "artifact_id", None),
        "rdf_config": (
            dict(asdict(config.safety.rdf), artifact=str(config.safety.rdf.artifact))
            if rdf_enabled
            else None
        ),
        "safe_ik_qp_time": None,
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
        search = (
            runtime.ik_search_factory()
            if runtime.ik_search_factory is not None
            else nullcontext(SafePoseIKSolver(config, kinematics, safety))
        )
        with search as ik_solver:
            timing_before = dict(getattr(safety, "total_timings", {}))
            ik_start = perf_counter()
            ik_result = ik_solver.solve(config.initial_q, require_safe=True)
            safe_ik_wall_time = perf_counter() - ik_start

            def record_ik_metrics():
                if rdf_enabled:
                    metadata["safe_ik_rdf"] = getattr(
                        ik_solver, "rdf_metrics", None
                    ) or {
                        "timings": {
                            k: v - timing_before.get(k, 0.0)
                            for k, v in safety.total_timings.items()
                        },
                        "minimum_self_barrier": safety.minimum_self_barrier,
                        "minimum_self_pair": safety.minimum_self_pair,
                    }

            record_ik_metrics()
            if ik_result.selected is None:
                diagnostic = ik_solver.solve(config.initial_q, require_safe=False)
                record_ik_metrics()
                failure = (
                    "pose_ik_unreachable"
                    if diagnostic.selected is None
                    else "safe_pose_ik_not_found"
                )
                ik_result = SafeIKResult((), None, failure)
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
    while steps < maximum_steps and (should_continue is None or should_continue()):
        keep_running = step()
        steps += 1
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
    minimum_barrier = min(minimum_barrier, final_status.minimum_barrier)
    trace = PandaControlTrace(
        q=np.asarray(trace_q, dtype=float).reshape(-1, config.dof),
        command=np.asarray(trace_command, dtype=float).reshape(-1, config.dof),
        time=np.asarray(trace_time, dtype=float),
        clf_value=np.asarray(trace_clf_value, dtype=float),
        clf_slack=np.asarray(trace_clf_slack, dtype=float),
        stage_index=np.asarray(trace_stage_index, dtype=int),
        stage_names=tuple(
            [f"{frame}_position" for frame in controller.stage_frames]
            + ["end_effector_pose"]
        ),
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

"""Backend-independent safe IK and trajectory orchestration."""

from __future__ import annotations

from contextlib import nullcontext
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
            ik_start = perf_counter()
            ik_result = ik_solver.solve(config.initial_q, require_safe=True)
            safe_ik_wall_time = perf_counter() - ik_start
            if ik_result.selected is None:
                diagnostic = ik_solver.solve(config.initial_q, require_safe=False)
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

    def step():
        nonlocal control_effort, minimum_barrier, total_qp_solve_time, maximum_clf_slack
        q = plant.q
        velocity = controller.get_velocity(q)
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
    )

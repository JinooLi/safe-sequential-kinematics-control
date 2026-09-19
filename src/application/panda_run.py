"""Safe pose-IK search followed by Panda CLF-CBF trajectory control."""

from __future__ import annotations

import argparse
import time
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from ..adapters.osqp_solver import OSQPSolver
from ..adapters.panda_kinematics import PandaKinematics
from ..control.panda_controller import (PandaControlStatus,
                                        PandaTrajectoryController)
from ..domain.safe_pose_ik import SafeIKResult, SafePoseIKSolver
from ..domain.spatial_safety import PandaCapsuleSafety
from .panda_configuration import PandaControllerConfig, load_panda_config


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


def build_panda_components(config: PandaControllerConfig):
    frames = {
        frame
        for capsule in config.safety.capsules
        for frame in (capsule.start_frame, capsule.end_frame)
    }
    frames.update(config.trajectory.stage_frames)
    kinematics = PandaKinematics(config.model_path, config.ee_frame, frames)
    if np.any(config.initial_q < kinematics.lower_limits) or np.any(
        config.initial_q > kinematics.upper_limits
    ):
        raise ValueError("robot.initial_q violates Panda joint limits")
    safety = PandaCapsuleSafety(
        kinematics,
        config.safety.capsules,
        config.obstacles,
        config.safety.margin,
        config.safety.self_collision_exclusions,
    )
    return kinematics, safety


def run_panda(config: PandaControllerConfig, visualize=False) -> PandaRunResult:
    """Find safe pose IK, then drive with task-space CLFs and hard CBFs."""
    kinematics, safety = build_panda_components(config)
    ik_skipped = config.trajectory.mode == "direct"
    if ik_skipped:
        ik_result = SafeIKResult((), None, None)
        safe_ik_wall_time = 0.0
        candidates = ()
    else:
        ik_solver = SafePoseIKSolver(config, kinematics, safety)
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
        OSQPSolver(
            max_iterations=config.qp.max_iterations,
            absolute_tolerance=config.qp.absolute_tolerance,
            relative_tolerance=config.qp.relative_tolerance,
        ),
    )
    model = mujoco.MjModel.from_xml_path(str(config.model_path))
    goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal_marker")
    if goal_id >= 0:
        model.site_pos[goal_id] = config.target.position
    for index, obstacle in enumerate(config.obstacles, start=1):
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_geom{index}"
        )
        if geom_id >= 0:
            model.geom_pos[geom_id] = obstacle.position
            model.geom_size[geom_id, 0] = obstacle.radius
            model.geom_rgba[geom_id, 3] = 0.6 if obstacle.enabled else 0.0
    data = mujoco.MjData(model)
    data.qpos[:] = config.initial_q
    mujoco.mj_forward(model, data)
    period = config.trajectory.integration_period
    maximum_steps = int(np.ceil(config.trajectory.duration / period))
    control_effort = 0.0
    minimum_barrier = safety.evaluate(config.initial_q).minimum
    total_qp_solve_time = 0.0
    maximum_clf_slack = 0.0
    trace_time: list[float] = []
    trace_clf_value: list[float] = []
    trace_clf_slack: list[float] = []
    trace_stage_index: list[int] = []
    trace_barrier_values: list[np.ndarray] = []

    def step():
        nonlocal control_effort, minimum_barrier, total_qp_solve_time, maximum_clf_slack
        q = data.qpos.copy()
        velocity = controller.get_velocity(q)
        for previous_stage, next_stage, reason in controller.consume_stage_transition_events():
            print(
                f"[Panda stage] t={data.time:.3f}s: "
                f"{previous_stage} -> {next_stage} ({reason})"
            )
        diagnostics = controller.safety_diagnostics(q)
        trace_time.append(float(data.time))
        trace_clf_value.append(float(diagnostics.clf_value))
        trace_clf_slack.append(float(controller.last_clf_slack))
        trace_stage_index.append(controller.stage_index)
        trace_barrier_values.append(diagnostics.barrier_values)
        control_effort += float(np.linalg.norm(velocity)) * period
        minimum_barrier = min(minimum_barrier, float(np.min(diagnostics.barrier_values)))
        total_qp_solve_time += controller.last_qp_solve_time
        if np.isfinite(controller.last_clf_slack):
            maximum_clf_slack = max(maximum_clf_slack, controller.last_clf_slack)
        if controller.last_status is not None and (
            controller.last_status.qp_infeasible
            or controller.last_status.qp_solver_failed
        ):
            return False
        data.qvel[:] = velocity
        mujoco.mj_integratePos(model, data.qpos, data.qvel, period)
        data.time += period
        mujoco.mj_forward(model, data)
        status = controller.last_status
        return not (
            status is not None and (status.success or status.candidates_exhausted)
        )

    steps = 0
    if visualize:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and steps < maximum_steps:
                keep_running = step()
                viewer.sync()
                steps += 1
                if not keep_running:
                    break

            # Keep the final simulation state visible until the user closes
            # the MuJoCo viewer window.
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
    else:
        while steps < maximum_steps:
            keep_running = step()
            steps += 1
            if not keep_running:
                break
    final_status = controller._status(data.qpos.copy())
    trace = PandaControlTrace(
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
        steps * period,
        steps,
        data.qpos.copy(),
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


def save_panda_control_plot(
    trace: PandaControlTrace | None, path: str | Path, *, show: bool = False
) -> bool:
    """Save CLF, slack, and CBF histories; optionally open the figure."""
    if trace is None or trace.time.size == 0:
        return False

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        3, 1, sharex=True, figsize=(11, 9), layout="constrained"
    )
    axes[0].plot(trace.time, trace.clf_value, color="tab:blue")
    axes[0].set_ylabel("CLF  V(q)")
    axes[0].set_title("Panda CLF-CBF QP diagnostics")

    axes[1].plot(trace.time, trace.clf_slack, color="tab:orange")
    axes[1].set_ylabel("CLF slack  δ")

    for values in trace.barrier_values.T:
        axes[2].plot(trace.time, values, color="tab:gray", alpha=0.32, linewidth=0.8)
    axes[2].plot(
        trace.time,
        np.min(trace.barrier_values, axis=1),
        color="black",
        linewidth=2.0,
        label="minimum CBF",
    )
    axes[2].axhline(
        0.0, color="tab:red", linestyle="--", linewidth=1.0, label="safety boundary"
    )
    axes[2].set_ylabel("CBF  hᵢ(q)")
    axes[2].legend(loc="best")

    for axis in axes:
        axis.grid(True, alpha=0.3)
    transition_samples = np.flatnonzero(np.diff(trace.stage_index) != 0) + 1
    for sample in transition_samples:
        for axis in axes:
            axis.axvline(trace.time[sample], color="tab:purple", alpha=0.35, linewidth=0.8)
        stage = int(trace.stage_index[sample])
        if stage < len(trace.stage_names):
            axes[0].annotate(
                trace.stage_names[stage],
                (trace.time[sample], trace.clf_value[sample]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7,
                rotation=90,
            )
    axes[-1].set_xlabel("time [s]")
    plot_path = Path(path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return True


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="settings/panda_controller.yaml")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--output", default="outputs/panda_run_summary.json")
    parser.add_argument(
        "--plot",
        default=None,
        help="control diagnostic PNG path (default: alongside --output)",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="open the diagnostic plot after the rollout",
    )
    args = parser.parse_args()
    config = load_panda_config(Path(args.config))
    result = run_panda(config, visualize=args.visualize)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_payload(result), indent=2), encoding="utf-8"
    )
    plot_path = (
        Path(args.plot)
        if args.plot is not None
        else output_path.with_name(f"{output_path.stem}_control.png")
    )
    trace_path = output_path.with_name(f"{output_path.stem}_control_trace.npz")
    if save_panda_control_plot(result.trace, plot_path, show=args.show_plot):
        save_panda_control_trace(result.trace, trace_path)
        print(f"control plot: {plot_path}")
        print(f"control trace: {trace_path}")
    if result.ik_skipped:
        print("safe IK: skipped (direct mode)")
    else:
        print(f"safe IK candidates: {len(result.ik.candidates)}")
    if result.ik.failure_reason:
        print(f"failure: {result.ik.failure_reason}")
        return
    print(f"trajectory time: {result.elapsed:.3f} s ({result.steps} control steps)")
    print(f"final q: {result.final_q}")
    print(f"status: {result.status}")
    print(f"summary: {output_path}")


if __name__ == "__main__":
    main()

"""Execute planar case pairs and coordinate worker processes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml

from ..adapters.osqp_solver import OSQPSolver
from ..adapters.planar_kinematics import RobotInterface
from ..application.artifacts import create_run_directory, save_run_provenance
from ..application.planar_configuration import (
    load_controller_config,
    validate_geometry_consistency,
)
from ..application.planar_simulation import run_simulation
from ..control.planar_controller import QPController, wrap_to_pi
from ..domain.planar_config import ControllerConfig
from ..logging.recorder import Recorder
from ..visualization.comparison import _write_paired_plot
from .analysis import (
    _failure_reason,
    _nonzero_slack_ratio_to_goal,
    _sample_interval,
    _stage_durations,
    classify_outcome,
)
from .models import (
    COMPARISONS,
    OUTCOME_LABELS,
    ComparisonWorkerInput,
    ExperimentCase,
    RunResult,
)
from .reporting import _write_summary
from .sampling import sample_case


def _case_config(
    base_config: ControllerConfig,
    case: ExperimentCase,
    sequential_enabled: bool,
):
    return replace(
        base_config,
        ee_target=np.asarray(case.target, dtype=float),
        ee_orientation=case.target_yaw,
        obstacles=case.obstacles,
        sequential_end_effector=replace(
            base_config.sequential_end_effector, enabled=sequential_enabled
        ),
    )


def _final_end_effector_trajectory(q, link_lengths):
    """Return final-link [x, y] samples and wrapped yaw for planar q samples."""
    q = np.asarray(q, dtype=float)
    angles = np.cumsum(q, axis=1)
    lengths = np.asarray(link_lengths, dtype=float)
    xy = np.stack(
        (
            np.sum(lengths * np.cos(angles), axis=1),
            np.sum(lengths * np.sin(angles), axis=1),
        ),
        axis=1,
    )
    yaw = (angles[:, -1] + np.pi) % (2.0 * np.pi) - np.pi
    return xy, yaw


def _save_trajectory(path: Path, recorder: Recorder, link_lengths):
    arrays = recorder.arrays()
    final_xy, final_yaw = _final_end_effector_trajectory(arrays["q"], link_lengths)
    saved = {
        "t": arrays["t"],
        "q": arrays["q"],
        "final_xy": final_xy,
        "final_yaw": final_yaw,
        "active_xy": arrays["ee"][:, :2],
        "barrier": arrays["barrier"],
    }
    for key in ("u", "clf", "active_link", "clf_slack", "qp_solve_time"):
        if key in arrays:
            saved[key] = arrays[key]
    np.savez_compressed(
        path,
        barrier_labels=np.asarray(recorder.barrier_labels, dtype=str),
        **saved,
    )
    return arrays, final_xy, final_yaw


def _execute_case(
    base_config: ControllerConfig,
    case: ExperimentCase,
    sequential_enabled: bool,
    method: str,
    *,
    mujoco_path: Path,
    urdf_path: Path,
    duration: float,
    experiment_goal_tolerance: float,
    yaw_tolerance: float,
    barrier_tolerance: float,
    slack_active_threshold: float,
    initial_q: tuple[float, ...],
    output_dir: Path,
) -> RunResult:
    config = _case_config(base_config, case, sequential_enabled)
    controller = QPController(config, OSQPSolver(), verbose=False)
    robot = RobotInterface(urdf_path, config.sequential_end_effector.frame_names)
    recorder = Recorder()
    trajectory_path = output_dir / f"case_{case.case_id:03d}_{method}.npz"
    run_simulation(
        mujoco_path,
        controller,
        robot,
        target_position=config.ee_target,
        obstacle_enabled=[obstacle.enabled for obstacle in config.obstacles],
        duration=duration,
        visualize=False,
        recorder=recorder,
        data_path=output_dir / "unused.npz",
        csv_path=output_dir / "unused.csv",
        plot_path=output_dir / "unused.png",
        show_recording_gui=False,
        verbose=False,
        save_recording=False,
        initial_q=initial_q,
        kinematic=config.simulation.kinematic,
    )
    arrays, final_xy, final_yaw = _save_trajectory(
        trajectory_path, recorder, config.safety.link_lengths
    )
    position_error = float(np.linalg.norm(final_xy[-1] - config.ee_target))
    yaw_error = abs(wrap_to_pi(final_yaw[-1] - config.ee_orientation))
    barrier = np.asarray(arrays["barrier"], dtype=float)
    min_barrier = float(np.min(barrier)) if barrier.size else float("inf")
    success = bool(
        position_error <= experiment_goal_tolerance
        and yaw_error <= yaw_tolerance
        and min_barrier >= barrier_tolerance
    )
    dt = _sample_interval(arrays["t"])
    command = np.asarray(arrays["u"], dtype=float)
    command_norm = np.linalg.norm(command, axis=1)
    goal_reached = (
        np.linalg.norm(final_xy - config.ee_target, axis=1) <= experiment_goal_tolerance
    ) & (
        np.abs(
            np.asarray([wrap_to_pi(yaw - config.ee_orientation) for yaw in final_yaw])
        )
        <= yaw_tolerance
    )
    completion_time = (
        float(arrays["t"][np.flatnonzero(goal_reached)[0]])
        if np.any(goal_reached)
        else None
    )
    slack = np.asarray(arrays.get("clf_slack", ()), dtype=float)
    nonzero_slack_ratio_to_goal = _nonzero_slack_ratio_to_goal(
        slack, goal_reached, slack_active_threshold
    )
    qp_time = np.asarray(arrays.get("qp_solve_time", ()), dtype=float)
    active_link_values = np.asarray(arrays.get("active_link", ()), dtype=int)
    switch_reasons = tuple(controller.stage_switch_reasons)
    failure_reason = _failure_reason(
        success,
        min_barrier,
        barrier_tolerance,
        controller.qp_infeasible_count,
        sequential_enabled,
        controller.end_effector_index,
        config.dof,
        switch_reasons,
        qp_solver_failure_count=controller.qp_solver_failure_count,
    )
    return RunResult(
        success=success,
        final_position=tuple(map(float, final_xy[-1])),
        final_yaw=float(final_yaw[-1]),
        position_error=position_error,
        yaw_error=yaw_error,
        min_barrier=min_barrier,
        active_link=controller.end_effector_index,
        switch_reason=controller.last_end_effector_switch_reason,
        failure_reason=failure_reason,
        completion_time=completion_time,
        control_effort=float(np.sum(command_norm) * dt),
        max_joint_velocity=float(np.max(command_norm)) if command_norm.size else 0.0,
        mean_clf_slack=float(np.nanmean(slack)) if slack.size else None,
        max_clf_slack=float(np.nanmax(slack)) if slack.size else None,
        nonzero_clf_slack_ratio_to_goal=nonzero_slack_ratio_to_goal,
        total_qp_solve_time=float(np.nansum(qp_time)),
        mean_qp_solve_time=float(np.nanmean(qp_time)) if qp_time.size else None,
        qp_infeasible_count=controller.qp_infeasible_count,
        qp_solver_failure_count=controller.qp_solver_failure_count,
        stage_durations=_stage_durations(
            active_link_values, dt, config.dof, sequential_enabled
        ),
        stage_switch_reasons=switch_reasons,
        trajectory_file=trajectory_path.name,
    )


def run_case_pair(worker_input: ComparisonWorkerInput):
    """Run direct and sequential task-decomposition control on one shared case."""
    common = {
        "mujoco_path": Path(worker_input.mujoco_path),
        "urdf_path": Path(worker_input.urdf_path),
        "duration": worker_input.duration,
        "experiment_goal_tolerance": worker_input.goal_tolerance,
        "yaw_tolerance": worker_input.yaw_tolerance,
        "barrier_tolerance": worker_input.barrier_tolerance,
        "slack_active_threshold": worker_input.slack_active_threshold,
        "initial_q": worker_input.initial_q,
        "output_dir": Path(worker_input.output_dir),
    }
    results = {
        "direct": _execute_case(
            worker_input.base_config, worker_input.case, False, "direct", **common
        ),
        "sequential": _execute_case(
            worker_input.base_config, worker_input.case, True, "sequential", **common
        ),
    }
    return worker_input.case, results


def run_comparison(args):
    root = Path(args.root).resolve()
    config_path = root / args.config
    mujoco_path = root / args.mujoco
    urdf_path = root / args.urdf
    base_config = load_controller_config(config_path)
    experiment = base_config.sequential_experiment
    overrides = {
        name: value
        for name, value in (
            ("cases", args.cases),
            ("workers", args.workers),
            ("duration", args.duration),
        )
        if value is not None
    }
    if overrides:
        experiment = replace(experiment, **overrides)
    if experiment.control_mode is not None:
        base_config = replace(base_config, control_mode=experiment.control_mode)
    validate_geometry_consistency(base_config, urdf_path, mujoco_path)

    if (
        experiment.cases < 1
        or experiment.workers < 1
        or not 0 < experiment.duration < float("inf")
    ):
        raise ValueError("cases, workers and duration must be positive and finite")
    # Validate frames before sampling or launching worker processes.
    RobotInterface(urdf_path, base_config.sequential_end_effector.frame_names)
    base_config = replace(base_config, sequential_experiment=experiment)
    output_dir = create_run_directory(root / experiment.output_dir, args.run_name)
    raw = yaml.safe_load(config_path.read_text())
    raw.setdefault("sequential_experiment", {}).update(asdict(experiment))
    raw["control_mode"] = base_config.control_mode
    resources = [mujoco_path, urdf_path]
    if base_config.safety.representation == "bernstein":
        resources.append(Path(base_config.safety.rdf_model_path))
    save_run_provenance(
        output_dir,
        config_path=config_path,
        config=base_config,
        root=root,
        resources=resources,
        effective_yaml=raw,
    )
    rng = np.random.default_rng(experiment.seed)
    worker_inputs = []
    for case_id in range(experiment.cases):
        # sample_case re-samples until a collision-free IK branch exists, so
        # impossible targets never enter either side of the comparison.
        case = sample_case(case_id, rng, base_config, experiment)

        worker_inputs.append(
            ComparisonWorkerInput(
                base_config=base_config,
                case=case,
                mujoco_path=str(mujoco_path),
                urdf_path=str(urdf_path),
                duration=experiment.duration,
                goal_tolerance=experiment.goal_tolerance,
                yaw_tolerance=experiment.yaw_tolerance,
                barrier_tolerance=experiment.barrier_tolerance,
                slack_active_threshold=experiment.slack_active_threshold,
                initial_q=experiment.initial_q,
                output_dir=str(output_dir),
            )
        )

    completed = []
    if experiment.workers == 1:
        for index, worker_input in enumerate(worker_inputs, start=1):
            completed.append(run_case_pair(worker_input))
            print(
                f"[{index}/{experiment.cases}] completed case {worker_input.case.case_id}"
            )
    else:
        with ProcessPoolExecutor(max_workers=experiment.workers) as executor:
            futures = [
                executor.submit(run_case_pair, worker_input)
                for worker_input in worker_inputs
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                case, results = future.result()
                completed.append((case, results))
                print(f"[{index}/{experiment.cases}] completed case {case.case_id}")

    records = []
    for case, results in sorted(completed, key=lambda item: item[0].case_id):
        record = {
            "case_id": case.case_id,
            "target": list(case.target),
            "target_yaw": case.target_yaw,
            "obstacles": [asdict(obstacle) for obstacle in case.obstacles],
            "ik_rejections_before_acceptance": case.ik_rejections_before_acceptance,
            **{method: asdict(result) for method, result in results.items()},
            "comparisons": {
                name: classify_outcome(
                    results[baseline].success, results[candidate].success
                )
                for name, (baseline, candidate) in COMPARISONS.items()
            },
        }
        records.append(record)

    _write_summary(output_dir, records, experiment, base_config)
    _write_paired_plot(output_dir, records)
    print(f"Saved comparison: {output_dir}")
    for label in OUTCOME_LABELS.values():
        ids = [
            record["case_id"]
            for record in records
            if record["comparisons"]["task_decomposition"] == label
        ]
        print(f"{label}: {len(ids)} cases {ids}")

"""Headless randomized comparison of sequential end-effector control.

Run a reproducible batch with the controller configuration currently in use:

    python -m src.application.compare_sequential run --cases 100 --seed 0

Render any saved case without starting a MuJoCo viewer:

    python -m src.application.compare_sequential visualize \
        --results-dir outputs/sequential_comparison/<run> --case-id 17

Render every saved case:

    python -m src.application.compare_sequential visualize-all \
        --results-dir outputs/sequential_comparison/<run>
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import strftime

import numpy as np

from ..adapters.osqp_solver import OSQPSolver
from ..adapters.pinocchio_kinematics import RobotInterface
from ..control.qp_controller import QPController, wrap_to_pi
from ..domain.goal_configuration import solve_goal_configurations
from ..domain.safety import obstacles_from_config
from ..logging.recorder import Recorder
from .configuration import (
    ControllerConfig,
    ObstacleConfig,
    SequentialExperimentConfig,
    load_controller_config,
    validate_geometry_consistency,
)
from .simulation import run_simulation


@dataclass(frozen=True)
class ExperimentCase:
    """One target/obstacle placement shared by all three ablation methods."""

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
    total_qp_solve_time: float
    mean_qp_solve_time: float | None
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


def classify_outcome(nonsequential_success: bool, sequential_success: bool) -> str:
    """Return one of the four comparison categories in a stable order."""
    return OUTCOME_LABELS[(bool(nonsequential_success), bool(sequential_success))]


def _json_default(value):
    """Convert NumPy scalar diagnostics to JSON-native scalar values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _distance_to_segment(point, start, end):
    segment = end - start
    length_squared = float(segment @ segment)
    if length_squared == 0.0:
        return float(np.linalg.norm(point - start))
    ratio = np.clip(((point - start) @ segment) / length_squared, 0.0, 1.0)
    return float(np.linalg.norm(point - (start + ratio * segment)))


def _initial_robot_clearance(point, initial_q, link_lengths):
    """Distance from a point to the configured initial serial-robot center line."""
    start = np.array([0.0, 0.0])
    clearance = np.inf
    for angle, length in zip(np.cumsum(initial_q), link_lengths):
        end = start + length * np.array([np.cos(angle), np.sin(angle)])
        clearance = min(clearance, _distance_to_segment(point, start, end))
        start = end
    return float(clearance)


def has_feasible_goal_configuration(
    target,
    target_yaw: float,
    obstacles: tuple[ObstacleConfig, ...],
    config: ControllerConfig,
) -> bool:
    """Return whether at least one safe IK branch exists for a sampled case."""
    candidates = solve_goal_configurations(
        target,
        target_yaw,
        config.safety.link_lengths,
        obstacles_from_config(
            [{"pos": obstacle.position, "radius": obstacle.radius} for obstacle in obstacles]
        ),
        link_radius=config.safety.link_radius,
        safe_distance=config.safety.safe_distance,
        joint_limits=config.safety.joint_limits,
    )
    return any(not candidate.colliding for candidate in candidates)


def sample_case(
    case_id: int,
    rng: np.random.Generator,
    base_config: ControllerConfig,
    experiment: SequentialExperimentConfig,
) -> ExperimentCase:
    """Sample a meaningful, initially safe target and two-obstacle placement."""
    obstacle_count = len(base_config.obstacles)
    if obstacle_count != 2:
        raise ValueError("random comparison currently requires exactly two obstacles")
    lengths = np.asarray(base_config.safety.link_lengths, dtype=float)
    robot_exclusion = (
        base_config.safety.link_radius
        + base_config.safety.safe_distance
        + experiment.initial_clearance_margin
    )

    ik_rejections = 0
    for _ in range(experiment.max_case_attempts):
        target_angle = rng.uniform(*experiment.target_angle_range)
        target_radius = rng.uniform(*experiment.target_radius_range)
        target = target_radius * np.array([np.cos(target_angle), np.sin(target_angle)])
        target_yaw = float(rng.uniform(*experiment.target_orientation_range))
        obstacles = []
        for _ in range(obstacle_count):
            for _ in range(experiment.max_obstacle_attempts):
                radius = float(rng.uniform(*experiment.obstacle_radius_range))
                position = rng.uniform(*experiment.obstacle_position_range, size=2)
                if _initial_robot_clearance(position, experiment.initial_q, lengths) <= radius + robot_exclusion:
                    continue
                if np.linalg.norm(position - target) <= (
                    radius
                    + base_config.safety.link_radius
                    + base_config.safety.safe_distance
                    + experiment.target_clearance_margin
                ):
                    continue
                if any(
                    np.linalg.norm(position - np.asarray(existing.position))
                    <= radius + existing.radius + experiment.target_clearance_margin
                    for existing in obstacles
                ):
                    continue
                obstacles.append(
                    ObstacleConfig((float(position[0]), float(position[1])), radius, True)
                )
                break
            else:
                break
        if len(obstacles) == obstacle_count:
            if not has_feasible_goal_configuration(
                target, target_yaw, tuple(obstacles), base_config
            ):
                ik_rejections += 1
                continue
            return ExperimentCase(
                case_id,
                (float(target[0]), float(target[1])),
                target_yaw,
                tuple(obstacles),
                ik_rejections,
            )
    raise RuntimeError("unable to sample a valid experiment case")


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
        sequential_end_effector=replace(base_config.sequential_end_effector, enabled=sequential_enabled),
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
        **saved,
    )
    return arrays, final_xy, final_yaw


def _sample_interval(times: np.ndarray) -> float:
    """Return the recorder's nominal simulation interval."""
    if len(times) < 2:
        return 0.0
    return float(np.median(np.diff(times)))


def _stage_durations(active_link: np.ndarray, dt: float, dof: int, sequential: bool) -> tuple[float, ...]:
    """Measure time spent with each sequential task endpoint active."""
    if not sequential or active_link.size == 0:
        return ()
    return tuple(float(np.count_nonzero(active_link == index) * dt) for index in range(1, dof + 1))


def _failure_reason(success: bool, min_barrier: float, barrier_tolerance: float, qp_infeasible_count: int,
                    sequential: bool, active_link: int, dof: int, switch_reasons: tuple[str, ...]) -> str | None:
    """Classify an unsuccessful rollout without conflating it with paired outcome."""
    if success:
        return None
    if min_barrier < barrier_tolerance:
        return "safety_violation"
    if qp_infeasible_count:
        return "qp_infeasible"
    if sequential and active_link < dof and any("CBF" in reason for reason in switch_reasons):
        return "cbf_boundary_stall"
    return "timeout"


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
    ) & (np.abs(np.asarray([wrap_to_pi(yaw - config.ee_orientation) for yaw in final_yaw])) <= yaw_tolerance)
    completion_time = float(arrays["t"][np.flatnonzero(goal_reached)[0]]) if np.any(goal_reached) else None
    slack = np.asarray(arrays.get("clf_slack", ()), dtype=float)
    qp_time = np.asarray(arrays.get("qp_solve_time", ()), dtype=float)
    active_link_values = np.asarray(arrays.get("active_link", ()), dtype=int)
    switch_reasons = tuple(controller.stage_switch_reasons)
    failure_reason = _failure_reason(
        success, min_barrier, barrier_tolerance, controller.qp_infeasible_count,
        sequential_enabled, controller.end_effector_index, config.dof, switch_reasons,
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
        total_qp_solve_time=float(np.nansum(qp_time)),
        mean_qp_solve_time=float(np.nanmean(qp_time)) if qp_time.size else None,
        qp_infeasible_count=controller.qp_infeasible_count,
        stage_durations=_stage_durations(active_link_values, dt, config.dof, sequential_enabled),
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
        "initial_q": worker_input.initial_q,
        "output_dir": Path(worker_input.output_dir),
    }
    results = {
        "direct": _execute_case(worker_input.base_config, worker_input.case, False, "direct", **common),
        "sequential": _execute_case(worker_input.base_config, worker_input.case, True, "sequential", **common),
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

    run_name = args.run_name or strftime("run_%Y%m%d_%H%M%S")
    output_dir = root / experiment.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
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
                initial_q=experiment.initial_q,
                output_dir=str(output_dir),
            )
        )

    completed = []
    if experiment.workers == 1:
        for index, worker_input in enumerate(worker_inputs, start=1):
            completed.append(run_case_pair(worker_input))
            print(f"[{index}/{experiment.cases}] completed case {worker_input.case.case_id}")
    else:
        with ProcessPoolExecutor(max_workers=experiment.workers) as executor:
            futures = [executor.submit(run_case_pair, worker_input) for worker_input in worker_inputs]
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
                name: classify_outcome(results[baseline].success, results[candidate].success)
                for name, (baseline, candidate) in COMPARISONS.items()
            },
        }
        records.append(record)

    _write_summary(output_dir, records, experiment, base_config)
    print(f"Saved comparison: {output_dir}")
    for label in OUTCOME_LABELS.values():
        ids = [
            record["case_id"]
            for record in records
            if record["comparisons"]["task_decomposition"] == label
        ]
        print(f"{label}: {len(ids)} cases {ids}")


def _distribution(values):
    """Summarize finite scalar values in a JSON-friendly form."""
    values = np.asarray([value for value in values if value is not None], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def _paired_analysis(records):
    """Return outcome, failure, and paired metric summaries for one run."""
    result = {
        "case_count": len(records),
        "methods": {},
        "comparisons": {},
    }
    metrics = (
        "position_error", "yaw_error", "min_barrier", "control_effort",
        "max_joint_velocity", "total_qp_solve_time", "mean_qp_solve_time",
        "mean_clf_slack", "max_clf_slack",
    )
    for method in METHODS:
        runs = [record[method] for record in records]
        failure_counts = {}
        for run in runs:
            if run["failure_reason"] is not None:
                failure_counts[run["failure_reason"]] = failure_counts.get(run["failure_reason"], 0) + 1
        result["methods"][method] = {
            "success_count": sum(run["success"] for run in runs),
            "success_rate": float(np.mean([run["success"] for run in runs])),
            "failure_reasons": failure_counts,
            "metrics": {metric: _distribution([run[metric] for run in runs]) for metric in metrics},
        }
    for name, (baseline, candidate) in COMPARISONS.items():
        outcomes = {label: 0 for label in OUTCOME_LABELS.values()}
        for record in records:
            outcomes[record["comparisons"][name]] += 1
        successful_pairs = [
            record for record in records
            if record[baseline]["success"] and record[candidate]["success"]
        ]
        deltas = {}
        for metric in ("completion_time", "control_effort", "min_barrier", "position_error", "yaw_error"):
            values = [
                record[candidate][metric] - record[baseline][metric]
                for record in successful_pairs
                if record[candidate][metric] is not None and record[baseline][metric] is not None
            ]
            deltas[f"{candidate}_minus_{baseline}_{metric}"] = _distribution(values)
        result["comparisons"][name] = {"baseline": baseline, "candidate": candidate, "outcomes": outcomes, "paired_both_success": deltas}
    return result


def _write_paired_plot(output_dir: Path, records):
    """Render direct-vs-sequential scatter plots for task decomposition."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 3.6), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    metrics = (
        ("completion_time", "Time to goal [s]"),
        ("control_effort", "Control effort [rad]"),
        ("min_barrier", "Minimum barrier [m]"),
    )
    baseline, candidate = COMPARISONS["task_decomposition"]
    for axis, (metric, title) in zip(axes, metrics):
        x, y = [], []
        for record in records:
            if record[baseline][metric] is not None and record[candidate][metric] is not None:
                x.append(record[baseline][metric])
                y.append(record[candidate][metric])
        if x:
            lower, upper = min(x + y), max(x + y)
            padding = max((upper - lower) * 0.05, 1e-6)
            axis.scatter(x, y, alpha=0.75, s=22)
            axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], "k--", linewidth=1)
            axis.set_xlim(lower - padding, upper + padding)
            axis.set_ylim(lower - padding, upper + padding)
        axis.set_title(title)
        axis.set_xlabel(baseline)
        axis.set_ylabel(candidate)
        axis.grid(True)
    figure.savefig(output_dir / "paired_metrics.png", dpi=180)


def _write_summary(output_dir: Path, records, experiment, config):
    analysis = _paired_analysis(records)
    payload = {
        "settings": {
            **asdict(experiment),
            "resolved_control_mode": config.control_mode,
            "kinematic": config.simulation.kinematic,
            "link_lengths": list(config.safety.link_lengths),
        },
        "cases": records,
        "analysis": analysis,
        "sampling": {
            "ik_rejections_before_acceptance": sum(
                record.get("ik_rejections_before_acceptance", 0) for record in records
            ),
            "note": "Only cases with at least one collision-free IK branch are executed in both methods.",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, default=_json_default), encoding="utf-8"
    )
    _write_paired_plot(output_dir, records)
    scalar_fields = ("success", "position_error", "yaw_error", "min_barrier", "active_link", "failure_reason", "completion_time", "control_effort", "max_joint_velocity", "mean_clf_slack", "total_qp_solve_time", "qp_infeasible_count")
    columns = ["case_id", "target_x", "target_y", "target_yaw"] + [f"{method}_{field}" for method in METHODS for field in scalar_fields]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {"case_id": record["case_id"], "target_x": record["target"][0], "target_y": record["target"][1], "target_yaw": record["target_yaw"]}
            for method in METHODS:
                row.update({f"{method}_{field}": record[method][field] for field in scalar_fields})
            writer.writerow(row)


def _joint_points(q, link_lengths):
    angles = np.cumsum(q)
    vectors = np.column_stack((np.cos(angles), np.sin(angles))) * np.asarray(link_lengths)[:, None]
    return np.vstack((np.zeros(2), np.cumsum(vectors, axis=0)))


def visualize_case(results_dir: Path, case_id: int, output_path: Path | None = None):
    """Render direct and sequential task-decomposition trajectories."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    record = next((item for item in summary["cases"] if item["case_id"] == case_id), None)
    if record is None:
        raise ValueError(f"case_id {case_id} was not found in {results_dir}")
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
    colors = {"direct": "tab:orange", "sequential": "tab:blue"}
    titles = {"direct": "Direct EE CLF + CBF", "sequential": "Sequential CLF"}
    link_lengths = summary["settings"]["link_lengths"]
    axis_limit = max(
        sum(link_lengths),
        max(abs(value) for value in summary["settings"]["obstacle_position_range"]),
        max(abs(value) for value in record["target"]),
    ) + 0.2
    for axis, mode in zip(axes, METHODS):
        title = titles[mode]
        result = record[mode]
        trace = np.load(results_dir / result["trajectory_file"])
        for obstacle in record["obstacles"]:
            axis.add_patch(Circle(obstacle["position"], obstacle["radius"], color="crimson", alpha=0.45))
        axis.plot(trace["final_xy"][:, 0], trace["final_xy"][:, 1], color=colors[mode], linewidth=1.5, label="final EE path")
        axis.scatter(*trace["final_xy"][0], color="black", s=25, label="start")
        axis.scatter(*trace["final_xy"][-1], color=colors[mode], s=45, label="final")
        axis.scatter(*record["target"], marker="*", color="tab:green", s=150, label="goal")
        arrow_scale = 0.30
        axis.arrow(
            *record["target"],
            arrow_scale * np.cos(record["target_yaw"]),
            arrow_scale * np.sin(record["target_yaw"]),
            color="tab:green", width=0.012, length_includes_head=True,
        )
        axis.arrow(
            *trace["final_xy"][-1],
            arrow_scale * np.cos(result["final_yaw"]),
            arrow_scale * np.sin(result["final_yaw"]),
            color=colors[mode], width=0.010, length_includes_head=True,
        )
        joints = _joint_points(trace["q"][-1], link_lengths)
        axis.plot(joints[:, 0], joints[:, 1], "o-", color="black", linewidth=2, label="final arm")
        axis.set_aspect("equal")
        axis.set_xlim(-axis_limit, axis_limit)
        axis.set_ylim(-axis_limit, axis_limit)
        axis.grid(True)
        axis.set_title(
            f"{title}\nsuccess={result['success']}, pos err={result['position_error']:.3f} m, "
            f"yaw err={result['yaw_error']:.3f} rad"
        )
        axis.legend(loc="upper right", fontsize=8)
    figure.suptitle(f"Case {case_id}: {record['comparisons']}")
    output_path = output_path or results_dir / f"case_{case_id:03d}_comparison.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved visualization: {output_path}")


def visualize_all_cases(results_dir: Path, output_dir: Path | None = None):
    """Render one comparison PNG for every case in a completed experiment."""
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    output_dir = output_dir or results_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in summary["cases"]:
        case_id = record["case_id"]
        visualize_case(
            results_dir,
            case_id,
            output_dir / f"case_{case_id:03d}_comparison.png",
        )
    print(f"Saved {len(summary['cases'])} visualizations: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run randomized direct/sequential ablation")
    run.add_argument("--root", default=".")
    run.add_argument("--config", default="settings/controller.yaml")
    run.add_argument("--mujoco", default="settings/mujoco.xml")
    run.add_argument("--urdf", default="settings/urdf.xml")
    run.add_argument("--run-name")
    run.add_argument("--cases", type=int, help="override sequential_experiment.cases")
    run.add_argument("--workers", type=int, help="override sequential_experiment.workers")
    run.add_argument("--duration", type=float, help="override sequential_experiment.duration [s]")
    run.set_defaults(func=run_comparison)
    visualize = subparsers.add_parser("visualize", help="render one saved comparison case")
    visualize.add_argument("--results-dir", type=Path, required=True)
    visualize.add_argument("--case-id", type=int, required=True)
    visualize.add_argument("--output", type=Path)
    visualize.set_defaults(func=lambda args: visualize_case(args.results_dir, args.case_id, args.output))
    visualize_all = subparsers.add_parser("visualize-all", help="render every saved comparison case")
    visualize_all.add_argument("--results-dir", type=Path, required=True)
    visualize_all.add_argument("--output-dir", type=Path)
    visualize_all.set_defaults(func=lambda args: visualize_all_cases(args.results_dir, args.output_dir))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

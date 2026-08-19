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
    """One target/obstacle placement shared by sequential-off and -on runs."""

    case_id: int
    target: tuple[float, float]
    target_yaw: float
    obstacles: tuple[ObstacleConfig, ...]


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
    trajectory_file: str


@dataclass(frozen=True)
class ComparisonWorkerInput:
    """Pickle-safe payload for one case's sequential-off/on pair."""

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
    (False, True): "sequential_only_success",
    (True, False): "nonsequential_only_success",
    (True, True): "both_succeeded",
}


def classify_outcome(nonsequential_success: bool, sequential_success: bool) -> str:
    """Return one of the four comparison categories in a stable order."""
    return OUTCOME_LABELS[(bool(nonsequential_success), bool(sequential_success))]


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
                continue
            return ExperimentCase(
                case_id,
                (float(target[0]), float(target[1])),
                target_yaw,
                tuple(obstacles),
            )
    raise RuntimeError("unable to sample a valid experiment case")


def _case_config(base_config: ControllerConfig, case: ExperimentCase, sequential_enabled: bool):
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
    np.savez_compressed(
        path,
        t=arrays["t"],
        q=arrays["q"],
        final_xy=final_xy,
        final_yaw=final_yaw,
        active_xy=arrays["ee"][:, :2],
        barrier=arrays["barrier"],
    )
    return arrays, final_xy, final_yaw


def _execute_case(
    base_config: ControllerConfig,
    case: ExperimentCase,
    sequential_enabled: bool,
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
    mode = "sequential_on" if sequential_enabled else "sequential_off"
    trajectory_path = output_dir / f"case_{case.case_id:03d}_{mode}.npz"
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
    min_barrier = float(np.min(arrays["barrier"]))
    success = (
        position_error <= experiment_goal_tolerance
        and yaw_error <= yaw_tolerance
        and min_barrier >= barrier_tolerance
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
        trajectory_file=trajectory_path.name,
    )


def run_case_pair(worker_input: ComparisonWorkerInput):
    """Run one shared case serially in a dedicated worker process."""
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
    sequential_off = _execute_case(
        worker_input.base_config, worker_input.case, False, **common
    )
    sequential_on = _execute_case(
        worker_input.base_config, worker_input.case, True, **common
    )
    return worker_input.case, sequential_off, sequential_on


def run_comparison(args):
    root = Path(args.root).resolve()
    config_path = root / args.config
    mujoco_path = root / args.mujoco
    urdf_path = root / args.urdf
    base_config = load_controller_config(config_path)
    experiment = base_config.sequential_experiment
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
                case, sequential_off, sequential_on = future.result()
                completed.append((case, sequential_off, sequential_on))
                print(f"[{index}/{experiment.cases}] completed case {case.case_id}")

    records = []
    for case, sequential_off, sequential_on in sorted(completed, key=lambda item: item[0].case_id):
        record = {
            "case_id": case.case_id,
            "target": list(case.target),
            "target_yaw": case.target_yaw,
            "obstacles": [asdict(obstacle) for obstacle in case.obstacles],
            "sequential_off": asdict(sequential_off),
            "sequential_on": asdict(sequential_on),
            "outcome": classify_outcome(sequential_off.success, sequential_on.success),
        }
        records.append(record)

    _write_summary(output_dir, records, experiment, base_config)
    print(f"Saved comparison: {output_dir}")
    for label in OUTCOME_LABELS.values():
        ids = [record["case_id"] for record in records if record["outcome"] == label]
        print(f"{label}: {len(ids)} cases {ids}")


def _write_summary(output_dir: Path, records, experiment, config):
    payload = {
        "settings": {
            **asdict(experiment),
            "resolved_control_mode": config.control_mode,
            "kinematic": config.simulation.kinematic,
            "link_lengths": list(config.safety.link_lengths),
        },
        "cases": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    columns = [
        "case_id", "outcome", "target_x", "target_y", "target_yaw", "off_success", "on_success",
        "off_position_error", "on_position_error", "off_yaw_error", "on_yaw_error",
        "off_min_barrier", "on_min_barrier", "off_active_link", "on_active_link",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            off, on = record["sequential_off"], record["sequential_on"]
            writer.writerow({
                "case_id": record["case_id"], "outcome": record["outcome"],
                "target_x": record["target"][0], "target_y": record["target"][1],
                "target_yaw": record["target_yaw"],
                "off_success": off["success"], "on_success": on["success"],
                "off_position_error": off["position_error"], "on_position_error": on["position_error"],
                "off_yaw_error": off["yaw_error"], "on_yaw_error": on["yaw_error"],
                "off_min_barrier": off["min_barrier"], "on_min_barrier": on["min_barrier"],
                "off_active_link": off["active_link"], "on_active_link": on["active_link"],
            })


def _joint_points(q, link_lengths):
    angles = np.cumsum(q)
    vectors = np.column_stack((np.cos(angles), np.sin(angles))) * np.asarray(link_lengths)[:, None]
    return np.vstack((np.zeros(2), np.cumsum(vectors, axis=0)))


def visualize_case(results_dir: Path, case_id: int, output_path: Path | None = None):
    """Render sequential-off/on trajectories, obstacles, and final arm poses."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    record = next((item for item in summary["cases"] if item["case_id"] == case_id), None)
    if record is None:
        raise ValueError(f"case_id {case_id} was not found in {results_dir}")
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
    colors = {"sequential_off": "tab:orange", "sequential_on": "tab:blue"}
    link_lengths = summary["settings"]["link_lengths"]
    axis_limit = max(
        sum(link_lengths),
        max(abs(value) for value in summary["settings"]["obstacle_position_range"]),
        max(abs(value) for value in record["target"]),
    ) + 0.2
    for axis, mode, title in zip(axes, ("sequential_off", "sequential_on"), ("Sequential OFF", "Sequential ON")):
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
    figure.suptitle(f"Case {case_id}: {record['outcome']}")
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
    run = subparsers.add_parser("run", help="run randomized sequential on/off comparisons")
    run.add_argument("--root", default=".")
    run.add_argument("--config", default="settings/controller.yaml")
    run.add_argument("--mujoco", default="settings/mujoco.xml")
    run.add_argument("--urdf", default="settings/urdf.xml")
    run.add_argument("--run-name")
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

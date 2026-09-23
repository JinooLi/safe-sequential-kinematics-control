"""Run one configured planar trajectory, optionally with the existing viewer."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from ..adapters.osqp_solver import OSQPSolver
from ..adapters.planar_kinematics import RobotInterface
from ..control.planar_controller import QPController
from ..logging.recorder import Recorder
from .artifacts import create_run_directory, save_run_provenance
from .paths import default_resource, project_root
from .planar_configuration import load_controller_config, validate_geometry_consistency
from .planar_simulation import run_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_resource("configs/planar.yaml"))
    parser.add_argument("--urdf", default=default_resource("assets/robots/planar.urdf"))
    parser.add_argument(
        "--mujoco", default=default_resource("assets/robots/planar.xml")
    )
    parser.add_argument("--mode", choices=("direct", "sequential"))
    parser.add_argument("--duration", type=float)
    parser.add_argument("--initial-q", nargs=3, type=float)
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--kinematic", dest="kinematic", action="store_true")
    model.add_argument("--physics", dest="kinematic", action="store_false")
    parser.set_defaults(kinematic=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--recording-gui", action="store_true")
    parser.add_argument("--camera", default="top_down")
    parser.add_argument("--output-dir", default=default_resource("outputs/planar"))
    parser.add_argument("--run-name")
    args = parser.parse_args()
    config = load_controller_config(args.config)
    raw = yaml.safe_load(Path(args.config).read_text())
    if args.mode is not None:
        enabled = args.mode == "sequential"
        config = replace(
            config,
            sequential_end_effector=replace(
                config.sequential_end_effector, enabled=enabled
            ),
        )
        raw.setdefault("sequential_end_effector", {})["enabled"] = enabled
    if args.kinematic is not None:
        config = replace(
            config, simulation=replace(config.simulation, kinematic=args.kinematic)
        )
        raw.setdefault("simulation", {})["kinematic"] = args.kinematic
    duration = (
        config.sequential_experiment.duration
        if args.duration is None
        else args.duration
    )
    initial_q = (
        config.sequential_experiment.initial_q
        if args.initial_q is None
        else tuple(args.initial_q)
    )
    if not np.isfinite(duration) or duration <= 0:
        parser.error("--duration must be finite and positive")
    if not np.all(np.isfinite(initial_q)):
        parser.error("--initial-q must be finite")
    config = replace(
        config,
        sequential_experiment=replace(
            config.sequential_experiment, duration=duration, initial_q=initial_q
        ),
    )
    raw.setdefault("sequential_experiment", {}).update(
        duration=duration, initial_q=list(initial_q)
    )
    validate_geometry_consistency(config, args.urdf, args.mujoco)
    frames = config.sequential_end_effector.frame_names
    robot = RobotInterface(
        args.urdf, frames if config.sequential_end_effector.enabled else frames[-1:]
    )
    controller = QPController(config, OSQPSolver())
    recorder = Recorder()
    directory = create_run_directory(args.output_dir, args.run_name)
    resources = [Path(args.urdf), Path(args.mujoco)]
    if config.safety.representation == "bernstein":
        resources.append(Path(config.safety.rdf_model_path))
    save_run_provenance(
        directory,
        config_path=Path(args.config),
        config=config,
        root=project_root(),
        resources=resources,
        effective_yaml=raw,
    )
    run_simulation(
        args.mujoco,
        controller,
        robot,
        target_position=config.ee_target,
        obstacle_enabled=[item.enabled for item in config.obstacles],
        initial_q=initial_q,
        duration=duration,
        kinematic=config.simulation.kinematic,
        visualize=args.visualize,
        camera_name=args.camera,
        recorder=recorder,
        save_recording=False,
        show_recording_gui=False,
    )
    recorder.save_npz(directory / "trajectory.npz")
    recorder.save_csv(directory / "trajectory.csv")
    recorder.plot(directory / "plots/control.png", show=args.show_plot)
    print(f"run artifacts: {directory}")
    if args.recording_gui:
        recorder.show_gui()


if __name__ == "__main__":
    main()

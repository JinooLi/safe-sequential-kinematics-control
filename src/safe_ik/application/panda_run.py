"""CLI for Panda safe IK and trajectory control."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml
from tqdm import tqdm

from ..logging.panda_results import result_payload, save_panda_control_trace
from ..visualization.panda import save_panda_control_plot
from ..visualization.panda_replay import replay_panda_result
from .artifacts import create_run_directory, save_run_provenance
from .panda_configuration import load_panda_config
from .panda_progress import PandaProgress
from .panda_rollout import run_panda
from .panda_setup import build_panda_runtime
from .paths import default_resource, project_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_resource("configs/panda.yaml"))
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="compute and save first, then replay the recorded motion",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="disable tqdm calculation progress"
    )
    parser.add_argument("--output-dir", default=default_resource("outputs/panda"))
    parser.add_argument("--run-name")
    parser.add_argument(
        "--output", help="optional extra summary JSON export; must not already exist"
    )
    parser.add_argument("--mode", choices=("direct", "sequential"))
    parser.add_argument(
        "--rdf-device",
        choices=("cpu", "cuda"),
        help="override RDF inference device; CUDA uses float64 and never falls back",
    )
    parser.add_argument(
        "--duration", type=float, help="override trajectory duration [s]"
    )
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
    raw = yaml.safe_load(Path(args.config).read_text())
    if args.rdf_device is not None:
        config = replace(
            config,
            safety=replace(
                config.safety,
                rdf=replace(config.safety.rdf, runtime_device=args.rdf_device),
            ),
        )
        raw.setdefault("safety", {}).setdefault("rdf", {})[
            "runtime_device"
        ] = args.rdf_device
    if args.mode is not None:
        config = replace(config, trajectory=replace(config.trajectory, mode=args.mode))
        raw["trajectory"]["mode"] = args.mode
    if args.duration is not None:
        if not 0 < args.duration < float("inf"):
            parser.error("--duration must be finite and positive")
        config = replace(
            config, trajectory=replace(config.trajectory, duration=args.duration)
        )
        raw["trajectory"]["duration"] = args.duration
    for extra in (args.output, args.plot):
        if extra and Path(extra).exists():
            parser.error(f"refusing to overwrite existing export: {extra}")
    directory = create_run_directory(args.output_dir, args.run_name)
    print(f"run artifacts: {directory}", flush=True)
    save_run_provenance(
        directory,
        config_path=Path(args.config),
        config=config,
        root=project_root(),
        resources=[config.model_path],
        effective_yaml=raw,
    )
    if config.trajectory.mode == "sequential":
        workers = min(config.safe_ik.workers, config.safe_ik.max_restarts)
        execution = "serial" if workers == 1 else f"{workers} worker processes"
        print(
            f"safe IK: {config.safe_ik.max_restarts} seeds, {execution}, "
            f"safety={config.ik_self_collision_model} "
            f"(trajectory={config.safety.self_collision_model})",
            flush=True,
        )
    try:
        print("Loading kinematics and safety models...", flush=True)
        runtime = build_panda_runtime(config)
    except (ValueError, OSError, RuntimeError) as exc:
        failure = {
            "failure": {
                "reason": "runtime_initialization_failed",
                "detail": str(exc),
                "time": 0.0,
                "q": config.initial_q.tolist(),
                "solver_status": "not_run",
            }
        }
        (directory / "summary.json").write_text(json.dumps(failure, indent=2))
        parser.exit(2, f"{exc}\nFailure report: {directory / 'summary.json'}\n")

    def report_transition(time, previous, following, reason):
        tqdm.write(f"[Panda stage] t={time:.3f}s: {previous} -> {following} ({reason})")

    # Always finish computation (and shut down IK workers) before opening a GUI.
    with PandaProgress(enabled=not args.no_progress) as progress:
        result = run_panda(
            config,
            runtime,
            on_transition=report_transition,
            progress=progress if not args.no_progress else None,
        )
    output_path = directory / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_payload(result), indent=2), encoding="utf-8"
    )
    if args.output:
        export = Path(args.output)
        export.parent.mkdir(parents=True, exist_ok=True)
        with export.open("x", encoding="utf-8") as file:
            file.write(json.dumps(result_payload(result), indent=2))
    plot_path = Path(args.plot) if args.plot else directory / "plots/control.png"
    trace_path = directory / "trajectory.npz"
    if save_panda_control_trace(result.trace, trace_path):
        print(f"control trace: {trace_path}")
    has_plot = save_panda_control_plot(result.trace, plot_path, show=False)
    if has_plot:
        print(f"control plot: {plot_path}")
    print(f"run artifacts: {directory}")
    if result.failure:
        print(f"stopped: {result.failure}")
    if result.ik_skipped:
        print("safe IK: skipped (direct mode)")
    else:
        print(f"safe IK candidates: {len(result.ik.candidates)}")
    if result.ik.failure_reason:
        print(f"failure: {result.ik.failure_reason}")
    else:
        print(f"trajectory time: {result.elapsed:.3f} s ({result.steps} control steps)")
        print(f"final q: {result.final_q}")
        print(f"status: {result.status}")
    print(f"summary: {output_path}")
    if args.visualize:
        print(
            "Calculation and saving complete. Replaying at 1x speed; close the viewer to exit playback.",
            flush=True,
        )
        replay_panda_result(runtime.plant, result)
    if args.show_plot and has_plot:
        # Plot rendering only: the rollout and safety checks are never rerun.
        print(
            "Showing saved-run diagnostics; close the plot window to exit.", flush=True
        )
        save_panda_control_plot(result.trace, plot_path, show=True)


if __name__ == "__main__":
    main()

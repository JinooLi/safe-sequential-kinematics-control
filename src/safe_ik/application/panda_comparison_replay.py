"""Replay saved Panda comparison cases without running IK, RDF, or control."""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from ..adapters.mujoco_plant import MujocoPlant
from ..experiments.panda_comparison import case_config
from .panda_configuration import parse_panda_config
from .paths import project_root


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_saved_cases(directory):
    directory = Path(directory).resolve()
    # Older campaigns contain a main batch beneath the selected output directory.
    if (
        not (directory / "cases.jsonl").is_file()
        and (directory / "main/cases.jsonl").is_file()
    ):
        directory /= "main"
    if not all((directory / name).is_file() for name in ("config.yaml", "cases.jsonl")):
        raise ValueError("RESULTS_DIR must contain saved config.yaml and cases.jsonl")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    for name in ("config.yaml", "cases.jsonl"):
        expected = manifest.get("data_sha256", {}).get(name)
        if expected and _digest(directory / name) != expected:
            raise ValueError(f"saved {name} hash mismatch")
    cases = [
        json.loads(line)
        for line in (directory / "cases.jsonl").read_text().splitlines()
    ]
    return directory, cases, manifest


def load_saved_replay(directory, case_id, method="sequential", *, model_path=None):
    if method not in ("direct", "sequential"):
        raise ValueError("method must be direct or sequential")
    directory, cases, manifest = read_saved_cases(directory)
    case = next((case for case in cases if case["case_id"] == case_id), None)
    if case is None or Path(case_id).name != case_id:
        raise ValueError(f"unknown case_id: {case_id}; use --list to see saved cases")
    folder = directory / case_id / method
    summary = json.loads((folder / "summary.json").read_text())
    trace_path = folder / "trajectory.npz"
    if not trace_path.is_file():
        raise ValueError(
            f"no saved trajectory: {case_id}/{method} ({summary.get('reason')})"
        )
    expected = summary.get("trace_sha256")
    if expected and _digest(trace_path) != expected:
        raise ValueError("saved trajectory hash mismatch")
    config_path = directory / "config.yaml"
    data = yaml.safe_load(config_path.read_text())
    recorded_model = (directory / data["robot"]["model_path"]).resolve()
    model = Path(model_path).resolve() if model_path else recorded_model
    if model_path is None and not model.is_file():
        model = project_root() / "assets/robots/panda.xml"
    identity = manifest.get("identity", {})
    if not model.is_file():
        raise ValueError("saved robot model is unavailable; specify --model-path")
    if identity.get("model_sha256") and _digest(model) != identity["model_sha256"]:
        raise ValueError("robot model differs from the recorded experiment")
    # Resolve mesh/include paths relative to the relocated model, not the old host.
    for name, expected in identity.get("model_resource_sha256", {}).items():
        relative = os.path.relpath(name, recorded_model.parent)
        resource = (model.parent / relative).resolve()
        if not resource.is_file() or _digest(resource) != expected:
            raise ValueError(
                f"recorded robot resource is missing or changed: {resource}"
            )
    data["robot"]["model_path"] = str(model)
    config = case_config(parse_panda_config(data, config_path), case)
    config = replace(config, trajectory=replace(config.trajectory, mode=method))
    with np.load(trace_path, allow_pickle=False) as trace:
        timestamps, positions = trace["time"].copy(), trace["q"].copy()
    if (
        timestamps.ndim != 1
        or not timestamps.size
        or positions.shape != (timestamps.size, config.dof)
        or not np.isfinite(timestamps).all()
        or not np.isfinite(positions).all()
        or np.any(np.diff(timestamps) < 0)
    ):
        raise ValueError(
            "saved replay needs finite ordered times and matching joint poses"
        )
    if not np.array_equal(positions[0], config.initial_q) or not np.array_equal(
        positions[-1], summary["final_q"]
    ):
        raise ValueError("saved trajectory does not match its initial/final state")
    return config, timestamps, positions, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument(
        "--method", choices=("direct", "sequential"), default="sequential"
    )
    parser.add_argument("--model-path", type=Path, help="robot XML on this computer")
    parser.add_argument(
        "--list", action="store_true", help="list recorded cases and outcomes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate saved inputs without opening a viewer",
    )
    args = parser.parse_args(argv)
    if not args.list and not args.case_id:
        parser.error(
            "--case-id is required for playback; use --list to see saved cases"
        )
    try:
        if args.list:
            directory, cases, _ = read_saved_cases(args.results_dir)
            print("case_id\tdirect\tsequential")
            for case in cases:
                outcomes = []
                for method in ("direct", "sequential"):
                    path = directory / case["case_id"] / method / "summary.json"
                    result = json.loads(path.read_text()) if path.is_file() else {}
                    outcomes.append(
                        f"{result.get('reason', 'not_recorded')} (success={result.get('success', '-')})"
                    )
                print("\t".join([case["case_id"], *outcomes]))
            return
        config, timestamps, positions, summary = load_saved_replay(
            args.results_dir, args.case_id, args.method, model_path=args.model_path
        )
        plant = MujocoPlant(config)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "case_id": args.case_id,
                "method": args.method,
                "reason": summary["reason"],
                "recorded_success": summary["success"],
                "samples": len(timestamps),
                "duration": float(timestamps[-1] - timestamps[0]),
                "model_path": str(config.model_path),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        parser.error(
            "a GUI display is required; copy the results to a desktop or use a graphical remote session"
        )
    print("Replaying saved poses at 1x speed; close the viewer to finish.", flush=True)
    plant.replay(timestamps, positions)


if __name__ == "__main__":
    main()

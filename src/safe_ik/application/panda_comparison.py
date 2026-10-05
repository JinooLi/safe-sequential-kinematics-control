"""Headless, resumable Panda Monte Carlo experiments and pilot/main campaigns.

Run with ``python -m safe_ik.application.panda_comparison --help``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import tarfile
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from ..adapters.mujoco_plant import MujocoPlant
from ..adapters.osqp_solver import OSQPSolver
from ..domain.ik_contracts import SafeIKCandidate
from ..domain.safety_contracts import BarrierEvaluation
from ..domain.spatial_safety import PandaCapsuleSafety
from ..experiments.panda_comparison import (
    case_config,
    generate_cases,
    measure_state,
    paired_analysis,
)
from .artifacts import _model_resources, json_value, save_run_provenance
from .panda_comparison_rollout import comparison_rollout
from .panda_components import build_panda_components
from .panda_configuration import load_panda_config
from .panda_runtime import PandaRuntime
from .parallel_ik import panda_ik_search
from .paths import project_root


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(json_value(value), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


class SceneSafety:
    """Share immutable RDF models while keeping per-scene obstacles/caches private."""

    def __init__(self, self_safety, obstacle_safety):
        self.self_safety, self.obstacle_safety = self_safety, obstacle_safety
        self.q = self.result = None
        self.last_timings = {}

    def evaluate(self, q):
        if self.q is not None and np.array_equal(q, self.q):
            return self.result
        own, obstacle = self.self_safety.evaluate(q), self.obstacle_safety.evaluate(q)
        self.last_timings = dict(getattr(self.self_safety, "last_timings", {}))
        self.result = BarrierEvaluation(
            own.labels + obstacle.labels,
            np.concatenate([own.values, obstacle.values]),
            np.concatenate([own.gradients, obstacle.gradients], axis=0),
        )
        self.q = np.array(q, copy=True)
        return self.result


def scene_safety(config, kin, self_safety):
    return SceneSafety(
        self_safety,
        PandaCapsuleSafety(
            kin,
            config.safety.capsules,
            config.obstacles,
            config.safety.margin,
            config.safety.self_collision_exclusions,
            include_self=False,
        ),
    )


def prepare_ik(config, kin, safety):
    start = perf_counter()
    ik_kin, ik_safety = build_panda_components(
        config, self_collision_model=config.ik_self_collision_model
    )
    with panda_ik_search(config, ik_kin, ik_safety) as search:
        raw = search.solve(config.initial_q, require_safe=True)
    accepted, checks = [], []
    for candidate in sorted(raw.candidates, key=lambda c: c.seed_index):
        barrier = safety.evaluate(candidate.q)
        # Recheck pose and joint limits as well as RDF endpoint safety.
        sample = measure_state(config, kin, candidate.q, barrier)
        passed = (
            sample["goal_reached"]
            and sample["joint_margin"] >= 0
            and barrier.minimum >= 0
        )
        checks.append(
            {
                "seed_index": candidate.seed_index,
                "passed": bool(passed),
                "minimum_barrier": barrier.minimum,
            }
        )
        if passed:
            accepted.append(candidate)
    accepted.sort(
        key=lambda c: (float(np.linalg.norm(c.q - config.initial_q)), c.seed_index)
    )
    selected = accepted[0] if accepted else None
    return {
        "selected": asdict(selected) if selected else None,
        "raw_candidates": [asdict(c) for c in raw.candidates],
        "accepted_candidates": [asdict(c) for c in accepted],
        "checks": checks,
        "reason": (
            None
            if selected
            else "rdf_candidate_rejected" if raw.candidates else "ik_budget_exhausted"
        ),
        "wall_time": perf_counter() - start,
        "diagnostic_pose_only_search": False,
    }


def fingerprint(config, config_path, seeds, cases_per_seed):
    root = project_root()
    source = {
        str(p.relative_to(root)): digest(p)
        for p in sorted((root / "src").rglob("*.py"))
    }
    protocol = root / "docs/experiments/panda_monte_carlo_protocol.md"
    return {
        "schema_version": 1,
        "config": json_value(config),
        "config_sha256": digest(config_path),
        "source_sha256": source,
        "model_sha256": digest(config.model_path),
        "model_resource_sha256": {
            str(path): digest(path)
            for path in sorted(_model_resources(config.model_path))
        },
        "rdf_sha256": digest(config.safety.rdf.artifact),
        "lock_sha256": digest(root / "uv.lock"),
        "protocol_sha256": digest(protocol),
        "seeds": seeds,
        "cases_per_seed": cases_per_seed,
        "scoring": "fresh_state_first_pose_hit_strict_barrier_joint_limits_v1",
        "subdivisions": 5,
        "candidate_policy": "one_nearest_no_switch",
    }


def verify_resume(directory, expected):
    path = directory / "manifest.json"
    actual = json.loads(path.read_text())
    if actual["identity"] != expected:
        raise ValueError(
            "resume refused: source, protocol, resources or experiment settings changed"
        )
    for name, expected_hash in actual.get("data_sha256", {}).items():
        if digest(directory / name) != expected_hash:
            raise ValueError(f"resume refused: {name} hash mismatch")
    return actual


def export_analysis(directory, rows, count, duration):
    analysis = paired_analysis(rows, count, duration)
    atomic_json(directory / "analysis.json", analysis)
    atomic_json(directory / "summary.json", rows)
    columns = [
        "case_id",
        "method",
        "ik_available",
        "success",
        "goal_reached",
        "reason",
        "completion_time",
        "minimum_barrier",
        "minimum_self_barrier",
        "minimum_obstacle_barrier",
        "interpolated_minimum_barrier",
        "rollout_wall_time",
        "ik_wall_time",
        "validation_wall_time",
    ]
    temporary = directory / "summary.csv.partial"
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(directory / "summary.csv")
    return analysis


def run_experiment(directory, config_path, config, seeds, cases_per_seed):
    """Persist each IK result and method once; identity checks protect resumes."""
    import fcntl

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(directory, config_path, config, seeds, cases_per_seed)


def _run_locked(directory, config_path, config, seeds, cases_per_seed):
    identity = fingerprint(config, config_path, seeds, cases_per_seed)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = verify_resume(directory, identity)
    else:
        if any(p.name != ".lock" for p in directory.iterdir()):
            raise ValueError(
                "refusing a nonempty experiment directory without manifest"
            )
        environment = {"hostname": platform.node(), "pid": os.getpid()}
        if config.safety.rdf.runtime_device == "cuda":
            import torch

            environment.update(
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0),
                cuda=torch.version.cuda,
            )
        manifest = {"identity": identity, "environment": environment, "data_sha256": {}}
        atomic_json(manifest_path, manifest)
        root = project_root()
        raw = yaml.safe_load(Path(config_path).read_text())
        raw["safety"]["rdf"]["runtime_device"] = config.safety.rdf.runtime_device
        raw["trajectory"]["duration"] = config.trajectory.duration
        save_run_provenance(
            directory,
            config_path=Path(config_path),
            config=config,
            root=root,
            resources=[config.model_path],
            effective_yaml=raw,
        )
        with tarfile.open(directory / "source_snapshot.tar.gz", "w:gz") as archive:
            for relative in identity["source_sha256"]:
                archive.add(root / relative, arcname=relative)
            archive.add(
                root / "docs/experiments/panda_monte_carlo_protocol.md",
                arcname="protocol.md",
            )
        (directory / "working_tree.patch").write_bytes(
            subprocess.check_output(
                ["git", "diff", "HEAD", "--", "src", "configs", "docs"], cwd=root
            )
        )
    print(f"Loading persistent RDF runtime: {directory}", flush=True)
    kin, self_safety = build_panda_components(replace(config, obstacles=()))
    cases_path = directory / "cases.jsonl"
    if cases_path.exists():
        if "cases.jsonl" not in manifest["data_sha256"]:
            raise ValueError(
                "cases file was not sealed in manifest; inspect interrupted generation"
            )
        cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    else:
        cases, sampling = generate_cases(
            config, kin, self_safety, seeds, cases_per_seed
        )
        atomic_json(directory / "sampling.json", sampling)
        temporary = cases_path.with_suffix(".jsonl.partial")
        temporary.write_text(
            "".join(json.dumps(c, allow_nan=False) + "\n" for c in cases)
        )
        temporary.replace(cases_path)
        manifest["data_sha256"] = {
            name: digest(directory / name) for name in ("cases.jsonl", "sampling.json")
        }
        atomic_json(manifest_path, manifest)
    if len(cases) != len(seeds) * cases_per_seed or len(
        {c["case_id"] for c in cases}
    ) != len(cases):
        raise ValueError("case count or identity mismatch")
    rows = []
    for index, case in enumerate(cases):
        case_dir = directory / case["case_id"]
        case_dir.mkdir(exist_ok=True)
        cfg = case_config(config, case)
        ik_path = case_dir / "ik.json"
        if ik_path.exists():
            ik = json.loads(ik_path.read_text())
        else:
            print(f"[{index + 1}/{len(cases)}] {case['case_id']} IK start", flush=True)
            ik = prepare_ik(cfg, kin, scene_safety(cfg, kin, self_safety))
            atomic_json(ik_path, ik)
        candidate = (
            SafeIKCandidate(**{**ik["selected"], "q": np.asarray(ik["selected"]["q"])})
            if ik["selected"]
            else None
        )
        methods = (
            ("direct", "sequential") if index % 2 == 0 else ("sequential", "direct")
        )
        for method in methods:
            result_dir = case_dir / method
            result_dir.mkdir(exist_ok=True)
            result_path = result_dir / "summary.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result["ik_sha256"] != digest(ik_path):
                    raise ValueError("cached result IK hash mismatch")
                if (
                    result.get("trace_sha256")
                    and digest(result_dir / "trajectory.npz") != result["trace_sha256"]
                ):
                    raise ValueError("cached trajectory hash mismatch")
            else:
                print(
                    f"[{index + 1}/{len(cases)}] {case['case_id']} {method} start; IK={candidate is not None}",
                    flush=True,
                )
                method_cfg = replace(
                    cfg, trajectory=replace(cfg.trajectory, mode=method)
                )
                if method == "sequential" and candidate is None:
                    result = {
                        "success": False,
                        "goal_reached": False,
                        "safety_violation": False,
                        "reason": ik["reason"],
                        "completion_time": None,
                    }
                else:
                    runtime_start = perf_counter()
                    runtime = PandaRuntime(
                        kin,
                        scene_safety(cfg, kin, self_safety),
                        OSQPSolver(
                            max_iterations=cfg.qp.max_iterations,
                            absolute_tolerance=cfg.qp.absolute_tolerance,
                            relative_tolerance=cfg.qp.relative_tolerance,
                        ),
                        MujocoPlant(method_cfg),
                    )
                    setup_wall = perf_counter() - runtime_start
                    result, trace = comparison_rollout(method_cfg, runtime, candidate)
                    temporary = result_dir / "trajectory.npz.partial"
                    with temporary.open("wb") as file:
                        np.savez_compressed(file, **trace)
                    temporary.replace(result_dir / "trajectory.npz")
                    result["trace_sha256"] = digest(result_dir / "trajectory.npz")
                    result["runtime_setup_wall_time"] = setup_wall
                result.update(
                    case_id=case["case_id"],
                    method=method,
                    ik_available=candidate is not None,
                    ik_wall_time=ik["wall_time"] if method == "sequential" else 0.0,
                    ik_sha256=digest(ik_path),
                )
                atomic_json(result_path, result)
            rows.append(result)
            print(
                f"[{index + 1}/{len(cases)}] {method}: {result['reason']}, success={result['success']}",
                flush=True,
            )
        analysis = export_analysis(
            directory, rows, len(cases), config.trajectory.duration
        )
        atomic_json(
            directory / "progress.json",
            {
                "completed_pairs": index + 1,
                "total_pairs": len(cases),
                "last_case": case["case_id"],
                "pid": os.getpid(),
            },
        )
    validate_completed_run(directory, cases)
    atomic_json(
        directory / "complete.json",
        {
            "validated_pairs": len(cases),
            "analysis_sha256": digest(directory / "analysis.json"),
        },
    )
    return analysis


def validate_completed_run(directory, cases):
    """Gate campaigns on artifact integrity, never on which controller wins."""
    for case in cases:
        for method in ("direct", "sequential"):
            root = directory / case["case_id"] / method
            result = json.loads((root / "summary.json").read_text())
            if result["case_id"] != case["case_id"] or result["method"] != method:
                raise ValueError("result identity mismatch")
            if not result.get("trace_sha256"):
                if (
                    method != "sequential"
                    or result["ik_available"]
                    or result["success"]
                ):
                    raise ValueError("missing trajectory for executed method")
                continue
            if digest(root / "trajectory.npz") != result["trace_sha256"]:
                raise ValueError("trajectory checksum mismatch")
            with np.load(root / "trajectory.npz") as trace:
                if (
                    not np.isfinite(trace["q"]).all()
                    or not np.isfinite(trace["barrier_values"]).all()
                ):
                    raise ValueError("nonfinite trajectory")
                if len(trace["q"]) != result["steps"] + 1:
                    raise ValueError("missing initial/final trajectory state")
                if int(trace["command_applied"].sum()) != result["steps"]:
                    raise ValueError("command/state alignment mismatch")
                if not np.array_equal(trace["q"][-1], result["final_q"]):
                    raise ValueError("final state mismatch")
                if not np.array_equal(trace["q"][0], case["initial_q"]):
                    raise ValueError("initial state mismatch")
                reached = bool(trace["goal_reached"][-1])
                safe = bool(
                    trace["barrier_values"].min() >= 0
                    and trace["joint_margin"].min() >= 0
                )
                if result["success"] != (reached and safe):
                    raise ValueError("success score disagrees with saved states")
                if result["interpolation_checks"] != result["steps"] * 4:
                    raise ValueError("incomplete interpolation validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "campaign"))
    parser.add_argument(
        "--config", type=Path, default=project_root() / "configs/panda.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rdf-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[2332, 2333, 2334, 2335, 2336]
    )
    parser.add_argument("--cases-per-seed", type=int, default=60)
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    if (
        args.cases_per_seed < 1
        or len(set(args.seeds)) != len(args.seeds)
        or min(args.seeds) < 0
    ):
        parser.error("provide positive case count and unique nonnegative seeds")
    config = load_panda_config(args.config)
    config = replace(
        config,
        safety=replace(
            config.safety,
            rdf=replace(config.safety.rdf, runtime_device=args.rdf_device),
        ),
    )
    if args.duration is not None:
        if not np.isfinite(args.duration) or args.duration <= 0:
            parser.error("duration must be positive and finite")
        config = replace(
            config, trajectory=replace(config.trajectory, duration=args.duration)
        )
    if args.command == "run":
        run_experiment(
            args.output, args.config, config, args.seeds, args.cases_per_seed
        )
        return
    # The locked campaign enforces a complete pilot before starting all 300 cases.
    import fcntl

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            atomic_json(
                args.output / "status.json", {"phase": "pilot", "pid": os.getpid()}
            )
            pilot = run_experiment(
                args.output / "pilot", args.config, config, [20261005], 20
            )
            if not pilot["complete"] or not pilot["ik_available_pairs"]:
                raise RuntimeError(
                    "pilot has no eligible paired cases; inspect IK preparation before main"
                )
            # Detect edits made while the pilot was running before starting the main run.
            verify_resume(
                args.output / "pilot", fingerprint(config, args.config, [20261005], 20)
            )
            atomic_json(
                args.output / "status.json",
                {"phase": "main", "pid": os.getpid(), "pilot": pilot},
            )
            main_result = run_experiment(
                args.output / "main",
                args.config,
                config,
                [2332, 2333, 2334, 2335, 2336],
                60,
            )
            atomic_json(
                args.output / "status.json",
                {"phase": "complete", "pilot": pilot, "main": main_result},
            )
        except Exception as exc:
            atomic_json(
                args.output / "status.json",
                {"phase": "stopped", "error": repr(exc), "pid": os.getpid()},
            )
            raise


if __name__ == "__main__":
    main()

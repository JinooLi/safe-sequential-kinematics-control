"""Compare Direct, Sequential and joint CLF on saved scenes and fixed safe IK.

No scenes or IK candidates are generated. Selected methods are rerun with a
common controller configuration and an optional joint-CLF-only slack override;
saved outcomes are used only to verify input hashes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import tarfile
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np

from ..adapters.mujoco_plant import MujocoPlant
from ..adapters.osqp_solver import OSQPSolver
from ..domain.ik_contracts import SafeIKCandidate
from ..experiments.panda_comparison import case_config, measure_state
from .artifacts import _model_resources, json_value
from .panda_comparison import atomic_json, digest, scene_safety
from .panda_comparison_replay import read_saved_cases
from .panda_comparison_rollout import comparison_rollout
from .panda_components import build_panda_components
from .panda_experiment_configuration import load_panda_experiment
from .panda_runtime import PandaRuntime

METHODS = ("direct", "sequential", "joint_clf")


def saved_inputs(reference, count):
    """Take the first count scenes in recorded order, without outcome filtering."""
    reference, cases, manifest = read_saved_cases(reference)
    if type(count) is not int or not 0 < count <= len(cases):
        raise ValueError("count must select a nonempty prefix of the saved scenes")
    for name in ("cases.jsonl", "config.yaml"):
        if digest(reference / name) != manifest.get("data_sha256", {}).get(name):
            raise ValueError(f"reference input is not sealed: {name}")
    chosen = cases[:count]
    names = [c["case_id"] for c in chosen]
    if len(set(names)) != count or any(
        Path(n).name != n or n in (".", "..") for n in names
    ):
        raise ValueError("invalid or duplicate saved scene IDs")
    records, hashes = {}, {}
    for case in chosen:
        name = case["case_id"]
        path = reference / name / "ik.json"
        hashes[name] = digest(path)
        for method in ("direct", "sequential"):
            outcome = json.loads(
                (reference / name / method / "summary.json").read_text()
            )
            if outcome["ik_sha256"] != hashes[name]:
                raise ValueError(f"saved IK hash mismatch: {name}")
        records[name] = json.loads(path.read_text())
    return reference, chosen, records, hashes


def fixed_candidate(ik, config, kin, safety):
    """Reuse the recorded nearest validated candidate, rechecking current safety."""
    item = ik["selected"]
    if item is None:
        return ()
    candidate = SafeIKCandidate(**{**item, "q": np.asarray(item["q"], dtype=float)})
    if candidate.q.shape != (config.dof,) or not np.isfinite(candidate.q).all():
        raise ValueError("invalid saved IK configuration")
    accepted = ik["accepted_candidates"]
    if not any(
        c["seed_index"] == candidate.seed_index and np.array_equal(c["q"], candidate.q)
        for c in accepted
    ):
        raise ValueError("selected IK is absent from validated candidates")
    nearest = min(
        accepted,
        key=lambda c: (
            float(np.linalg.norm(np.asarray(c["q"]) - config.initial_q)),
            c["seed_index"],
        ),
    )
    if nearest["seed_index"] != candidate.seed_index or not np.array_equal(
        nearest["q"], candidate.q
    ):
        raise ValueError("saved selected IK is not the nearest validated candidate")
    barrier = safety.evaluate(candidate.q)
    sample = measure_state(config, kin, candidate.q, barrier)
    if not sample["goal_reached"] or sample["joint_margin"] < 0 or barrier.minimum < 0:
        raise ValueError(
            "saved IK fails current pose, joint-limit or barrier validation"
        )
    return (candidate,)


def method_config(config, method, joint_clf_slack_weight=None):
    if method not in METHODS:
        raise ValueError(f"unknown comparison method: {method}")
    trajectory = replace(config.trajectory, mode=method)
    if joint_clf_slack_weight is not None:
        if not np.isfinite(joint_clf_slack_weight) or joint_clf_slack_weight <= 0:
            raise ValueError("joint CLF slack weight must be finite and positive")
        if method == "joint_clf":
            trajectory = replace(
                trajectory, clf_slack_weight=float(joint_clf_slack_weight)
            )
    return replace(config, trajectory=trajectory)


def analyze(rows, expected_cases, methods=METHODS):
    grouped = {}
    for row in rows:
        group = grouped.setdefault(row["case_id"], {})
        if row["method"] in group:
            raise ValueError("duplicate result")
        group[row["method"]] = row
    complete = [g for g in grouped.values() if set(g) == set(methods)]
    stats = {}
    for method in methods:
        selected = [g[method] for g in complete]
        times = [r["completion_time"] for r in selected if r["success"]]
        stats[method] = {
            "cases": len(selected),
            "goal_reached": sum(r["goal_reached"] for r in selected),
            "successes": sum(r["success"] for r in selected),
            "success_rate": (
                float(np.mean([r["success"] for r in selected])) if selected else None
            ),
            "safety_violations": sum(r["safety_violation"] for r in selected),
            "termination_reasons": dict(Counter(r["reason"] for r in selected)),
            "successful_completion_time_mean": float(np.mean(times)) if times else None,
        }
    paired = {}
    for a, b in combinations(methods, 2):
        paired[f"{a}_vs_{b}"] = {
            "both_success": sum(g[a]["success"] and g[b]["success"] for g in complete),
            "first_only": sum(
                g[a]["success"] and not g[b]["success"] for g in complete
            ),
            "second_only": sum(
                not g[a]["success"] and g[b]["success"] for g in complete
            ),
            "both_failed": sum(
                not g[a]["success"] and not g[b]["success"] for g in complete
            ),
        }
    return {
        "complete": len(complete) == expected_cases,
        "expected_cases": expected_cases,
        "completed_cases": len(complete),
        "ik_available_cases": sum(g[methods[0]]["ik_available"] for g in complete),
        "methods": stats,
        "paired": paired,
    }


def verify_trace(result, trace, case, candidate, subdivisions):
    """Check scoring, fixed IK and CLF logging independently of the controller."""
    if (
        not np.isfinite(trace["q"]).all()
        or not np.isfinite(trace["barrier_values"]).all()
    ):
        raise ValueError("nonfinite recorded states or barriers")
    if (
        len(trace["q"]) != result["steps"] + 1
        or int(trace["command_applied"].sum()) != result["steps"]
    ):
        raise ValueError("state/command alignment mismatch")
    if not np.array_equal(trace["q"][0], case["initial_q"]) or not np.array_equal(
        trace["q"][-1], result["final_q"]
    ):
        raise ValueError("initial/final configuration mismatch")
    tolerance = result["success_barrier_tolerance"]
    safe = bool(
        trace["barrier_values"].min() >= -tolerance and trace["joint_margin"].min() >= 0
    )
    reached = bool(trace["goal_reached"][-1])
    if result["success"] != (reached and safe) or result["goal_reached"] != reached:
        raise ValueError("success scoring mismatch")
    dense_safe = bool(
        result["interpolated_minimum_barrier"] >= -tolerance
        and trace["joint_margin"].min() >= 0
    )
    if result["interpolated_success"] != (reached and dense_safe):
        raise ValueError("interpolated success scoring mismatch")
    if (
        result["interpolation_checks"] != result["steps"] * (subdivisions - 1)
        or result["interpolation_subdivisions"] != subdivisions
    ):
        raise ValueError("incomplete safety validation")
    if result["candidate_switch_count"] != 0:
        raise ValueError("fixed-IK protocol forbids candidate switching")
    if candidate is not None:
        errors = np.linalg.norm(trace["q"] - candidate.q, axis=1)
        np.testing.assert_allclose(
            trace["reference_joint_error"], errors, rtol=0, atol=1e-12
        )
        if result["method"] != "direct":
            np.testing.assert_array_equal(trace["candidate_q"], candidate.q[None])
            np.testing.assert_array_equal(
                trace["candidate_index"], np.zeros(len(trace["q"]), dtype=int)
            )
        if result["method"] == "joint_clf":
            np.testing.assert_allclose(
                trace["clf_value"],
                errors[: len(trace["clf_value"])] ** 2,
                rtol=1e-12,
                atol=1e-12,
            )


def export(directory, rows, count, methods=METHODS):
    analysis = analyze(rows, count, methods)
    atomic_json(directory / "summary.json", rows)
    atomic_json(directory / "analysis.json", analysis)
    fields = [
        "case_id",
        "method",
        "clf_slack_weight",
        "ik_available",
        "success",
        "goal_reached",
        "reason",
        "completion_time",
        "elapsed",
        "final_joint_error",
        "final_position_error",
        "final_orientation_error",
        "minimum_barrier",
        "minimum_joint_margin",
        "safety_violation",
        "candidate_switch_count",
        "rollout_wall_time",
    ]
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return analysis


def run(
    reference,
    experiment,
    output,
    count=10,
    *,
    methods=METHODS,
    joint_clf_slack_weight=None,
):
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("select a nonempty list of distinct methods")
    resolved_configs = {
        m: method_config(experiment.config, m, joint_clf_slack_weight) for m in methods
    }
    reference, cases, records, ik_hashes = saved_inputs(reference, count)
    # This experiment always fixes one candidate, independently of the ordinary
    # two-method campaign's default policy. The resolved snapshot records this.
    experiment = replace(experiment, use_all_ik_candidates=False)
    config = experiment.config
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    try:
        snapshot = experiment.snapshot(output)
        source_root = Path(__file__).resolve().parents[2]
        sources = {
            str(p.relative_to(source_root)): digest(p)
            for p in sorted(source_root.rglob("*.py"))
        }
        with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as bundle:
            for relative in sources:
                bundle.add(source_root / relative, arcname="src/" + relative)
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reference": str(reference),
            "reference_cases_sha256": digest(reference / "cases.jsonl"),
            "reference_ik_sha256": ik_hashes,
            "selection": "first recorded scenes in file order; no outcome filtering",
            "methods": methods,
            "method_configs": {m: json_value(c) for m, c in resolved_configs.items()},
            "joint_clf_slack_weight_override": joint_clf_slack_weight,
            "candidate_policy": "one_nearest_no_switch",
            "config": json_value(config),
            "source_sha256": sources,
            "resource_sha256": {
                str(p): digest(p)
                for p in sorted(
                    _model_resources(config.model_path) | {config.safety.rdf.artifact}
                )
            },
            "data_sha256": snapshot,
            "hostname": platform.node(),
            "pid": os.getpid(),
            "scoring": "same first EE pose hit and whole-trace barrier/joint safety for all methods",
            "joint_clf": "V=(q-q_star)^T(q-q_star); gradient=2*(q-q_star); unscaled CLF; slack weight recorded in method_configs",
            "stopping": "existing applied-velocity and normalized EE/stage task-progress rule",
        }
        (output / "cases.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in cases)
        )
        manifest["data_sha256"]["cases.jsonl"] = digest(output / "cases.jsonl")
        atomic_json(output / "manifest.json", manifest)
        kin, self_safety = build_panda_components(replace(config, obstacles=()))
        candidates = {}
        # Validate every saved endpoint before starting any method. Missing IK
        # remains a failure in both IK-dependent methods and stays in denominators.
        for case in cases:
            cfg = case_config(config, case)
            candidates[case["case_id"]] = fixed_candidate(
                records[case["case_id"]], cfg, kin, scene_safety(cfg, kin, self_safety)
            )
        rows = []
        for index, case in enumerate(cases):
            name = case["case_id"]
            cfg = case_config(config, case)
            fixed = candidates[name]
            case_dir = output / name
            case_dir.mkdir()
            # Copy exact bytes, preserving the preflight-sealed IK data.
            raw = (reference / name / "ik.json").read_bytes()
            (case_dir / "ik.json").write_bytes(raw)
            if digest(case_dir / "ik.json") != ik_hashes[name]:
                raise ValueError("reference IK changed after preflight")
            offset = index % len(methods)
            order = methods[offset:] + methods[:offset]
            for method in order:
                print(f"[{index + 1}/{count}] {name} {method} start", flush=True)
                result_dir = case_dir / method
                result_dir.mkdir()
                method_cfg = method_config(cfg, method, joint_clf_slack_weight)
                if method != "direct" and not fixed:
                    result = dict(
                        success=False,
                        goal_reached=False,
                        safety_violation=False,
                        reason=records[name].get("reason") or "no_validated_ik",
                        completion_time=None,
                        elapsed=0.0,
                        candidate_switch_count=0,
                        final_joint_error=None,
                    )
                else:
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
                    result, trace = comparison_rollout(
                        method_cfg, runtime, fixed, subdivisions=experiment.subdivisions
                    )
                    result["method"] = method
                    candidate = fixed[0] if fixed else None
                    result["final_joint_error"] = (
                        float(
                            np.linalg.norm(np.asarray(result["final_q"]) - candidate.q)
                        )
                        if candidate is not None
                        else None
                    )
                    if candidate is not None:
                        trace["reference_q_star"] = candidate.q.copy()
                        trace["reference_joint_error"] = np.linalg.norm(
                            trace["q"] - candidate.q, axis=1
                        )
                    verify_trace(
                        result, trace, case, candidate, experiment.subdivisions
                    )
                    np.savez_compressed(result_dir / "trajectory.npz", **trace)
                    result["trace_sha256"] = digest(result_dir / "trajectory.npz")
                result.update(
                    case_id=name,
                    method=method,
                    clf_slack_weight=method_cfg.trajectory.clf_slack_weight,
                    ik_available=bool(fixed),
                    candidate_policy=(
                        "not_used" if method == "direct" else "one_nearest_no_switch"
                    ),
                    ik_sha256=ik_hashes[name],
                    reference_q_star=fixed[0].q.tolist() if fixed else None,
                )
                atomic_json(result_dir / "summary.json", result)
                rows.append(result)
                analysis = export(output, rows, count, methods)
                atomic_json(
                    output / "progress.json",
                    {
                        "completed_rollouts": len(rows),
                        "total_rollouts": count * len(methods),
                        "completed_cases": analysis["completed_cases"],
                        "last_case": name,
                        "last_method": method,
                    },
                )
                print(
                    f"[{index + 1}/{count}] {name} {method}: {result['reason']}, success={result['success']}, time={result['completion_time']}",
                    flush=True,
                )
        if not analysis["complete"]:
            raise ValueError("comparison is incomplete")
        experiment.identity()
        for relative, expected in sources.items():
            if digest(source_root / relative) != expected:
                raise ValueError("source changed during experiment")
        for name, expected in manifest["resource_sha256"].items():
            if digest(name) != expected:
                raise ValueError("resource changed during experiment")
        for name, expected in manifest["data_sha256"].items():
            if digest(output / name) != expected:
                raise ValueError("saved input changed during experiment")
        atomic_json(
            output / "complete.json",
            {
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "validated_cases": count,
                "results": len(rows),
                "elapsed_wall_seconds": perf_counter() - start,
                "data_sha256": {
                    str(p.relative_to(output)): digest(p)
                    for p in sorted(output.rglob("*"))
                    if p.is_file()
                },
            },
        )
        return analysis
    except BaseException as error:
        atomic_json(
            output / "failed.json",
            {"error": repr(error), "at_utc": datetime.now(timezone.utc).isoformat()},
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--joint-clf-slack-weight", type=float)
    args = parser.parse_args(argv)
    experiment = load_panda_experiment(
        args.experiment_config, cases_per_seed=args.count
    )
    result = run(
        args.reference_dir,
        experiment,
        args.output,
        args.count,
        methods=args.methods,
        joint_clf_slack_weight=args.joint_clf_slack_weight,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Reproducible Panda scenes and method-independent experiment measurements."""

from collections import Counter
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import binomtest

from ..domain.panda_config import SphereObstacleConfig
from ..domain.panda_sampling_config import PandaSamplingConfig
from ..domain.rotations import quaternion_orientation_error
from ..domain.spatial_safety import PandaCapsuleSafety


def case_config(base, case):
    quaternion = np.asarray(case["target_quaternion"], dtype=float)
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    return replace(
        base,
        initial_q=np.asarray(case["initial_q"], dtype=float),
        target=replace(
            base.target,
            position=np.asarray(case["target_position"], dtype=float),
            quaternion=quaternion,
            approach=rotation[:, 2],
            up=rotation[:, 1],
        ),
        obstacles=tuple(
            SphereObstacleConfig(
                item["name"], np.asarray(item["position"]), item["radius"], True
            )
            for item in case["obstacles"]
        ),
        safe_ik=replace(base.safe_ik, random_seed=case["ik_seed"]),
    )


def generate_cases(
    config,
    kinematics,
    self_safety,
    seeds,
    cases_per_seed,
    sampling=PandaSamplingConfig(),
):
    """Reject scenes before IK/control; never condition on controller outcomes."""
    cases, statistics = [], {}
    if not sampling.randomize_initial_q:
        p0 = kinematics.pose_and_jacobian(config.initial_q)[0]
        initial_self = _scene_barrier_minimum(self_safety, config.initial_q)
        if initial_self < sampling.minimum_barrier:
            raise ValueError("initial self barrier cannot satisfy sampling margin")
    lower = kinematics.lower_limits + sampling.joint_limit_margin
    upper = kinematics.upper_limits - sampling.joint_limit_margin
    if np.any(lower >= upper):
        raise ValueError("sampling joint margin leaves no valid joint range")
    for seed in seeds:
        rng = np.random.default_rng(seed)
        counts = Counter()
        accepted = 0
        for proposal in range(sampling.max_proposals_per_seed):
            # The default path consumes no extra RNG draws, preserving old scenes.
            initial_q = (
                rng.uniform(lower, upper)
                if sampling.randomize_initial_q
                else config.initial_q
            )
            witness = rng.uniform(lower, upper)
            centers = rng.uniform(
                sampling.workspace_lower,
                sampling.workspace_upper,
                size=(sampling.obstacle_count, 3),
            )
            radii = rng.uniform(
                *sampling.obstacle_radius_range, size=sampling.obstacle_count
            )
            position, quaternion, _ = kinematics.pose_and_jacobian(witness)
            if sampling.randomize_initial_q:
                p0 = kinematics.pose_and_jacobian(initial_q)[0]
            counts["proposals"] += 1
            if np.any(position < sampling.workspace_lower) or np.any(
                position > sampling.workspace_upper
            ):
                counts["outside_workspace"] += 1
                continue
            if np.linalg.norm(position - p0) < sampling.minimum_target_distance:
                counts["target_too_close"] += 1
                continue
            if sampling.randomize_initial_q:
                initial_self = _scene_barrier_minimum(self_safety, initial_q)
                if initial_self < sampling.minimum_barrier:
                    counts["initial_self_margin"] += 1
                    continue
            obstacles = tuple(
                SphereObstacleConfig(
                    f"obstacle{i + 1}", centers[i], float(radii[i]), True
                )
                for i in range(sampling.obstacle_count)
            )
            safety = PandaCapsuleSafety(
                kinematics,
                config.safety.capsules,
                obstacles,
                config.safety.margin,
                config.safety.self_collision_exclusions,
                include_self=False,
            )
            initial_obstacle = _scene_barrier_minimum(safety, initial_q)
            if initial_obstacle < sampling.minimum_barrier:
                counts["initial_margin"] += 1
                continue
            if (
                min(
                    _scene_barrier_minimum(self_safety, witness),
                    _scene_barrier_minimum(safety, witness),
                )
                < sampling.minimum_barrier
            ):
                counts["goal_margin"] += 1
                continue
            cases.append(
                {
                    "case_id": f"seed{seed}_{accepted:04d}",
                    "seed": seed,
                    "proposal_index": proposal,
                    "ik_seed": int(
                        np.random.SeedSequence([seed, accepted, 1]).generate_state(1)[0]
                    ),
                    "initial_q": initial_q.tolist(),
                    "witness_q": witness.tolist(),
                    "target_position": position.tolist(),
                    "target_quaternion": quaternion.tolist(),
                    "obstacles": [
                        {
                            "name": item.name,
                            "position": item.position.tolist(),
                            "radius": item.radius,
                        }
                        for item in obstacles
                    ],
                }
            )
            if sampling.randomize_initial_q:
                cases[-1]["initial_safety"] = {
                    "self_minimum_barrier": initial_self,
                    "obstacle_minimum_barrier": initial_obstacle,
                    "joint_margin": float(
                        min(
                            np.min(initial_q - kinematics.lower_limits),
                            np.min(kinematics.upper_limits - initial_q),
                        )
                    ),
                }
            accepted += 1
            if accepted == cases_per_seed:
                break
        counts["accepted"] = accepted
        statistics[str(seed)] = dict(counts)
        if accepted != cases_per_seed:
            raise RuntimeError(f"sampling budget exhausted: {statistics}")
    return cases, statistics


def _scene_barrier_minimum(safety, q):
    evaluation = safety.evaluate(q)
    if not (
        np.isfinite(evaluation.values).all()
        and np.isfinite(evaluation.gradients).all()
        and np.isfinite(evaluation.minimum)
    ):
        raise FloatingPointError("nonfinite scene safety evaluation")
    return evaluation.minimum


def measure_state(config, kinematics, q, barrier):
    if not (
        np.isfinite(q).all()
        and np.isfinite(barrier.values).all()
        and np.isfinite(barrier.gradients).all()
    ):
        raise FloatingPointError("nonfinite trajectory state or barrier")
    position, quaternion, _ = kinematics.pose_and_jacobian(q)
    position_error = float(np.linalg.norm(position - config.target.position))
    orientation_error = float(
        np.linalg.norm(
            quaternion_orientation_error(config.target.quaternion, quaternion)
        )
    )
    if not np.isfinite([position_error, orientation_error]).all():
        raise FloatingPointError("nonfinite pose error")
    joint_margin = float(
        min(np.min(q - kinematics.lower_limits), np.min(kinematics.upper_limits - q))
    )
    return {
        "position_error": position_error,
        "orientation_error": orientation_error,
        "rotation_angle": float(2 * np.arcsin(np.clip(orientation_error, 0, 1))),
        "pose_clf": 0.5
        * (
            position_error**2
            + config.trajectory.orientation_weight * orientation_error**2
        ),
        "joint_margin": joint_margin,
        "goal_reached": bool(
            position_error <= config.target.position_tolerance
            and orientation_error <= config.target.orientation_tolerance
        ),
    }


def paired_analysis(rows, expected_cases, duration):
    """Keep paired denominators, IK eligibility and both-success times explicit."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case_id"], {})[row["method"]] = row
    pairs = [pair for pair in grouped.values() if set(pair) == {"direct", "sequential"}]

    def analyze(items):
        n = len(items)
        if not n:
            return {"case_count": 0}
        direct = np.array([p["direct"]["success"] for p in items], dtype=int)
        seq = np.array([p["sequential"]["success"] for p in items], dtype=int)
        delta = seq - direct
        rng = np.random.default_rng(20261005)
        boot = delta[rng.integers(0, n, size=(10000, n))].mean(axis=1)
        seq_only, direct_only = int(np.sum(delta == 1)), int(np.sum(delta == -1))
        times = [
            p["sequential"]["completion_time"] - p["direct"]["completion_time"]
            for p in items
            if p["direct"]["success"] and p["sequential"]["success"]
        ]
        result = {
            "case_count": n,
            "success_rate_difference": float(delta.mean()),
            "difference_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "outcomes": {
                "both_success": int(np.sum(direct * seq)),
                "sequential_only": seq_only,
                "direct_only": direct_only,
                "both_failed": int(np.sum((1 - direct) * (1 - seq))),
            },
            "mcnemar_exact_p": (
                float(binomtest(seq_only, seq_only + direct_only).pvalue)
                if seq_only + direct_only
                else 1.0
            ),
            "both_success_time_difference": {
                "count": len(times),
                "median": float(np.median(times)) if times else None,
                "mean": float(np.mean(times)) if times else None,
            },
        }
        for method in ("direct", "sequential"):
            selected = [p[method] for p in items]
            grid = np.linspace(0, duration, 13)
            result[method] = {
                "success_count": sum(r["success"] for r in selected),
                "success_rate": sum(r["success"] for r in selected) / n,
                "goal_reached_count": sum(r["goal_reached"] for r in selected),
                "termination_reasons": dict(Counter(r["reason"] for r in selected)),
                "safety_violation_count": sum(r["safety_violation"] for r in selected),
                "time_grid": grid.tolist(),
                "success_fraction_by_time": [
                    sum(r["success"] and r["completion_time"] <= t for r in selected)
                    / n
                    for t in grid
                ],
            }
        return result

    eligible = [p for p in pairs if p["sequential"]["ik_available"]]
    return {
        "expected_cases": expected_cases,
        "completed_pairs": len(pairs),
        "complete": len(pairs) == expected_cases,
        "ik_available_pairs": len(eligible),
        "all_scenes": analyze(pairs),
        "ik_conditional": analyze(eligible),
    }

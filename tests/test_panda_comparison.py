"""Protect paired scoring against stale barriers, stage gates and biased cohorts."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from safe_ik.application.panda_comparison import (
    SceneSafety,
    atomic_json,
    digest,
    verify_resume,
)
from safe_ik.application.panda_comparison_rollout import comparison_rollout
from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.application.panda_runtime import PandaRuntime
from safe_ik.domain.ik_contracts import SafeIKCandidate
from safe_ik.domain.safety_contracts import BarrierEvaluation
from safe_ik.domain.types import QPSolution, QPStatus
from safe_ik.experiments.panda_comparison import paired_analysis

ROOT = Path(__file__).resolve().parents[1]


class Kinematics:
    lower_limits = np.full(7, -10.0)
    upper_limits = np.full(7, 10.0)

    def pose_and_jacobian(self, q):
        return q[:3].copy(), np.array([1.0, 0.0, 0.0, 0.0]), np.eye(6, 7)

    def frame_point_and_jacobian(self, name, q, local_offset=None):
        return q[3:6].copy(), np.eye(7)[3:6]


class Safety:
    def __init__(self, function=lambda q: 1.0, label="self_test"):
        self.function, self.label = function, label

    def evaluate(self, q):
        return BarrierEvaluation(
            (self.label,), np.array([self.function(q)]), np.zeros((1, 7))
        )


class Plant:
    def __init__(self):
        self.q = np.zeros(7)
        self.time = 0.0

    def advance(self, velocity, period):
        self.q += velocity * period
        self.time += period


class Solver:
    def __init__(self, status=QPStatus.SOLVED):
        self.status = status

    def solve(self, problem):
        return QPSolution(
            self.status,
            (
                np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                if self.status == QPStatus.SOLVED
                else None
            ),
            0.001,
            "test",
        )


class PandaComparisonTest(unittest.TestCase):
    def setUp(self):
        config = load_panda_config(ROOT / "configs/panda.yaml")
        self.config = replace(
            config,
            initial_q=np.zeros(7),
            target=replace(
                config.target,
                position=np.array([0.02, 0.0, 0.0]),
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                position_tolerance=0.001,
            ),
            trajectory=replace(
                config.trajectory, mode="direct", duration=0.02, integration_period=0.02
            ),
        )

    def run_case(
        self, safety=None, config=None, candidate=None, status=QPStatus.SOLVED
    ):
        runtime = PandaRuntime(
            Kinematics(), safety or Safety(), Solver(status), Plant()
        )
        return comparison_rollout(config or self.config, runtime, candidate)

    def test_goal_on_final_step_is_counted_and_recorded(self):
        result, trace = self.run_case()
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 1)
        self.assertEqual(len(trace["q"]), 2)
        np.testing.assert_array_equal(trace["q"][-1], result["final_q"])
        self.assertAlmostEqual(result["completion_time"], 0.02)

    def test_newly_unsafe_final_state_is_not_success(self):
        result, trace = self.run_case(Safety(lambda q: 0.01 - q[0]))
        self.assertTrue(result["goal_reached"])
        self.assertFalse(result["success"])
        self.assertLess(trace["barrier_values"][-1, 0], 0)

    def test_interpolation_detects_crossing_between_safe_samples(self):
        result, _ = self.run_case(Safety(lambda q: (q[0] - 0.01) ** 2 - 0.00003))
        self.assertTrue(result["success"])
        self.assertFalse(result["interpolated_success"])
        self.assertEqual(result["interpolation_checks"], 4)

    def test_goal_does_not_require_intermediate_stage_completion(self):
        config = replace(
            self.config, trajectory=replace(self.config.trajectory, mode="sequential")
        )
        candidate = SafeIKCandidate(
            np.array([0.02, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), 0, 1, 0.0, 0.0, 1.0
        )
        result, trace = self.run_case(config=config, candidate=candidate)
        self.assertTrue(result["success"])
        self.assertEqual(trace["stage_index"][0], 0)
        self.assertGreater(
            trace["stage_errors"][-1, 0], config.trajectory.stage_position_tolerance
        )

    def test_failed_qp_is_recorded_without_integrating_dummy_command(self):
        result, trace = self.run_case(status=QPStatus.INFEASIBLE)
        self.assertEqual(result["reason"], "qp_infeasible")
        self.assertFalse(result["success"])
        self.assertEqual(len(trace["q"]), 1)
        self.assertEqual(len(trace["command"]), 1)
        self.assertFalse(trace["command_applied"][0])

    def test_scene_obstacles_do_not_leak_through_shared_self_cache(self):
        shared = Safety()
        first = SceneSafety(shared, Safety(lambda q: 0.2, "obstacle"))
        second = SceneSafety(shared, Safety(lambda q: -0.1, "obstacle"))
        q = np.zeros(7)
        self.assertEqual(first.evaluate(q).minimum, 0.2)
        self.assertEqual(second.evaluate(q).minimum, -0.1)

    def test_ik_failures_stay_in_all_scene_denominator(self):
        rows = []
        for case, ik, direct, sequential in (
            ("a", True, False, True),
            ("b", False, True, False),
        ):
            for method, success in (("direct", direct), ("sequential", sequential)):
                rows.append(
                    dict(
                        case_id=case,
                        method=method,
                        ik_available=ik,
                        success=success,
                        goal_reached=success,
                        reason="goal_reached" if success else "timeout",
                        safety_violation=False,
                        completion_time=1.0 if success else None,
                    )
                )
        result = paired_analysis(rows, 2, 60.0)
        self.assertEqual(result["all_scenes"]["case_count"], 2)
        self.assertEqual(result["ik_conditional"]["case_count"], 1)
        self.assertEqual(result["all_scenes"]["success_rate_difference"], 0.0)
        self.assertEqual(result["ik_conditional"]["success_rate_difference"], 1.0)

    def test_resume_rejects_changed_settings_and_tampered_cases(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cases.jsonl").write_text("{}\n")
            atomic_json(
                root / "manifest.json",
                {
                    "identity": {"seed": 1},
                    "data_sha256": {"cases.jsonl": digest(root / "cases.jsonl")},
                },
            )
            verify_resume(root, {"seed": 1})
            with self.assertRaisesRegex(ValueError, "settings changed"):
                verify_resume(root, {"seed": 2})
            (root / "cases.jsonl").write_text(json.dumps({"changed": True}))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_resume(root, {"seed": 1})


if __name__ == "__main__":
    unittest.main()

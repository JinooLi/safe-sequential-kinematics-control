"""Random initial configurations must be safe, reproducible and replayable."""

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.domain.panda_sampling_config import PandaSamplingConfig
from safe_ik.domain.safety_contracts import BarrierEvaluation
from safe_ik.experiments.panda_comparison import case_config, generate_cases

ROOT = Path(__file__).resolve().parents[1]


class Kinematics:
    lower_limits = np.full(7, -1.0)
    upper_limits = np.full(7, 1.0)

    def pose_and_jacobian(self, q):
        return q[:3], np.array([1.0, 0, 0, 0]), np.eye(6, 7)


class Safety:
    def __init__(self, component=0, offset=0.2, gradient=1.0):
        self.component = component
        self.offset = offset
        self.gradient = gradient

    def evaluate(self, q):
        gradients = np.zeros((1, 7))
        gradients[0, self.component] = self.gradient
        return BarrierEvaluation(
            ("test",), np.array([q[self.component] + self.offset]), gradients
        )


class PandaRandomInitialSamplingTest(unittest.TestCase):
    def setUp(self):
        self.config = replace(
            load_panda_config(ROOT / "tests/fixtures/panda.yaml"),
            initial_q=np.zeros(7),
        )
        self.kin = Kinematics()
        self.self_safety = Safety(0)
        self.obstacle_safety = Safety(1)
        self.sampling = PandaSamplingConfig(
            randomize_initial_q=True,
            workspace_lower=(-1, -1, -1),
            workspace_upper=(1, 1, 1),
            obstacle_count=10,
            minimum_target_distance=0.5,
        )
        self.addCleanup(patch.stopall)
        patch(
            "safe_ik.experiments.panda_comparison.PandaCapsuleSafety",
            return_value=self.obstacle_safety,
        ).start()

    def generate(self, seeds=(19,), count=12, **overrides):
        return generate_cases(
            self.config,
            self.kin,
            self.self_safety,
            seeds,
            count,
            replace(self.sampling, **overrides),
        )

    def test_random_starts_pass_joint_self_obstacle_and_distance_checks(self):
        cases, stats = self.generate()
        self.assertGreater(stats["19"]["initial_self_margin"], 0)
        self.assertGreater(stats["19"]["initial_margin"], 0)
        self.assertEqual(len({tuple(c["initial_q"]) for c in cases}), len(cases))
        for case in cases:
            q = np.asarray(case["initial_q"])
            witness = np.asarray(case["witness_q"])
            joint_margin = min(np.min(q + 1), np.min(1 - q))
            self.assertGreaterEqual(joint_margin, self.sampling.joint_limit_margin)
            self.assertGreaterEqual(
                np.linalg.norm(q[:3] - case["target_position"]),
                self.sampling.minimum_target_distance,
            )
            for state in (q, witness):
                for evaluator in (self.self_safety, self.obstacle_safety):
                    self.assertGreaterEqual(
                        evaluator.evaluate(state).minimum, self.sampling.minimum_barrier
                    )
            evidence = case["initial_safety"]
            self.assertEqual(evidence["joint_margin"], joint_margin)
            self.assertEqual(
                evidence["self_minimum_barrier"], self.self_safety.evaluate(q).minimum
            )
            self.assertEqual(
                evidence["obstacle_minimum_barrier"],
                self.obstacle_safety.evaluate(q).minimum,
            )
            np.testing.assert_array_equal(case_config(self.config, case).initial_q, q)

    def test_same_seed_and_separate_seed_workers_produce_identical_cases(self):
        together, stats = self.generate(seeds=(19, 23), count=5)
        self.assertEqual(self.generate(seeds=(19, 23), count=5), (together, stats))
        first, first_stats = self.generate(seeds=(19,), count=5)
        second, second_stats = self.generate(seeds=(23,), count=5)
        self.assertEqual(together, first + second)
        self.assertEqual(stats, first_stats | second_stats)

    def test_fixed_mode_preserves_configured_initial_configuration(self):
        cases, _ = self.generate(randomize_initial_q=False, count=3)
        for case in cases:
            self.assertEqual(case["initial_q"], self.config.initial_q.tolist())
            self.assertNotIn("initial_safety", case)
        self.assertFalse(PandaSamplingConfig().randomize_initial_q)

    def test_random_mode_does_not_require_unused_configured_start_to_be_safe(self):
        self.config = replace(self.config, initial_q=np.full(7, -0.9))
        cases, _ = self.generate(count=1)
        self.assertGreaterEqual(
            cases[0]["initial_safety"]["self_minimum_barrier"], 0.005
        )
        with self.assertRaisesRegex(ValueError, "initial self barrier"):
            self.generate(randomize_initial_q=False, count=1)

    def test_unsafe_starts_exhaust_budget_without_relaxing_margin(self):
        for evaluator in (self.self_safety, self.obstacle_safety):
            with (
                self.subTest(evaluator=evaluator),
                patch.object(evaluator, "offset", -2),
            ):
                with self.assertRaisesRegex(RuntimeError, "sampling budget exhausted"):
                    self.generate(
                        count=1, max_proposals_per_seed=4, minimum_target_distance=0
                    )

    def test_positive_barrier_below_start_margin_is_rejected(self):
        for evaluator in (self.self_safety, self.obstacle_safety):
            for minimum in (0.004, 0.005):
                result = BarrierEvaluation(
                    ("test",), np.array([minimum]), np.zeros((1, 7))
                )
                with (
                    self.subTest(evaluator=evaluator, minimum=minimum),
                    patch.object(evaluator, "evaluate", return_value=result),
                ):
                    if minimum < self.sampling.minimum_barrier:
                        with self.assertRaisesRegex(RuntimeError, "sampling budget"):
                            self.generate(
                                count=1,
                                max_proposals_per_seed=4,
                                minimum_target_distance=0,
                            )
                    else:
                        self.assertEqual(len(self.generate(count=1)[0]), 1)

    def test_nonfinite_safety_values_or_gradients_abort_instead_of_accepting(self):
        for evaluator in (self.self_safety, self.obstacle_safety):
            for attribute in ("offset", "gradient"):
                with (
                    self.subTest(evaluator=evaluator, attribute=attribute),
                    patch.object(evaluator, attribute, float("nan")),
                    self.assertRaisesRegex(
                        FloatingPointError, "nonfinite scene safety"
                    ),
                ):
                    self.generate(count=1, minimum_target_distance=0)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.application.compare_sequential import (
    classify_outcome,
    has_feasible_goal_configuration,
    sample_case,
)
from src.application.configuration import load_controller_config


ROOT = Path(__file__).resolve().parents[1]


class CompareSequentialTest(unittest.TestCase):
    def test_outcome_classification_covers_all_four_comparisons(self):
        self.assertEqual(classify_outcome(False, False), "both_failed")
        self.assertEqual(classify_outcome(False, True), "sequential_only_success")
        self.assertEqual(classify_outcome(True, False), "nonsequential_only_success")
        self.assertEqual(classify_outcome(True, True), "both_succeeded")

    def test_sampling_is_seeded_and_target_is_reachable(self):
        config = load_controller_config(ROOT / "settings/controller.yaml")
        first = sample_case(7, np.random.default_rng(11), config, config.sequential_experiment)
        second = sample_case(7, np.random.default_rng(11), config, config.sequential_experiment)

        self.assertEqual(first, second)
        self.assertEqual(len(first.obstacles), 2)
        self.assertLessEqual(np.linalg.norm(first.target), 2.7)
        self.assertGreaterEqual(np.linalg.norm(first.target), 0.6)
        self.assertGreaterEqual(first.target_yaw, -np.pi)
        self.assertLessEqual(first.target_yaw, np.pi)

    def test_missing_ik_solution_is_not_feasible_for_comparison_sampling(self):
        config = load_controller_config(ROOT / "settings/controller.yaml")
        with patch("src.application.compare_sequential.solve_goal_configurations", return_value=()):
            feasible = has_feasible_goal_configuration(
                np.array([1.0, 1.0]),
                0.0,
                tuple(config.obstacles),
                config,
            )
        self.assertFalse(feasible)

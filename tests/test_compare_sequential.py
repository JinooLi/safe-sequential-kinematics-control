from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.application.compare_sequential import (
    _failure_reason,
    _paired_analysis,
    classify_outcome,
    has_feasible_goal_configuration,
    sample_case,
)
from src.application.configuration import load_controller_config


ROOT = Path(__file__).resolve().parents[1]


class CompareSequentialTest(unittest.TestCase):
    def test_outcome_classification_covers_all_four_comparisons(self):
        self.assertEqual(classify_outcome(False, False), "both_failed")
        self.assertEqual(classify_outcome(False, True), "candidate_only_success")
        self.assertEqual(classify_outcome(True, False), "baseline_only_success")
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

    def test_failure_taxonomy_prioritizes_safety_then_qp_then_timeout(self):
        self.assertEqual(
            _failure_reason(False, -0.01, -0.001, 2, True, 1, 3, ()),
            "safety_violation",
        )
        self.assertEqual(
            _failure_reason(False, 0.02, -0.001, 1, True, 1, 3, ()),
            "qp_infeasible",
        )
        self.assertEqual(
            _failure_reason(False, 0.02, -0.001, 0, True, 1, 3, ("stopped at CBF boundary",)),
            "cbf_boundary_stall",
        )
        self.assertEqual(
            _failure_reason(False, 0.02, -0.001, 0, False, 3, 3, ()),
            "timeout",
        )

    def test_paired_analysis_uses_only_jointly_successful_cases_for_deltas(self):
        def run(success, completion_time):
            return {
                "success": success, "failure_reason": None if success else "timeout",
                "completion_time": completion_time, "control_effort": 1.0,
                "min_barrier": 0.1, "position_error": 0.01, "yaw_error": 0.02,
                "max_joint_velocity": 1.0, "total_qp_solve_time": 0.01,
                "mean_qp_solve_time": 0.001, "mean_clf_slack": 0.0, "max_clf_slack": 0.0,
            }

        analysis = _paired_analysis([
            {
                "direct": run(True, 2.0),
                "sequential": run(True, 3.0),
                "comparisons": {"task_decomposition": "both_succeeded"},
            },
            {
                "direct": run(False, None),
                "sequential": run(True, 1.0),
                "comparisons": {"task_decomposition": "candidate_only_success"},
            },
        ])
        delta = analysis["comparisons"]["task_decomposition"]["paired_both_success"]["sequential_minus_direct_completion_time"]
        self.assertEqual(delta["count"], 1)
        self.assertEqual(delta["mean"], 1.0)

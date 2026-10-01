"""An explicit per-seed budget preserves the old numerical IK behavior."""

import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.application.panda_setup import build_panda_components
from safe_ik.domain.safe_pose_ik import (
    IKSeed,
    SafeIKCandidate,
    SafePoseIKSolver,
    damped_pseudoinverse,
)
from safe_ik.domain.spatial_safety import BarrierEvaluation

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/panda.yaml"


class IKIterationNumberTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load_panda_config(FIXTURE)
        cls.kinematics, cls.safety = build_panda_components(cls.base)

    def load_with_ik_settings(self, settings):
        raw = yaml.safe_load(FIXTURE.read_text())
        raw["safe_ik"] = settings
        with patch(
            "safe_ik.application.panda_configuration.yaml.safe_load", return_value=raw
        ):
            return load_panda_config(FIXTURE)

    def test_new_configuration_and_snapshot_only_use_iteration_number(self):
        settings = yaml.safe_load(FIXTURE.read_text())["safe_ik"]
        for count in (1, 4, 1000):
            with self.subTest(count=count):
                config = self.load_with_ik_settings(
                    dict(settings, iteration_number=count)
                )
                snapshot = asdict(config.safe_ik)
                self.assertEqual(snapshot["iteration_number"], count)
                self.assertNotIn("integration_period", snapshot)
                self.assertNotIn("duration_per_seed", snapshot)
                self.assertEqual(config.safe_ik.convergence_hold_steps, 300)
                self.assertEqual(config.trajectory.integration_period, 0.05)

    def test_iteration_number_must_be_a_positive_integer(self):
        settings = yaml.safe_load(FIXTURE.read_text())["safe_ik"]
        for value in (0, -1, 1.5, 2.0, True, False, "2", None, np.inf, np.nan):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "safe_ik.iteration_number"):
                    self.load_with_ik_settings(dict(settings, iteration_number=value))
                with self.assertRaisesRegex(ValueError, "safe_ik.iteration_number"):
                    replace(self.base.safe_ik, iteration_number=value)

    def test_legacy_fields_are_rejected_even_alongside_iteration_number(self):
        settings = yaml.safe_load(FIXTURE.read_text())["safe_ik"]
        for legacy in (
            {"integration_period": 0.01},
            {"duration_per_seed": 10.0},
            {"integration_period": 0.01, "duration_per_seed": 10.0},
        ):
            for keep_new_field in (False, True):
                with self.subTest(legacy=legacy, keep_new_field=keep_new_field):
                    mixed = dict(settings, **legacy)
                    if not keep_new_field:
                        mixed.pop("iteration_number")
                    with self.assertRaisesRegex(ValueError, "safe_ik.iteration_number"):
                        self.load_with_ik_settings(mixed)

    def test_failed_search_uses_exact_budget_for_each_seed_and_both_modes(self):
        for count in (1, 4, 7):
            for require_safe in (False, True):
                with self.subTest(count=count, require_safe=require_safe):
                    config = replace(
                        self.base,
                        safe_ik=replace(
                            self.base.safe_ik, iteration_number=count, max_restarts=2
                        ),
                    )
                    solver = SafePoseIKSolver(config, self.kinematics, self.safety)
                    with patch.object(solver, "_candidate", return_value=None) as check:
                        result = solver.solve(require_safe=require_safe)
                    self.assertIsNone(result.selected)
                    self.assertEqual(
                        [(call.args[1], call.args[2]) for call in check.call_args_list],
                        [
                            (seed, step)
                            for seed in range(2)
                            for step in range(1, count + 1)
                        ],
                    )

    def test_hold_must_fit_budget_and_can_complete_on_last_iteration(self):
        for count in (2, 3):
            with self.subTest(count=count):
                config = replace(
                    self.base,
                    safe_ik=replace(
                        self.base.safe_ik,
                        iteration_number=count,
                        convergence_hold_duration=0.03,
                    ),
                )
                solver = SafePoseIKSolver(config, self.kinematics, self.safety)

                def candidate(q, seed, step, require_safe, **_evaluation):
                    return SafeIKCandidate(q.copy(), seed, step, 0.0, 0.0, 0.1)

                with patch.object(solver, "_candidate", side_effect=candidate) as check:
                    result = solver.solve_seed(IKSeed(0, config.initial_q))
                self.assertEqual(check.call_count, count)
                if count < 3:
                    self.assertIsNone(result)
                else:
                    self.assertEqual(result.iterations, 3)

    def test_update_scale_stays_at_old_001_for_different_budgets(self):
        position_error = np.array([0.002, -0.001, 0.003])
        orientation_error = np.zeros(3)
        jacobian = np.column_stack((np.eye(6), np.zeros(6)))
        gradient = np.arange(1, 8) * 0.01
        ik = self.base.safe_ik
        primary = damped_pseudoinverse(jacobian, ik.damping) @ (
            ik.pose_gain * np.concatenate((position_error, 2.0 * orientation_error))
        )
        null_projector = np.eye(7) - np.linalg.pinv(jacobian) @ jacobian
        for count in (2, 4):
            for require_safe in (False, True):
                with self.subTest(count=count, require_safe=require_safe):
                    config = replace(
                        self.base,
                        safe_ik=replace(ik, iteration_number=count),
                    )
                    solver = SafePoseIKSolver(config, self.kinematics, self.safety)
                    with (
                        patch.object(solver, "_candidate", return_value=None) as check,
                        patch.object(
                            solver,
                            "_pose_errors",
                            return_value=(position_error, orientation_error, jacobian),
                        ),
                        patch.object(
                            self.safety,
                            "evaluate",
                            return_value=BarrierEvaluation(
                                ("test",), np.array([0.0]), gradient[None]
                            ),
                        ),
                    ):
                        solver.solve_seed(
                            IKSeed(0, config.initial_q), require_safe=require_safe
                        )
                    secondary = (
                        null_projector @ (ik.safety_gradient_gain * gradient)
                        if require_safe
                        else np.zeros(7)
                    )
                    expected = config.initial_q.copy()
                    for call in check.call_args_list:
                        np.testing.assert_array_equal(call.args[0], expected)
                        expected = np.clip(
                            expected + 0.01 * (primary + secondary),
                            self.kinematics.lower_limits,
                            self.kinematics.upper_limits,
                        )

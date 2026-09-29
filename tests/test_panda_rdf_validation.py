"""Exterior accuracy uses mesh truth; interior training and derivative checks remain."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from safe_ik.application.panda_rdf_training import validate_model
from safe_ik.application.panda_rdf_validation import (
    ACCURACY_DOMAIN,
    accuracy_masks,
    distance_accuracy_report,
)
from safe_ik.domain.panda_rdf import BernsteinRDF, ExtendedRDF


class ExteriorAccuracyTest(unittest.TestCase):
    def report(self, truth, prediction, **kwargs):
        return distance_accuracy_report(
            np.zeros((len(truth), 3)),
            truth,
            prediction,
            kwargs.pop("error_margin", 0.005),
            global_count=kwargs.pop("global_count", len(truth)),
            **kwargs,
        )

    def test_exact_surface_included_and_any_negative_truth_excluded(self):
        outside, near = accuracy_masks(np.array([-0.03, -1e-15, 0, 0.06, 0.0600001]))
        np.testing.assert_array_equal(outside, [False, False, True, True, True])
        np.testing.assert_array_equal(near, [False, False, True, True, False])

    def test_internal_error_excluded_from_all_accuracy_statistics(self):
        report = self.report(
            [-0.02, 0, 0.03, 0.08], [100, -0.002, 0.033, 0.081], global_count=2
        )
        self.assertEqual(report["accuracy_domain"], ACCURACY_DOMAIN)
        self.assertEqual(report["evaluated_count"], 3)
        self.assertEqual(report["interior_excluded_count"], 1)
        self.assertEqual(report["near_count"], 2)
        self.assertEqual(report["global_evaluated_count"], 1)
        self.assertAlmostEqual(report["near_max_error"], 0.003)
        self.assertAlmostEqual(report["near_mean_error"], 0.0025)
        self.assertAlmostEqual(report["mean_error"], 0.002)
        self.assertAlmostEqual(report["global_max_error"], 0.002)
        self.assertEqual(report["sign_errors"], 1)
        self.assertTrue(report["distance_passed"])
        self.assertTrue(all(p["truth"] >= 0 for p in report["worst_near_points"]))
        json.dumps(report, allow_nan=False)

    def test_negative_prediction_at_exterior_point_is_not_filtered(self):
        report = self.report([-0.03, 0.02], [0.03, -0.04])
        self.assertEqual(report["sign_errors"], 1)
        self.assertAlmostEqual(report["near_max_error"], 0.06)
        self.assertFalse(report["distance_passed"])

    def test_10mm_limit_accepts_new_range_but_rejects_larger_error(self):
        for prediction, accepted in (
            (0.007113405, True),
            (0.010, True),
            (0.010001, False),
        ):
            with self.subTest(prediction=prediction):
                old = self.report([0.0], [prediction], error_margin=0.005)
                new = self.report([0.0], [prediction], error_margin=0.010)
                self.assertFalse(old["distance_passed"])
                self.assertEqual(new["distance_passed"], accepted)
                self.assertEqual(old["near_max_error"], new["near_max_error"])

    def test_empty_exterior_near_region_never_vacuously_passes(self):
        for truth in ([-0.01, -0.02], [0.061, 0.1]):
            with (
                self.subTest(truth=truth),
                self.assertRaisesRegex(ValueError, "no exterior near"),
            ):
                self.report(truth, truth)

    def test_no_exterior_global_points_is_reported_as_null(self):
        report = self.report([-0.03, 0.02], [10, 0.021], global_count=1)
        self.assertIsNone(report["global_max_error"])
        self.assertEqual(report["global_evaluated_count"], 0)
        json.dumps(report, allow_nan=False)

    def test_nonfinite_field_values_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid distance accuracy"):
            self.report([-0.03, 0.02], [np.nan, 0.021])
        with self.assertRaisesRegex(ValueError, "reference distances"):
            self.report([np.nan, 0.02], [0, 0.021])

    def validation(self, bad_derivative=False, error_margin=0.005):
        mesh = trimesh.creation.box(extents=[0.1] * 3)
        model = ExtendedRDF(
            BernsteinRDF(np.zeros(3), 0.3, np.full((2,) * 3, 0.003)), 0.2, 0.003
        )
        truth = np.full(5000, 0.002)
        truth[:100] = -1
        gradient = np.zeros((5000, 3))
        ambiguous = np.zeros(5000, dtype=bool)
        ambiguous[:100] = True
        with patch(
            "safe_ik.application.panda_rdf_training.mesh_truth",
            return_value=(truth, gradient, ambiguous),
        ):
            if bad_derivative:
                with patch(
                    "safe_ik.application.panda_rdf_training.numerical_gradient",
                    side_effect=lambda model, points: np.ones_like(points),
                ):
                    return validate_model(
                        mesh, model, np.random.default_rng(2), error_margin
                    )
            return validate_model(mesh, model, np.random.default_rng(2), error_margin)

    def test_validate_model_uses_exterior_policy_and_preserves_generated_count(self):
        report = self.validation()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["sample_counts"], [3000, 2000])
        self.assertEqual(report["near_count"], 4900)
        self.assertEqual(report["interior_excluded_count"], 100)
        self.assertEqual(report["ambiguous_gradient_count"], 0)

    def test_derivative_failure_still_rejects_good_exterior_accuracy(self):
        for margin in (0.005, 0.010):
            with self.subTest(error_margin=margin):
                report = self.validation(bad_derivative=True, error_margin=margin)
                self.assertTrue(report["distance_passed"])
                self.assertFalse(report["derivatives_passed"])
                self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()

"""Directional limits affect approval, never clearance, geometry or training."""

import copy
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import test_panda_rdf_integration as integration

from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.application.panda_rdf_artifact import (
    geometry_settings,
    load_bundle,
    read_candidate_for_validation,
    save_bundle,
)
from safe_ik.application.panda_rdf_training import validate_saved_model
from safe_ik.application.panda_rdf_validation import (
    ASYMMETRIC_POLICY,
    distance_accuracy_report,
    validation_policy,
)
from safe_ik.application.revalidate_panda_rdf import check_coverage
from safe_ik.domain.panda_config import PandaRDFConfig
from safe_ik.domain.panda_rdf_safety import PandaRDFSafety

ROOT = Path(__file__).resolve().parents[1]


class DirectionalLimitsTest(unittest.TestCase):
    def report(self, prediction, **kwargs):
        return distance_accuracy_report(
            np.zeros((1, 3)),
            [0.0],
            [prediction],
            0.010,
            underestimation_limit=0.020,
            global_count=1,
            **kwargs,
        )

    def test_direction_and_exact_boundaries(self):
        for error, accepted in [
            (-0.020, True),
            (-0.020001, False),
            (0.010, True),
            (0.010001, False),
            (-0.015, True),
            (0.015, False),
            (0.0, True),
        ]:
            with self.subTest(error=error):
                r = self.report(error)
                self.assertEqual(r["distance_passed"], accepted)
                self.assertEqual(r["near_max_underestimate"], max(-error, 0))
                self.assertEqual(r["near_max_overestimate"], max(error, 0))
                self.assertEqual(
                    r["near_underestimation_exceedance_count"], int(error < -0.020)
                )
                self.assertEqual(
                    r["near_overestimation_exceedance_count"], int(error > 0.010)
                )
                self.assertEqual(r["accuracy_policy"], ASYMMETRIC_POLICY)

    def test_interior_excluded_and_negative_prediction_retained(self):
        r = distance_accuracy_report(
            np.zeros((4, 3)),
            [-0.001, 0, 0.020, 0.061],
            [100, -0.015, -0.001, 100],
            0.010,
            global_count=4,
            underestimation_limit=0.020,
        )
        self.assertEqual(r["near_count"], 2)
        self.assertEqual(r["interior_excluded_count"], 1)
        self.assertEqual(r["sign_errors"], 2)
        self.assertFalse(r["distance_passed"])
        self.assertAlmostEqual(r["near_max_underestimate"], 0.021)

    def test_bad_limits_rejected(self):
        for bad in [True, "0.020", -0.020, 0, np.nan, np.inf]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    PandaRDFConfig(underestimation_limit=bad)
                with self.assertRaises(ValueError):
                    distance_accuracy_report(
                        np.zeros((1, 3)), [0], [0], 0.01, underestimation_limit=bad
                    )

    def test_config_has_separate_limits_and_unchanged_radii_inputs(self):
        options = load_panda_config(ROOT / "configs/panda.yaml").safety.rdf
        self.assertEqual(options.underestimation_limit, 0.020)
        self.assertEqual(options.error_margin, 0.010)
        # Runtime clearance may be reduced; the directional validation limit
        # must never be added to it or change the geometry inputs.
        self.assertAlmostEqual(
            options.effective_margin, options.physical_margin + options.error_margin
        )
        legacy = replace(options, underestimation_limit=None)
        self.assertEqual(options.effective_margin, legacy.effective_margin)
        self.assertEqual(geometry_settings(options), geometry_settings(legacy))
        r = distance_accuracy_report(
            np.zeros((1, 3)), [0], [-0.015], 0.010, global_count=1
        )
        self.assertFalse(r["distance_passed"])
        self.assertNotIn("accuracy_policy", r)


class AsymmetricLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        integration.PandaRDFIntegrationTest.setUpClass()
        cls.fixture = integration.PandaRDFIntegrationTest()
        cls.geometry = integration.PandaRDFIntegrationTest.geometry

    def bundle(self):
        bundle, options = self.fixture.bundle()
        options = replace(options, underestimation_limit=0.020)
        bundle.metadata["validation_policy"] = validation_policy(options)
        for r in bundle.metadata["validation"].values():
            r.update(
                accuracy_policy=ASYMMETRIC_POLICY,
                overestimation_limit=0.010,
                underestimation_limit=0.020,
                near_max_error=0.015,
                near_max_underestimate=0.015,
                near_max_overestimate=0.007,
                near_overestimation_exceedance_count=0,
                near_underestimation_exceedance_count=0,
            )
        return bundle, options

    def test_legacy_report_requires_revalidation_but_arrays_are_readable(self):
        b, old = self.fixture.bundle()
        new = replace(old, underestimation_limit=0.020)
        with TemporaryDirectory() as d:
            p = Path(d) / "legacy.npz"
            save_bundle(b, p)
            with self.assertRaisesRegex(ValueError, "accuracy policy.*revalidation"):
                load_bundle(p, self.geometry, new)
            offline = read_candidate_for_validation(p, self.geometry, new)
            np.testing.assert_array_equal(
                offline.models["link0"].rdf.weights, b.models["link0"].rdf.weights
            )

    def test_asymmetric_artifact_is_not_legacy_approval(self):
        b, opt = self.bundle()
        with TemporaryDirectory() as d:
            p = Path(d) / "candidate.npz"
            save_bundle(b, p)
            load_bundle(p, self.geometry, opt)
            for changed in (
                replace(opt, underestimation_limit=0.030),
                replace(opt, underestimation_limit=None),
            ):
                with self.assertRaises(ValueError):
                    load_bundle(p, self.geometry, changed)

    def test_fail_closed_mutation_matrix(self):
        changes = [
            ("near_max_underestimate", 0.020001),
            ("near_max_overestimate", 0.010001),
            ("near_max_error", 0.014),
            ("near_max_overestimate", float("nan")),
            ("near_max_underestimate", True),
            ("near_overestimation_exceedance_count", 1),
            ("near_underestimation_exceedance_count", False),
            ("underestimation_limit", 0.03),
            ("overestimation_limit", 0.02),
            ("accuracy_policy", "invented"),
            ("derivatives_passed", False),
            ("distance_passed", False),
            ("near_count", 0),
            ("interior_excluded_count", 0),
        ]
        for key, value in changes:
            with self.subTest(key=key), TemporaryDirectory() as d:
                b, opt = self.bundle()
                b.metadata["validation"]["link0"][key] = value
                # JSON forbids NaN, independently of runtime approval.
                if isinstance(value, float) and np.isnan(value):
                    with self.assertRaises(ValueError):
                        save_bundle(b, Path(d) / "bad.npz")
                    continue
                p = Path(d) / "bad.npz"
                save_bundle(b, p)
                with self.assertRaisesRegex(ValueError, "validation failed"):
                    load_bundle(p, self.geometry, opt)

    def test_both_holdouts_required_even_when_summary_passes(self):
        for mutation in (
            "fail",
            "missing",
            "no_requirements",
            "drop_link",
            "null_requirement",
        ):
            with self.subTest(mutation=mutation), TemporaryDirectory() as d:
                b, opt = self.bundle()
                b.metadata["validation_set_requirements"] = {
                    n: ["regression", "archived_purpose102"] for n in b.models
                }
                for r in b.metadata["validation"].values():
                    r["validation_sets"] = {
                        k: copy.deepcopy(r)
                        for k in ("regression", "archived_purpose102")
                    }
                p = Path(d) / "paired.npz"
                save_bundle(b, p)
                load_bundle(p, self.geometry, opt)
                target = b.metadata["validation"]["link0"]
                if mutation == "fail":
                    target["validation_sets"]["archived_purpose102"][
                        "derivatives_passed"
                    ] = False
                elif mutation == "missing":
                    target["validation_sets"].pop("archived_purpose102")
                elif mutation == "drop_link":
                    b.metadata["validation_set_requirements"].pop("link0")
                elif mutation == "null_requirement":
                    b.metadata["validation_set_requirements"]["link0"] = None
                    target.pop("validation_sets")
                else:
                    b.metadata.pop("validation_set_requirements")
                save_bundle(b, p)
                with self.assertRaises(ValueError):
                    load_bundle(p, self.geometry, opt)

    def test_underestimate_limit_does_not_change_barrier_or_gradient(self):
        b, opt = self.bundle()
        q = self.fixture.config.initial_q
        old = PandaRDFSafety(
            self.fixture.kinematics,
            b.models,
            b.queries,
            b.pairs,
            replace(opt, underestimation_limit=None),
        ).evaluate(q)
        new = PandaRDFSafety(
            self.fixture.kinematics, b.models, b.queries, b.pairs, opt
        ).evaluate(q)
        np.testing.assert_array_equal(old.values, new.values)
        np.testing.assert_array_equal(old.gradients, new.gradients)

    def test_saved_validation_never_resamples_and_derivative_gate_remains(self):
        b, opt = self.bundle()
        model = b.models["link0"]
        mesh = b.reconstructed["link0"]
        points = np.zeros((5000, 3))
        prediction = model.rdf.query(points, False)[0]
        with (
            patch(
                "safe_ik.application.panda_rdf_training.mesh_truth",
                return_value=(
                    prediction + 0.015,
                    np.zeros_like(points),
                    np.ones(5000, dtype=bool),
                ),
            ),
            patch(
                "safe_ik.application.panda_rdf_training.numerical_gradient",
                side_effect=lambda model, points: np.ones_like(points),
            ),
        ):
            # Force truth within near range so this isolates the derivative gate.
            model.rdf.weights[:] = 0.005
            with patch(
                "safe_ik.application.panda_rdf_training.mesh_truth",
                return_value=(
                    np.full(5000, 0.020),
                    np.zeros_like(points),
                    np.ones(5000, dtype=bool),
                ),
            ):
                with TemporaryDirectory() as d:
                    p = Path(d) / "evidence.npz"
                    r = validate_saved_model(
                        mesh,
                        model,
                        points,
                        np.random.default_rng(42),
                        0.01,
                        underestimation_limit=0.02,
                        evidence_path=p,
                    )
                    self.assertTrue(r["distance_passed"])
                    self.assertFalse(r["derivatives_passed"])
                    self.assertFalse(r["passed"])
                    with np.load(p) as a:
                        np.testing.assert_array_equal(a["points"], points)

    def test_source_paired_requirement_cannot_be_dropped(self):
        b, _ = self.bundle()
        b.metadata["validation"]["link0"]["fresh_and_regression_passed"] = False
        manifest = {"evidence": {n: {"regression": {}} for n in b.models}}
        with self.assertRaisesRegex(ValueError, "archived holdout"):
            check_coverage(manifest, b)
        manifest["evidence"]["link0"]["archived_purpose102"] = {}
        check_coverage(manifest, b)


if __name__ == "__main__":
    unittest.main()

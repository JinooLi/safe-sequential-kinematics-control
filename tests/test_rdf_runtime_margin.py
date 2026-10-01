"""Reduced runtime clearance must not resize or reapprove a learned RDF."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import test_panda_rdf_integration as integration

from safe_ik.application.panda_rdf_artifact import (
    load_bundle,
    read_candidate_for_validation,
    save_bundle,
)
from safe_ik.application.panda_setup import build_panda_components
from safe_ik.domain.panda_config import PandaRDFConfig


class RuntimePhysicalMarginTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        integration.PandaRDFIntegrationTest.setUpClass()
        cls.fixture = integration.PandaRDFIntegrationTest()
        cls.geometry = cls.fixture.geometry

    def test_smaller_and_equal_margins_preserve_fields_and_provenance(self):
        bundle, trained = self.fixture.bundle()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rdf.npz"
            identifier = save_bundle(bundle, path)
            original_bytes = path.read_bytes()
            original = load_bundle(path, self.geometry, trained)
            for physical in (0.0, 0.010, 0.020, trained.physical_margin):
                with self.subTest(physical=physical):
                    runtime = replace(trained, physical_margin=physical)
                    loaded = load_bundle(path, self.geometry, runtime)
                    self.assertEqual(loaded.identifier, identifier)
                    self.assertEqual(loaded.metadata, original.metadata)
                    self.assertEqual(loaded.pairs, original.pairs)
                    for name, model in loaded.models.items():
                        old = original.models[name]
                        self.assertEqual(model.inner_radius, old.inner_radius)
                        self.assertEqual(model.rdf.radius, old.rdf.radius)
                        self.assertEqual(model.surface_maximum, old.surface_maximum)
                        np.testing.assert_array_equal(model.rdf.center, old.rdf.center)
                        np.testing.assert_array_equal(
                            model.rdf.weights, old.rdf.weights
                        )
                        np.testing.assert_array_equal(
                            loaded.queries[name], original.queries[name]
                        )
                        points = np.array(
                            [
                                [0.0, 0.0, 0.0],
                                [old.inner_radius, 0.0, 0.0],
                                [(old.inner_radius + old.rdf.radius) / 2, 0.0, 0.0],
                                [old.rdf.radius + 0.1, 0.0, 0.0],
                            ]
                        )
                        for actual, expected in zip(
                            model.query(points), old.query(points)
                        ):
                            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(path.read_bytes(), original_bytes)

    def test_upper_limit_comes_from_each_artifacts_training_value(self):
        for trained_physical in (0.0, 0.015, 0.030, 0.050):
            with (
                self.subTest(trained=trained_physical),
                TemporaryDirectory() as directory,
            ):
                bundle, trained = self.fixture.bundle(
                    PandaRDFConfig(query_density=10, physical_margin=trained_physical)
                )
                path = Path(directory) / "rdf.npz"
                save_bundle(bundle, path)
                load_bundle(path, self.geometry, trained)
                load_bundle(path, self.geometry, replace(trained, physical_margin=0.0))
                above = np.nextafter(trained_physical, np.inf)
                with self.assertRaisesRegex(
                    ValueError, "physical_margin.*exceeds training"
                ):
                    load_bundle(
                        path, self.geometry, replace(trained, physical_margin=above)
                    )

    def test_invalid_runtime_and_training_margins_are_rejected(self):
        for physical in (-0.001, np.nan, np.inf, True, "0.03"):
            with self.subTest(runtime=physical), self.assertRaises(ValueError):
                PandaRDFConfig(physical_margin=physical)
        for physical in (None, -0.001, np.nan, np.inf, True, "0.03"):
            with self.subTest(training=physical), TemporaryDirectory() as directory:
                bundle, trained = self.fixture.bundle()
                path = Path(directory) / "rdf.npz"
                save_bundle(bundle, path)
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key].copy() for key in archive.files}
                metadata = json.loads(str(arrays["metadata"]))
                if physical is None:
                    metadata["settings"].pop("physical_margin")
                else:
                    metadata["settings"]["physical_margin"] = physical
                # Deliberately bypass the writer's NaN/Inf guard to test the
                # runtime loader on an externally corrupted artifact.
                arrays["metadata"] = np.array(json.dumps(metadata))
                np.savez_compressed(path, **arrays)
                with self.assertRaisesRegex(
                    ValueError, "invalid training physical_margin"
                ):
                    load_bundle(
                        path, self.geometry, replace(trained, physical_margin=0.0)
                    )

    def test_reduced_margin_does_not_relax_other_settings(self):
        bundle, trained = self.fixture.bundle()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rdf.npz"
            save_bundle(bundle, path)
            for key, value in (
                ("error_margin", 0.005),
                ("error_margin", 0.020),
                ("transition_ratio", 0.7),
                ("query_density", 20),
                ("seed", 1),
                ("underestimation_limit", 0.020),
            ):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    runtime = replace(trained, physical_margin=0.010, **{key: value})
                    load_bundle(path, self.geometry, runtime)

    def test_radii_are_checked_against_training_not_runtime_margin(self):
        bundle, trained = self.fixture.bundle()
        runtime = replace(trained, physical_margin=0.010)
        # A field resized for the smaller runtime margin must still be rejected,
        # even when saved with fresh array hashes and passing validation flags.
        model = bundle.models["link0"]
        model.inner_radius = 0.2 + runtime.effective_margin
        model.rdf.radius = (1 + runtime.transition_ratio) * model.inner_radius
        with TemporaryDirectory() as directory:
            path = Path(directory) / "resized.npz"
            save_bundle(bundle, path)
            with self.assertRaisesRegex(ValueError, "link0: radii mismatch"):
                load_bundle(path, self.geometry, runtime)

    def test_relabeling_training_margin_or_waiving_validation_is_rejected(self):
        for change in (
            "physical_margin",
            "derivatives",
            "joint_validation",
            "incomplete",
        ):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                bundle, trained = self.fixture.bundle()
                if change == "physical_margin":
                    bundle.metadata["settings"]["physical_margin"] = 0.020
                elif change == "derivatives":
                    bundle.metadata["validation"]["link0"]["derivatives_passed"] = False
                elif change == "joint_validation":
                    bundle.metadata["joint_validation"]["passed"] = False
                else:
                    bundle.metadata["complete"] = False
                path = Path(directory) / "invalid.npz"
                save_bundle(bundle, path)
                with self.assertRaises(ValueError):
                    load_bundle(
                        path, self.geometry, replace(trained, physical_margin=0.010)
                    )

    def test_offline_validation_still_requires_original_training_settings(self):
        bundle, trained = self.fixture.bundle()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rdf.npz"
            save_bundle(bundle, path)
            for refresh in (False, True):
                with (
                    self.subTest(refresh=refresh),
                    self.assertRaisesRegex(ValueError, "candidate settings mismatch"),
                ):
                    read_candidate_for_validation(
                        path,
                        self.geometry,
                        replace(trained, physical_margin=0.020),
                        refresh_queries=refresh,
                    )

    def test_runtime_reduction_only_shifts_self_barriers_not_gradients_or_obstacles(
        self,
    ):
        bundle, trained = self.fixture.bundle()
        base = self.fixture.config
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rdf.npz"
            save_bundle(bundle, path)
            config = replace(
                base,
                safety=replace(
                    base.safety,
                    self_collision_model="rdf",
                    rdf=replace(trained, artifact=path),
                ),
            )
            _, original = build_panda_components(config)
            for physical in (0.0, 0.020):
                with self.subTest(physical=physical):
                    changed = replace(
                        config,
                        safety=replace(
                            config.safety,
                            rdf=replace(config.safety.rdf, physical_margin=physical),
                        ),
                    )
                    _, reduced = build_panda_components(changed)
                    for q in (config.initial_q, config.initial_q + 0.01):
                        before, after = original.evaluate(q), reduced.evaluate(q)
                        self.assertEqual(before.labels, after.labels)
                        indices = np.array(
                            [label.startswith("self_") for label in before.labels]
                        )
                        np.testing.assert_allclose(
                            after.values[indices],
                            before.values[indices] + trained.physical_margin - physical,
                            atol=1e-14,
                            rtol=0,
                        )
                        np.testing.assert_array_equal(
                            after.values[~indices], before.values[~indices]
                        )
                        np.testing.assert_array_equal(after.gradients, before.gradients)


if __name__ == "__main__":
    unittest.main()

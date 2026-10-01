"""Query densification preserves trained fields and never bypasses approval."""

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import test_panda_rdf_integration as integration
import trimesh

from safe_ik.application.augment_panda_rdf_queries import (
    append_surface_queries,
    augment,
    surface_residual,
    verify_preserved_arrays,
)
from safe_ik.application.panda_rdf_artifact import load_bundle, save_bundle

MODULE = "safe_ik.application.augment_panda_rdf_queries"


class SurfaceAppendTest(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
        self.original = trimesh.sample.sample_surface(self.mesh, 10, seed=3)[0]

    def test_preserves_prefix_is_deterministic_and_adds_distinct_surface_points(self):
        first = append_surface_queries(
            self.mesh, self.original, 40, np.random.default_rng(0)
        )
        second = append_surface_queries(
            self.mesh, self.original, 40, np.random.default_rng(0)
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:10], self.original)
        self.assertEqual(first.shape, (40, 3))
        self.assertEqual(len(np.unique(first, axis=0)), 40)
        self.assertLess(surface_residual(self.mesh, first), 1e-12)

    def test_equal_count_returns_independent_unchanged_array(self):
        result = append_surface_queries(
            self.mesh, self.original, 10, np.random.default_rng(0)
        )
        np.testing.assert_array_equal(result, self.original)
        self.assertFalse(np.shares_memory(result, self.original))

    def test_refuses_dropping_points_and_invalid_inputs(self):
        for count in (9, True, 10.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                append_surface_queries(
                    self.mesh, self.original, count, np.random.default_rng(0)
                )
        for points in (np.zeros((0, 3)), np.zeros((2, 2)), np.full((2, 3), np.nan)):
            with self.assertRaises(ValueError):
                append_surface_queries(self.mesh, points, 40, np.random.default_rng(0))

    def test_off_surface_is_not_accepted(self):
        with self.assertRaisesRegex(ValueError, "not on the visual surface"):
            surface_residual(self.mesh, np.array([[1.0, 1.0, 1.0]]))

    def test_tiny_triangle_interior_is_not_moved_to_an_edge(self):
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1e-5, 0, 0], [0, 1e-5, 0]],
            faces=[[0, 1, 2]],
            process=False,
        )
        points = np.array([[3e-6, 3e-6, 0]])
        self.assertLess(surface_residual(mesh, points), 1e-15)
        with self.assertRaisesRegex(ValueError, "not on the visual surface"):
            surface_residual(mesh, points + [0, 0, 1e-7])


class QueryArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        integration.PandaRDFIntegrationTest.setUpClass()
        cls.fixture = integration.PandaRDFIntegrationTest()
        cls.geometry = cls.fixture.geometry

    def prepare(self, directory):
        bundle, options = self.fixture.bundle()
        # Preserve multiple holdout identities and their full reports verbatim.
        requirements = {}
        for name, report in bundle.metadata["validation"].items():
            reports = {"regression": copy.deepcopy(report)}
            if name in ("link0", "link6", "hand"):
                reports["archived_purpose102"] = copy.deepcopy(report)
            report["validation_sets"] = reports
            requirements[name] = list(reports)
        bundle.metadata["validation_set_requirements"] = requirements
        source = Path(directory) / "source.npz"
        save_bundle(bundle, source)
        source.with_suffix(".report.json").write_text(json.dumps(bundle.metadata))
        config = replace(
            self.fixture.config,
            safety=replace(
                self.fixture.config.safety,
                rdf=replace(
                    options, artifact=source, query_density=20, physical_margin=0.010
                ),
            ),
        )
        return bundle, source, config

    def test_new_artifact_preserves_arrays_reports_radii_and_reduced_runtime_margin(
        self,
    ):
        with TemporaryDirectory() as d:
            original, source, config = self.prepare(d)
            model_before = source.read_bytes()
            report_before = source.with_suffix(".report.json").read_bytes()
            output = Path(d) / "dense.npz"
            # Synthetic coefficients are NOT a mesh approval. The real joint
            # checker is exercised on the actual generated artifact separately.
            with (
                patch(
                    MODULE + ".validate_joint_gradients", return_value={"passed": True}
                ) as check,
                patch(MODULE + ".surface_residual", return_value=0.0),
                patch(
                    "safe_ik.application.panda_rdf_training.train_model",
                    side_effect=AssertionError("must not train"),
                ),
            ):
                summary = augment(config, source, output)
            self.assertEqual(check.call_count, 1)
            self.assertEqual(check.call_args.args[2].top_k, config.safety.rdf.top_k)
            loaded = load_bundle(output, self.geometry, config.safety.rdf)
            self.assertEqual(source.read_bytes(), model_before)
            self.assertEqual(
                source.with_suffix(".report.json").read_bytes(), report_before
            )
            self.assertEqual(
                loaded.metadata["validation"], original.metadata["validation"]
            )
            self.assertEqual(
                loaded.metadata["validation_set_requirements"],
                original.metadata["validation_set_requirements"],
            )
            self.assertEqual(loaded.metadata["settings"]["physical_margin"], 0.03)
            self.assertEqual(loaded.metadata["settings"]["query_density"], 20)
            for name, model in loaded.models.items():
                self.assertEqual(
                    len(loaded.queries[name]),
                    int(np.ceil(20 * 4 * np.pi * model.inner_radius**2)),
                )
                np.testing.assert_array_equal(
                    loaded.queries[name][: len(original.queries[name])],
                    original.queries[name],
                )
                np.testing.assert_array_equal(
                    model.rdf.weights, original.models[name].rdf.weights
                )
                self.assertEqual(model.inner_radius, original.models[name].inner_radius)
            self.assertTrue(summary["loader_approved"])
            self.assertFalse(summary["training_performed"])
            self.assertFalse(summary["runtime_installed"])
            self.assertEqual(
                json.loads(output.with_suffix(".report.json").read_text()),
                loaded.metadata,
            )
            verify_preserved_arrays(source, output)
            with self.assertRaisesRegex(ValueError, "query_density"):
                load_bundle(
                    output, self.geometry, replace(config.safety.rdf, query_density=10)
                )

    def test_never_overwrites_any_destination(self):
        for suffix in (".npz", ".report.json", ".augmentation.json"):
            with self.subTest(suffix=suffix), TemporaryDirectory() as d:
                _, source, config = self.prepare(d)
                output = Path(d) / "dense.npz"
                protected = output.with_suffix(suffix)
                protected.write_bytes(b"keep me")
                with self.assertRaisesRegex(ValueError, "never overwrite"):
                    augment(config, source, output)
                self.assertEqual(protected.read_bytes(), b"keep me")
                with self.assertRaisesRegex(ValueError, "never overwrite"):
                    augment(config, source, source)

    def test_invalid_density_or_changed_non_density_settings_cannot_publish(self):
        with TemporaryDirectory() as d:
            _, source, config = self.prepare(d)
            output = Path(d) / "dense.npz"
            for density in (0, -1, 10, 5, np.nan, np.inf, True):
                with self.subTest(density=density), self.assertRaises(ValueError):
                    augment(config, source, output, density=density)
            for field, value in (
                ("seed", 1),
                ("error_margin", 0.02),
                ("physical_margin", 0.04),
            ):
                changed = replace(
                    config,
                    safety=replace(
                        config.safety, rdf=replace(config.safety.rdf, **{field: value})
                    ),
                )
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(ValueError, "settings mismatch"),
                ):
                    augment(changed, source, output)
            self.assertFalse(output.exists())

    def test_rejected_source_or_mismatched_report_cannot_publish(self):
        with TemporaryDirectory() as d:
            bundle, source, config = self.prepare(d)
            output = Path(d) / "dense.npz"
            source.with_suffix(".report.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "sidecar"):
                augment(config, source, output)
            bundle.metadata["complete"] = False
            save_bundle(bundle, source)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                augment(config, source, output)
            self.assertFalse(output.exists())

    def test_failed_joint_check_does_not_publish_or_change_source(self):
        with TemporaryDirectory() as d:
            _, source, config = self.prepare(d)
            before = source.read_bytes()
            output = Path(d) / "dense.npz"
            with (
                patch(
                    MODULE + ".validate_joint_gradients", return_value={"passed": False}
                ),
                patch(MODULE + ".surface_residual", return_value=0.0),
            ):
                with self.assertRaisesRegex(ValueError, "joint validation failed"):
                    augment(config, source, output)
            self.assertFalse(output.exists())
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

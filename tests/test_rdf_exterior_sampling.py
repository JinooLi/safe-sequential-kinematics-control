"""Exterior sampling does not change labels, reuse validation points, or grow batches."""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import trimesh

from safe_ik.adapters.rdf_checkpoint import file_sha256
from safe_ik.application.panda_rdf_training import TrainingOptions
from safe_ik.application.rdf_exterior_sampling import (
    ExteriorSamplingOptions,
    generate_exterior_pool,
    load_exterior_pool,
    mix_exterior_batches,
    sample_exterior_points,
)


class ExteriorSamplingTest(unittest.TestCase):
    def options(self, **overrides):
        return replace(
            ExteriorSamplingOptions(
                pool_size=120,
                near_batch=2,
                random_batch=1,
                exterior_batch=3,
                candidate_batch=40,
            ),
            **overrides,
        )

    def test_exterior_points_are_in_band_sphere_and_reproducible(self):
        mesh = trimesh.creation.box(extents=[0.10, 0.12, 0.08])
        mesh.apply_translation([0.2, -0.1, 0.3])
        options = self.options()
        p, d, report = sample_exterior_points(
            mesh, np.array([0.2, -0.1, 0.3]), 0.3, options
        )
        p2, d2, _ = sample_exterior_points(
            mesh, np.array([0.2, -0.1, 0.3]), 0.3, options
        )
        self.assertEqual(p.shape, (120, 3))
        self.assertTrue(np.all((d >= 0) & (d <= 0.06)))
        self.assertTrue(np.all(np.linalg.norm(p - [0.2, -0.1, 0.3], axis=1) <= 0.3))
        np.testing.assert_allclose(-trimesh.proximity.signed_distance(mesh, p), d)
        np.testing.assert_array_equal(p, p2)
        np.testing.assert_array_equal(d, d2)
        self.assertEqual(sum(report["distance_bin_counts"]), 120)
        json.dumps(report, allow_nan=False)

    def test_exterior_classification_uses_true_mesh_distance(self):
        options = self.options(pool_size=12, max_candidate_multiplier=2)
        mesh = trimesh.creation.box(extents=[0.1] * 3)
        with patch(
            "trimesh.proximity.signed_distance",
            side_effect=lambda mesh, p: np.full(len(p), 0.01),
        ):
            with self.assertRaisesRegex(ValueError, "acceptance"):
                sample_exterior_points(mesh, np.zeros(3), 0.3, options)

    def test_sampling_fails_for_invalid_mesh_or_sphere(self):
        mesh = trimesh.creation.box(extents=[0.1] * 3)
        with self.assertRaises(ValueError):
            sample_exterior_points(mesh, np.zeros(3), -0.3, self.options())
        mesh.update_faces(np.arange(len(mesh.faces) - 1))
        with self.assertRaisesRegex(ValueError, "closed"):
            sample_exterior_points(mesh, np.zeros(3), 0.3, self.options())

    def test_same_scan_label_method_and_normalization_not_exact_truth_targets(self):
        mesh = trimesh.creation.box(extents=[0.1] * 3)
        center = np.zeros(3)
        radius = 0.3
        cloud = SimpleNamespace(
            get_sdf=Mock(side_effect=lambda p, sample_count: np.full(len(p), 0.123))
        )
        getter = Mock(return_value=cloud)
        with patch.dict(
            sys.modules,
            {"mesh_to_sdf": SimpleNamespace(get_surface_point_cloud=getter)},
        ):
            p, labels, truth, report = generate_exterior_pool(
                mesh, center, radius, TrainingOptions(), self.options(), lambda _: None
            )
        np.testing.assert_allclose(labels, 0.123 * radius)
        self.assertFalse(np.allclose(labels, truth))
        self.assertFalse(report["reference_truth_used_as_training_labels"])
        np.testing.assert_allclose(
            getter.call_args.args[0].vertices, mesh.vertices / radius
        )
        self.assertEqual(getter.call_args.kwargs["scan_count"], 100)
        self.assertEqual(getter.call_args.kwargs["scan_resolution"], 400)
        self.assertEqual(cloud.get_sdf.call_args.kwargs["sample_count"], 100)
        np.testing.assert_allclose(cloud.get_sdf.call_args.args[0], p / radius)
        json.dumps(report, allow_nan=False)

    def test_nonfinite_scan_labels_fail(self):
        mesh = trimesh.creation.box(extents=[0.1] * 3)
        cloud = SimpleNamespace(get_sdf=lambda p, sample_count: np.full(len(p), np.nan))
        with patch.dict(
            sys.modules,
            {
                "mesh_to_sdf": SimpleNamespace(
                    get_surface_point_cloud=lambda *a, **k: cloud
                )
            },
        ):
            with self.assertRaisesRegex(ValueError, "scan labels"):
                generate_exterior_pool(
                    mesh,
                    np.zeros(3),
                    0.3,
                    TrainingOptions(),
                    self.options(),
                    lambda _: None,
                )

    def test_mixed_batches_keep_paired_subsets_and_total_size(self):
        near = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        random = np.array([[0, 1], [2, 3]])
        indices, report = mix_exterior_batches(near, random, self.options())
        np.testing.assert_array_equal(indices["near"], near[:, :2])
        np.testing.assert_array_equal(indices["random"], random[:, :1])
        self.assertEqual(indices["exterior"].shape, (2, 3))
        self.assertEqual(report["consumed_samples"], 12)
        self.assertFalse(report["full_baseline_batches_preserved"])
        again, again_report = mix_exterior_batches(near, random, self.options())
        np.testing.assert_array_equal(indices["exterior"], again["exterior"])
        self.assertEqual(report, again_report)
        for row in indices["exterior"]:
            self.assertEqual(len(set(row)), 3)
            self.assertTrue(np.all((row >= 0) & (row < 120)))
        with self.assertRaisesRegex(ValueError, "batch size"):
            mix_exterior_batches(near, random, self.options(exterior_batch=4))
        json.dumps(report, allow_nan=False)

    def test_options_reject_invalid_numbers(self):
        for change in (
            {"pool_size": 0},
            {"pool_size": 2},
            {"seed": -1},
            {"distance_max": np.nan},
            {"truth_batch": False},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.options(**change)

    def test_longer_mixed_plan_preserves_complete_prefix(self):
        near = np.tile(np.arange(4), (8, 1))
        random = np.tile(np.arange(2), (8, 1))
        short, _ = mix_exterior_batches(near[:4], random[:4], self.options())
        long, _ = mix_exterior_batches(near, random, self.options())
        for key in short:
            np.testing.assert_array_equal(short[key], long[key][:4])

    def test_saved_exterior_pool_reuse_and_mismatch_rejection(self):
        from dataclasses import asdict

        options = self.options(pool_size=3)
        training = TrainingOptions()
        manifest = {"link": "link4", "sha256": {"reference": "fixed"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = np.array([[0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0]])
            # Negative scan labels are intentionally retained.
            labels = np.array([-0.001, 0.02, 0.03])
            np.savez_compressed(
                root / "exterior_pool.npz",
                points=points,
                labels=labels,
                selection_truth=np.array([0.01, 0.02, 0.03]),
            )
            report = {
                "options": asdict(options),
                "source_reference_mesh_digest": "mesh",
                "reference_truth_used_as_training_labels": False,
                "scan_count": 100,
                "scan_resolution": 400,
                "normal_count": 100,
                "pool_sha256": file_sha256(root / "exterior_pool.npz"),
            }
            (root / "exterior_pool.report.json").write_text(json.dumps(report))
            (root / "summary.json").write_text(
                json.dumps({"inputs": manifest, "exterior_sampling": report})
            )

            def load(**changes):
                args = dict(
                    root=root,
                    manifest=manifest,
                    mesh_digest="mesh",
                    center=np.zeros(3),
                    radius=0.3,
                    training=training,
                    options=options,
                )
                return load_exterior_pool(**(args | changes))

            p, y, _, r = load()
            np.testing.assert_array_equal(p, points)
            np.testing.assert_array_equal(y, labels)
            self.assertEqual(r, report)
            for changes in (
                {"manifest": {}},
                {"mesh_digest": "other"},
                {"radius": 0.001},
                {"options": replace(options, seed=1)},
                {"training": replace(training, normal_count=10)},
            ):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    load(**changes)
            with (root / "exterior_pool.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load()


if __name__ == "__main__":
    unittest.main()

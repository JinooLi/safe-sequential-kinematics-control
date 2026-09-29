"""Real spawned workers must preserve point/label bytes, order and RNG state."""

import importlib.util
import multiprocessing
import os
import unittest
from dataclasses import asdict
from unittest.mock import Mock, patch

import numpy as np
import trimesh

from safe_ik.application.panda_rdf_training import TrainingOptions, training_pools
from safe_ik.application.rdf_data_workers import (
    data_execution,
    default_data_workers,
    mesh_distance_evaluator,
    scan_label_evaluator,
)
from safe_ik.application.rdf_exterior_sampling import (
    ExteriorSamplingOptions,
    generate_exterior_pool,
    sample_exterior_points,
)


def cloud_fixture():
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from mesh_to_sdf.surface_point_cloud import SurfacePointCloud

    # Deliberate duplicate points exercise KD-tree ties and normal majority votes.
    rng = np.random.default_rng(81)
    points = rng.normal(size=(256, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    points *= 0.25
    points = np.repeat(points, 2, axis=0)
    return SurfacePointCloud(None, points, normals=points / 0.25)


class DataWorkersTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "optional torch not installed"
    )
    def test_spawned_data_queries_after_parent_cuda_initialization(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float64)
        cloud = cloud_fixture()
        points = np.zeros((21, 3))
        with scan_label_evaluator(cloud, 2) as query:
            actual = np.concatenate(list(query.chunks(points, 10, 100)))
        np.testing.assert_array_equal(actual, cloud.get_sdf(points, sample_count=100))
        self.assertEqual(float(tensor @ tensor), 14.0)

    def test_default_respects_cpu_affinity_and_invalid_workers_fail(self):
        for cpus, expected in (({0}, 1), ({2, 3}, 2), (set(range(24)), 4)):
            with patch("os.sched_getaffinity", return_value=cpus, create=True):
                self.assertEqual(default_data_workers(), expected)
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                data_execution(workers)
        self.assertEqual(data_execution(4)["start_method"], "spawn")
        self.assertNotIn("data_workers", asdict(TrainingOptions()))
        self.assertNotIn("data_workers", asdict(ExteriorSamplingOptions()))

    def test_serial_mode_does_not_construct_executor(self):
        mesh = trimesh.creation.box()
        with patch(
            "safe_ik.application.rdf_data_workers.ProcessPoolExecutor",
            side_effect=AssertionError("serial must not spawn"),
        ):
            with mesh_distance_evaluator(mesh, 1) as query:
                result = list(query.chunks(np.array([[2.0, 0.0, 0.0]]), 1))
                self.assertEqual(result[0][0], 1.5)

    def test_spawned_mesh_queries_match_serial_and_order(self):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
        points = np.random.default_rng(7).uniform(-0.2, 0.2, (257, 3))
        before = {p.pid for p in multiprocessing.active_children()}
        with mesh_distance_evaluator(mesh, 2) as query:
            for size in (31, 64):
                actual = np.concatenate(list(query.chunks(points, size)))
                expected = np.concatenate(
                    [
                        -trimesh.proximity.signed_distance(mesh, points[i : i + size])
                        for i in range(0, len(points), size)
                    ]
                )
                np.testing.assert_array_equal(actual, expected)
        self.assertEqual(before, {p.pid for p in multiprocessing.active_children()})

    def test_exterior_sampling_preserves_rejections_points_truth_and_histogram(self):
        mesh = trimesh.creation.annulus(
            r_min=0.025, r_max=0.08, height=0.06, sections=16
        )
        options = ExteriorSamplingOptions(
            pool_size=513, exterior_batch=3, candidate_batch=128, truth_batch=32
        )
        single = sample_exterior_points(mesh, np.zeros(3), 0.10, options, link_index=2)
        multi = sample_exterior_points(
            mesh, np.zeros(3), 0.10, options, link_index=2, data_workers=2
        )
        np.testing.assert_array_equal(single[0], multi[0])
        np.testing.assert_array_equal(single[1], multi[1])
        self.assertEqual(single[2], multi[2])
        self.assertGreater(single[2]["rejected_outside_sphere"], 0)
        self.assertGreater(single[2]["rejected_outside_band"], 0)

    def test_normal_labels_keep_exact_kdtree_and_do_not_pickle_renderer(self):
        cloud = cloud_fixture()
        scans = [lambda: None]  # Unpicklable renderer stand-in; must stay in parent.
        cloud.scans = scans
        points = np.random.default_rng(71).uniform(-0.4, 0.4, (4211, 3))
        expected = np.concatenate(
            [
                cloud.get_sdf(points[i : i + 2048], sample_count=100)
                for i in range(0, len(points), 2048)
            ]
        )
        with scan_label_evaluator(cloud, 2) as query:
            actual = np.concatenate(list(query.chunks(points, 2048, 100)))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertIs(cloud.scans, scans)
        self.assertTrue(np.any(actual < 0))

    def test_near_random_pools_and_next_rng_draw_match(self):
        cloud = cloud_fixture()
        options = TrainingOptions(pool_size=4211)
        mesh = trimesh.creation.box(extents=[0.1] * 3)
        outputs = []
        with patch("mesh_to_sdf.get_surface_point_cloud", return_value=cloud):
            for workers in (1, 2):
                rng = np.random.default_rng(91)
                pools = training_pools(
                    mesh,
                    np.array([0.1, -0.2, 0.3]),
                    0.4,
                    rng,
                    options,
                    data_workers=workers,
                )
                outputs.append((pools, rng.integers(1000000, size=100)))
        for single, multi in zip(outputs[0][0], outputs[1][0]):
            np.testing.assert_array_equal(single, multi)
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])

    def test_exterior_generation_preserves_scan_labels_and_execution_provenance(self):
        cloud = cloud_fixture()
        mesh = trimesh.creation.box(extents=[0.1] * 3)
        options = ExteriorSamplingOptions(pool_size=2113, candidate_batch=256)
        values = []
        getter = Mock(return_value=cloud)
        with patch("mesh_to_sdf.get_surface_point_cloud", getter):
            for workers in (1, 2):
                values.append(
                    generate_exterior_pool(
                        mesh,
                        np.zeros(3),
                        0.3,
                        TrainingOptions(),
                        options,
                        progress=lambda _: None,
                        data_workers=workers,
                    )
                )
        for single, multi in zip(values[0][:3], values[1][:3]):
            np.testing.assert_array_equal(single, multi)
        for value, workers in zip(values, (1, 2)):
            self.assertEqual(
                value[3]["data_generation_execution"], data_execution(workers)
            )
            self.assertFalse(value[3]["reference_truth_used_as_training_labels"])
        self.assertEqual(getter.call_count, 2)  # Exactly one renderer per parent run.

    def test_worker_error_propagates_and_pool_is_closed(self):
        cloud = cloud_fixture()
        before = {p.pid for p in multiprocessing.active_children()}
        with self.assertRaises(ValueError):
            with scan_label_evaluator(cloud, 2) as query:
                list(query.chunks(np.zeros((100, 3)), 10, 100000))
        self.assertEqual(before, {p.pid for p in multiprocessing.active_children()})

    def test_cli_rejects_zero_workers_before_creating_outputs(self):
        from safe_ik.application.train_panda_rdf import main

        with patch("sys.argv", ["train", "--data-workers", "0"]):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

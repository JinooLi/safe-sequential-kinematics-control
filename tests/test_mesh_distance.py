"""The sign oracle is query-order independent and never consumes random state."""

import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from safe_ik.adapters.mesh_distance import (
    SIGN_ORACLE,
    AmbiguousMeshSignError,
    _fixed_ray_inside,
    mesh_distance,
    signed_mesh_distance,
    winding_number,
)
from safe_ik.application.panda_rdf_training import mesh_truth
from safe_ik.application.rdf_data_workers import mesh_distance_evaluator


class MeshDistanceTest(unittest.TestCase):
    def test_box_sign_and_surface_convention(self):
        mesh = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
        p = np.array(
            [[0.08, 0.01, 0.01], [0.04, 0.01, 0.01], [-0.09, 0.01, 0.01], [0.05, 0, 0]]
        )
        q = mesh_distance(mesh, p)
        np.testing.assert_allclose(q.signed, [0.03, -0.01, 0.04, 0], atol=1e-12)
        np.testing.assert_array_equal(np.abs(q.signed), q.unsigned)
        np.testing.assert_array_equal(mesh_truth(mesh, p)[0], q.signed)
        self.assertFalse(q.winding_fallback.any())

    def test_winding_known_closed_shapes_and_reversed_orientation(self):
        mesh = trimesh.creation.box()
        p = np.array([[0, 0, 0], [2, 0, 0], [0.49, 0.02, 0.01]])
        np.testing.assert_allclose(
            np.abs(winding_number(mesh, p)), [1, 0, 1], atol=1e-12
        )
        mesh.invert()
        np.testing.assert_allclose(
            np.abs(winding_number(mesh, p)), [1, 0, 1], atol=1e-12
        )
        mesh.apply_scale(0.001)
        mesh.apply_translation([0.2, -0.3, 0.5])
        np.testing.assert_allclose(
            np.abs(winding_number(mesh, p * 0.001 + [0.2, -0.3, 0.5])),
            [1, 0, 1],
            atol=1e-12,
        )

    def test_winding_hollow_mesh_and_open_mesh_failure(self):
        mesh = trimesh.creation.annulus(r_min=0.2, r_max=0.5, height=0.5, sections=24)
        p = np.array([[0, 0, 0], [0.35, 0, 0], [1, 0, 0]])
        np.testing.assert_allclose(
            np.abs(winding_number(mesh, p)), [0, 1, 0], atol=1e-12
        )
        mesh.update_faces(np.arange(len(mesh.faces) - 1))
        with self.assertRaisesRegex(AmbiguousMeshSignError, "closed oriented"):
            winding_number(mesh, p)

    def test_conflicting_rays_use_winding_without_rng(self):
        mesh = trimesh.creation.box()
        # Forward parity odd / backward parity even, neither direction empty.
        intersections = (np.empty((3, 3)), np.array([0, 1, 1]), np.zeros(3, dtype=int))
        with (
            patch.object(mesh.ray, "intersects_location", return_value=intersections),
            patch(
                "trimesh.util.random_generator",
                side_effect=AssertionError("RNG forbidden"),
            ),
        ):
            for _ in range(3):
                inside, fallback = _fixed_ray_inside(mesh, np.array([[0.0, 0.0, 0.0]]))
                self.assertTrue(inside[0])
                self.assertTrue(fallback[0])

    def test_noninteger_winding_fails_instead_of_guessing(self):
        mesh = trimesh.creation.box()
        with patch(
            "safe_ik.adapters.mesh_distance.np.arctan2", return_value=np.array([np.pi])
        ):
            with self.assertRaisesRegex(AmbiguousMeshSignError, "noninteger"):
                winding_number(mesh, np.array([[0.0, 0.0, 0.0]]))

    def test_repeat_order_chunk_and_worker_invariance(self):
        mesh = trimesh.creation.annulus(r_min=0.2, r_max=0.5, height=0.5, sections=24)
        p = np.random.default_rng(91).uniform(-0.6, 0.6, (201, 3))
        with patch(
            "trimesh.util.random_generator", side_effect=AssertionError("RNG forbidden")
        ):
            expected = signed_mesh_distance(mesh, p)
            np.testing.assert_array_equal(
                expected, signed_mesh_distance(mesh, p[::-1])[::-1]
            )
            np.testing.assert_array_equal(
                expected,
                np.concatenate(
                    [signed_mesh_distance(mesh, b) for b in np.array_split(p, 7)]
                ),
            )
        with mesh_distance_evaluator(mesh, 2) as query:
            for size in [17, 64]:
                np.testing.assert_array_equal(
                    expected, np.concatenate(list(query.chunks(p, size)))
                )

    def test_empty_and_invalid_queries(self):
        mesh = trimesh.creation.box()
        self.assertEqual(mesh_distance(mesh, np.empty((0, 3))).signed.shape, (0,))
        for p in ([[float("nan"), 0, 0]], [1, 2, 3], [[1, 2]]):
            with self.assertRaises(ValueError):
                mesh_distance(mesh, p)
        self.assertEqual(SIGN_ORACLE, "triangle_normal_fixed_ray_integer_winding_v1")


if __name__ == "__main__":
    unittest.main()

"""Numerical checks independent of learned meshes/artifact acceptance."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from safe_ik.domain.panda_rdf import (
    BernsteinRDF,
    ExtendedRDF,
    recursive_ridge,
    relative_point_jacobian,
    soft_min,
    top_indices,
)


class PandaRDFPrimitivesTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.model = BernsteinRDF(
            np.array([0.1, -0.2, 0.3]), 1.0, self.rng.normal(size=(4, 4, 4))
        )

    def finite_difference(self, function, p):
        return np.column_stack(
            [
                (
                    function(p + np.eye(3)[i] * 1e-6)[0]
                    - function(p - np.eye(3)[i] * 1e-6)[0]
                )
                / 2e-6
                for i in range(3)
            ]
        )

    def test_analytic_gradient_and_normalization(self):
        p = self.rng.uniform(-0.5, 0.5, (25, 3)) + self.model.center
        d, g = self.model.query(p)
        np.testing.assert_allclose(
            d, self.model.features(p) @ self.model.weights.ravel()
        )
        np.testing.assert_allclose(
            g, self.finite_difference(self.model.query, p), atol=2e-8
        )

    def test_basis_endpoints_empty_and_values_only(self):
        p = self.model.center + np.array([[-1, -1, -1], [1, 1, 1]])
        d, g = self.model.query(p)
        np.testing.assert_allclose(
            d, [self.model.weights[0, 0, 0], self.model.weights[-1, -1, -1]]
        )
        self.assertTrue(np.isfinite(g).all())
        self.assertEqual(self.model.query(np.empty((0, 3)))[1].shape, (0, 3))
        np.testing.assert_allclose(self.model.query(p, False)[0], d)

    def test_recursive_matches_batch_ridge(self):
        model = BernsteinRDF(np.zeros(3), 2.0, np.zeros((3, 3, 3)))
        p = self.rng.uniform(-0.8, 0.8, (75, 3))
        targets = np.linalg.norm(p, axis=1) - 0.3
        phi = model.features(p)
        lam = 1e-4
        expected = np.linalg.solve(phi.T @ phi + lam * np.eye(27), phi.T @ targets)
        batches = [(p[i : i + 15], targets[i : i + 15]) for i in range(0, len(p), 15)]
        recursive_ridge(model, batches, lam)
        np.testing.assert_allclose(model.weights.ravel(), expected, atol=1e-8)

    def test_extension_chain_rule_and_boundaries(self):
        model = ExtendedRDF(self.model, 0.6, 1.0)
        dirs = self.rng.normal(size=(20, 3))
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        for radius in [0.2, 0.8, 1.3]:
            p = self.model.center + radius * dirs
            np.testing.assert_allclose(
                model.query(p)[1], self.finite_difference(model.query, p), atol=2e-7
            )
        for radius in [0.6, 1.0]:
            before = model.query(self.model.center + (radius - 1e-8) * dirs)
            after = model.query(self.model.center + (radius + 1e-8) * dirs)
            np.testing.assert_allclose(before[0], after[0], atol=2e-7)
            np.testing.assert_allclose(before[1], after[1], atol=2e-6)
        np.testing.assert_allclose(model.query(self.model.center + 2 * dirs)[1], dirs)
        self.assertTrue(np.isfinite(model.query(self.model.center[None])[1]).all())

    def test_extension_never_queries_exterior_polynomial(self):
        original = self.model.query

        def guarded(p, gradient=True):
            self.assertTrue(
                np.all(np.linalg.norm(p - self.model.center, axis=1) <= 1.0)
            )
            return original(p, gradient)

        self.model.query = guarded
        ExtendedRDF(self.model, 0.6, 1.0).query(
            np.array([[100.0, 0, 0], [0.1, -0.2, 0.3]])
        )

    def test_softmin_bound_weights_and_top_k(self):
        for n in [1, 3, 20, 100]:
            values = self.rng.normal(size=n)
            d, w = soft_min(values)
            self.assertGreaterEqual(d, values.min() - 0.005 - 1e-12)
            self.assertLessEqual(d, values.min() + 1e-12)
            self.assertAlmostEqual(w.sum(), 1.0)
        self.assertAlmostEqual(soft_min(np.ones(20))[0], 0.995)
        np.testing.assert_array_equal(top_indices([2.0, 1.0, 1.0], 2), [1, 2])
        np.testing.assert_array_equal(top_indices([2.0, 1.0], None), [0, 1])
        with self.assertRaises(ValueError):
            soft_min([])

    def test_relative_jacobian_both_frames_move(self):
        local = np.array([0.2, 0.3, -0.1])

        def transforms(q):
            ri = Rotation.from_rotvec([0, 0, q[0]]).as_matrix()
            rj = Rotation.from_rotvec([0, q[1], 0]).as_matrix()
            ti = np.array([q[1], 0, 0])
            x = rj @ local + np.array([0, q[0], 0])
            return ri, ti, x

        q = np.array([0.4, -0.2])
        ri, ti, x = transforms(q)
        ji = np.array([[0.0, 1], [0, 0], [0, 0]])
        wi = np.array([[0.0, 0], [0, 0], [1, 0]])
        rotated = Rotation.from_rotvec([0, q[1], 0]).apply(local)
        jx = np.column_stack(([0.0, 1, 0], np.cross([0, 1, 0], rotated)))
        analytic = relative_point_jacobian(ri, ti, ji, wi, x[None], jx[None])[0]

        def p(q):
            r, t, point = transforms(q)
            return r.T @ (point - t)

        numeric = np.column_stack(
            [
                (p(q + np.eye(2)[i] * 1e-6) - p(q - np.eye(2)[i] * 1e-6)) / 2e-6
                for i in range(2)
            ]
        )
        np.testing.assert_allclose(analytic, numeric, atol=1e-9)


if __name__ == "__main__":
    unittest.main()

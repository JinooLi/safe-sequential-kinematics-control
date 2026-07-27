import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.domain.rdf import PlanarBernsteinRobot, capsule_sdf_local


class BernsteinRDFTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = PlanarBernsteinRobot.train_from_capsules(
            [1.0, 1.0, 1.0], 0.05, degree=10, sample_count=5000,
            rng=np.random.default_rng(7),
        )

    def test_approximates_reference_sdf(self):
        model = self.robot.models[0]
        rng = np.random.default_rng(3)
        points = rng.uniform(model.bounds[:, 0], model.bounds[:, 1], size=(1000, 2))
        prediction, _ = model.query(points)
        truth = capsule_sdf_local(points, 1.0, 0.05)
        self.assertLess(np.mean(np.abs(prediction - truth)), 0.03)

    def test_spatial_gradient_matches_finite_difference(self):
        model = self.robot.models[0]
        point = np.array([0.55, 0.18])
        _, analytic = model.query(point)
        epsilon = 1e-6
        numerical = []
        for axis in range(2):
            delta = np.zeros(2)
            delta[axis] = epsilon
            plus = model.query(point + delta)[0]
            minus = model.query(point - delta)[0]
            numerical.append((plus - minus) / (2.0 * epsilon))
        np.testing.assert_allclose(analytic, numerical, atol=1e-5)

    def test_joint_gradient_and_model_round_trip(self):
        q = np.array([0.2, -0.35, 0.1])
        point = np.array([1.25, 0.65])
        _, analytic = self.robot.link_sdf_and_gradient(point, q, 1)
        epsilon = 1e-6
        numerical = []
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = epsilon
            plus = self.robot.link_sdf_and_gradient(point, q + delta, 1)[0]
            minus = self.robot.link_sdf_and_gradient(point, q - delta, 1)[0]
            numerical.append((plus - minus) / (2.0 * epsilon))
        np.testing.assert_allclose(analytic, numerical, atol=1e-5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rdf.npz"
            self.robot.save(path)
            loaded = PlanarBernsteinRobot.load(path)
            self.assertAlmostEqual(
                loaded.link_sdf_and_gradient(point, q, 1)[0],
                self.robot.link_sdf_and_gradient(point, q, 1)[0],
            )

    def test_level_set_samples_satisfy_requested_value(self):
        model = self.robot.models[0]
        samples = model.sample_level_set(20, level=0.05, rng=np.random.default_rng(2))
        values, _ = model.query(samples)
        np.testing.assert_allclose(values, 0.05, atol=1e-5)

    def test_query_outside_training_box_uses_bounded_extension(self):
        model = self.robot.models[0]
        point = np.array([-0.5, 0.15])
        value, gradient = model.query(point)
        self.assertTrue(np.isfinite(value))
        self.assertTrue(np.all(np.isfinite(gradient)))
        # A point farther from the fitted box cannot appear closer to the link.
        boundary_value, _ = model.query(np.array([-0.3, 0.15]))
        self.assertGreater(value, boundary_value)


if __name__ == "__main__":
    unittest.main()

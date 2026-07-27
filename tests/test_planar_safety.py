import unittest

import numpy as np

from src.domain.safety import CircleObstacle, PlanarCapsuleRobot


class PlanarSafetyTest(unittest.TestCase):
    def setUp(self):
        self.robot = PlanarCapsuleRobot([1.0, 1.0, 1.0], link_radius=0.05)

    def test_capsule_distance_for_extended_link(self):
        sdf, _ = self.robot.link_sdf_and_gradient(
            np.array([0.5, 0.2]), np.zeros(3), link_index=0
        )
        self.assertAlmostEqual(sdf, 0.15, places=12)

    def test_analytic_joint_gradient_matches_finite_difference(self):
        q = np.array([0.35, -0.4, 0.2])
        point = np.array([1.2, 0.75])
        _, analytic = self.robot.link_sdf_and_gradient(point, q, link_index=1)
        epsilon = 1e-7
        numerical = np.zeros(3)
        for index in range(3):
            dq = np.zeros(3)
            dq[index] = epsilon
            plus = self.robot.link_sdf_and_gradient(point, q + dq, 1)[0]
            minus = self.robot.link_sdf_and_gradient(point, q - dq, 1)[0]
            numerical[index] = (plus - minus) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, numerical, atol=1e-6)

    def test_cbf_lower_bound_increases_when_collision_is_near(self):
        obstacle = CircleObstacle(center=np.array([0.5, 0.1]), radius=0.1)
        _, lower, h = self.robot.cbf_constraints(
            np.zeros(3), [obstacle], safe_distance=0.05, alpha=8.0
        )
        self.assertLess(h[0], 0.0)
        self.assertGreater(lower[0], 0.0)


if __name__ == "__main__":
    unittest.main()

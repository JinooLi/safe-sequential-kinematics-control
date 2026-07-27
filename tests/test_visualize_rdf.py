import unittest

import numpy as np

from src.application.visualize_rdf import evaluate_robot_distance_field
from src.domain.safety import PlanarCapsuleRobot


class VisualizeRDFTest(unittest.TestCase):
    def test_distance_field_matches_capsule_query(self):
        robot = PlanarCapsuleRobot([1.0, 1.0, 1.0], link_radius=0.05)
        q = np.zeros(3)
        points = np.array([[0.5, 0.2], [0.5, 0.0], [3.0, 0.4]])
        values = evaluate_robot_distance_field(robot, q, points)
        np.testing.assert_allclose(values, [0.15, -0.05, 0.35], atol=1e-12)


if __name__ == "__main__":
    unittest.main()

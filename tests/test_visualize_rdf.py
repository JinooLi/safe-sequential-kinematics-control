import unittest

import numpy as np

from src.application.visualize_rdf import evaluate_robot_distance_field
from src.application.visualize_clf import evaluate_clf_in_q_space, marching_tetrahedra
from src.domain.safety import PlanarCapsuleRobot


class VisualizeRDFTest(unittest.TestCase):
    def test_distance_field_matches_capsule_query(self):
        robot = PlanarCapsuleRobot([1.0, 1.0, 1.0], link_radius=0.05)
        q = np.zeros(3)
        points = np.array([[0.5, 0.2], [0.5, 0.0], [3.0, 0.4]])
        values = evaluate_robot_distance_field(robot, q, points)
        np.testing.assert_allclose(values, [0.15, -0.05, 0.35], atol=1e-12)

    def test_clf_q_space_value_matches_controller_definition(self):
        value = evaluate_clf_in_q_space(
            np.zeros(3), [1.0, 1.0, 1.0], ee_target=[3.0, 0.0], ee_orientation=0.0
        )
        self.assertEqual(value, 0.0)

    def test_marching_tetrahedra_extracts_requested_level(self):
        axis = np.array([0.0, 1.0])
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        faces = marching_tetrahedra((axis, axis, axis), grid[..., 0], level=0.5)
        self.assertGreater(len(faces), 0)
        np.testing.assert_allclose(faces[..., 0], 0.5, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

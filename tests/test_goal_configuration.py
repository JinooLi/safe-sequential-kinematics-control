import unittest

import numpy as np

from src.domain.goal_configuration import link1_link3_crosses, solve_goal_configurations


class GoalConfigurationTest(unittest.TestCase):
    def test_link1_link3_cross_detects_only_interior_intersection(self):
        crossing = np.array([[1.0, 0.0], [0.5, 1.0], [0.5, -1.0]])
        non_crossing = np.array([[1.0, 0.0], [0.5, 1.0], [1.5, 1.0]])

        self.assertTrue(link1_link3_crosses(crossing))
        self.assertFalse(link1_link3_crosses(non_crossing))

    def test_goal_ik_marks_link1_link3_crossing_branch_as_colliding(self):
        candidates = solve_goal_configurations(
            target_position=[0.5, 1.0 - np.sqrt(3.0)],
            target_yaw=-np.pi / 2.0,
            link_lengths=[1.0, 1.0, 1.0],
            joint_limits=(None, None, None),
        )

        elbow_down = next(candidate for candidate in candidates if candidate.name == "elbow_down")
        self.assertTrue(link1_link3_crosses(elbow_down.joint_positions))
        self.assertTrue(elbow_down.colliding)


if __name__ == "__main__":
    unittest.main()

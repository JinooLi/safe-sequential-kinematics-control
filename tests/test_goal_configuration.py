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

    def test_goal_ik_rejects_base_to_link3_self_collision(self):
        candidates = solve_goal_configurations(
            target_position=[0.0, 0.0],
            target_yaw=0.0,
            link_lengths=[1.0, 1.0, 1.0],
            link_radius=0.05,
            self_collision_distance=0.13,
            joint_limits=(None, None, None),
        )

        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(candidate.colliding for candidate in candidates))


if __name__ == "__main__":
    unittest.main()

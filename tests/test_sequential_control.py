from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from src.application.configuration import load_controller_config
from src.control.qp_controller import QPController


ROOT = Path(__file__).resolve().parents[1]


class CaptureSolver:
    def solve(self, problem):
        self.problem = problem
        return np.zeros(len(problem.q))


class SequentialControlTest(unittest.TestCase):
    def setUp(self):
        self.config = load_controller_config(ROOT / "settings/controller.yaml")
        self.q = np.zeros(self.config.dof)

    def test_targets_are_link_endpoints_of_selected_goal_configuration(self):
        controller = QPController(self.config, CaptureSolver(), verbose=False)
        controller._select_goal_configuration(self.q)
        selected = controller.selected_goal_configuration
        self.assertIsNotNone(selected)
        for link_index, expected in enumerate(selected.joint_positions, start=1):
            controller.end_effector_index = link_index
            np.testing.assert_allclose(controller.active_target_position(), expected)

    def test_all_clf_terms_use_configured_weight(self):
        controller = QPController(self.config, CaptureSolver(), verbose=False)
        controller.end_effector_index = self.config.dof
        controller._select_goal_configuration(self.q)
        target = controller.active_target_position()
        task_state = np.array([target[0] + 0.2, target[1], self.config.ee_orientation + 1.0])
        task_jacobian = np.zeros((3, self.config.dof))
        task_jacobian[2] = 1.0

        value, gradient = controller._task_clf_terms(task_state, task_jacobian)

        weight = self.config.safety.clf_weight
        yaw_weight = self.config.sequential_end_effector.final_yaw_weight
        self.assertAlmostEqual(value, 2.0 * (0.2**2 + yaw_weight) * weight)
        np.testing.assert_allclose(gradient, 4.0 * yaw_weight * weight)

        controller.end_effector_index = 1
        target = controller.active_target_position()
        position_error = 0.2
        task_state = np.array([target[0] + position_error, target[1], 0.0])
        value, gradient = controller._task_clf_terms(task_state, task_jacobian)
        self.assertAlmostEqual(value, 2.0 * position_error**2 * weight)
        np.testing.assert_allclose(gradient, 0.0)

    def test_link1_reverse_uses_the_opposite_winding_and_a_joint_clf(self):
        controller = QPController(self.config, CaptureSolver(), verbose=False)
        controller._select_goal_configuration(self.q)
        controller._start_link1_reverse(self.q)

        goal_q1 = controller.selected_goal_configuration.q[0]
        normal_target = goal_q1 + 2.0 * np.pi * round((self.q[0] - goal_q1) / (2.0 * np.pi))
        normal_direction = np.sign(normal_target - self.q[0]) or 1.0
        self.assertTrue(controller.link1_reverse_active)
        self.assertLess((controller.link1_reverse_target - self.q[0]) * normal_direction, 0.0)

        value, gradient = controller._link1_reverse_clf_terms(self.q)
        expected_error = self.q[0] - controller.link1_reverse_target
        self.assertAlmostEqual(
            value,
            self.config.safety.clf_weight
            * self.config.sequential_end_effector.link1_reverse_weight
            * expected_error**2,
        )
        np.testing.assert_allclose(
            gradient,
            [
                2.0
                * self.config.safety.clf_weight
                * self.config.sequential_end_effector.link1_reverse_weight
                * expected_error,
                0.0,
                0.0,
            ],
        )

    def test_link1_reverse_blocked_by_cbf_advances_to_stage2(self):
        sequence = replace(self.config.sequential_end_effector, link1_stuck_cycles=1)
        config = replace(self.config, sequential_end_effector=sequence)
        controller = QPController(config, CaptureSolver(), verbose=False)
        controller._select_goal_configuration(self.q)
        controller._link1_joint_stopped_by_barrier = lambda q, input: True

        controller._handle_barrier_stop(self.q, np.zeros(config.dof))
        self.assertTrue(controller.link1_reverse_active)
        self.assertEqual(controller.end_effector_index, 1)

        controller._handle_barrier_stop(self.q, np.zeros(config.dof))
        self.assertFalse(controller.link1_reverse_active)
        self.assertEqual(controller.end_effector_index, 2)
        self.assertEqual(controller.last_end_effector_switch_reason, "link1 reverse blocked by CBF")

if __name__ == "__main__":
    unittest.main()

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

if __name__ == "__main__":
    unittest.main()

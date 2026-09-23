import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from safe_ik.application.planar_configuration import load_controller_config
from safe_ik.control.planar_controller import QPController
from safe_ik.domain.types import QPSolution, QPStatus

ROOT = Path(__file__).resolve().parents[1]


class CaptureSolver:
    def solve(self, problem):
        self.problem = problem
        return QPSolution(QPStatus.SOLVED, np.zeros(len(problem.q)), 0.0)


class SequentialControlTest(unittest.TestCase):
    def setUp(self):
        self.config = load_controller_config(ROOT / "tests/fixtures/planar.yaml")
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
        task_state = np.array(
            [target[0] + 0.2, target[1], self.config.ee_orientation + 1.0]
        )
        task_jacobian = np.zeros((3, self.config.dof))
        task_jacobian[2] = 1.0

        value, gradient = controller._task_clf_terms(task_state, task_jacobian)

        weight = self.config.safety.clf_weight
        yaw_weight = self.config.sequential_end_effector.final_yaw_weight
        self.assertAlmostEqual(
            value, 2.0 * (0.2**2 + yaw_weight * (1.0 - np.cos(1.0))) * weight
        )
        np.testing.assert_allclose(gradient, 2.0 * yaw_weight * np.sin(1.0) * weight)

        task_state = np.array(
            [target[0], target[1], self.config.ee_orientation + np.pi]
        )
        value, gradient = controller._task_clf_terms(task_state, task_jacobian)
        self.assertAlmostEqual(value, 4.0 * yaw_weight * weight)
        np.testing.assert_allclose(gradient, 0.0, atol=1e-12)

        controller.end_effector_index = 1
        target = controller.active_target_position()
        position_error = 0.2
        task_state = np.array([target[0] + position_error, target[1], 0.0])
        value, gradient = controller._task_clf_terms(task_state, task_jacobian)
        self.assertAlmostEqual(value, 2.0 * position_error**2 * weight)
        np.testing.assert_allclose(gradient, 0.0)

    def test_clf_slack_has_no_upper_bound(self):
        solver = CaptureSolver()
        controller = QPController(self.config, solver, verbose=False)

        controller.get_velocity(
            self.q,
            np.zeros((6, self.config.dof)),
            np.zeros(3),
            0.0,
        )

        self.assertEqual(solver.problem.lower[-1], 0.0)
        self.assertTrue(np.isposinf(solver.problem.upper[-1]))


if __name__ == "__main__":
    unittest.main()

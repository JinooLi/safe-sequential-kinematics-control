from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from src.application.configuration import load_controller_config
from src.control.qp_controller import QPController
from src.domain.clbf import CLBFModule, build_cumulative_clbfs, end_effector_barrier
from src.domain.safety import CircleObstacle


ROOT = Path(__file__).resolve().parents[1]


class CaptureSolver:
    def solve(self, problem):
        self.problem = problem
        return np.zeros(len(problem.q))


def planar_end_effector(q):
    """Return planar [x, y, yaw] and its task Jacobian for unit links."""
    q = np.asarray(q, dtype=float)
    angles = np.cumsum(q)
    p = np.array([np.cos(angles).sum(), np.sin(angles).sum(), angles[-1]])
    jacobian = np.zeros((3, len(q)))
    for joint in range(len(q)):
        jacobian[0, joint] = -np.sin(angles[joint:]).sum()
        jacobian[1, joint] = np.cos(angles[joint:]).sum()
        jacobian[2, joint] = 1.0
    return p, jacobian


class CLBFTest(unittest.TestCase):
    def setUp(self):
        self.obstacles = (
            CircleObstacle(np.array([2.85, 0.35]), 0.2),
            CircleObstacle(np.array([2.70, 0.50]), 0.15),
        )
        self.modules = (
            CLBFModule(1, epsilon=0.01, lambda_=1.2, kappa=0.05),
            CLBFModule(2, epsilon=0.02, lambda_=0.8, kappa=0.1),
        )

    def evaluate_clbf(self, q):
        p, task_jacobian = planar_end_effector(q)
        barriers = tuple(
            end_effector_barrier(
                p,
                task_jacobian,
                obstacle,
                end_effector_radius=0.06,
                safe_distance=0.08,
                epsilon=module.epsilon,
            )
            for obstacle, module in zip(self.obstacles, self.modules)
        )
        return build_cumulative_clbfs(
            clf_value=0.0,
            clf_gradient_q=np.zeros(3),
            barriers=barriers,
            gamma=1.0,
            modules=self.modules,
        )[-1]

    def test_final_end_effector_clbf_gradient_matches_finite_difference(self):
        q = np.array([0.2, -0.3, 0.4])
        epsilon = 1e-7
        result = self.evaluate_clbf(q)
        numerical = np.zeros(3)
        for index in range(3):
            delta = np.zeros(3)
            delta[index] = epsilon
            numerical[index] = (
                self.evaluate_clbf(q + delta).value - self.evaluate_clbf(q - delta).value
            ) / (2.0 * epsilon)
        self.assertTrue(np.isfinite(result.value))
        np.testing.assert_allclose(result.gradient_q, numerical, atol=1e-6)

    def test_qp_uses_one_final_clbf_row_or_legacy_clf_row_by_mode(self):
        config = load_controller_config(ROOT / "settings/controller.yaml")
        q = np.zeros(config.dof)
        J = np.zeros((6, config.dof))
        p_ee = np.array([*config.ee_target, 0.0])

        clbf_solver = CaptureSolver()
        QPController(config, clbf_solver).get_velocity(q, J, p_ee, config.ee_orientation)
        clbf_problem = clbf_solver.problem
        active_obstacles = sum(obstacle.enabled for obstacle in config.obstacles)
        self.assertEqual(
            clbf_problem.A.shape[0], 1 + active_obstacles * config.dof + config.dof + 1
        )
        self.assertEqual(clbf_problem.A[0, -1], 0.0)
        self.assertEqual(clbf_problem.upper[-1], 0.0)

        clf_config = replace(config, control_mode="clf")
        clf_solver = CaptureSolver()
        QPController(clf_config, clf_solver).get_velocity(q, J, p_ee, config.ee_orientation)
        clf_problem = clf_solver.problem
        self.assertEqual(clf_problem.A[0, -1], -1.0)
        self.assertGreaterEqual(clf_problem.upper[-1], 0.0)


if __name__ == "__main__":
    unittest.main()

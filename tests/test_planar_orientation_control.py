from pathlib import Path
import unittest

import numpy as np

from src.adapters.pinocchio_kinematics import RobotInterface
from src.control.qp_controller import planar_task_jacobian, wrap_to_pi


ROOT = Path(__file__).resolve().parents[1]


class PlanarOrientationControlTest(unittest.TestCase):
    def test_task_jacobian_uses_angular_z_not_linear_z(self):
        spatial_jacobian = np.arange(18, dtype=float).reshape(6, 3)

        task_jacobian = planar_task_jacobian(spatial_jacobian)

        np.testing.assert_array_equal(task_jacobian, spatial_jacobian[[0, 1, 5]])
        np.testing.assert_array_equal(task_jacobian[2], spatial_jacobian[5])
        self.assertFalse(np.array_equal(task_jacobian[2], spatial_jacobian[2]))

    def test_angular_z_jacobian_matches_yaw_finite_difference(self):
        robot = RobotInterface(ROOT / "settings/urdf.xml", "ee_joint")
        q = np.array([0.2, -0.4, 0.3])
        direction = np.array([0.6, -0.2, 0.5])
        epsilon = 1e-7

        _, spatial_jacobian = robot.update_kinematics(q)
        yaw = robot.end_effector_yaw()
        robot.update_kinematics(q + epsilon * direction)
        yaw_plus = robot.end_effector_yaw()

        numerical_yaw_rate = wrap_to_pi(yaw_plus - yaw) / epsilon
        analytic_yaw_rate = planar_task_jacobian(spatial_jacobian)[2] @ direction
        self.assertAlmostEqual(analytic_yaw_rate, numerical_yaw_rate, places=6)

    def test_yaw_error_wraps_across_pi_boundary(self):
        self.assertAlmostEqual(wrap_to_pi(-np.pi + 0.01 - (np.pi - 0.01)), 0.02)

    def test_can_select_successive_link_tip_frames(self):
        robot = RobotInterface(ROOT / "settings/urdf.xml", ("joint2", "joint3", "ee_joint"))
        q = np.zeros(3)

        robot.set_end_effector(1)
        first_tip, _ = robot.update_kinematics(q)
        robot.set_end_effector(2)
        second_tip, _ = robot.update_kinematics(q)
        robot.set_end_effector(3)
        final_tip, _ = robot.update_kinematics(q)

        np.testing.assert_allclose(first_tip, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(second_tip, [2.0, 0.0, 0.0])
        np.testing.assert_allclose(final_tip, [3.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

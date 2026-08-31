from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from src.application.configuration import (
    load_controller_config,
    validate_geometry_consistency,
)
from src.control.qp_controller import QPController


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.config = load_controller_config(ROOT / "settings/controller.yaml")
        self.urdf = ROOT / "settings/urdf.xml"
        self.mujoco = ROOT / "settings/mujoco.xml"

    def test_current_robot_descriptions_are_consistent(self):
        validate_geometry_consistency(self.config, self.urdf, self.mujoco)

    def test_first_joint_has_no_position_limits_in_urdf_or_mujoco(self):
        urdf_joint = ET.parse(self.urdf).find("./joint[@name='joint1']")
        mujoco_joint = ET.parse(self.mujoco).find(".//joint[@name='joint1']")

        self.assertEqual(urdf_joint.get("type"), "revolute")
        self.assertIsNone(urdf_joint.find("limit").get("lower"))
        self.assertIsNone(urdf_joint.find("limit").get("upper"))
        self.assertEqual(mujoco_joint.get("limited"), "false")
        self.assertIsNone(mujoco_joint.get("range"))
        self.assertIsNone(self.config.safety.joint_limits[0])
        self.assertEqual(self.config.safety.joint_limits[1:], ((-3.14, 3.14), (-3.14, 3.14)))

    def test_joint_limit_cbfs_exclude_joint1_and_push_inward_at_each_bound(self):
        controller = QPController(self.config, solver=None)
        rows, lower = controller._joint_limit_cbf_constraints(
            np.array([0.0, 3.0, -3.0])
        )

        np.testing.assert_allclose(
            rows,
            [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        )
        np.testing.assert_allclose(lower, [-6.14, -0.14, -0.14, -6.14])

    def test_simulation_kinematic_mode_is_loaded_from_yaml(self):
        self.assertTrue(self.config.simulation.kinematic)
        text = (ROOT / "settings/controller.yaml").read_text().replace(
            "kinematic: true", "kinematic: false"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.yaml"
            path.write_text(text)
            config = load_controller_config(path)
        self.assertFalse(config.simulation.kinematic)

    def test_end_effector_orientation_is_loaded(self):
        self.assertTrue(np.isfinite(self.config.ee_orientation))

    def test_clf_mode_is_loaded(self):
        self.assertEqual(self.config.control_mode, "clf")

    def test_sequential_end_effector_configuration_uses_each_link_tip(self):
        sequence = self.config.sequential_end_effector
        self.assertEqual(sequence.frame_names, ("joint2", "joint3", "ee_joint"))
        self.assertGreater(sequence.goal_tolerance, 0.0)

    def test_sequential_experiment_configuration_is_loaded(self):
        experiment = self.config.sequential_experiment
        self.assertGreaterEqual(experiment.cases, 1)
        self.assertGreaterEqual(experiment.workers, 1)
        self.assertGreater(experiment.duration, 0.0)
        self.assertEqual(len(experiment.initial_q), self.config.dof)
        self.assertEqual(len(experiment.target_radius_range), 2)
        self.assertEqual(len(experiment.obstacle_radius_range), 2)

    def test_sequential_end_effector_requires_one_frame_per_link(self):
        text = (ROOT / "settings/controller.yaml").read_text().replace(
            "frame_names: [joint2, joint3, ee_joint]", "frame_names: [joint2, ee_joint]"
        )
        text = text.replace("enabled: false", "enabled: true", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.yaml"
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "one frame for each link"):
                load_controller_config(path)

    def test_clf_mode_is_selectable_from_yaml(self):
        text = (ROOT / "settings/controller.yaml").read_text().replace(
            "control_mode: clf", "control_mode: invalid"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.yaml"
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "control_mode"):
                load_controller_config(path)

    def test_controller_geometry_mismatch_fails_fast(self):
        wrong_safety = replace(self.config.safety, link_radius=0.06)
        wrong_config = replace(self.config, safety=wrong_safety)
        with self.assertRaisesRegex(ValueError, "geometry mismatch for link1"):
            validate_geometry_consistency(wrong_config, self.urdf, self.mujoco)

    def test_disabled_obstacle_is_excluded_from_cbf_controller(self):
        disabled_obstacle = replace(self.config.obstacles[0], enabled=False)
        disabled_config = replace(
            self.config,
            obstacles=(disabled_obstacle, *self.config.obstacles[1:]),
            control_mode="clf",
        )

        controller = QPController(disabled_config, solver=None)

        self.assertEqual(len(controller.obstacles), len(self.config.obstacles) - 1)
        self.assertEqual(
            controller.barrier_labels,
            [f"obs2_link{index}" for index in range(1, self.config.dof + 1)],
        )

    def test_multiple_obstacles_create_a_barrier_for_each_link(self):
        controller = QPController(self.config, solver=None)
        active_obstacle_count = sum(obstacle.enabled for obstacle in self.config.obstacles)

        _, _, barrier_values = controller._cbf_constraints(np.zeros(self.config.dof))

        self.assertEqual(len(barrier_values), active_obstacle_count * self.config.dof)
        self.assertEqual(
            len(controller.barrier_labels), active_obstacle_count * self.config.dof
        )


if __name__ == "__main__":
    unittest.main()

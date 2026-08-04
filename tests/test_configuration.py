from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

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

    def test_clf_slack_ratio_is_loaded(self):
        self.assertGreater(self.config.qp.d, 0.0)
        self.assertLessEqual(self.config.qp.d, 1.0)

    def test_clf_slack_ratio_one_is_valid(self):
        config_path = ROOT / "settings/controller.yaml"
        text = config_path.read_text().replace(f"d: {self.config.qp.d}", "d: 1.0")
        with tempfile.TemporaryDirectory() as directory:
            temporary_path = Path(directory) / "controller.yaml"
            temporary_path.write_text(text)
            loaded = load_controller_config(temporary_path)
        self.assertEqual(loaded.qp.d, 1.0)

    def test_end_effector_orientation_is_loaded(self):
        self.assertTrue(np.isfinite(self.config.ee_orientation))

    def test_control_mode_and_clbf_parameters_are_loaded(self):
        self.assertEqual(self.config.control_mode, "clbf")
        self.assertGreater(self.config.clbf.gamma, 0.0)
        self.assertEqual(
            len(self.config.clbf.barriers),
            sum(obstacle.enabled for obstacle in self.config.obstacles),
        )
        self.assertTrue(self.config.clbf.barriers[1].auto_from_previous)

    def test_clf_mode_is_selectable_from_yaml(self):
        text = (ROOT / "settings/controller.yaml").read_text().replace(
            "control_mode: clbf", "control_mode: clf"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.yaml"
            path.write_text(text)
            config = load_controller_config(path)
        self.assertEqual(config.control_mode, "clf")

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

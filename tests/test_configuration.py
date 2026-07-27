from dataclasses import replace
from pathlib import Path
import unittest

from src.application.configuration import load_controller_config, validate_geometry_consistency
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
        self.assertLess(self.config.qp.d, 1.0)

    def test_controller_geometry_mismatch_fails_fast(self):
        wrong_safety = replace(self.config.safety, link_radius=0.06)
        wrong_config = replace(self.config, safety=wrong_safety)
        with self.assertRaisesRegex(ValueError, "geometry mismatch for link1"):
            validate_geometry_consistency(wrong_config, self.urdf, self.mujoco)

    def test_disabled_obstacle_is_excluded_from_cbf_controller(self):
        disabled_obstacle = replace(self.config.obstacles[0], enabled=False)
        disabled_config = replace(self.config, obstacles=(disabled_obstacle,))

        controller = QPController(disabled_config, solver=None)

        self.assertEqual(controller.obstacles, [])
        self.assertEqual(controller.barrier_labels, [])


if __name__ == "__main__":
    unittest.main()

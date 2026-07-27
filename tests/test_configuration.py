from dataclasses import replace
from pathlib import Path
import unittest

from src.application.configuration import load_controller_config, validate_geometry_consistency


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.config = load_controller_config(ROOT / "settings/controller.yaml")
        self.urdf = ROOT / "settings/urdf.xml"
        self.mujoco = ROOT / "settings/mujoco.xml"

    def test_current_robot_descriptions_are_consistent(self):
        validate_geometry_consistency(self.config, self.urdf, self.mujoco)

    def test_controller_geometry_mismatch_fails_fast(self):
        wrong_safety = replace(self.config.safety, link_radius=0.06)
        wrong_config = replace(self.config, safety=wrong_safety)
        with self.assertRaisesRegex(ValueError, "geometry mismatch for link1"):
            validate_geometry_consistency(wrong_config, self.urdf, self.mujoco)


if __name__ == "__main__":
    unittest.main()

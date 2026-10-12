"""The Panda scene displays all configured obstacles without altering the robot."""

import unittest
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from safe_ik.adapters.mujoco_plant import MujocoPlant
from safe_ik.adapters.panda_kinematics import PandaKinematics
from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.domain.panda_config import SphereObstacleConfig

ROOT = Path(__file__).resolve().parents[1]


class MujocoPlantTest(unittest.TestCase):
    def setUp(self):
        self.config = load_panda_config(ROOT / "tests/fixtures/panda.yaml")

    def test_scene_matches_obstacle_count_including_no_obstacles(self):
        for count in (0, 1, 2, 5):
            with self.subTest(count=count):
                obstacles = tuple(
                    SphereObstacleConfig(
                        f"sphere{index}",
                        np.array([0.1 * index, -0.3, 0.4]),
                        0.05 + 0.01 * index,
                        True,
                    )
                    for index in range(count)
                )
                plant = MujocoPlant(replace(self.config, obstacles=obstacles))
                names = {
                    plant.model.geom(index).name
                    for index in range(plant.model.ngeom)
                    if plant.model.geom(index).name.startswith("obstacle_geom")
                }
                self.assertEqual(
                    names, {f"obstacle_geom{i}" for i in range(1, count + 1)}
                )
                for index, obstacle in enumerate(obstacles, start=1):
                    geom = plant.model.geom(f"obstacle_geom{index}")
                    np.testing.assert_array_equal(geom.pos, obstacle.position)
                    self.assertEqual(geom.size[0], obstacle.radius)
                    self.assertEqual(geom.type[0], mujoco.mjtGeom.mjGEOM_SPHERE)
                    self.assertGreater(geom.rgba[3], 0.0)
                    self.assertEqual(geom.contype[0], 0)
                    self.assertEqual(geom.conaffinity[0], 0)

    def test_disabled_obstacle_is_hidden(self):
        obstacle = replace(self.config.obstacles[0], enabled=False)
        plant = MujocoPlant(replace(self.config, obstacles=(obstacle,)))
        self.assertEqual(plant.model.geom("obstacle_geom1").rgba[3], 0.0)

    def test_scene_changes_preserve_robot_kinematics_and_integration(self):
        original = mujoco.MjModel.from_xml_path(str(self.config.model_path))
        plant = MujocoPlant(replace(self.config, obstacles=()))
        self.assertEqual(plant.model.nq, original.nq)
        self.assertEqual(plant.model.nv, original.nv)
        for name in ("jnt_range", "body_mass", "body_inertia"):
            np.testing.assert_array_equal(
                getattr(plant.model, name), getattr(original, name)
            )
        velocity = np.full(original.nv, 0.1)
        plant.advance(velocity, 0.01)
        reference = mujoco.MjData(original)
        reference.qpos[:] = self.config.initial_q + velocity * 0.01
        mujoco.mj_forward(original, reference)
        np.testing.assert_allclose(plant.q, reference.qpos)
        np.testing.assert_allclose(plant.data.xpos, reference.xpos)
        np.testing.assert_allclose(plant.data.xquat, reference.xquat)
        np.testing.assert_array_equal(plant.last_applied_velocity, velocity)
        self.assertFalse(plant.last_joint_limit_clipped.any())

    def test_all_joint_limits_stop_outward_motion_and_allow_return(self):
        kin = PandaKinematics(self.config.model_path, self.config.ee_frame)
        for direction, boundary in ((-1, kin.lower_limits), (1, kin.upper_limits)):
            with self.subTest(direction=direction):
                initial = boundary - direction * 1e-5
                plant = MujocoPlant(replace(self.config, initial_q=initial))
                command = np.full(7, direction * 0.1)
                for _ in range(3):
                    before = plant.q
                    plant.advance(command, 0.02)
                    np.testing.assert_array_equal(plant.q, boundary)
                    self.assertTrue(plant.last_joint_limit_clipped.all())
                    np.testing.assert_allclose(
                        plant.data.qvel, (plant.q - before) / 0.02
                    )
                    np.testing.assert_array_equal(
                        plant.data.qvel, plant.last_applied_velocity
                    )
                np.testing.assert_array_equal(command, np.full(7, direction * 0.1))
                plant.advance(-command, 0.02)
                self.assertTrue(np.all(plant.q > kin.lower_limits))
                self.assertTrue(np.all(plant.q < kin.upper_limits))
                self.assertFalse(plant.last_joint_limit_clipped.any())
                self.assertAlmostEqual(plant.time, 0.08)

    def test_out_of_range_initial_state_is_rejected_without_projection(self):
        kin = PandaKinematics(self.config.model_path, self.config.ee_frame)
        for initial in (kin.lower_limits - 1e-12, kin.upper_limits + 1e-12):
            with self.assertRaisesRegex(ValueError, "initial joint positions"):
                MujocoPlant(replace(self.config, initial_q=initial))

    def test_invalid_command_or_period_does_not_change_state(self):
        plant = MujocoPlant(self.config)
        for velocity, period in (
            (np.full(7, np.nan), 0.02),
            (np.zeros(6), 0.02),
            (np.zeros(7), 0),
            (np.zeros(7), -0.02),
            (np.zeros(7), np.inf),
        ):
            with self.assertRaises(ValueError):
                plant.advance(velocity, period)
            np.testing.assert_array_equal(plant.q, self.config.initial_q)
            self.assertEqual(plant.time, 0.0)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

import mujoco
import numpy as np

from safe_ik.application.planar_simulation import (
    load_mujoco_model,
    run_simulation,
    set_goal_marker_position,
    set_obstacle_visibility,
)
from safe_ik.domain.types import SafetyDiagnostics
from safe_ik.logging.recorder import Recorder

ROOT = Path(__file__).resolve().parents[1]


class _ConstantVelocityController:
    dof = 3
    sequential_end_effector_enabled = False

    def get_velocity(self, q, J, p_ee, ee_yaw):
        return np.array([0.5, -1.0, 0.25])

    def safety_diagnostics(self, q, J, p_ee, ee_yaw):
        return SafetyDiagnostics((), np.empty(0))


class _StaticRobot:
    def update_kinematics(self, q):
        return np.zeros(3), np.zeros((6, 3))

    def end_effector_yaw(self):
        return 0.0


class GoalMarkerTest(unittest.TestCase):
    def test_goal_marker_is_visual_only_and_tracks_target_position(self):
        model = load_mujoco_model(ROOT / "assets/robots/planar.xml")
        marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker")

        self.assertNotEqual(marker_id, -1)
        self.assertEqual(model.geom_contype[marker_id], 0)
        self.assertEqual(model.geom_conaffinity[marker_id], 0)

        set_goal_marker_position(model, [1.0, 1.3])
        np.testing.assert_allclose(model.geom_pos[marker_id], [1.0, 1.3, 0.06])

    def test_disabled_obstacle_is_hidden(self):
        model = load_mujoco_model(ROOT / "assets/robots/planar.xml")
        obstacle_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_geom1"
        )

        set_obstacle_visibility(model, [False])

        self.assertEqual(model.geom_rgba[obstacle_id, 3], 0.0)

    def test_viewer_sync_interval_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            run_simulation(
                ROOT / "assets/robots/planar.xml",
                controller=None,
                robot=None,
                duration=0.0,
                visualize=False,
                viewer_sync_interval_steps=0,
            )

    def test_kinematic_mode_integrates_the_qp_velocity_directly(self):
        recorder = Recorder()
        run_simulation(
            ROOT / "assets/robots/planar.xml",
            controller=_ConstantVelocityController(),
            robot=_StaticRobot(),
            duration=0.004,
            visualize=False,
            recorder=recorder,
            save_recording=False,
            kinematic=True,
            verbose=False,
        )
        arrays = recorder.arrays()
        self.assertEqual(len(arrays["q"]), 2)
        np.testing.assert_allclose(
            arrays["q"][1],
            np.array([0.5, -1.0, 0.25]) * 0.002,
            atol=1e-12,
        )
        np.testing.assert_allclose(arrays["dq"][1], [0.5, -1.0, 0.25])


if __name__ == "__main__":
    unittest.main()

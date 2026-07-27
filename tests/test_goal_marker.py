from pathlib import Path
import unittest

import mujoco
import numpy as np

from src.application.simulation import (
    load_mujoco_model,
    set_goal_marker_position,
    set_obstacle_visibility,
)


ROOT = Path(__file__).resolve().parents[1]


class GoalMarkerTest(unittest.TestCase):
    def test_goal_marker_is_visual_only_and_tracks_target_position(self):
        model = load_mujoco_model(ROOT / "settings/mujoco.xml")
        marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker")

        self.assertNotEqual(marker_id, -1)
        self.assertEqual(model.geom_contype[marker_id], 0)
        self.assertEqual(model.geom_conaffinity[marker_id], 0)

        set_goal_marker_position(model, [1.0, 1.3])
        np.testing.assert_allclose(model.geom_pos[marker_id], [1.0, 1.3, 0.06])

    def test_disabled_obstacle_is_hidden(self):
        model = load_mujoco_model(ROOT / "settings/mujoco.xml")
        obstacle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_geom1")

        set_obstacle_visibility(model, [False])

        self.assertEqual(model.geom_rgba[obstacle_id, 3], 0.0)


if __name__ == "__main__":
    unittest.main()

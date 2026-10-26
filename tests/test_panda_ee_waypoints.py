"""Waypoint geometry, inherited QP, and final-goal / total-budget contracts."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from safe_ik.adapters.panda_kinematics import PandaKinematics
from safe_ik.application.panda_comparison_rollout import comparison_rollout
from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.application.panda_runtime import PandaRuntime
from safe_ik.control.panda_controller import PandaTrajectoryController
from safe_ik.control.panda_ee_waypoints import (
    PandaEEWaypointController,
    interpolate_ee_targets,
)
from safe_ik.domain.safety_contracts import BarrierEvaluation
from safe_ik.domain.types import QPSolution, QPStatus

ROOT = Path(__file__).resolve().parents[1]


class SafeScene:
    def evaluate(self, q):
        return BarrierEvaluation(("test",), np.ones(1), np.zeros((1, 7)))


class RecordSolver:
    def __init__(self, speed=0.0):
        self.speed, self.problems = speed, []

    def solve(self, problem):
        self.problems.append(problem)
        return QPSolution(QPStatus.SOLVED, np.r_[self.speed, np.zeros(7)], 0.0)


class LinearKinematics:
    lower_limits = np.full(7, -10.0)
    upper_limits = np.full(7, 10.0)

    def pose_and_jacobian(self, q):
        jac = np.zeros((6, 7))
        jac[:3, :3] = np.eye(3)
        return q[:3].copy(), np.array([1.0, 0.0, 0.0, 0.0]), jac

    def frame_pose_and_jacobian(self, frame, q):
        p, _, j = self.pose_and_jacobian(q)
        return np.eye(3), p, j


class Plant:
    def __init__(self, q):
        self.q, self.time = q.copy(), 0.0

    def advance(self, velocity, period):
        self.q += velocity * period
        self.time += period


class PandaEEWaypointTest(unittest.TestCase):
    def setUp(self):
        base = load_panda_config(ROOT / "tests/fixtures/panda.yaml")
        self.config = replace(
            base,
            trajectory=replace(
                base.trajectory,
                mode="direct",
                stage_task="pose",
                orientation_weight=1.0,
                stage_orientation_weight=1.0,
            ),
        )
        self.kin = PandaKinematics(base.model_path, base.ee_frame)

    def controller(self, count, solver=None):
        return PandaEEWaypointController(
            self.config,
            self.kin,
            SafeScene(),
            (),
            solver or RecordSolver(),
            stages=count,
        )

    def test_interpolation_has_equal_translation_and_shortest_rotation(self):
        initial = Rotation.from_euler("z", 170, degrees=True).as_quat()[[3, 0, 1, 2]]
        target_q = Rotation.from_euler("z", -170, degrees=True).as_quat()[[3, 0, 1, 2]]
        target = replace(
            self.config.target, position=np.array([0.8, 0.0, 0.0]), quaternion=target_q
        )
        positions, quats = interpolate_ee_targets(np.zeros(3), initial, target, 4)
        np.testing.assert_allclose(positions[:, 0], [0.2, 0.4, 0.6, 0.8])
        rotations = Rotation.from_quat(np.vstack([initial, quats])[:, [1, 2, 3, 0]])
        step_angles = (rotations[:-1].inv() * rotations[1:]).magnitude()
        np.testing.assert_allclose(step_angles, np.deg2rad(5), atol=1e-12)
        _, antipodal = interpolate_ee_targets(np.zeros(3), -initial, target, 4)
        np.testing.assert_allclose(
            Rotation.from_quat(antipodal[:, [1, 2, 3, 0]]).as_matrix(),
            rotations[1:].as_matrix(),
            atol=1e-12,
        )
        np.testing.assert_array_equal(quats[-1], target.quaternion)

    def test_one_stage_is_identical_to_direct_qp(self):
        a, b = RecordSolver(), RecordSolver()
        direct = PandaTrajectoryController(self.config, self.kin, SafeScene(), (), a)
        waypoint = self.controller(1, b)
        q = self.config.initial_q.copy()
        direct.get_velocity(q)
        waypoint.get_velocity(q)
        for name in ("P", "q", "A", "lower", "upper"):
            np.testing.assert_array_equal(
                getattr(a.problems[0], name), getattr(b.problems[0], name)
            )
        self.assertEqual(waypoint.stage_frames, ())
        self.assertEqual(waypoint.candidates, ())

    def test_intermediate_clf_gradient_matches_finite_difference(self):
        ctrl = self.controller(4)
        q = self.config.initial_q.copy()
        value, gradient, _ = ctrl._active_clf(q)
        eps = 1e-7
        numerical = np.array(
            [
                (ctrl._active_clf(q + d)[0] - ctrl._active_clf(q - d)[0]) / (2 * eps)
                for d in np.eye(7) * eps
            ]
        )
        self.assertGreater(value, 0)
        np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=1e-5)

    def test_only_clf_row_changes_in_qp_and_final_target_is_unchanged(self):
        a, b = RecordSolver(), RecordSolver()
        direct = PandaTrajectoryController(self.config, self.kin, SafeScene(), (), a)
        ctrl = self.controller(2, b)
        q = self.config.initial_q.copy()
        direct.get_velocity(q)
        ctrl.get_velocity(q)
        x, y = a.problems[0], b.problems[0]
        for name in ("P", "q"):
            np.testing.assert_array_equal(getattr(x, name), getattr(y, name))
        for name in ("A", "lower", "upper"):
            np.testing.assert_array_equal(getattr(x, name)[1:], getattr(y, name)[1:])
        self.assertFalse(np.allclose(x.A[0], y.A[0]))
        self.assertIs(ctrl.config.target, self.config.target)
        ctrl.stage_index = len(ctrl.stage_frames)
        np.testing.assert_allclose(ctrl._active_clf(q)[0], direct._active_clf(q)[0])
        np.testing.assert_allclose(ctrl._active_clf(q)[1], direct._active_clf(q)[1])

    def linear_rollout(self, duration):
        cfg = replace(
            self.config,
            initial_q=np.zeros(7),
            target=replace(
                self.config.target,
                position=np.array([0.2, 0.0, 0.0]),
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                position_tolerance=0.001,
                orientation_tolerance=0.001,
            ),
            trajectory=replace(
                self.config.trajectory,
                duration=duration,
                integration_period=0.1,
                stage_position_tolerance=0.001,
                stop_window=0.5,
                stop_progress_window=0.5,
            ),
        )
        runtime = PandaRuntime(
            LinearKinematics(), SafeScene(), RecordSolver(0.1), Plant(cfg.initial_q)
        )
        factory = lambda *args: PandaEEWaypointController(*args, stages=2)
        return comparison_rollout(
            cfg, runtime, subdivisions=1, controller_factory=factory
        )

    def test_intermediate_goal_is_not_final_success_and_budget_is_global(self):
        summary, trace = self.linear_rollout(1.0)
        self.assertEqual(summary["reason"], "timeout")
        self.assertFalse(summary["success"])
        self.assertAlmostEqual(summary["elapsed"], 1.0)
        self.assertAlmostEqual(trace["q"][-1, 0], 0.1)
        self.assertEqual(trace["stage_errors"].shape, (11, 1))

    def test_stage_transition_does_not_restart_time_or_create_ik(self):
        summary, trace = self.linear_rollout(3.0)
        self.assertTrue(summary["success"])
        self.assertAlmostEqual(summary["elapsed"], 2.0)
        self.assertEqual(len(summary["stage_transitions"]), 1)
        self.assertAlmostEqual(summary["stage_transitions"][0]["time"], 1.0)
        self.assertEqual(
            summary["stage_transitions"][0]["previous"], "ee_waypoint_01_pose"
        )
        self.assertEqual(summary["candidate_count"], 0)
        self.assertEqual(summary["candidate_switch_count"], 0)
        self.assertTrue((trace["candidate_index"] == -1).all())


if __name__ == "__main__":
    unittest.main()

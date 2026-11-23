"""Regression tests for numerical goal acceptance and premature IK switching."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from safe_ik.adapters.osqp_solver import OSQPSolver
from safe_ik.application.panda_configuration import (
    load_panda_config,
    parse_panda_config,
)
from safe_ik.control.panda_controller import PandaTrajectoryController
from safe_ik.domain.ik_contracts import SafeIKCandidate
from safe_ik.domain.safety_contracts import BarrierEvaluation
from safe_ik.domain.types import QPSolution, QPStatus

ROOT = Path(__file__).resolve().parents[1]


class Kinematics:
    lower_limits = np.full(7, -10.0)
    upper_limits = np.full(7, 10.0)

    def pose_and_jacobian(self, q):
        quaternion = np.array([np.cos(q[6] / 2), 0, 0, np.sin(q[6] / 2)])
        jacobian = np.zeros((6, 7))
        jacobian[:3, :3] = np.eye(3)
        jacobian[5, 6] = 1.0
        return q[:3].copy(), quaternion, jacobian

    def frame_point_and_jacobian(self, name, q):
        return q[:3].copy(), np.eye(3, 7)


class Safety:
    def __init__(self, value=1.0):
        self.value = value

    def evaluate(self, q):
        return BarrierEvaluation(
            ("obstacle_test",),
            np.array([1.0 if q[0] == 0 else self.value]),
            np.zeros((1, 7)),
        )


class Solver:
    def __init__(self, joint=0):
        self.joint = joint
        self.problems = []

    def solve(self, problem):
        self.problems.append(problem)
        solution = np.zeros(8)
        solution[self.joint] = 1e-4
        return QPSolution(QPStatus.SOLVED, solution, 0.0)


class PandaStoppingTest(unittest.TestCase):
    def setUp(self):
        config = load_panda_config(ROOT / "tests/fixtures/panda.yaml")
        self.config = replace(
            config,
            initial_q=np.zeros(7),
            safety=replace(config.safety, self_collision_model="rdf"),
            target=replace(
                config.target,
                position=np.array([0.0105, 0, 0]),
                quaternion=np.array([1.0, 0, 0, 0]),
            ),
            trajectory=replace(
                config.trajectory,
                mode="direct",
                integration_period=0.02,
                stage_frames=("link3", "link5"),
            ),
        )

    def controller(self, config=None, safety=None, candidates=(), solver=None):
        return PandaTrajectoryController(
            config or self.config,
            Kinematics(),
            safety or Safety(),
            candidates,
            solver or Solver(),
        )

    def test_goal_accepts_tiny_negative_residual_without_solving_or_switching(self):
        solver = Solver()
        controller = self.controller(safety=Safety(-4.43e-11), solver=solver)
        q = np.array([0.0105, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(controller.get_velocity(q), np.zeros(7))
        self.assertTrue(controller.last_status.success)
        self.assertEqual(controller.last_status.minimum_barrier, -4.43e-11)
        self.assertEqual(controller.candidate_switch_count, 0)
        self.assertEqual(solver.problems, [])

    def test_larger_violation_is_rejected_and_cbf_is_not_relaxed(self):
        solver = Solver()
        controller = self.controller(safety=Safety(-2e-6), solver=solver)
        controller.get_velocity(np.array([0.0105, 0, 0, 0, 0, 0, 0]))
        self.assertFalse(controller.last_status.success)
        self.assertAlmostEqual(
            solver.problems[-1].lower[1], self.config.safety.alpha * 2e-6
        )

    def test_zero_tolerance_restores_strict_goal_acceptance(self):
        config = replace(
            self.config,
            trajectory=replace(self.config.trajectory, success_barrier_tolerance=0),
        )
        controller = self.controller(config, Safety(-4.43e-11))
        controller.get_velocity(np.array([0.0105, 0, 0, 0, 0, 0, 0]))
        self.assertFalse(controller.last_status.success)

    def test_status_at_new_state_uses_fresh_rdf_barrier(self):
        controller = self.controller(safety=Safety(-2e-6))
        controller.get_velocity(np.zeros(7))
        status = controller.status(np.array([0.0105, 0, 0, 0, 0, 0, 0]))
        self.assertFalse(status.success)
        self.assertEqual(status.minimum_barrier, -2e-6)

    def test_slow_position_progress_waits_until_goal_at_both_control_periods(self):
        for period in (0.02, 0.01):
            with self.subTest(period=period):
                config = replace(
                    self.config,
                    trajectory=replace(
                        self.config.trajectory, integration_period=period
                    ),
                )
                controller = self.controller(config)
                q = np.zeros(7)
                for _ in range(int(6 / period)):
                    velocity = controller.get_velocity(q)
                    self.assertFalse(controller.last_status.stalled)
                    if controller.last_status.success:
                        break
                    q += period * velocity
                    controller.record_applied_velocity(velocity)
                self.assertTrue(controller.last_status.success)

    def test_slow_orientation_progress_also_prevents_stall(self):
        angle = 0.021
        config = replace(
            self.config,
            target=replace(
                self.config.target,
                position=np.zeros(3),
                quaternion=np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)]),
            ),
        )
        controller = self.controller(config, solver=Solver(joint=6))
        q = np.zeros(7)
        for _ in range(200):
            velocity = controller.get_velocity(q)
            q += config.trajectory.integration_period * velocity
            controller.record_applied_velocity(velocity)
            self.assertFalse(controller.last_status.stalled)
        self.assertFalse(controller.last_status.success)

    def test_no_progress_stalls_after_elapsed_window_not_number_of_queries(self):
        for period in (0.02, 0.01):
            config = replace(
                self.config,
                trajectory=replace(self.config.trajectory, integration_period=period),
            )
            controller = self.controller(config)
            q = np.zeros(7)
            for _ in range(int(config.trajectory.stop_progress_window / period)):
                controller.get_velocity(q)
                for _ in range(3):
                    self.assertFalse(controller.status(q).stalled)
                controller.record_applied_velocity(np.zeros(7))
            controller.get_velocity(q)
            self.assertTrue(controller.last_status.candidates_exhausted)

    def test_oscillation_without_a_new_best_error_does_not_count_as_progress(self):
        controller = self.controller()
        q = np.zeros(7)
        for i in range(110):
            q[0] = 1e-5 * (i % 2)
            controller.get_velocity(q)
            if controller.last_status.stalled:
                break
            controller.record_applied_velocity(np.zeros(7))
        self.assertTrue(controller.last_status.stalled)

    def test_reached_stages_reset_progress_and_velocity_history(self):
        candidate = SafeIKCandidate(np.array([0.1, 0, 0, 0, 0, 0, 0]), 0, 1, 0, 0, 1)
        for frames in (("link3",), ("link3", "link5")):
            with self.subTest(frames=frames):
                config = replace(
                    self.config,
                    trajectory=replace(
                        self.config.trajectory,
                        mode="sequential",
                        stage_frames=frames,
                        stage_position_tolerance=0.005,
                    ),
                )
                controller = self.controller(config, candidates=(candidate,))
                for _ in range(100):
                    controller.get_velocity(np.zeros(7))
                    controller.record_applied_velocity(np.zeros(7))
                controller.get_velocity(candidate.q)
                self.assertEqual(controller.active_stage_name, "end_effector_pose")
                self.assertEqual(controller.stage_index, len(frames))
                self.assertFalse(controller.last_status.stalled)
                self.assertEqual(len(controller._progress_best_errors), 1)
                self.assertEqual(len(controller.applied_velocity_norms), 0)

    def test_real_stall_switches_candidate_and_resets_progress(self):
        config = replace(
            self.config, trajectory=replace(self.config.trajectory, mode="sequential")
        )
        candidates = tuple(
            SafeIKCandidate(np.array([x, 0, 0, 0, 0, 0, 0]), i, 1, 0, 0, 1)
            for i, x in enumerate((0.1, 0.2))
        )
        controller = self.controller(config, candidates=candidates)
        for _ in range(101):
            controller.get_velocity(np.zeros(7))
            controller.record_applied_velocity(np.zeros(7))
        self.assertEqual(controller.candidate_index, 1)
        self.assertEqual(controller.candidate_switch_count, 1)
        self.assertFalse(controller.last_status.stalled)
        self.assertEqual(len(controller._progress_best_errors), 1)
        self.assertAlmostEqual(controller._best_task_error, 4.0)

    def test_applied_motion_drives_stall_independently_of_commands(self):
        for actual_speed, stalled in ((0.0, True), (0.1, False)):
            config = replace(
                self.config,
                trajectory=replace(
                    self.config.trajectory,
                    stop_window=0.04,
                    stop_progress_window=0.04,
                ),
            )
            solver = Solver()
            controller = self.controller(config, solver=solver)
            for _ in range(2):
                controller.get_velocity(np.zeros(7))
                actual = np.zeros(7)
                actual[0] = actual_speed
                controller.record_applied_velocity(actual)
                np.testing.assert_array_equal(controller.last_applied_velocity, actual)
            controller.get_velocity(np.zeros(7))
            self.assertEqual(controller.last_status.stalled, stalled)
            if not stalled:
                np.testing.assert_array_equal(
                    solver.problems[-1].upper[-7:], config.trajectory.velocity_limits
                )

    def test_speed_can_reach_limit_and_reverse_without_acceleration_ramp(self):
        target = np.array([0.5, 0, 0, 0, 0, 0, 0])
        candidate = SafeIKCandidate(target, 0, 1, 0, 0, 1)
        for mode in ("direct", "sequential", "joint_clf"):
            for period in (0.001, 0.02):
                with self.subTest(mode=mode, period=period):
                    config = replace(
                        self.config,
                        target=replace(self.config.target, position=target[:3]),
                        trajectory=replace(
                            self.config.trajectory,
                            mode=mode,
                            integration_period=period,
                        ),
                    )
                    controller = self.controller(
                        config,
                        candidates=() if mode == "direct" else (candidate,),
                        solver=OSQPSolver(
                            max_iterations=20_000,
                            absolute_tolerance=1e-8,
                            relative_tolerance=1e-8,
                        ),
                    )
                    # Start from rest, then observe the opposite side of the goal.
                    # Both commands should reach the speed bound in a single tick.
                    for q, sign in ((np.zeros(7), 1), (2 * target, -1)):
                        command = controller.get_velocity(q)
                        self.assertEqual(controller.last_qp_status, "solved")
                        expected = np.zeros(7)
                        expected[0] = sign * config.trajectory.velocity_limits[0]
                        np.testing.assert_allclose(command, expected, atol=1e-6)
                        controller.record_applied_velocity(command)

    def test_joint_cbf_still_limits_outward_speed_near_both_joint_bounds(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                config = replace(
                    self.config,
                    target=replace(
                        self.config.target, position=np.array([sign * 10.5, 0, 0])
                    ),
                )
                controller = self.controller(
                    config,
                    solver=OSQPSolver(
                        max_iterations=20_000,
                        absolute_tolerance=1e-8,
                        relative_tolerance=1e-8,
                    ),
                )
                q = np.zeros(7)
                q[0] = sign * (10.0 - 0.001)
                command = controller.get_velocity(q)
                self.assertEqual(controller.last_qp_status, "solved")
                self.assertAlmostEqual(
                    sign * command[0], config.safety.alpha * 0.001, places=6
                )

    def test_legacy_yaml_acceleration_limit_no_longer_narrows_velocity_bounds(self):
        path = ROOT / "tests/fixtures/panda.yaml"
        data = yaml.safe_load(path.read_text())
        data["trajectory"].update(mode="direct", acceleration_limits=[1e-9] * 7)
        config = parse_panda_config(data, path)
        solver = Solver()
        self.controller(config, solver=solver).get_velocity(np.zeros(7))
        np.testing.assert_array_equal(
            solver.problems[-1].lower[-7:], -config.trajectory.velocity_limits
        )
        np.testing.assert_array_equal(
            solver.problems[-1].upper[-7:], config.trajectory.velocity_limits
        )

    def test_unapplied_commands_do_not_fill_motion_window(self):
        controller = self.controller()
        for _ in range(150):
            controller.get_velocity(np.zeros(7))
        self.assertFalse(controller.last_status.stalled)
        self.assertEqual(len(controller.applied_velocity_norms), 0)
        for invalid in (np.zeros(6), np.full(7, np.nan), np.full(7, np.inf)):
            with self.assertRaises(ValueError):
                controller.record_applied_velocity(invalid)
        self.assertEqual(len(controller.applied_velocity_norms), 0)

    def test_invalid_stopping_parameters_are_rejected(self):
        for field in (
            "success_barrier_tolerance",
            "stop_velocity_tolerance",
            "stop_progress_window",
            "stop_progress_tolerance",
        ):
            for value in (-1, float("nan"), float("inf"), True):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    replace(self.config.trajectory, **{field: value})
        with self.assertRaises(ValueError):
            replace(self.config.trajectory, stop_progress_window=0)


if __name__ == "__main__":
    unittest.main()

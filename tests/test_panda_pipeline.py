import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from safe_ik.adapters.osqp_solver import OSQPSolver
from safe_ik.adapters.panda_kinematics import PandaKinematics
from safe_ik.application.panda_configuration import (
    load_panda_config,
    orientation_vectors_to_wxyz,
)
from safe_ik.application.panda_rollout import run_panda
from safe_ik.application.panda_setup import build_panda_components, build_panda_runtime
from safe_ik.control.panda_controller import PandaTrajectoryController
from safe_ik.domain.panda_types import PandaControlTrace
from safe_ik.domain.rotations import quaternion_orientation_error
from safe_ik.domain.safe_pose_ik import (
    SafeIKCandidate,
    SafePoseIKSolver,
    damped_pseudoinverse,
)
from safe_ik.domain.types import QPSolution, QPStatus
from safe_ik.logging.panda_results import save_panda_control_trace
from safe_ik.visualization.panda import save_panda_control_plot

ROOT = Path(__file__).resolve().parents[1]


class PandaPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_panda_config(ROOT / "tests/fixtures/panda.yaml")
        cls.kinematics, cls.safety = build_panda_components(cls.config)

    def test_fixed_gripper_model_has_seven_arm_dofs_and_tip_frame(self):
        self.assertEqual(self.kinematics.model.nq, 7)
        self.assertEqual(self.kinematics.model.nv, 7)
        self.assertIn("gripper_tip", self.kinematics.frame_ids)

    def test_timing_configuration_matches_clarified_specification(self):
        self.assertEqual(self.config.safe_ik.iteration_number, 1000)
        self.assertEqual(self.config.safe_ik.convergence_hold_duration, 3.0)
        self.assertEqual(self.config.safe_ik.convergence_hold_steps, 300)
        self.assertEqual(self.config.trajectory.integration_period, 0.05)
        self.assertGreater(self.config.trajectory.duration, 0.0)
        self.assertEqual(self.config.trajectory.control_frequency, 20.0)
        self.assertEqual(self.config.trajectory.stop_window, 0.5)
        self.assertEqual(self.config.qp.max_iterations, 20_000)

    def test_quaternion_error_is_zero_for_both_equivalent_signs(self):
        desired = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(
            quaternion_orientation_error(desired, desired), np.zeros(3), atol=1e-12
        )
        np.testing.assert_allclose(
            quaternion_orientation_error(-desired, desired), np.zeros(3), atol=1e-12
        )

    def test_non_normalized_orientation_vectors_construct_tip_axes(self):
        quaternion, approach, up = orientation_vectors_to_wxyz(
            [0.0, 0.0, -3.0], [2.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(approach, [0.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(up, [1.0, 0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0)
        np.testing.assert_allclose(
            self.config.target.approach, [0.0, 1.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(self.config.target.up, [1.0, 0.0, 0.0], atol=1e-12)

    def test_control_trace_exports_plot_and_named_data(self):
        trace = PandaControlTrace(
            time=np.array([0.0, 0.05]),
            clf_value=np.array([1.0, 0.5]),
            clf_slack=np.array([0.1, 0.0]),
            stage_index=np.array([0, 1]),
            stage_names=("link2_position", "end_effector_pose"),
            barrier_labels=("obstacle:link1", "self:link1-link3"),
            barrier_values=np.array([[0.2, 0.3], [0.1, 0.25]]),
        )
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            self.assertTrue(save_panda_control_plot(trace, output / "plot.png"))
            self.assertTrue(save_panda_control_trace(trace, output / "trace.npz"))
            self.assertTrue((output / "plot.png").is_file())
            with np.load(output / "trace.npz") as trace_data:
                np.testing.assert_array_equal(
                    trace_data["barrier_labels"], np.array(trace.barrier_labels)
                )
                np.testing.assert_array_equal(trace_data["stage_index"], [0, 1])

    def test_damped_inverse_has_task_to_joint_shape(self):
        _, _, jacobian = self.kinematics.pose_and_jacobian(self.config.initial_q)
        inverse = damped_pseudoinverse(jacobian, self.config.safe_ik.damping)
        self.assertEqual(inverse.shape, (7, 6))
        self.assertTrue(np.all(np.isfinite(inverse)))

    def test_safe_ik_ends_each_seed_after_hold_but_evaluates_all_seeds(self):
        config = replace(
            self.config,
            safe_ik=replace(
                self.config.safe_ik,
                convergence_hold_duration=0.03,
                max_restarts=2,
            ),
        )
        candidate = SafeIKCandidate(config.initial_q.copy(), 0, 1, 0.0, 0.0, 0.1)
        solver = SafePoseIKSolver(config, self.kinematics, self.safety)
        with patch.object(solver, "_candidate", return_value=candidate) as check:
            result = solver.solve(config.initial_q, require_safe=True)
        self.assertIs(result.selected, candidate)
        self.assertEqual(result.candidates, (candidate, candidate))
        self.assertEqual(check.call_count, 6)

    def test_safe_ik_hold_resets_when_acceptance_condition_is_lost(self):
        config = replace(
            self.config,
            safe_ik=replace(
                self.config.safe_ik,
                convergence_hold_duration=0.03,
                max_restarts=1,
            ),
        )
        candidate = SafeIKCandidate(config.initial_q.copy(), 0, 1, 0.0, 0.0, 0.1)
        solver = SafePoseIKSolver(config, self.kinematics, self.safety)
        sequence = [candidate, None, candidate, candidate, candidate]
        with patch.object(solver, "_candidate", side_effect=sequence) as check:
            result = solver.solve(config.initial_q, require_safe=True)
        self.assertIs(result.selected, candidate)
        self.assertEqual(check.call_count, 5)

    def test_safe_ik_selects_candidate_nearest_initial_configuration(self):
        config = replace(
            self.config,
            safe_ik=replace(
                self.config.safe_ik,
                convergence_hold_duration=0.01,
                max_restarts=2,
            ),
        )
        far = SafeIKCandidate(config.initial_q + 0.5, 0, 1, 0.0, 0.0, 0.1)
        near = SafeIKCandidate(config.initial_q + 0.1, 1, 1, 0.0, 0.0, 0.1)
        solver = SafePoseIKSolver(config, self.kinematics, self.safety)

        def candidate_for_seed(
            _q, seed_index, _iteration, _require_safe, **_evaluation
        ):
            return far if seed_index == 0 else near

        with patch.object(solver, "_candidate", side_effect=candidate_for_seed):
            result = solver.solve(config.initial_q, require_safe=True)
        self.assertEqual(result.candidates, (far, near))
        self.assertIs(result.selected, near)

    def test_safe_ik_first_seed_starts_at_initial_configuration(self):
        config = replace(
            self.config,
            safe_ik=replace(
                self.config.safe_ik,
                convergence_hold_duration=0.01,
                max_restarts=1,
            ),
        )
        candidate = SafeIKCandidate(config.initial_q.copy(), 0, 1, 0.0, 0.0, 0.1)
        first_checked_q = []

        def capture_first_q(q, _seed_index, _iteration, _require_safe, **_evaluation):
            first_checked_q.append(np.asarray(q).copy())
            return candidate

        solver = SafePoseIKSolver(config, self.kinematics, self.safety)
        with patch.object(solver, "_candidate", side_effect=capture_first_q):
            solver.solve(config.initial_q, require_safe=True)
        np.testing.assert_allclose(first_checked_q[0], config.initial_q, atol=0.0)

    def test_softmin_safety_gradient_matches_finite_difference(self):
        q = self.config.initial_q + np.array(
            [0.03, -0.02, 0.01, 0.0, 0.02, -0.01, 0.01]
        )
        _, analytic, _ = self.safety.softmin(q, self.config.safe_ik.softmin_temperature)
        epsilon = 1e-7
        numerical = np.zeros(7)
        for joint in range(7):
            dq = np.zeros(7)
            dq[joint] = epsilon
            plus = self.safety.softmin(q + dq, self.config.safe_ik.softmin_temperature)[
                0
            ]
            minus = self.safety.softmin(
                q - dq, self.config.safe_ik.softmin_temperature
            )[0]
            numerical[joint] = (plus - minus) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, numerical, atol=2e-5)

    def test_controller_stops_immediately_at_a_safe_target(self):
        position, quaternion, _ = self.kinematics.pose_and_jacobian(
            self.config.initial_q
        )
        target = replace(
            self.config.target,
            position=position,
            quaternion=quaternion,
        )
        config = replace(self.config, target=target)
        barrier = self.safety.evaluate(config.initial_q)
        candidate = SafeIKCandidate(
            config.initial_q.copy(), 0, 1, 0.0, 0.0, barrier.minimum
        )
        controller = PandaTrajectoryController(
            config,
            self.kinematics,
            self.safety,
            (candidate,),
            OSQPSolver(),
        )
        np.testing.assert_allclose(
            controller.get_velocity(config.initial_q), np.zeros(7), atol=1e-12
        )
        self.assertTrue(controller.last_status.success)

    def test_final_pose_clf_gradient_matches_finite_difference(self):
        trajectory = replace(self.config.trajectory, mode="direct")
        config = replace(self.config, trajectory=trajectory)
        barrier = self.safety.evaluate(config.initial_q)
        candidate = SafeIKCandidate(
            config.initial_q + 0.1, 0, 1, 0.0, 0.0, barrier.minimum
        )
        controller = PandaTrajectoryController(
            config, self.kinematics, self.safety, (candidate,), OSQPSolver()
        )
        q = config.initial_q + np.array([0.02, -0.01, 0.01, 0, 0, 0, 0])
        _, analytic, _ = controller._active_clf(q)
        epsilon = 1e-7
        numerical = np.zeros(7)
        for joint in range(7):
            dq = np.zeros(7)
            dq[joint] = epsilon
            plus = controller._active_clf(q + dq)[0]
            minus = controller._active_clf(q - dq)[0]
            numerical[joint] = (plus - minus) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, numerical, atol=2e-5)

    def test_direct_mode_skips_safe_ik_search(self):
        position, quaternion, _ = self.kinematics.pose_and_jacobian(
            self.config.initial_q
        )
        config = replace(
            self.config,
            target=replace(
                self.config.target, position=position, quaternion=quaternion
            ),
            trajectory=replace(self.config.trajectory, mode="direct"),
        )
        with patch("safe_ik.application.panda_rollout.SafePoseIKSolver") as ik_solver:
            result = run_panda(config, build_panda_runtime(config))
        ik_solver.assert_not_called()
        self.assertTrue(result.ik_skipped)
        self.assertEqual(result.safe_ik_wall_time, 0.0)
        self.assertEqual(result.ik.candidates, ())
        self.assertTrue(result.status.success)

    def test_sequential_clf_uses_link_endpoint_target_from_candidate(self):
        config = replace(
            self.config, trajectory=replace(self.config.trajectory, mode="sequential")
        )
        target_q = self.config.initial_q + np.array(
            [0.2, -0.1, 0.1, 0.0, 0.0, 0.0, 0.0]
        )
        candidate = SafeIKCandidate(target_q, 0, 1, 0.0, 0.0, 0.1)
        controller = PandaTrajectoryController(
            config, self.kinematics, self.safety, (candidate,), OSQPSolver()
        )
        self.assertEqual(controller.active_stage_name, "link2_position")
        value_at_target, _, error_at_target = controller._active_clf(target_q)
        self.assertAlmostEqual(value_at_target, 0.0, places=12)
        self.assertAlmostEqual(error_at_target, 0.0, places=12)

    def test_reached_stage_emits_named_transition_event(self):
        config = replace(
            self.config, trajectory=replace(self.config.trajectory, mode="sequential")
        )
        candidate = SafeIKCandidate(self.config.initial_q.copy(), 0, 1, 0.0, 0.0, 0.1)
        controller = PandaTrajectoryController(
            config, self.kinematics, self.safety, (candidate,), OSQPSolver()
        )
        controller._advance_reached_stages(self.config.initial_q)
        events = controller.consume_stage_transition_events()
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(
            events[0], ("link2_position", "link3_position", "target_reached")
        )
        self.assertEqual(controller.consume_stage_transition_events(), ())

    def test_maximum_iteration_failure_is_not_reported_as_infeasible(self):
        class MaximumIterationSolver:

            @staticmethod
            def solve(_problem):
                return QPSolution(QPStatus.ITERATION_LIMIT, None, 0.0)

        barrier = self.safety.evaluate(self.config.initial_q)
        target = self.config.initial_q + 0.1
        candidate = SafeIKCandidate(target, 0, 1, 0.0, 0.0, barrier.minimum)
        controller = PandaTrajectoryController(
            self.config,
            self.kinematics,
            self.safety,
            (candidate,),
            MaximumIterationSolver(),
        )
        np.testing.assert_allclose(
            controller.get_velocity(self.config.initial_q), np.zeros(7), atol=1e-12
        )
        self.assertFalse(controller.last_status.qp_infeasible)
        self.assertTrue(controller.last_status.qp_solver_failed)
        self.assertEqual(controller.qp_infeasible_count, 0)
        self.assertEqual(controller.qp_solver_failure_count, 1)


if __name__ == "__main__":
    unittest.main()

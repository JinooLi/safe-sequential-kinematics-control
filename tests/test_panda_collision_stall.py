"""Fixed-path advancement on stalls associated with binding collision CBFs."""

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from safe_ik.adapters.osqp_solver import OSQPSolver
from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.control.panda_controller import PandaTrajectoryController
from safe_ik.domain.ik_contracts import SafeIKCandidate
from safe_ik.domain.safety_contracts import BarrierEvaluation
from safe_ik.domain.types import QPProblem, QPSolution, QPStatus


class Kinematics:
    lower_limits = np.full(7, -10.0)
    upper_limits = np.full(7, 10.0)

    def pose_and_jacobian(self, q):
        return q[:3].copy(), np.array([1.0, 0, 0, 0]), np.eye(6, 7)

    def frame_point_and_jacobian(self, frame, q):
        joint = int(frame[4:]) - 2
        jacobian = np.zeros((3, 7))
        jacobian[0, joint] = 1.0
        return np.array([q[joint], 0, 0]), jacobian


class Safety:
    def __init__(self, label="obstacle1_arm", h=0.0):
        self.label, self.h = label, h

    def evaluate(self, q):
        gradient = np.zeros((1, 7))
        gradient[0, 0] = 1.0
        return BarrierEvaluation((self.label,), np.array([self.h]), gradient)


class ControlSolver:
    def __init__(self, status=QPStatus.SOLVED):
        self.status, self.problems = status, []

    def solve(self, problem):
        self.problems.append(problem)
        return QPSolution(
            self.status, np.zeros(8) if self.status == QPStatus.SOLVED else None, 0.0
        )


class DiagnosticSolver:
    def __init__(self, mode="active"):
        self.mode, self.problems = mode, []

    def solve(self, problem):
        self.problems.append(problem)
        if self.mode == "raise":
            raise RuntimeError("diagnostic-only failure")
        if self.mode == "infeasible":
            return QPSolution(QPStatus.INFEASIBLE, None, 0.0)
        x = np.zeros(8)
        dual = np.zeros(len(problem.lower))
        if self.mode != "missing":
            dual[2 if self.mode == "joint_only" else 1] = -2.0
        if self.mode == "tiny_dual":
            dual[1] = -1e-6
        if self.mode == "upper_sign":
            dual[1] = 2.0
        if self.mode == "nan":
            dual[1] = np.nan
        if self.mode == "wrong_shape":
            dual = dual[:-1]
        if self.mode == "match_bound":
            x[0] = problem.lower[1]
        return QPSolution(
            QPStatus.SOLVED,
            x,
            0.002,
            "diagnostic solved",
            None if self.mode == "missing" else dual,
        )


class CollisionStallTests(unittest.TestCase):
    def setUp(self):
        base = load_panda_config(Path(__file__).parent / "fixtures/panda.yaml")
        self.config = replace(
            base,
            initial_q=np.zeros(7),
            target=replace(
                base.target,
                position=np.full(3, 5.0),
                quaternion=np.array([1.0, 0, 0, 0]),
            ),
            trajectory=replace(
                base.trajectory,
                mode="sequential",
                stage_frames=("link2", "link4"),
                stage_task="position",
                integration_period=0.02,
                stop_window=0.04,
                stop_progress_window=0.04,
            ),
        )
        self.candidate = SafeIKCandidate(np.ones(7), 0, 1, 0, 0, 1)

    def controller(
        self, *, safety=None, diagnostic=None, enabled=True, config=None, control=None
    ):
        control = control or ControlSolver()
        diagnostic = diagnostic or DiagnosticSolver()
        controller = PandaTrajectoryController(
            config or self.config,
            Kinematics(),
            safety or Safety(),
            (self.candidate,),
            control,
            advance_on_collision_stall=enabled,
            collision_stall_solver=diagnostic,
        )
        return controller, control, diagnostic

    def reach_stall(self, controller, *, advance_state=None):
        q = np.zeros(7)
        for _ in range(2):
            controller.get_velocity(q)
            controller.record_applied_velocity(np.zeros(7))
        if advance_state is not None:
            q[:] = advance_state
        controller.get_velocity(q)
        return q

    def test_obstacle_and_self_collision_can_skip_without_marking_reached_or_changing_ik(
        self,
    ):
        for label in ("obstacle1_arm", "self_link2_link5"):
            with self.subTest(label=label):
                c, main, diagnostic = self.controller(safety=Safety(label))
                q = self.reach_stall(c)
                self.assertEqual(c.stage_index, 1)
                self.assertEqual(c.active_stage_name, "link4_position")
                self.assertEqual(c.last_completed_link, 0)
                self.assertEqual(c.candidate_switch_count, 0)
                np.testing.assert_array_equal(c.target_q, self.candidate.q)
                self.assertEqual(len(diagnostic.problems), 1)
                self.assertEqual(len(main.problems), 3)
                self.assertEqual(
                    c.consume_stage_transition_events(),
                    (
                        (
                            "link2_position",
                            "link4_position",
                            "collision_constrained_stall",
                        ),
                    ),
                )
                (event,) = c.consume_collision_stall_events()
                self.assertEqual(event["control_call_index"], 2)
                self.assertEqual(event["q"], q.tolist())
                self.assertEqual(event["active_collision_labels"], [label])
                self.assertTrue(event["skipped"])
                self.assertTrue(event["diagnostic_valid"])
                self.assertEqual(
                    event["collision_constraints"][0]["lower_multiplier"], 2
                )
                self.assertEqual(c.consume_collision_stall_events(), ())
                self.assertFalse(c.last_status.stalled)
                self.assertEqual(len(c.applied_velocity_norms), 0)
                self.assertEqual(len(c._progress_best_errors), 1)
                c.get_velocity(q)
                self.assertEqual(len(diagnostic.problems), 1)

    def test_slack_collision_and_joint_only_binding_do_not_skip(self):
        for safety, mode in (
            (Safety(h=1), "active"),
            (Safety(), "joint_only"),
            (Safety(), "tiny_dual"),
            (Safety(), "upper_sign"),
        ):
            with self.subTest(h=safety.h, mode=mode):
                c, _, _ = self.controller(
                    safety=safety, diagnostic=DiagnosticSolver(mode)
                )
                self.reach_stall(c)
                self.assertEqual(c.stage_index, 0)
                self.assertTrue(c.last_status.candidates_exhausted)
                (event,) = c.consume_collision_stall_events()
                self.assertFalse(event["skipped"])
                self.assertEqual(event["active_collision_labels"], [])

    def test_missing_invalid_dual_or_diagnostic_failure_preserves_stall(self):
        for mode in ("missing", "nan", "wrong_shape", "infeasible", "raise"):
            with self.subTest(mode=mode):
                c, control, _ = self.controller(diagnostic=DiagnosticSolver(mode))
                self.reach_stall(c)
                self.assertEqual(c.stage_index, 0)
                self.assertTrue(c.last_status.candidates_exhausted)
                self.assertEqual(len(control.problems), 2)
                (event,) = c.consume_collision_stall_events()
                self.assertFalse(event["diagnostic_valid"])
                self.assertFalse(event["skipped"])
                self.assertEqual(c.qp_solver_failure_count, 0)
                self.assertFalse(c.qp_infeasible_stopped)

    def test_disabled_policy_does_not_call_diagnostic_solver(self):
        c, _, diagnostic = self.controller(enabled=False)
        self.reach_stall(c)
        self.assertTrue(c.last_status.candidates_exhausted)
        self.assertEqual(diagnostic.problems, [])
        self.assertEqual(c.consume_collision_stall_events(), ())

    def test_reach_has_priority_over_stale_stall_window(self):
        c, _, diagnostic = self.controller()
        q = np.zeros(7)
        q[0] = 1.0
        self.reach_stall(c, advance_state=q)
        self.assertEqual(c.stage_index, 1)
        self.assertEqual(diagnostic.problems, [])
        self.assertEqual(
            c.consume_stage_transition_events(),
            (("link2_position", "link4_position", "target_reached"),),
        )
        self.assertFalse(c.last_status.stalled)

    def test_progress_or_motion_prevents_diagnostic(self):
        for progress in (True, False):
            c, _, diagnostic = self.controller()
            q = np.zeros(7)
            for _ in range(5):
                c.get_velocity(q)
                if progress:
                    q[0] += 0.01
                    c.record_applied_velocity(np.zeros(7))
                else:
                    c.record_applied_velocity(np.ones(7))
            self.assertEqual(diagnostic.problems, [])

    def test_last_link_skips_to_ee_and_ee_never_skips_again(self):
        config = replace(
            self.config,
            trajectory=replace(self.config.trajectory, stage_frames=("link2",)),
        )
        c, _, diagnostic = self.controller(config=config)
        self.reach_stall(c)
        self.assertEqual(c.active_stage_name, "end_effector_pose")
        self.reach_stall(c)
        self.assertEqual(len(diagnostic.problems), 1)
        self.assertTrue(c.last_status.candidates_exhausted)

    def test_live_qp_failure_remains_terminal(self):
        c, _, diagnostic = self.controller(control=ControlSolver(QPStatus.FAILED))
        self.reach_stall(c)
        self.assertTrue(c.qp_infeasible_stopped)
        self.assertEqual(c.stage_index, 0)
        self.assertEqual(diagnostic.problems, [])

    def test_rdf_self_gain_and_original_units_tolerance_are_used(self):
        cfg = replace(
            self.config,
            safety=replace(
                self.config.safety,
                alpha=0.2,
                self_collision_model="rdf",
                rdf=replace(self.config.safety.rdf, cbf_gain=1.0),
            ),
        )
        for label, gain in (("self_link2_link5", 1.0), ("obstacle1_arm", 0.2)):
            c, _, _ = self.controller(
                config=cfg,
                safety=Safety(label, h=0.02),
                diagnostic=DiagnosticSolver("match_bound"),
            )
            self.reach_stall(c)
            (event,) = c.consume_collision_stall_events()
            row = event["collision_constraints"][0]
            self.assertAlmostEqual(row["gain_times_h"], gain * 0.02)
            self.assertAlmostEqual(row["residual"], 0.0)
            self.assertAlmostEqual(
                row["tolerance"],
                cfg.qp.absolute_tolerance + cfg.qp.relative_tolerance * gain * 0.02,
            )
            self.assertTrue(event["skipped"])

    def test_policy_rejects_reselection_multiple_ik_and_shared_solver(self):
        solver = ControlSolver()
        base = (self.config, Kinematics(), Safety(), (self.candidate,), solver)
        with self.assertRaisesRegex(ValueError, "separate diagnostic"):
            PandaTrajectoryController(*base, advance_on_collision_stall=True)
        with self.assertRaisesRegex(ValueError, "separate diagnostic"):
            PandaTrajectoryController(
                *base, advance_on_collision_stall=True, collision_stall_solver=solver
            )
        with self.assertRaisesRegex(ValueError, "fixed sequence"):
            PandaTrajectoryController(
                *base,
                stage_selector=lambda q: (),
                advance_on_collision_stall=True,
                collision_stall_solver=DiagnosticSolver(),
            )
        with self.assertRaisesRegex(ValueError, "fixed sequence"):
            PandaTrajectoryController(
                self.config,
                Kinematics(),
                Safety(),
                (self.candidate, self.candidate),
                solver,
                advance_on_collision_stall=True,
                collision_stall_solver=DiagnosticSolver(),
            )


class OriginalUnitDualTests(unittest.TestCase):
    def test_real_osqp_dual_sign_and_objective_scale_on_setup_update_rebuild(self):
        solver = OSQPSolver(absolute_tolerance=1e-9, relative_tolerance=1e-9)
        for weight, target in ((5000.0, -2.0), (5000.0, -3.0), (20.0, -2.0)):
            problem = QPProblem(
                np.array([[weight]]),
                np.array([-weight * target]),
                np.ones((1, 1)),
                np.zeros(1),
                np.full(1, np.inf),
            )
            result = solver.solve(problem)
            self.assertEqual(result.status, QPStatus.SOLVED, result.message)
            self.assertAlmostEqual(float(result.x[0]), 0.0, places=7)
            self.assertIsNotNone(result.dual)
            self.assertAlmostEqual(float(result.dual[0]), weight * target, places=4)
            np.testing.assert_allclose(
                problem.P @ result.x + problem.q + problem.A.T @ result.dual,
                0.0,
                atol=1e-4,
            )


if __name__ == "__main__":
    unittest.main()

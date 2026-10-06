"""Intermediate pose CLFs, orientation progress, and candidate-aware diagnostics."""

import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from safe_ik.adapters.osqp_solver import OSQPSolver
from safe_ik.adapters.panda_kinematics import PandaKinematics
from safe_ik.application.panda_comparison_rollout import comparison_rollout
from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.application.panda_rollout import run_panda
from safe_ik.application.panda_runtime import PandaRuntime
from safe_ik.control.panda_controller import PandaTrajectoryController
from safe_ik.domain.ik_contracts import SafeIKCandidate, SafeIKResult
from safe_ik.domain.safety_contracts import BarrierEvaluation
from safe_ik.domain.types import QPSolution, QPStatus
from safe_ik.logging.panda_results import result_payload, save_panda_control_trace

ROOT = Path(__file__).resolve().parents[1]


class SafeScene:
    def evaluate(self, q):
        return BarrierEvaluation(("test",), np.ones(1), np.zeros((1, 7)))


class FixedVelocity:
    def __init__(self, speed=0.0):
        self.speed = speed

    def solve(self, problem):
        return QPSolution(QPStatus.SOLVED, np.r_[self.speed, np.zeros(7)], 0.0)


class Plant:
    def __init__(self, q):
        self.q, self.time = q.copy(), 0.0

    def advance(self, velocity, period):
        self.q += velocity * period
        self.time += period


class PandaIntermediatePoseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = load_panda_config(ROOT / "tests/fixtures/panda.yaml")
        cls.config = replace(
            base,
            trajectory=replace(
                base.trajectory,
                stage_task="pose",
                stage_frames=("link2",),
                stage_orientation_weight=0.7,
                stage_orientation_tolerance=0.01,
                integration_period=0.01,
                duration=0.1,
                stop_window=0.02,
                stop_progress_window=0.04,
                stop_progress_tolerance=1e-5,
            ),
        )
        cls.kin = PandaKinematics(
            base.model_path, base.ee_frame, ("link2", "link3", "link5")
        )

    def candidates(self):
        return tuple(
            SafeIKCandidate(
                self.config.initial_q + np.array([angle, 0, 0, 0, 0, 0, 0]),
                seed,
                1,
                0,
                0,
                1,
            )
            for angle, seed in ((0.3, 4), (0.6, 8))
        )

    def controller(self, config=None, candidates=None, solver=None):
        return PandaTrajectoryController(
            config or self.config,
            self.kin,
            SafeScene(),
            self.candidates() if candidates is None else candidates,
            solver or FixedVelocity(),
        )

    def test_actual_frame_pose_clf_gradient_matches_finite_difference(self):
        target_q = self.config.initial_q + np.array(
            [0.2, -0.15, 0.1, 0.1, -0.1, 0.1, 0]
        )
        candidate = SafeIKCandidate(target_q, 4, 1, 0, 0, 1)
        q = self.config.initial_q + np.array([-0.02, 0.03, -0.04, 0, 0.02, 0, 0])
        for frame in ("link2", "link3", "link5"):
            for weight in (0.2, 2.0):
                with self.subTest(frame=frame, weight=weight):
                    config = replace(
                        self.config,
                        trajectory=replace(
                            self.config.trajectory,
                            stage_frames=(frame,),
                            stage_orientation_weight=weight,
                        ),
                    )
                    controller = self.controller(config, (candidate,))
                    _, gradient, _ = controller._active_clf(q)
                    epsilon = 1e-7
                    numerical = np.array(
                        [
                            (
                                controller._active_clf(q + delta)[0]
                                - controller._active_clf(q - delta)[0]
                            )
                            / (2 * epsilon)
                            for delta in np.eye(7) * epsilon
                        ]
                    )
                    np.testing.assert_allclose(
                        gradient, numerical, atol=1e-8, rtol=1e-5
                    )
                    value, at_target, _ = controller._active_clf(target_q)
                    self.assertAlmostEqual(value, 0.0, places=12)
                    np.testing.assert_allclose(at_target, 0.0, atol=1e-12)

    def test_position_match_requires_orientation_only_in_pose_mode(self):
        pose = self.controller()
        q = self.config.initial_q
        position_error, orientation_error = pose.stage_error_norms(q, 0)
        self.assertAlmostEqual(position_error, 0.0)
        self.assertAlmostEqual(orientation_error, np.sin(0.15))
        pose._advance_reached_stages(q)
        self.assertEqual(pose.active_stage_name, "link2_pose")
        pose._advance_reached_stages(self.candidates()[0].q)
        self.assertEqual(pose.active_stage_name, "end_effector_pose")
        baseline = self.controller(
            replace(
                self.config,
                trajectory=replace(
                    self.config.trajectory,
                    stage_task="position",
                ),
            )
        )
        baseline._advance_reached_stages(q)
        self.assertEqual(baseline.active_stage_name, "end_effector_pose")

    def test_real_qp_command_reduces_orientation_when_position_already_matches(self):
        controller = self.controller(
            solver=OSQPSolver(
                absolute_tolerance=1e-8,
                relative_tolerance=1e-8,
            )
        )
        q = self.config.initial_q.copy()
        before = controller._active_clf(q)[0]
        velocity = controller.get_velocity(q)
        self.assertFalse(controller.last_status.qp_solver_failed)
        self.assertFalse(controller.last_status.qp_infeasible)
        self.assertGreater(np.linalg.norm(velocity), 0)
        self.assertLess(controller._active_clf(q + 0.01 * velocity)[0], before)

    def test_orientation_improvement_prevents_low_speed_stall(self):
        controller = self.controller(solver=FixedVelocity(1e-4))
        q = self.config.initial_q.copy()
        for _ in range(100):
            q += self.config.trajectory.integration_period * controller.get_velocity(q)
            self.assertFalse(controller.last_status.stalled)
        self.assertEqual(controller.candidate_switch_count, 0)
        self.assertEqual(controller.stage_index, 0)

    def test_comparison_records_new_orientation_target_on_candidate_switch(self):
        runtime = PandaRuntime(
            self.kin, SafeScene(), FixedVelocity(), Plant(self.config.initial_q)
        )
        result, trace = comparison_rollout(
            self.config, runtime, self.candidates(), subdivisions=1
        )
        self.assertEqual(result["reason"], "stalled")
        self.assertEqual(result["candidate_switch_count"], 1)
        self.assertEqual(result["final_stage"], "link2_pose")
        np.testing.assert_allclose(trace["stage_errors"], 0.0)
        self.assertEqual(trace["stage_orientation_errors"].shape, (len(trace["q"]), 1))
        for index, error in zip(
            trace["candidate_index"], trace["stage_orientation_errors"][:, 0]
        ):
            self.assertAlmostEqual(error, np.sin((0.3, 0.6)[index] / 2))
        self.assertEqual(trace["candidate_seed_index"][-1], 8)

    def test_regular_rollout_exports_pose_stage_labels_and_orientation_errors(self):
        candidates = self.candidates()
        runtime = PandaRuntime(
            self.kin,
            SafeScene(),
            FixedVelocity(),
            Plant(self.config.initial_q),
            ik_search_factory=lambda: nullcontext(
                SimpleNamespace(
                    solve=lambda *args, **kwargs: SafeIKResult(
                        candidates, candidates[0], None
                    ),
                )
            ),
        )
        result = run_panda(self.config, runtime)
        self.assertEqual(result.trace.stage_names, ("link2_pose", "end_effector_pose"))
        self.assertEqual(
            result_payload(result)["trajectory"]["final_stage_orientation_error"],
            result.status.stage_orientation_error,
        )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.npz"
            save_panda_control_trace(result.trace, path)
            with np.load(path) as trace:
                self.assertEqual(
                    trace["stage_orientation_error"].shape, trace["time"].shape
                )
                self.assertTrue(np.all(np.isfinite(trace["stage_orientation_error"])))
                self.assertGreater(trace["stage_orientation_error"][-1], 0.1)

    def test_invalid_stage_pose_options_are_rejected(self):
        for value in ("orientation", True, None):
            with self.subTest(stage_task=value), self.assertRaises(ValueError):
                replace(self.config.trajectory, stage_task=value)
        for name in ("stage_orientation_weight", "stage_orientation_tolerance"):
            for value in (0, -1, True, float("nan"), float("inf")):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    replace(self.config.trajectory, **{name: value})
        with self.assertRaises(ValueError):
            replace(self.config.trajectory, stage_orientation_tolerance=1.0)


if __name__ == "__main__":
    unittest.main()

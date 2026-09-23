"""Run the Panda orchestration with no MuJoCo or Pinocchio backend."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.application.panda_rollout import run_panda
from safe_ik.application.panda_runtime import PandaRuntime
from safe_ik.domain.spatial_safety import BarrierEvaluation
from safe_ik.domain.types import QPSolution, QPStatus

ROOT = Path(__file__).resolve().parents[1]


class LinearKinematics:
    lower_limits = np.full(7, -10.0)
    upper_limits = np.full(7, 10.0)

    def pose_and_jacobian(self, q):
        return q[:3].copy(), np.array([1.0, 0.0, 0.0, 0.0]), np.eye(6, 7)

    def frame_point_and_jacobian(self, frame_name, q, local_offset=None):
        return q[:3].copy(), np.eye(3, 7)


class ConstantSafety:
    def evaluate(self, q):
        return BarrierEvaluation(("test_barrier",), np.array([1.0]), np.zeros((1, 7)))

    def softmin(self, q, temperature):
        return 1.0, np.zeros(7), self.evaluate(q)


class EulerPlant:
    def __init__(self, q):
        self.q = q.copy()
        self.time = 0.0
        self.commands = []

    def advance(self, velocity, period):
        self.commands.append(velocity.copy())
        self.q += velocity * period
        self.time += period


class FixedSolver:
    def __init__(self, status=QPStatus.SOLVED):
        self.status = status

    def solve(self, problem):
        x = np.r_[np.full(7, 0.01), 0.0] if self.status == QPStatus.SOLVED else None
        return QPSolution(self.status, x, 0.002, "backend-independent test result")


class PandaRuntimeTest(unittest.TestCase):
    def setUp(self):
        base = load_panda_config(ROOT / "tests/fixtures/panda.yaml")
        self.config = replace(
            base,
            target=replace(
                base.target,
                position=base.initial_q[:3] + 0.5,
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
            trajectory=replace(base.trajectory, mode="direct", duration=0.1),
        )

    def runtime(self, status=QPStatus.SOLVED):
        return PandaRuntime(
            LinearKinematics(),
            ConstantSafety(),
            FixedSolver(status),
            EulerPlant(self.config.initial_q),
        )

    def test_injected_plant_controls_state_time_and_trace(self):
        runtime = self.runtime()
        seen = []
        result = run_panda(
            self.config, runtime, observe=lambda: seen.append(runtime.plant.time)
        )
        np.testing.assert_allclose(result.final_q, self.config.initial_q + 0.001)
        self.assertEqual(len(seen), 2)
        self.assertEqual(result.steps, 2)
        self.assertAlmostEqual(result.elapsed, 0.1)
        self.assertEqual(result.trace.q.shape, (2, 7))
        np.testing.assert_allclose(result.trace.command, 0.01)
        self.assertAlmostEqual(result.total_qp_solve_time, 0.004)

    def test_failure_stops_without_integrating_and_survives_final_status_query(self):
        runtime = self.runtime(QPStatus.ITERATION_LIMIT)
        result = run_panda(self.config, runtime)
        self.assertTrue(result.status.qp_solver_failed)
        self.assertFalse(result.status.qp_infeasible)
        self.assertEqual(result.qp_solver_failure_count, 1)
        self.assertEqual(result.qp_infeasible_count, 0)
        self.assertEqual(runtime.plant.commands, [])
        self.assertEqual(result.elapsed, 0.0)
        np.testing.assert_array_equal(result.final_q, self.config.initial_q)

    def test_caller_can_stop_before_any_control_step(self):
        result = run_panda(self.config, self.runtime(), should_continue=lambda: False)
        self.assertEqual(result.steps, 0)
        self.assertEqual(result.trace.q.shape, (0, 7))
        self.assertEqual(result.elapsed, 0.0)

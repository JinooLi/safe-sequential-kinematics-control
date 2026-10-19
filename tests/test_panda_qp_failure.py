"""Rejected backend solutions stop motion and retain actionable diagnostics."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import test_panda_runtime as runtime_fixture

from safe_ik.adapters.osqp_solver import OSQPSolver
from safe_ik.application.panda_comparison_rollout import comparison_rollout
from safe_ik.application.panda_rollout import run_panda
from safe_ik.application.panda_runtime import PandaRuntime
from safe_ik.logging.panda_results import save_panda_control_trace


class PandaQPFailureTest(unittest.TestCase):
    def setUp(self):
        fixture = runtime_fixture.PandaRuntimeTest()
        fixture.setUp()
        self.config = fixture.config

    def runtime(self):
        return PandaRuntime(
            runtime_fixture.LinearKinematics(),
            runtime_fixture.ConstantSafety(),
            OSQPSolver(),
            runtime_fixture.EulerPlant(self.config.initial_q),
        )

    @staticmethod
    def backend():
        backend = Mock()
        backend.solve.return_value = SimpleNamespace(
            info=SimpleNamespace(status_val=1, status="solved"),
            x=np.r_[np.full(7, 10.0), 0.0],
        )
        return backend

    def test_single_rollout_stops_before_integration_and_exports_failure(self):
        with patch("safe_ik.adapters.osqp_solver.osqp.OSQP", self.backend):
            runtime = self.runtime()
            result = run_panda(self.config, runtime)
        self.assertTrue(result.status.qp_solver_failed)
        self.assertFalse(result.status.success)
        self.assertIn("qp_constraint_violation", result.failure["detail"])
        self.assertEqual(runtime.plant.commands, [])
        self.assertEqual(result.elapsed, 0.0)
        self.assertFalse(result.trace.command_applied.any())
        with TemporaryDirectory() as folder:
            path = Path(folder) / "trace.npz"
            save_panda_control_trace(result.trace, path)
            with np.load(path, allow_pickle=False) as saved:
                self.assertIn("qp_constraint_violation", saved["qp_message"][0])
                self.assertEqual(saved["qp_status"][0], "failed")

    def test_comparison_rollout_keeps_failure_label_message_and_unapplied_command(self):
        with patch("safe_ik.adapters.osqp_solver.osqp.OSQP", self.backend):
            runtime = self.runtime()
            result, trace = comparison_rollout(self.config, runtime, ())
        self.assertEqual(result["reason"], "qp_solver_failure")
        self.assertFalse(result["success"])
        self.assertIn("qp_constraint_violation", result["qp_failure_message"])
        self.assertEqual(runtime.plant.commands, [])
        self.assertEqual(result["steps"], 0)
        self.assertEqual(len(trace["q"]), 1)
        self.assertFalse(trace["command_applied"].any())
        self.assertEqual(trace["qp_message"][0], result["qp_failure_message"])


if __name__ == "__main__":
    unittest.main()

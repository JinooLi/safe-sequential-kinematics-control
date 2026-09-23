"""Backend-neutral QP results and workspace reuse behavior."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from safe_ik.adapters.osqp_solver import OSQPSolver
from safe_ik.domain.types import QPProblem, QPStatus


def problem(weight=2.0, rows=1):
    return QPProblem(
        np.array([[weight]]),
        np.array([-2.0]),
        np.ones((rows, 1)),
        np.zeros(rows),
        np.full(rows, np.inf),
    )


class SolverContractTest(unittest.TestCase):
    def test_reused_solver_honors_changed_hessian_and_constraint_count(self):
        solver = OSQPSolver(absolute_tolerance=1e-8, relative_tolerance=1e-8)
        for weight, rows in [(2.0, 1), (4.0, 1), (4.0, 2), (4.0, 2)]:
            result = solver.solve(problem(weight, rows))
            self.assertEqual(result.status, QPStatus.SOLVED)
            self.assertAlmostEqual(result.x[0], 2.0 / weight, places=5)
            self.assertGreaterEqual(result.solve_time, 0.0)

    def test_infeasible_problem_has_no_solution(self):
        infeasible = QPProblem(
            np.eye(1),
            np.zeros(1),
            np.ones((2, 1)),
            np.array([1.0, -np.inf]),
            np.array([np.inf, 0.0]),
        )
        result = OSQPSolver().solve(infeasible)
        self.assertEqual(result.status, QPStatus.INFEASIBLE)
        self.assertIsNone(result.x)

    def test_status_mapping_does_not_depend_on_backend_message(self):
        solver = OSQPSolver()
        solver.solve(problem())
        with patch.object(
            solver._solver,
            "solve",
            return_value=SimpleNamespace(
                info=SimpleNamespace(status_val=7, status="arbitrary backend message"),
                x=None,
            ),
        ):
            result = solver.solve(problem())
        self.assertEqual(result.status, QPStatus.ITERATION_LIMIT)

    def test_backend_exception_is_a_typed_failure(self):
        solver = OSQPSolver()
        solver.solve(problem())
        with patch.object(
            solver._solver, "solve", side_effect=RuntimeError("backend unavailable")
        ):
            result = solver.solve(problem())
        self.assertEqual(result.status, QPStatus.FAILED)
        self.assertIn("backend unavailable", result.message)
        self.assertEqual(solver.solve(problem()).status, QPStatus.SOLVED)

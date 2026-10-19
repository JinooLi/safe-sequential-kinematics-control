"""Backend-neutral QP results and workspace reuse behavior."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
    def test_reused_solver_updates_vectors_and_matrix(self):
        solver = OSQPSolver(absolute_tolerance=1e-8, relative_tolerance=1e-8)
        self.assertEqual(solver.solve(problem()).status, QPStatus.SOLVED)
        workspace = solver._solver
        changed = replace(problem(), A=np.array([[2.0]]), upper=np.array([0.6]))
        result = solver.solve(changed)
        self.assertIs(solver._solver, workspace)
        self.assertEqual(result.status, QPStatus.SOLVED)
        self.assertAlmostEqual(result.x[0], 0.3, places=6)
        result = solver.solve(replace(changed, q=np.array([-0.2])))
        self.assertEqual(result.status, QPStatus.SOLVED)
        self.assertAlmostEqual(result.x[0], 0.1, places=6)

    def test_invalid_inputs_do_not_update_or_solve_a_previous_problem(self):
        invalid = [
            replace(problem(), lower=np.array([0.5]), upper=np.array([0.1])),
            replace(problem(), q=np.array([np.nan])),
            replace(problem(), P=np.array([[np.inf]])),
            replace(problem(), A=np.array([[np.nan]])),
            replace(problem(), lower=np.array([np.nan])),
            replace(problem(), lower=np.array([np.inf])),
            replace(problem(), upper=np.array([-np.inf])),
            replace(problem(), A=np.ones((1, 2))),
            replace(problem(), upper=np.zeros(2)),
        ]
        for data in invalid:
            with self.subTest(data=data):
                solver = OSQPSolver()
                self.assertEqual(solver.solve(problem()).status, QPStatus.SOLVED)
                workspace = solver._solver
                with (
                    patch.object(solver, "_checked_update") as update,
                    patch.object(workspace, "solve") as solve,
                ):
                    result = solver.solve(data)
                    update.assert_not_called()
                    solve.assert_not_called()
                self.assertEqual(result.status, QPStatus.FAILED)
                self.assertIsNone(result.x)
                self.assertIn("invalid_qp_", result.message)
                self.assertEqual(solver.solve(problem()).status, QPStatus.SOLVED)
                self.assertIsNot(solver._solver, workspace)

    def test_rejected_update_never_solves_stale_or_partially_updated_data(self):
        for operation in ("update_data_vec", "update_data_mat"):
            with self.subTest(operation=operation):
                solver = OSQPSolver()
                solver.solve(problem())
                workspace = solver._solver
                backend = Mock(wraps=workspace._solver)
                getattr(backend, operation).return_value = 1
                with (
                    patch.object(workspace, "_solver", backend),
                    patch.object(workspace, "solve") as solve,
                ):
                    result = solver.solve(problem())
                    solve.assert_not_called()
                    if operation == "update_data_vec":
                        backend.update_data_mat.assert_not_called()
                self.assertEqual(result.status, QPStatus.FAILED)
                self.assertIsNone(result.x)
                self.assertIn("qp_update_failed", result.message)
                self.assertEqual(solver.solve(problem()).status, QPStatus.SOLVED)
                self.assertIsNot(solver._solver, workspace)

    def test_solved_status_does_not_override_local_constraint_residuals(self):
        # The second variable models a large CLF slack. It cannot make a speed
        # violation acceptable, nor can the opposite side's huge finite bound.
        for lower, upper, x, side in [
            ([-np.inf, 0], [0.145, np.inf], [0.272, 10000], "upper"),
            ([-0.145, 0], [1e10, np.inf], [-0.272, 10000], "lower"),
            ([0, 0], [0, np.inf], [0.1, 10000], "upper"),
            ([-1, 0], [1, np.inf], [0, -0.1], "lower"),
        ]:
            with self.subTest(side=side, x=x):
                data = QPProblem(np.eye(2), np.zeros(2), np.eye(2), lower, upper)
                solver = OSQPSolver()
                self.assertEqual(solver.solve(data).status, QPStatus.SOLVED)
                solved = SimpleNamespace(
                    info=SimpleNamespace(status_val=1, status="solved"),
                    x=np.array(x),
                )
                with patch.object(solver._solver, "solve", return_value=solved):
                    result = solver.solve(data)
                self.assertEqual(result.status, QPStatus.FAILED)
                self.assertIsNone(result.x)
                self.assertIn("qp_constraint_violation", result.message)
                self.assertIn(f"side={side}", result.message)
                self.assertIn("residual=", result.message)
                self.assertIn("tolerance=", result.message)

    def test_numerical_tolerance_is_per_row_and_keeps_small_residuals(self):
        data = replace(problem(), upper=np.array([0.145]))
        solver = OSQPSolver()
        solver.solve(data)
        solved = SimpleNamespace(
            info=SimpleNamespace(status_val=1, status="solved"), x=np.array([0.1455])
        )
        with patch.object(solver._solver, "solve", return_value=solved):
            self.assertEqual(solver.solve(data).status, QPStatus.SOLVED)

    def test_nonfinite_or_wrong_shape_solutions_are_failures(self):
        for x in ([np.nan], [np.inf], [1.0, 2.0], [[1.0]], None):
            with self.subTest(x=x):
                solver = OSQPSolver()
                solver.solve(problem())
                solved = SimpleNamespace(
                    info=SimpleNamespace(status_val=1, status="solved"), x=x
                )
                with patch.object(solver._solver, "solve", return_value=solved):
                    result = solver.solve(problem())
                self.assertEqual(result.status, QPStatus.FAILED)
                self.assertIsNone(result.x)
                self.assertIn("invalid_qp_solution", result.message)

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

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
    def test_large_clf_slack_does_not_cause_a_false_solved_speed_violation(self):
        # Analytic optimum: v=-0.145, slack=100+v. The old global relative
        # stopping criterion returned v about -0.194 and our row check rejected
        # this feasible QP. Exercise the real backend, including a warm update.
        data = QPProblem(
            np.diag([1.0, 5000.0]),
            np.zeros(2),
            np.array([[1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]),
            np.array([-np.inf, -0.145, 0.0]),
            np.array([-100.0, 0.145, np.inf]),
        )
        solver = OSQPSolver(max_iterations=20000)
        for clf_bound in (100.0, 20.0):
            current = replace(data, upper=np.array([-clf_bound, 0.145, np.inf]))
            result = solver.solve(current)
            self.assertEqual(result.status, QPStatus.SOLVED, result.message)
            np.testing.assert_allclose(result.x, [-0.145, clf_bound - 0.145], atol=1e-5)
            values = current.A @ result.x
            self.assertLessEqual(float(np.max(current.lower - values)), 1e-5)
            self.assertLessEqual(float(np.max(values - current.upper)), 1e-5)

    def test_objective_normalization_preserves_linear_term_on_reuse_and_rebuild(self):
        # The same known unconstrained minimizer at three objective scales.
        # Omitting q's scale on either setup or update changes this optimum.
        solver = OSQPSolver(absolute_tolerance=1e-8, relative_tolerance=1e-8)
        for factor, target in ((5000.0, 0.2), (5000.0, -0.3), (20.0, 0.4)):
            data = QPProblem(
                np.array([[factor]]),
                np.array([-factor * target]),
                np.ones((1, 1)),
                np.array([-1.0]),
                np.array([1.0]),
            )
            result = solver.solve(data)
            self.assertEqual(result.status, QPStatus.SOLVED, result.message)
            self.assertAlmostEqual(result.x[0], target, places=6)

    def test_constraint_refinement_keeps_budget_and_restores_settings_on_reuse(self):
        data = replace(problem(), upper=np.array([0.145]))
        solver = OSQPSolver(max_iterations=100)
        solver.solve(data)
        workspace = solver._solver
        bad = SimpleNamespace(
            info=SimpleNamespace(status_val=1, status="solved", iter=25),
            x=np.array([0.2]),
        )
        good = SimpleNamespace(
            info=SimpleNamespace(status_val=1, status="solved", iter=50),
            x=np.array([0.145]),
        )
        with (
            patch.object(workspace, "solve", side_effect=[bad, good, good]) as solve,
            patch.object(
                workspace, "update_settings", wraps=workspace.update_settings
            ) as settings,
        ):
            result = solver.solve(data)
            self.assertEqual(result.status, QPStatus.SOLVED)
            self.assertAlmostEqual(result.x[0], 0.145)
            self.assertIn("constraint refinement", result.message)
            settings.assert_any_call(eps_rel=0.0, max_iter=75)
            self.assertEqual(solve.call_count, 2)
            self.assertEqual(solver.solve(data).status, QPStatus.SOLVED)
            settings.assert_called_with(eps_rel=solver.relative_tolerance, max_iter=100)

    def test_refinement_never_accepts_an_inaccurate_or_still_violating_vector(self):
        data = replace(problem(), upper=np.array([0.145]))
        for final_status in (1, 2, 7):
            with self.subTest(final_status=final_status):
                solver = OSQPSolver(max_iterations=100)
                solver.solve(data)
                bad = SimpleNamespace(
                    info=SimpleNamespace(status_val=1, status="solved", iter=25),
                    x=np.array([0.2]),
                )
                final = SimpleNamespace(
                    info=SimpleNamespace(
                        status_val=final_status, status="final", iter=75
                    ),
                    x=np.array([0.2 if final_status == 1 else 0.145]),
                )
                with patch.object(
                    solver._solver, "solve", side_effect=[bad, final]
                ) as solve:
                    result = solver.solve(data)
                    self.assertEqual(solve.call_count, 2)
                self.assertNotEqual(result.status, QPStatus.SOLVED)
                self.assertIsNone(result.x)
                self.assertFalse(solver._initialized)

    def test_exhausted_budget_does_not_start_a_refinement(self):
        data = replace(problem(), upper=np.array([0.145]))
        solver = OSQPSolver(max_iterations=100)
        solver.solve(data)
        bad = SimpleNamespace(
            info=SimpleNamespace(status_val=1, status="solved", iter=100),
            x=np.array([0.2]),
        )
        with patch.object(solver._solver, "solve", return_value=bad) as solve:
            result = solver.solve(data)
            self.assertEqual(solve.call_count, 1)
        self.assertEqual(result.status, QPStatus.FAILED)
        self.assertIsNone(result.x)

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

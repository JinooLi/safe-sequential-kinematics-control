"""OSQP implementation of the control-layer QPSolver interface."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import osqp
import scipy.sparse as sparse

from ..domain.types import QPProblem, QPSolution, QPStatus


class OSQPSolver:
    def __init__(
        self,
        *,
        max_iterations: int = 4_000,
        absolute_tolerance: float = 1e-3,
        relative_tolerance: float = 1e-3,
    ):
        self._solver = osqp.OSQP()
        self._initialized = False
        self.max_iterations = int(max_iterations)
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self._P = None
        self._shape = None

    @staticmethod
    def _full_pattern_csc(matrix: np.ndarray):
        """CSC with a stable dense pattern, including numerically zero entries."""
        matrix = np.asarray(matrix, dtype=float)
        rows, columns = matrix.shape
        return sparse.csc_matrix(
            (
                np.ravel(matrix, order="F"),
                np.tile(np.arange(rows), columns),
                np.arange(0, rows * columns + 1, rows),
            ),
            shape=(rows, columns),
        )

    def solve(self, problem: QPProblem) -> QPSolution:
        start = perf_counter()
        try:
            # Rebuild when P or the constraint dimensions change. Otherwise reuse
            # OSQP's workspace and warm start with a fixed full A sparsity pattern.
            A = self._full_pattern_csc(problem.A)
            P = np.asarray(problem.P, dtype=float)
            q = np.asarray(problem.q, dtype=float).ravel()
            lower = np.asarray(problem.lower, dtype=float).ravel()
            upper = np.asarray(problem.upper, dtype=float).ravel()
            if (
                not self._initialized
                or self._shape != A.shape
                or not np.array_equal(self._P, P)
            ):
                self._initialized = False
                self._solver = osqp.OSQP()
                self._solver.setup(
                    P=sparse.csc_matrix(P),
                    q=q,
                    A=A,
                    l=lower,
                    u=upper,
                    verbose=False,
                    max_iter=self.max_iterations,
                    eps_abs=self.absolute_tolerance,
                    eps_rel=self.relative_tolerance,
                )
                self._P = P.copy()
                self._shape = A.shape
                self._initialized = True
            else:
                self._solver.update(q=q, l=lower, u=upper, Ax=A.data)
            result = self._solver.solve(raise_error=False)
            status = {
                1: QPStatus.SOLVED,
                3: QPStatus.INFEASIBLE,
                4: QPStatus.INFEASIBLE,
                5: QPStatus.INFEASIBLE,
                6: QPStatus.INFEASIBLE,
                7: QPStatus.ITERATION_LIMIT,
            }.get(int(result.info.status_val), QPStatus.FAILED)
            # Preserve the existing policy: an inaccurate solution is not accepted.
            return QPSolution(
                status,
                np.asarray(result.x).copy() if status == QPStatus.SOLVED else None,
                perf_counter() - start,
                str(result.info.status),
            )
        except Exception as error:
            self._initialized = False
            return QPSolution(QPStatus.FAILED, None, perf_counter() - start, str(error))

"""OSQP implementation of the control-layer QPSolver interface."""

from __future__ import annotations

import numpy as np
import osqp
import scipy.sparse as sparse

from ..domain.types import QPProblem


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
        self.last_status: str | None = None

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

    def solve(self, problem: QPProblem) -> np.ndarray | None:
        A = self._full_pattern_csc(problem.A)
        q = np.asarray(problem.q, dtype=float).ravel()
        lower = np.asarray(problem.lower, dtype=float).ravel()
        upper = np.asarray(problem.upper, dtype=float).ravel()
        try:
            if not self._initialized:
                self._solver.setup(
                    P=sparse.csc_matrix(problem.P),
                    q=q,
                    A=A,
                    l=lower,
                    u=upper,
                    verbose=False,
                    max_iter=self.max_iterations,
                    eps_abs=self.absolute_tolerance,
                    eps_rel=self.relative_tolerance,
                )
                self._initialized = True
            else:
                self._solver.update(q=q, l=lower, u=upper, Ax=A.data)
        except Exception as error:
            self.last_status = f"setup/update failed: {error}"
            print(f"QP {self.last_status}")
            return None
        result = self._solver.solve()
        self.last_status = str(result.info.status)
        return result.x if self.last_status == "solved" else None

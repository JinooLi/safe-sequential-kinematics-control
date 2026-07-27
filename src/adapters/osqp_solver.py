"""OSQP implementation of the control-layer QPSolver interface."""

from __future__ import annotations

import numpy as np
import osqp
import scipy.sparse as sparse

from ..domain.types import QPProblem


class OSQPSolver:
    def __init__(self):
        self._solver = osqp.OSQP()
        self._initialized = False

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
                    P=sparse.csc_matrix(problem.P), q=q, A=A, l=lower, u=upper, verbose=False
                )
                self._initialized = True
            else:
                self._solver.update(q=q, l=lower, u=upper, Ax=A.data)
        except Exception as error:
            print(f"QP setup/update failed: {error}")
            return None
        result = self._solver.solve()
        return result.x if result.info.status == "solved" else None

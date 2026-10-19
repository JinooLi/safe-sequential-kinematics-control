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

    @staticmethod
    def _validated_arrays(problem: QPProblem):
        P, q, A, lower, upper = (
            np.asarray(value, dtype=float)
            for value in (problem.P, problem.q, problem.A, problem.lower, problem.upper)
        )
        if q.ndim != 1 or not q.size or P.shape != (q.size, q.size):
            raise ValueError("invalid_qp_data: incompatible P/q dimensions")
        if (
            A.ndim != 2
            or A.shape[1] != q.size
            or lower.shape != (A.shape[0],)
            or upper.shape != lower.shape
        ):
            raise ValueError("invalid_qp_data: incompatible A/bound dimensions")
        if not all(np.isfinite(value).all() for value in (P, q, A)):
            raise ValueError("invalid_qp_data: nonfinite P, q or A")
        if not np.allclose(P, P.T, rtol=1e-12, atol=1e-12):
            raise ValueError("invalid_qp_data: P must be symmetric")
        if (
            np.isnan(lower).any()
            or np.isnan(upper).any()
            or np.isposinf(lower).any()
            or np.isneginf(upper).any()
        ):
            raise ValueError("invalid_qp_data: invalid nonfinite bound")
        inverted = np.flatnonzero(lower > upper)
        if inverted.size:
            row = int(inverted[0])
            raise ValueError(
                f"invalid_qp_bounds: row={row} lower={lower[row]:.12g} "
                f"> upper={upper[row]:.12g}"
            )
        return P, q, A, lower, upper

    def _checked_update(self, q, lower, upper, A):
        # OSQP 1.x's public update() discards these return codes. In particular,
        # a rejected vector update may otherwise be followed by solving stale data.
        backend = self._solver._solver
        infinity = self._solver.constant("OSQP_INFTY")
        code = backend.update_data_vec(
            q=q, l=np.maximum(lower, -infinity), u=np.minimum(upper, infinity)
        )
        if code != 0:
            raise RuntimeError(f"qp_update_failed: vector update returned {code!r}")
        code = backend.update_data_mat(P_x=None, P_i=None, A_x=A.data, A_i=None)
        if code != 0:
            raise RuntimeError(f"qp_update_failed: matrix update returned {code!r}")

    def _check_solution(self, x, A, lower, upper):
        if x.shape != (A.shape[1],) or not np.isfinite(x).all():
            raise ValueError("invalid_qp_solution: wrong shape or nonfinite x")
        values = A @ x
        if not np.isfinite(values).all():
            raise ValueError("invalid_qp_solution: nonfinite A @ x")
        # Check each finite side in its own original units. A large CLF slack or
        # an unrelated row must not enlarge the tolerance of a velocity/CBF row.
        for side, bounds, sign in (("lower", lower, -1), ("upper", upper, 1)):
            rows = np.flatnonzero(np.isfinite(bounds))
            residual = sign * (values[rows] - bounds[rows])
            tolerance = self.absolute_tolerance + self.relative_tolerance * np.maximum(
                np.abs(values[rows]), np.abs(bounds[rows])
            )
            bad = np.flatnonzero(residual > tolerance)
            if bad.size:
                index = int(bad[np.argmax((residual - tolerance)[bad])])
                row = int(rows[index])
                raise ValueError(
                    f"qp_constraint_violation: row={row} side={side} "
                    f"value={values[row]:.12g} bound={bounds[row]:.12g} "
                    f"residual={residual[index]:.12g} "
                    f"tolerance={tolerance[index]:.12g}; backend=solved"
                )

    def solve(self, problem: QPProblem) -> QPSolution:
        start = perf_counter()
        try:
            # Rebuild when P or the constraint dimensions change. Otherwise reuse
            # OSQP's workspace and warm start with a fixed full A sparsity pattern.
            P, q, dense_A, lower, upper = self._validated_arrays(problem)
            A = self._full_pattern_csc(dense_A)
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
                self._checked_update(q, lower, upper, A)
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
            x = None
            if status == QPStatus.SOLVED:
                x = np.asarray(result.x, dtype=float).copy()
                self._check_solution(x, dense_A, lower, upper)
            else:
                self._initialized = False
            return QPSolution(
                status,
                x,
                perf_counter() - start,
                str(result.info.status),
            )
        except Exception as error:
            self._initialized = False
            return QPSolution(QPStatus.FAILED, None, perf_counter() - start, str(error))

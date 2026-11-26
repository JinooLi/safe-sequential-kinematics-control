"""Typed values exchanged between control, simulation, and logging."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(frozen=True)
class QPProblem:
    P: np.ndarray
    q: np.ndarray
    A: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class SafetyDiagnostics:
    """Named barriers, active CLF, and optional per-link CLF diagnostics."""

    barrier_labels: tuple[str, ...]
    barrier_values: np.ndarray
    clf_value: float | None = None
    clf_by_link: np.ndarray | None = None

    def __post_init__(self):
        values = np.asarray(self.barrier_values, dtype=float).reshape(-1)
        if len(values) != len(self.barrier_labels):
            raise ValueError(
                "barrier_labels and barrier_values must have the same length"
            )
        object.__setattr__(self, "barrier_values", values.copy())
        if self.clf_by_link is not None:
            object.__setattr__(
                self, "clf_by_link", np.asarray(self.clf_by_link, dtype=float).copy()
            )


@dataclass(frozen=True)
class SimulationStep:
    """One sampled simulation state, command, and safety diagnostic."""

    time: float
    q: np.ndarray
    dq: np.ndarray
    end_effector: np.ndarray
    command: np.ndarray
    safety: SafetyDiagnostics
    end_effector_yaw: float = 0.0
    active_link: int | None = None
    clf_slack: float | None = None
    qp_solve_time: float | None = None


class QPStatus(str, Enum):
    SOLVED = "solved"
    INFEASIBLE = "infeasible"
    ITERATION_LIMIT = "iteration_limit"
    FAILED = "failed"


@dataclass(frozen=True)
class QPSolution:
    """Backend-neutral solver result, including setup/update/solve wall time."""

    status: QPStatus
    x: np.ndarray | None
    solve_time: float
    message: str = ""
    # Signed multipliers for lower <= A @ x <= upper, in the original
    # objective units. A binding lower bound has a negative multiplier.
    dual: np.ndarray | None = None

    def __post_init__(self):
        if self.status == QPStatus.SOLVED and self.x is None:
            raise ValueError("a solved QP must include a solution")

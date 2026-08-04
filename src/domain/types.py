"""Typed values exchanged between control, simulation, and logging."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """Barrier values in a stable, explicitly named order."""

    barrier_labels: tuple[str, ...]
    barrier_values: np.ndarray
    clf_value: float | None = None
    clbf_labels: tuple[str, ...] = ()
    clbf_values: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __post_init__(self):
        values = np.asarray(self.barrier_values, dtype=float).reshape(-1)
        if len(values) != len(self.barrier_labels):
            raise ValueError("barrier_labels and barrier_values must have the same length")
        object.__setattr__(self, "barrier_values", values.copy())
        clbf_values = np.asarray(self.clbf_values, dtype=float).reshape(-1)
        if len(clbf_values) != len(self.clbf_labels):
            raise ValueError("clbf_labels and clbf_values must have the same length")
        object.__setattr__(self, "clbf_values", clbf_values.copy())


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

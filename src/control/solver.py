"""Solver boundary used by the QP controller."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..domain.types import QPProblem


class QPSolver(Protocol):
    def solve(self, problem: QPProblem) -> np.ndarray | None: ...

"""Solver boundary used by the QP controller."""

from __future__ import annotations

from typing import Protocol

from ..domain.types import QPProblem, QPSolution


class QPSolver(Protocol):
    def solve(self, problem: QPProblem) -> QPSolution: ...

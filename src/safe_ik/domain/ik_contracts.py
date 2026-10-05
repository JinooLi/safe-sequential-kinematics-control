"""IK candidates, seed/progress messages and the search interface."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SafeIKCandidate:
    q: np.ndarray
    seed_index: int
    iterations: int
    position_error: float
    orientation_error: float
    minimum_barrier: float


@dataclass(frozen=True)
class SafeIKResult:
    candidates: tuple[SafeIKCandidate, ...]
    selected: SafeIKCandidate | None
    failure_reason: str | None


@dataclass(frozen=True)
class IKSeed:
    index: int
    q: np.ndarray


SeedMap = Callable[[list[IKSeed], bool], Iterable[SafeIKCandidate | None]]


@dataclass(frozen=True)
class IKProgress:
    seed_index: int
    iteration: int
    finished: bool = False
    candidate_found: bool = False


IKProgressCallback = Callable[[IKProgress], None]


class PoseIKSearch(Protocol):
    def solve(
        self,
        initial_q=None,
        *,
        require_safe: bool = True,
        on_progress: IKProgressCallback | None = None,
    ) -> SafeIKResult: ...

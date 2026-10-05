"""Safety evaluation values and the backend-independent evaluator contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BarrierEvaluation:
    labels: tuple[str, ...]
    values: np.ndarray
    gradients: np.ndarray

    @property
    def minimum(self) -> float:
        return float(np.min(self.values)) if self.values.size else float("inf")

    def softmin(self, temperature: float) -> tuple[float, np.ndarray]:
        """Aggregate this evaluation without querying geometry or kinematics.

        ``temperature`` is tau in -tau * log(sum(exp(-h / tau))), not its
        reciprocal. Shift by the minimum to retain the existing stable formula.
        """
        if temperature <= 0.0:
            raise ValueError("soft-min temperature must be positive")
        if not self.values.size:
            return float("inf"), np.zeros(self.gradients.shape[1])
        minimum = float(np.min(self.values))
        exponentials = np.exp(-(self.values - minimum) / temperature)
        weights = exponentials / np.sum(exponentials)
        value = minimum - temperature * np.log(np.sum(exponentials))
        gradient = weights @ self.gradients
        return float(value), gradient


class SafetyEvaluator(Protocol):
    def evaluate(self, q: np.ndarray) -> BarrierEvaluation: ...


def evaluate_softmin(
    evaluator: SafetyEvaluator, q: np.ndarray, temperature: float
) -> tuple[float, np.ndarray, BarrierEvaluation]:
    """Compatibility entry point; reuse BarrierEvaluation.softmin in new loops."""
    if temperature <= 0.0:
        raise ValueError("soft-min temperature must be positive")
    evaluation = evaluator.evaluate(q)
    value, gradient = evaluation.softmin(temperature)
    return value, gradient, evaluation

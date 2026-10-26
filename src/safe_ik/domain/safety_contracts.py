"""Safety evaluation values and the backend-independent evaluator contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BarrierValues:
    labels: tuple[str, ...]
    values: np.ndarray

    @property
    def minimum(self) -> float:
        return float(np.min(self.values)) if self.values.size else float("inf")


@dataclass(frozen=True)
class BarrierEvaluation(BarrierValues):
    gradients: np.ndarray

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


def barrier_values_batch(evaluator, states):
    """Use a value-only backend when available; retain other evaluator contracts."""
    states = list(states)
    if not states:
        return []
    method = getattr(evaluator, "evaluate_values_batch", None)
    if method is not None:
        return method(states)
    return [BarrierValues(e.labels, e.values) for e in map(evaluator.evaluate, states)]


def evaluate_softmin(
    evaluator: SafetyEvaluator, q: np.ndarray, temperature: float
) -> tuple[float, np.ndarray, BarrierEvaluation]:
    """Compatibility entry point; reuse BarrierEvaluation.softmin in new loops."""
    if temperature <= 0.0:
        raise ValueError("soft-min temperature must be positive")
    evaluation = evaluator.evaluate(q)
    value, gradient = evaluation.softmin(temperature)
    return value, gradient, evaluation

"""Replaceable 3D capsule safety model for Panda obstacle/self collision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .interfaces import PoseKinematics
from .panda_config import CapsuleConfig, SphereObstacleConfig


@dataclass(frozen=True)
class CapsuleState:
    name: str
    start: np.ndarray
    end: np.ndarray
    start_jacobian: np.ndarray
    end_jacobian: np.ndarray
    radius: float


@dataclass(frozen=True)
class BarrierEvaluation:
    labels: tuple[str, ...]
    values: np.ndarray
    gradients: np.ndarray

    @property
    def minimum(self) -> float:
        return float(np.min(self.values)) if self.values.size else float("inf")


class SafetyEvaluator(Protocol):
    def evaluate(self, q: np.ndarray) -> BarrierEvaluation: ...
    def softmin(
        self, q: np.ndarray, temperature: float
    ) -> tuple[float, np.ndarray, BarrierEvaluation]: ...


def _unit_or_default(vector: np.ndarray) -> tuple[float, np.ndarray]:
    distance = float(np.linalg.norm(vector))
    if distance < 1e-12:
        return 0.0, np.array([1.0, 0.0, 0.0])
    return distance, vector / distance


def _point_segment_parameter(point, start, end) -> float:
    segment = end - start
    denominator = float(segment @ segment)
    if denominator < 1e-16:
        return 0.0
    return float(np.clip((point - start) @ segment / denominator, 0.0, 1.0))


def _segment_parameters(p1, q1, p2, q2) -> tuple[float, float]:
    """Closest parameters on two finite 3D segments."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = float(d1 @ d1), float(d2 @ d2)
    if a < 1e-16 and e < 1e-16:
        return 0.0, 0.0
    if a < 1e-16:
        return 0.0, float(np.clip((d2 @ r) / e, 0.0, 1.0))
    c = float(d1 @ r)
    if e < 1e-16:
        return float(np.clip(-c / a, 0.0, 1.0)), 0.0
    b, f = float(d1 @ d2), float(d2 @ r)
    denominator = a * e - b * b
    s = (
        0.0
        if abs(denominator) < 1e-16
        else np.clip((b * f - c * e) / denominator, 0.0, 1.0)
    )
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = np.clip(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = np.clip((b - c) / a, 0.0, 1.0)
    return float(s), float(t)


class PandaCapsuleSafety:
    """Obstacle and non-adjacent self-collision signed-distance barriers.

    Geometry access is isolated here so a learned robot distance field can
    replace this capsule implementation without changing IK or QP control.
    """

    def __init__(
        self,
        kinematics: PoseKinematics,
        capsules,
        obstacles,
        margin: float,
        self_collision_exclusions=(),
    ):
        self.kinematics = kinematics
        self.capsules: tuple[CapsuleConfig, ...] = tuple(capsules)
        self.obstacles: tuple[SphereObstacleConfig, ...] = tuple(
            obstacle for obstacle in obstacles if obstacle.enabled
        )
        if margin < 0.0:
            raise ValueError("safety margin must be non-negative")
        self.margin = float(margin)
        self.self_collision_exclusions = frozenset(self_collision_exclusions)

    def _states(self, q) -> tuple[CapsuleState, ...]:
        states = []
        for capsule in self.capsules:
            start, start_jacobian = self.kinematics.frame_point_and_jacobian(
                capsule.start_frame, q, capsule.start_offset
            )
            end, end_jacobian = self.kinematics.frame_point_and_jacobian(
                capsule.end_frame, q, capsule.end_offset
            )
            states.append(
                CapsuleState(
                    capsule.name,
                    start,
                    end,
                    start_jacobian,
                    end_jacobian,
                    capsule.radius,
                )
            )
        return tuple(states)

    def evaluate(self, q) -> BarrierEvaluation:
        states = self._states(q)
        labels, values, gradients = [], [], []
        for obstacle in self.obstacles:
            for capsule in states:
                ratio = _point_segment_parameter(
                    obstacle.position, capsule.start, capsule.end
                )
                closest = (1.0 - ratio) * capsule.start + ratio * capsule.end
                closest_jacobian = (
                    1.0 - ratio
                ) * capsule.start_jacobian + ratio * capsule.end_jacobian
                distance, normal = _unit_or_default(closest - obstacle.position)
                labels.append(f"{obstacle.name}_{capsule.name}")
                values.append(distance - capsule.radius - obstacle.radius - self.margin)
                gradients.append(normal @ closest_jacobian)

        # Capsule order follows the kinematic chain. Direct neighbours share
        # a joint and are deliberately excluded from self-collision checks.
        for first in range(len(states)):
            for second in range(first + 2, len(states)):
                left, right = states[first], states[second]
                if frozenset((left.name, right.name)) in self.self_collision_exclusions:
                    continue
                s, t = _segment_parameters(left.start, left.end, right.start, right.end)
                point_left = (1.0 - s) * left.start + s * left.end
                point_right = (1.0 - t) * right.start + t * right.end
                jacobian_left = (1.0 - s) * left.start_jacobian + s * left.end_jacobian
                jacobian_right = (
                    1.0 - t
                ) * right.start_jacobian + t * right.end_jacobian
                distance, normal = _unit_or_default(point_left - point_right)
                labels.append(f"self_{left.name}_{right.name}")
                values.append(distance - left.radius - right.radius - self.margin)
                gradients.append(normal @ (jacobian_left - jacobian_right))

        dof = len(self.kinematics.lower_limits)
        if not values:
            return BarrierEvaluation((), np.empty(0), np.empty((0, dof)))
        return BarrierEvaluation(
            tuple(labels), np.asarray(values), np.vstack(gradients)
        )

    def softmin(self, q, temperature: float):
        """Return log-sum-exp soft minimum and its joint gradient."""
        if temperature <= 0.0:
            raise ValueError("soft-min temperature must be positive")
        evaluation = self.evaluate(q)
        if not evaluation.values.size:
            return float("inf"), np.zeros(len(self.kinematics.lower_limits)), evaluation
        minimum = float(np.min(evaluation.values))
        exponentials = np.exp(-(evaluation.values - minimum) / temperature)
        weights = exponentials / np.sum(exponentials)
        value = minimum - temperature * np.log(np.sum(exponentials))
        gradient = weights @ evaluation.gradients
        return float(value), gradient, evaluation

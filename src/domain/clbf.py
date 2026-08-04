"""End-effector hierarchical CLBF construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CLBFResult:
    """One cumulative CLBF value and its joint-space gradient."""

    label: str
    value: float
    gradient_q: np.ndarray
    gamma: float


@dataclass(frozen=True)
class EndEffectorBarrier:
    """Composed barrier that is lower bounded by ``-epsilon``."""

    label: str
    value: float
    gradient_q: np.ndarray


@dataclass(frozen=True)
class CLBFModule:
    """Resolved numerical parameters of one hierarchical CLBF term."""

    obstacle_index: int
    epsilon: float
    lambda_: float
    kappa: float


def end_effector_barrier(
    p_ee, task_jacobian, obstacle, end_effector_radius, safe_distance, epsilon
):
    """Build ``B_i`` and ``dB_i/dq`` for one circular obstacle.

    Given the raw, unsafe-positive clearance violation

    ``b_i = r_safe - ||p_ee - p_obs,i||``,

    this implements the composed barrier requested in the paper workflow:

    ``B_i = max(b_i, 0)^2 - epsilon_i``.

    Hence ``B_i >= -epsilon_i`` everywhere, while the unsafe set represented
    by this CLBF term is ``b_i > sqrt(epsilon_i)``.  Only the x/y rows of the
    planar task Jacobian participate.
    """
    p_ee = np.asarray(p_ee, dtype=float).reshape(-1)
    task_jacobian = np.asarray(task_jacobian, dtype=float)
    if p_ee.shape[0] < 2 or task_jacobian.ndim != 2 or task_jacobian.shape[0] < 2:
        raise ValueError("end-effector position and task Jacobian must contain x/y")
    delta = p_ee[:2] - np.asarray(obstacle.center, dtype=float)
    distance = float(np.linalg.norm(delta))
    if epsilon <= 0.0:
        raise ValueError("CLBF barrier epsilon must be positive")
    clearance_radius = float(obstacle.radius + end_effector_radius + safe_distance)
    if distance < 1e-12:
        # The distance gradient is undefined at the obstacle centre.  Zero is
        # deterministic; the link-wise CBF hard constraints still provide
        # the recovery direction in this degenerate configuration.
        raw_gradient = np.zeros(task_jacobian.shape[1])
    else:
        raw_gradient = -(delta / distance) @ task_jacobian[:2, :]
    raw_value = clearance_radius - distance
    if raw_value <= 0.0:
        value = -float(epsilon)
        gradient = np.zeros(task_jacobian.shape[1])
    else:
        value = raw_value * raw_value - float(epsilon)
        gradient = 2.0 * raw_value * raw_gradient
    return EndEffectorBarrier(
        label="",
        value=float(value),
        gradient_q=np.asarray(gradient, dtype=float),
    )


def boundary_distance_squared(
    target_position, obstacle, end_effector_radius, safe_distance, epsilon
):
    """Compute c_{i,3} on the composed-barrier boundary ``B_i=0``."""
    target_position = np.asarray(target_position, dtype=float)
    clearance_radius = float(obstacle.radius + end_effector_radius + safe_distance)
    radius = clearance_radius - float(np.sqrt(epsilon))
    if radius <= 0.0:
        raise ValueError(
            "CLBF epsilon is too large: max(b, 0)^2 - epsilon has no positive unsafe set"
        )
    distance = float(np.linalg.norm(target_position - obstacle.center))
    if distance <= radius:
        raise ValueError(f"end-effector target is inside CLBF obstacle {obstacle.center.tolist()}")
    return (distance - radius) ** 2


def build_cumulative_clbfs(
    clf_value: float,
    clf_gradient_q,
    barriers,
    gamma: float,
    modules,
):
    """Build ``W_k = W_(k-1) + lambda_k*B_k + kappa_k`` in order."""
    if len(barriers) != len(modules):
        raise ValueError("each end-effector barrier must have one CLBF module")
    clf_gradient_q = np.asarray(clf_gradient_q, dtype=float)
    value = float(clf_value)
    gradient = clf_gradient_q.copy()
    results = []
    for barrier, module in zip(barriers, modules):
        value += module.lambda_ * barrier.value + module.kappa
        gradient += module.lambda_ * barrier.gradient_q
        results.append(
            CLBFResult(
                label=f"W_obs{module.obstacle_index}",
                value=float(value),
                gradient_q=gradient.copy(),
                gamma=float(gamma),
            )
        )
    return tuple(results)

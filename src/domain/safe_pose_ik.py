"""Multi-start null-space safe pose-IK search for the 7-DoF Panda."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..adapters.panda_kinematics import quaternion_orientation_error


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


def damped_pseudoinverse(jacobian: np.ndarray, damping: float) -> np.ndarray:
    """Right damped least-squares inverse for a 6-by-7 task Jacobian."""
    jacobian = np.asarray(jacobian, dtype=float)
    task_dimension = jacobian.shape[0]
    return jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + damping**2 * np.eye(task_dimension),
        np.eye(task_dimension),
    )


class SafePoseIKSolver:
    """Search pose IK and improve all safety barriers in its null space."""

    def __init__(self, config, kinematics, safety):
        self.config = config
        self.kinematics = kinematics
        self.safety = safety

    def _pose_errors(self, q):
        position, quaternion, jacobian = self.kinematics.pose_and_jacobian(q)
        position_error = self.config.target.position - position
        orientation_error = quaternion_orientation_error(
            self.config.target.quaternion, quaternion
        )
        return position_error, orientation_error, jacobian

    def _candidate(self, q, seed_index, iterations, require_safe):
        position_error, orientation_error, _ = self._pose_errors(q)
        barriers = self.safety.evaluate(q)
        position_norm = float(np.linalg.norm(position_error))
        orientation_norm = float(np.linalg.norm(orientation_error))
        minimum_barrier = barriers.minimum
        pose_reached = (
            position_norm <= self.config.target.position_tolerance
            and orientation_norm <= self.config.target.orientation_tolerance
        )
        safe = minimum_barrier >= 0.0
        if not pose_reached or (require_safe and not safe):
            return None
        return SafeIKCandidate(
            np.asarray(q, dtype=float).copy(),
            seed_index,
            iterations,
            position_norm,
            orientation_norm,
            minimum_barrier,
        )

    def solve(self, initial_q=None, *, require_safe: bool = True) -> SafeIKResult:
        """Evaluate every seed, then select the candidate nearest ``initial_q``."""
        ik = self.config.safe_ik
        rng = np.random.default_rng(ik.random_seed)
        lower, upper = self.kinematics.lower_limits, self.kinematics.upper_limits
        reference = (
            self.config.initial_q
            if initial_q is None
            else np.asarray(initial_q, dtype=float)
        )
        candidates = []
        for seed_index in range(ik.max_restarts):
            # Seed 0 starts from the actual initial configuration. Remaining
            # seeds retain the joint-limit uniform multi-start exploration.
            q = reference.copy() if seed_index == 0 else rng.uniform(lower, upper)
            consecutive_acceptable_steps = 0
            for iteration in range(1, ik.iterations_per_seed + 1):
                candidate = self._candidate(q, seed_index, iteration, require_safe)
                if candidate is not None:
                    consecutive_acceptable_steps += 1
                    if consecutive_acceptable_steps >= ik.convergence_hold_steps:
                        candidates.append(candidate)
                        break
                else:
                    consecutive_acceptable_steps = 0

                position_error, orientation_error, jacobian = self._pose_errors(q)
                # For DeltaQ = Qd * Qe^-1, twice the vector part is the
                # small-angle world-frame orientation correction.
                task_error = np.concatenate((position_error, 2.0 * orientation_error))
                primary_inverse = damped_pseudoinverse(jacobian, ik.damping)
                primary_velocity = primary_inverse @ (ik.pose_gain * task_error)

                null_projector = (
                    np.eye(self.config.dof) - np.linalg.pinv(jacobian) @ jacobian
                )
                if require_safe:
                    _, safety_gradient, _ = self.safety.softmin(
                        q, ik.softmin_temperature
                    )
                    safety_velocity = null_projector @ (
                        ik.safety_gradient_gain * safety_gradient
                    )
                else:
                    safety_velocity = np.zeros(self.config.dof)
                q = np.clip(
                    q + ik.integration_period * (primary_velocity + safety_velocity),
                    lower,
                    upper,
                )

        selected = (
            min(candidates, key=lambda item: np.linalg.norm(item.q - reference))
            if candidates
            else None
        )
        failure = (
            None
            if selected is not None
            else ("safe_pose_ik_not_found" if require_safe else "pose_ik_unreachable")
        )
        return SafeIKResult(tuple(candidates), selected, failure)

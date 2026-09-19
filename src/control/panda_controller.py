"""Sequential task-space CLF and hard-CBF trajectory controller for Panda."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import block_diag

from ..adapters.panda_kinematics import quaternion_orientation_error_components
from ..domain.types import QPProblem, SafetyDiagnostics


@dataclass(frozen=True)
class PandaControlStatus:
    success: bool
    stalled: bool
    candidates_exhausted: bool
    qp_infeasible: bool
    qp_solver_failed: bool
    stage_index: int
    stage_name: str
    stage_position_error: float
    joint_error: float | None
    position_error: float
    orientation_error: float
    minimum_barrier: float


class PandaTrajectoryController:
    """Track safe IK candidates and switch when a moving-average stall occurs."""

    def __init__(self, config, kinematics, safety, candidates, solver):
        if not candidates and config.trajectory.mode != "direct":
            raise ValueError("at least one safe IK candidate is required")
        self.config = config
        self.kinematics = kinematics
        self.safety = safety
        self.candidates = tuple(candidates)
        self.solver = solver
        self.candidate_index = 0
        self.candidate_switch_count = 0
        self.stage_frames = (
            config.trajectory.stage_frames
            if config.trajectory.mode == "sequential"
            else ()
        )
        self.stage_index = 0
        self.stage_transition_count = 0
        self._stage_transition_events: list[tuple[str, str, str]] = []
        self._candidate_stage_targets = tuple(
            tuple(
                self.kinematics.frame_point_and_jacobian(frame, candidate.q)[0]
                for frame in self.stage_frames
            )
            for candidate in self.candidates
        )
        self.previous_velocity = np.zeros(config.dof)
        window_samples = max(
            1,
            int(
                np.ceil(
                    config.trajectory.stop_window / config.trajectory.integration_period
                )
            ),
        )
        self.velocity_norms = deque(maxlen=window_samples)
        self.qp_infeasible_count = 0
        self.qp_solver_failure_count = 0
        self.qp_infeasible_stopped = False
        self.last_qp_solve_time = 0.0
        self.last_clf_slack = 0.0
        self.last_status: PandaControlStatus | None = None
        self.barrier_labels = self.safety.evaluate(config.initial_q).labels

    @property
    def target_q(self) -> np.ndarray:
        if not self.candidates:
            raise RuntimeError("direct mode without safe IK has no target configuration")
        return self.candidates[self.candidate_index].q

    @property
    def active_stage_name(self) -> str:
        return self._stage_name(self.stage_index)

    def _stage_name(self, stage_index: int) -> str:
        if stage_index < len(self.stage_frames):
            return f"{self.stage_frames[stage_index]}_position"
        return "end_effector_pose"

    def consume_stage_transition_events(self) -> tuple[tuple[str, str, str], ...]:
        """Return and clear stage transitions since the previous control step."""
        events = tuple(self._stage_transition_events)
        self._stage_transition_events.clear()
        return events

    def _pose_errors(self, q):
        position, quaternion, jacobian = self.kinematics.pose_and_jacobian(q)
        position_error = position - self.config.target.position
        orientation_scalar, orientation_error = quaternion_orientation_error_components(
            self.config.target.quaternion, quaternion
        )
        return (
            position_error,
            orientation_scalar,
            orientation_error,
            jacobian,
        )

    def _active_clf(self, q):
        """Return task-space CLF value, joint gradient, and position error."""
        if self.stage_index < len(self.stage_frames):
            frame = self.stage_frames[self.stage_index]
            position, jacobian = self.kinematics.frame_point_and_jacobian(frame, q)
            target = self._candidate_stage_targets[self.candidate_index][self.stage_index]
            error = position - target
            return 0.5 * float(error @ error), error @ jacobian, float(np.linalg.norm(error))

        position_error, eta, orientation_error, jacobian = self._pose_errors(q)
        weight = self.config.trajectory.orientation_weight
        value = 0.5 * float(position_error @ position_error) + 0.5 * weight * float(
            orientation_error @ orientation_error
        )
        # For DeltaQ = Qd * Qe^-1 and world angular velocity omega,
        # d(0.5 ||epsilon||^2)/dt = -0.5 eta epsilon^T omega.
        gradient = position_error @ jacobian[:3] - (
            0.5 * weight * eta * orientation_error @ jacobian[3:]
        )
        return value, gradient, float(np.linalg.norm(position_error))

    def _advance_reached_stages(self, q) -> None:
        while self.stage_index < len(self.stage_frames):
            _, _, position_error = self._active_clf(q)
            if position_error > self.config.trajectory.stage_position_tolerance:
                break
            previous_name = self.active_stage_name
            self.stage_index += 1
            self.stage_transition_count += 1
            self._stage_transition_events.append(
                (previous_name, self.active_stage_name, "target_reached")
            )
            self.velocity_norms.clear()

    def _status(
        self, q, qp_infeasible: bool = False, qp_solver_failed: bool = False
    ) -> PandaControlStatus:
        q = np.asarray(q, dtype=float)
        pose_position_error, _, pose_orientation_error, _ = self._pose_errors(q)
        position_error = float(np.linalg.norm(pose_position_error))
        orientation_error = float(np.linalg.norm(pose_orientation_error))
        _, _, stage_position_error = self._active_clf(q)
        joint_error = (
            None
            if not self.candidates
            else float(np.linalg.norm(q - self.target_q))
        )
        minimum_barrier = self.safety.evaluate(q).minimum
        success = (
            self.stage_index == len(self.stage_frames)
            and position_error <= self.config.target.position_tolerance
            and orientation_error <= self.config.target.orientation_tolerance
            and minimum_barrier >= 0.0
        )
        stalled = (
            len(self.velocity_norms) == self.velocity_norms.maxlen
            and float(np.mean(self.velocity_norms))
            <= self.config.trajectory.stop_velocity_tolerance
            and not success
        )
        return PandaControlStatus(
            success,
            stalled,
            stalled
            and (
                not self.candidates
                or self.candidate_index + 1 >= len(self.candidates)
            ),
            qp_infeasible,
            qp_solver_failed,
            self.stage_index,
            self.active_stage_name,
            stage_position_error,
            joint_error,
            position_error,
            orientation_error,
            minimum_barrier,
        )

    def _switch_candidate_if_stalled(self, q) -> bool:
        status = self._status(q)
        if not status.stalled or status.candidates_exhausted:
            self.last_status = status
            return False
        previous_stage_name = self.active_stage_name
        self.candidate_index += 1
        self.candidate_switch_count += 1
        self.stage_index = 0
        if previous_stage_name != self.active_stage_name:
            self._stage_transition_events.append(
                (previous_stage_name, self.active_stage_name, "candidate_switch")
            )
        self.velocity_norms.clear()
        self.previous_velocity.fill(0.0)
        self._advance_reached_stages(q)
        self.last_status = self._status(q)
        return True

    def get_velocity(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if self.qp_infeasible_stopped:
            return np.zeros(self.config.dof)
        self._advance_reached_stages(q)
        self._switch_candidate_if_stalled(q)
        if self.last_status is not None and (
            self.last_status.success or self.last_status.candidates_exhausted
        ):
            return np.zeros(self.config.dof)

        trajectory = self.config.trajectory
        clf_value, clf_gradient, _ = self._active_clf(q)
        barrier = self.safety.evaluate(q)
        lower_limits, upper_limits = (
            self.kinematics.lower_limits,
            self.kinematics.upper_limits,
        )

        rows = [np.concatenate((clf_gradient, [-1.0]))]
        lower = [-np.inf]
        upper = [-trajectory.clf_rate * clf_value]
        for gradient, value in zip(barrier.gradients, barrier.values):
            rows.append(np.concatenate((gradient, [0.0])))
            lower.append(-self.config.safety.alpha * value)
            upper.append(np.inf)
        for joint in range(self.config.dof):
            row = np.zeros(self.config.dof + 1)
            row[joint] = 1.0
            rows.append(row)
            lower.append(-self.config.safety.alpha * (q[joint] - lower_limits[joint]))
            upper.append(np.inf)
            rows.append(-row)
            lower.append(-self.config.safety.alpha * (upper_limits[joint] - q[joint]))
            upper.append(np.inf)

        # CLF relaxation remains nonnegative and cannot relax any CBF.
        slack_row = np.zeros(self.config.dof + 1)
        slack_row[-1] = 1.0
        rows.append(slack_row)
        lower.append(0.0)
        upper.append(np.inf)

        velocity_lower = -trajectory.velocity_limits
        velocity_upper = trajectory.velocity_limits
        delta = trajectory.acceleration_limits * trajectory.integration_period
        velocity_lower = np.maximum(velocity_lower, self.previous_velocity - delta)
        velocity_upper = np.minimum(velocity_upper, self.previous_velocity + delta)
        for joint in range(self.config.dof):
            row = np.zeros(self.config.dof + 1)
            row[joint] = 1.0
            rows.append(row)
            lower.append(velocity_lower[joint])
            upper.append(velocity_upper[joint])

        problem = QPProblem(
            block_diag(
                self.config.qp.joint_velocity_weight * np.eye(self.config.dof),
                np.array([[trajectory.clf_slack_weight]]),
            ),
            np.zeros(self.config.dof + 1),
            np.vstack(rows),
            np.asarray(lower),
            np.asarray(upper),
        )
        start = perf_counter()
        result = self.solver.solve(problem)
        self.last_qp_solve_time = perf_counter() - start
        if result is None:
            solver_status = str(getattr(self.solver, "last_status", "unknown")).lower()
            qp_infeasible = "infeasible" in solver_status
            if qp_infeasible:
                self.qp_infeasible_count += 1
            else:
                self.qp_solver_failure_count += 1
            self.qp_infeasible_stopped = True
            self.last_clf_slack = np.nan
            velocity = np.zeros(self.config.dof)
            self.velocity_norms.append(0.0)
            self.last_status = self._status(
                q,
                qp_infeasible=qp_infeasible,
                qp_solver_failed=not qp_infeasible,
            )
            return velocity

        velocity = np.asarray(result[: self.config.dof], dtype=float)
        self.previous_velocity = velocity.copy()
        self.last_clf_slack = float(result[-1])
        self.velocity_norms.append(float(np.linalg.norm(velocity)))
        self.last_status = self._status(q)
        return velocity

    def safety_diagnostics(self, q) -> SafetyDiagnostics:
        barrier = self.safety.evaluate(q)
        clf_value, _, _ = self._active_clf(np.asarray(q, dtype=float))
        return SafetyDiagnostics(
            barrier.labels,
            barrier.values,
            clf_value=clf_value,
        )

"""Sequential task-space CLF and hard-CBF trajectory controller for Panda."""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.linalg import block_diag

from ..domain.ik_contracts import SafeIKCandidate
from ..domain.interfaces import FramePoseKinematics, QPSolver
from ..domain.panda_config import PandaControllerConfig
from ..domain.panda_types import PandaControlStatus
from ..domain.rotations import (
    quaternion_orientation_error_components,
    rotation_to_wxyz,
)
from ..domain.safety_contracts import SafetyEvaluator
from ..domain.types import QPProblem, QPStatus, SafetyDiagnostics


class PandaTrajectoryController:
    """Track safe IK candidates; switch only on low speed without task progress."""

    def __init__(
        self,
        config: PandaControllerConfig,
        kinematics: FramePoseKinematics,
        safety: SafetyEvaluator,
        candidates: tuple[SafeIKCandidate, ...],
        solver: QPSolver,
    ):
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
        self._candidate_stage_orientations = (
            tuple(
                tuple(
                    rotation_to_wxyz(
                        self.kinematics.frame_pose_and_jacobian(frame, candidate.q)[0]
                    )
                    for frame in self.stage_frames
                )
                for candidate in self.candidates
            )
            if config.trajectory.stage_task == "pose"
            else ()
        )
        self.previous_command_velocity = np.zeros(config.dof)
        self.last_applied_velocity = np.zeros(config.dof)
        window_samples = max(
            1,
            int(
                np.ceil(
                    config.trajectory.stop_window / config.trajectory.integration_period
                )
            ),
        )
        self.applied_velocity_norms = deque(maxlen=window_samples)
        progress_samples = max(
            1,
            int(
                np.ceil(
                    config.trajectory.stop_progress_window
                    / config.trajectory.integration_period
                )
            ),
        )
        # N intervals need N+1 states. status() never advances this history.
        self._progress_best_errors = deque(maxlen=progress_samples + 1)
        self._best_task_error = float("inf")
        self.qp_infeasible_count = 0
        self.qp_solver_failure_count = 0
        self.qp_infeasible_stopped = False
        self.last_qp_solve_time = 0.0
        self.last_clf_slack = 0.0
        self.last_qp_status = "not_run"
        self.last_barrier = self.safety.evaluate(config.initial_q)
        self.last_barrier_q = config.initial_q.copy()
        if config.safety.self_collision_model == "rdf" and np.any(
            self.last_barrier.values < 0
        ):
            violations = {
                label: float(value)
                for label, value in zip(
                    self.last_barrier.labels, self.last_barrier.values
                )
                if value < 0
            }
            raise ValueError(f"unsafe initial trajectory state: {violations}")
        self.last_status: PandaControlStatus | None = None
        self.barrier_labels = self.last_barrier.labels

    @property
    def target_q(self) -> np.ndarray:
        if not self.candidates:
            raise RuntimeError(
                "direct mode without safe IK has no target configuration"
            )
        return self.candidates[self.candidate_index].q

    @property
    def active_stage_name(self) -> str:
        return self._stage_name(self.stage_index)

    def _stage_name(self, stage_index: int) -> str:
        if stage_index < len(self.stage_frames):
            return (
                f"{self.stage_frames[stage_index]}_{self.config.trajectory.stage_task}"
            )
        return "end_effector_pose"

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(self._stage_name(i) for i in range(len(self.stage_frames) + 1))

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

    def _intermediate_pose_errors(self, q, stage_index):
        rotation, position, jacobian = self.kinematics.frame_pose_and_jacobian(
            self.stage_frames[stage_index], q
        )
        target = self._candidate_stage_targets[self.candidate_index][stage_index]
        desired = self._candidate_stage_orientations[self.candidate_index][stage_index]
        eta, orientation_error = quaternion_orientation_error_components(
            desired, rotation_to_wxyz(rotation)
        )
        return position - target, eta, orientation_error, jacobian

    def stage_error_norms(self, q, stage_index):
        """Intermediate errors relative to the active IK; orientation is optional."""
        if self.config.trajectory.stage_task == "pose":
            position, _, orientation, _ = self._intermediate_pose_errors(q, stage_index)
            return float(np.linalg.norm(position)), float(np.linalg.norm(orientation))
        position, _ = self.kinematics.frame_point_and_jacobian(
            self.stage_frames[stage_index], q
        )
        target = self._candidate_stage_targets[self.candidate_index][stage_index]
        return float(np.linalg.norm(position - target)), None

    @staticmethod
    def _pose_clf(position_error, eta, orientation_error, jacobian, weight):
        value = 0.5 * float(position_error @ position_error) + 0.5 * weight * float(
            orientation_error @ orientation_error
        )
        # For DeltaQ = Qd * Qe^-1 and world angular velocity omega,
        # d(0.5 ||epsilon||^2)/dt = -0.5 eta epsilon^T omega.
        gradient = position_error @ jacobian[:3] - (
            0.5 * weight * eta * orientation_error @ jacobian[3:]
        )
        return value, gradient, float(np.linalg.norm(position_error))

    def _active_clf(self, q):
        """Return task-space CLF value, joint gradient, and position error."""
        if self.stage_index < len(self.stage_frames):
            if self.config.trajectory.stage_task == "pose":
                return self._pose_clf(
                    *self._intermediate_pose_errors(q, self.stage_index),
                    self.config.trajectory.stage_orientation_weight,
                )
            frame = self.stage_frames[self.stage_index]
            position, jacobian = self.kinematics.frame_point_and_jacobian(frame, q)
            target = self._candidate_stage_targets[self.candidate_index][
                self.stage_index
            ]
            error = position - target
            return (
                0.5 * float(error @ error),
                error @ jacobian,
                float(np.linalg.norm(error)),
            )

        return self._pose_clf(
            *self._pose_errors(q), self.config.trajectory.orientation_weight
        )

    def _advance_reached_stages(self, q) -> None:
        while self.stage_index < len(self.stage_frames):
            position_error, orientation_error = self.stage_error_norms(
                q, self.stage_index
            )
            if position_error > self.config.trajectory.stage_position_tolerance or (
                orientation_error is not None
                and orientation_error
                > self.config.trajectory.stage_orientation_tolerance
            ):
                break
            previous_name = self.active_stage_name
            self.stage_index += 1
            self.stage_transition_count += 1
            self._stage_transition_events.append(
                (previous_name, self.active_stage_name, "target_reached")
            )
            self._reset_stall_history()

    def _reset_stall_history(self) -> None:
        self.applied_velocity_norms.clear()
        self._progress_best_errors.clear()
        self._best_task_error = float("inf")

    def record_applied_velocity(self, velocity) -> None:
        """Record measured motion once after each completed integration interval.

        Command generation alone is not evidence of motion. Keep the command
        history used by the QP acceleration bounds separate from this feedback.
        """
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (self.config.dof,) or not np.isfinite(velocity).all():
            raise ValueError("applied velocity must be a finite vector of length dof")
        self.last_applied_velocity = velocity.copy()
        self.applied_velocity_norms.append(float(np.linalg.norm(velocity)))

    def _record_task_progress(self, q) -> None:
        """Record once per control tick, for the active candidate and stage."""
        if self.stage_index < len(self.stage_frames):
            position, orientation = self.stage_error_norms(q, self.stage_index)
            error = position / self.config.trajectory.stage_position_tolerance
            if orientation is not None:
                error = max(
                    error,
                    orientation / self.config.trajectory.stage_orientation_tolerance,
                )
        else:
            position, _, orientation, _ = self._pose_errors(q)
            error = max(
                float(np.linalg.norm(position)) / self.config.target.position_tolerance,
                float(np.linalg.norm(orientation))
                / self.config.target.orientation_tolerance,
            )
        # Repeated oscillations must not count as new progress toward the goal.
        self._best_task_error = min(self._best_task_error, error)
        self._progress_best_errors.append(self._best_task_error)

    def status(
        self, q, qp_infeasible: bool = False, qp_solver_failed: bool = False
    ) -> PandaControlStatus:
        q = np.asarray(q, dtype=float)
        pose_position_error, _, pose_orientation_error, _ = self._pose_errors(q)
        position_error = float(np.linalg.norm(pose_position_error))
        orientation_error = float(np.linalg.norm(pose_orientation_error))
        if self.stage_index < len(self.stage_frames):
            stage_position_error, stage_orientation_error = self.stage_error_norms(
                q, self.stage_index
            )
        else:
            stage_position_error, stage_orientation_error = (
                position_error,
                orientation_error,
            )
        joint_error = (
            None if not self.candidates else float(np.linalg.norm(q - self.target_q))
        )
        minimum_barrier = (
            self.last_barrier.minimum
            if np.array_equal(q, self.last_barrier_q)
            else self.safety.evaluate(q).minimum
        )
        success = (
            self.stage_index == len(self.stage_frames)
            and position_error <= self.config.target.position_tolerance
            and orientation_error <= self.config.target.orientation_tolerance
            and minimum_barrier >= -self.config.trajectory.success_barrier_tolerance
        )
        stalled = (
            len(self.applied_velocity_norms) == self.applied_velocity_norms.maxlen
            and float(np.mean(self.applied_velocity_norms))
            <= self.config.trajectory.stop_velocity_tolerance
            and len(self._progress_best_errors) == self._progress_best_errors.maxlen
            and self._progress_best_errors[0] - self._progress_best_errors[-1]
            <= self.config.trajectory.stop_progress_tolerance
            and not success
        )
        return PandaControlStatus(
            success,
            stalled,
            stalled
            and (
                not self.candidates or self.candidate_index + 1 >= len(self.candidates)
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
            stage_orientation_error=stage_orientation_error,
        )

    def _switch_candidate_if_stalled(self, q) -> bool:
        status = self.status(q)
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
        self._reset_stall_history()
        self.previous_command_velocity.fill(0.0)
        self._advance_reached_stages(q)
        self._record_task_progress(q)
        self.last_status = self.status(q)
        return True

    def get_velocity(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if self.qp_infeasible_stopped:
            return np.zeros(self.config.dof)
        self.last_barrier = self.safety.evaluate(q)
        self.last_barrier_q = q.copy()
        self.last_qp_status = "not_run"
        self.last_qp_solve_time = 0.0
        self._advance_reached_stages(q)
        self._record_task_progress(q)
        self._switch_candidate_if_stalled(q)
        if self.last_status is not None and (
            self.last_status.success or self.last_status.candidates_exhausted
        ):
            return np.zeros(self.config.dof)

        trajectory = self.config.trajectory
        clf_value, clf_gradient, _ = self._active_clf(q)
        barrier = self.last_barrier
        lower_limits, upper_limits = (
            self.kinematics.lower_limits,
            self.kinematics.upper_limits,
        )

        rows = [np.concatenate((clf_gradient, [-1.0]))]
        lower = [-np.inf]
        upper = [-trajectory.clf_rate * clf_value]
        for label, gradient, value in zip(
            barrier.labels, barrier.gradients, barrier.values
        ):
            rows.append(np.concatenate((gradient, [0.0])))
            gain = (
                self.config.safety.rdf.cbf_gain
                if self.config.safety.self_collision_model == "rdf"
                and label.startswith("self_")
                else self.config.safety.alpha
            )
            lower.append(-gain * value)
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
        velocity_lower = np.maximum(
            velocity_lower, self.previous_command_velocity - delta
        )
        velocity_upper = np.minimum(
            velocity_upper, self.previous_command_velocity + delta
        )
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
        result = self.solver.solve(problem)
        self.last_qp_solve_time = result.solve_time
        solver_status = result.status
        if solver_status == QPStatus.SOLVED and (
            np.asarray(result.x).shape != (self.config.dof + 1,)
            or not np.isfinite(result.x).all()
        ):
            solver_status = QPStatus.FAILED
        self.last_qp_status = solver_status.value
        if solver_status != QPStatus.SOLVED:
            qp_infeasible = solver_status == QPStatus.INFEASIBLE
            if qp_infeasible:
                self.qp_infeasible_count += 1
            else:
                self.qp_solver_failure_count += 1
            self.qp_infeasible_stopped = True
            self.last_clf_slack = np.nan
            velocity = np.zeros(self.config.dof)
            self.last_status = self.status(
                q,
                qp_infeasible=qp_infeasible,
                qp_solver_failed=not qp_infeasible,
            )
            return velocity

        velocity = np.asarray(result.x[: self.config.dof], dtype=float)
        self.previous_command_velocity = velocity.copy()
        self.last_clf_slack = float(result.x[-1])
        self.last_status = self.status(q)
        return velocity

    def safety_diagnostics(self, q) -> SafetyDiagnostics:
        barrier = self.last_barrier
        clf_value, _, _ = self._active_clf(np.asarray(q, dtype=float))
        return SafetyDiagnostics(
            barrier.labels,
            barrier.values,
            clf_value=clf_value,
        )

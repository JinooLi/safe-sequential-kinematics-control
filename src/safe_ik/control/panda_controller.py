"""Task-space or IK-configuration CLF with hard-CBF constraints for Panda."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence

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
        *,
        stage_selector: Callable[[np.ndarray], Sequence[str]] | None = None,
        advance_on_collision_stall: bool = False,
        collision_stall_solver: QPSolver | None = None,
        periodic_stage_selector: Callable[[np.ndarray], Sequence[str]] | None = None,
        periodic_reselection_period: float = 0.1,
        periodic_reselection_confirmations: int = 5,
        periodic_reselection_max_switches: int | None = None,
        periodic_reselection_lock_on_ee: bool = False,
        initial_stage_sequence: Sequence[str] | None = None,
    ):
        if not candidates and config.trajectory.mode != "direct":
            raise ValueError("at least one safe IK candidate is required")
        if stage_selector is not None and (
            config.trajectory.mode != "sequential" or len(candidates) != 1
        ):
            raise ValueError(
                "stage reselection requires sequential mode and one fixed IK"
            )
        if advance_on_collision_stall and (
            config.trajectory.mode != "sequential"
            or len(candidates) != 1
            or stage_selector is not None
        ):
            raise ValueError(
                "collision-stall advancement needs a fixed sequence and one IK"
            )
        if advance_on_collision_stall and (
            collision_stall_solver is None or collision_stall_solver is solver
        ):
            raise ValueError(
                "collision-stall advancement needs a separate diagnostic solver"
            )
        if periodic_stage_selector is not None and (
            config.trajectory.mode != "sequential"
            or len(candidates) != 1
            or stage_selector is not None
            or advance_on_collision_stall
        ):
            raise ValueError(
                "periodic reselection requires sequential mode, one fixed IK, "
                "and no other reselection policy"
            )
        if periodic_stage_selector is None and initial_stage_sequence is not None:
            raise ValueError("initial_stage_sequence requires periodic reselection")
        if not isinstance(periodic_reselection_lock_on_ee, (bool, np.bool_)):
            raise ValueError("periodic EE lock must be a boolean")
        if periodic_reselection_lock_on_ee and periodic_stage_selector is None:
            raise ValueError("a periodic EE lock requires periodic reselection")
        if periodic_reselection_max_switches is not None:
            if periodic_stage_selector is None:
                raise ValueError(
                    "a periodic switch limit requires periodic reselection"
                )
            if (
                isinstance(periodic_reselection_max_switches, (bool, np.bool_))
                or not isinstance(periodic_reselection_max_switches, (int, np.integer))
                or periodic_reselection_max_switches < 0
            ):
                raise ValueError("periodic max switches must be a nonnegative integer")
        self.config = config
        self.kinematics = kinematics
        self.safety = safety
        self.candidates = tuple(candidates)
        self.solver = solver
        self.advance_on_collision_stall = bool(advance_on_collision_stall)
        self.collision_stall_solver = collision_stall_solver
        self._collision_stall_events: list[dict] = []
        self._control_call_index = -1
        self.candidate_index = 0
        self.candidate_switch_count = 0
        self.stage_frames = (
            config.trajectory.stage_frames
            if config.trajectory.mode == "sequential"
            else ()
        )
        self.stage_selector = stage_selector
        self.periodic_stage_selector = periodic_stage_selector
        self.periodic_reselection_count = 0
        self.periodic_reselection_lock_on_ee = bool(periodic_reselection_lock_on_ee)
        self.periodic_ee_locked = False
        self.periodic_ee_lock_event: dict | None = None
        self.periodic_reselection_max_switches = (
            None
            if periodic_reselection_max_switches is None
            else int(periodic_reselection_max_switches)
        )
        self._periodic_stage_selection_events: list[dict] = []
        self._periodic_proposal: tuple[str, ...] | None = None
        self._periodic_streak = 0
        self.active_sequence = tuple(self.stage_frames)
        self._sequence_position = 0
        if periodic_stage_selector is not None:
            period = float(periodic_reselection_period)
            dt = float(config.trajectory.integration_period)
            ticks = period / dt
            if (
                not np.isfinite(period)
                or period <= 0
                or not np.isfinite(ticks)
                or ticks < 1
                or not np.isclose(ticks, round(ticks), rtol=0.0, atol=1e-9)
            ):
                raise ValueError(
                    "periodic reselection period must be a positive integer "
                    "multiple of the integration period"
                )
            if (
                isinstance(periodic_reselection_confirmations, (bool, np.bool_))
                or not isinstance(periodic_reselection_confirmations, (int, np.integer))
                or periodic_reselection_confirmations < 1
            ):
                raise ValueError("periodic confirmations must be a positive integer")
            if len(set(self.stage_frames)) != len(self.stage_frames):
                raise ValueError("periodic reselection needs a unique frame catalog")
            if initial_stage_sequence is None:
                raise ValueError(
                    "periodic reselection needs an initial selected sequence"
                )
            self._periodic_ticks = int(round(ticks))
            self.periodic_reselection_period = period
            self.periodic_reselection_confirmations = int(
                periodic_reselection_confirmations
            )
            self.active_sequence = self._validate_periodic_sequence(
                initial_stage_sequence
            )
        self.last_completed_link = 0
        self._stage_selector_initialized = False
        self._stage_selection_events: list[dict] = []
        if stage_selector is not None:
            try:
                link_numbers = tuple(
                    int(frame.removeprefix("link")) if frame.startswith("link") else 0
                    for frame in self.stage_frames
                )
            except ValueError as error:
                raise ValueError(
                    "stage reselection needs numbered link frames"
                ) from error
            if (
                any(number <= 0 for number in link_numbers)
                or tuple(sorted(set(link_numbers))) != link_numbers
            ):
                raise ValueError("stage reselection needs unique ascending link frames")
            self._stage_link_numbers = dict(zip(self.stage_frames, link_numbers))
        self.stage_index = (
            self.stage_frames.index(self.active_sequence[0])
            if periodic_stage_selector is not None and self.active_sequence
            else len(self.stage_frames) if periodic_stage_selector is not None else 0
        )
        self.periodic_initial_direct = (
            periodic_stage_selector is not None and not self.active_sequence
        )
        if self.periodic_initial_direct:
            self._lock_periodic_ee("initial_direct")
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
        self.last_qp_message = ""
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
        if self.config.trajectory.mode == "joint_clf":
            return "joint_configuration"
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

    def consume_stage_selection_events(self) -> tuple[dict, ...]:
        """Return decisions made initially and after intermediate target reaches.

        An empty selected action, or an action containing only completed/earlier
        links, commits to the final EE stage. That stage never invokes selection.
        """
        events = tuple(self._stage_selection_events)
        self._stage_selection_events.clear()
        return events

    def consume_collision_stall_events(self) -> tuple[dict, ...]:
        """Return all diagnostic checks, including checks that did not skip.

        control_call_index identifies the corresponding command-time entry in
        a rollout. A skipped target is never reported as a reached target.
        """
        events = tuple(self._collision_stall_events)
        self._collision_stall_events.clear()
        return events

    def consume_periodic_stage_selection_events(self) -> tuple[dict, ...]:
        """Return scheduled polls before any EE lock, including capped proposals.

        The initial selection is supplied by the caller and is not a poll.
        ``control_call_index * integration_period`` is simulated time; neither
        inference wall time nor stage transitions change the polling schedule.
        """
        events = tuple(self._periodic_stage_selection_events)
        self._periodic_stage_selection_events.clear()
        return events

    def _validate_periodic_sequence(self, frames: Sequence[str]) -> tuple[str, ...]:
        """Require an ordered catalog subsequence; the empty sequence is Direct."""
        chosen = tuple(frames)
        unknown = set(chosen).difference(self.stage_frames)
        if unknown:
            raise ValueError(
                f"periodic selector returned unknown link frames: {sorted(unknown)}"
            )
        indices = tuple(self.stage_frames.index(frame) for frame in chosen)
        if tuple(sorted(set(indices))) != indices:
            raise ValueError(
                "periodic selector requires unique frames in catalog order"
            )
        return chosen

    def _lock_periodic_ee(self, reason: str) -> None:
        """Make the first EE entry absorbing without adding synthetic poll events."""
        if (
            not self.periodic_reselection_lock_on_ee
            or self.periodic_ee_locked
            or self.stage_index != len(self.stage_frames)
        ):
            return
        self.periodic_ee_locked = True
        self._periodic_proposal = None
        self._periodic_streak = 0
        index = max(0, self._control_call_index)
        self.periodic_ee_lock_event = dict(
            control_call_index=index,
            simulated_time=index * self.config.trajectory.integration_period,
            reason=reason,
            initial_direct=self.periodic_initial_direct,
        )

    def _poll_periodic_sequence(self, q) -> None:
        if (
            self.periodic_stage_selector is None
            or self.periodic_ee_locked
            or self._control_call_index <= 0
            or self._control_call_index % self._periodic_ticks
        ):
            return
        if self.status(q).success:
            return
        previous_stage = self.active_stage_name
        previous_sequence = self.active_sequence
        previous_candidate = self._periodic_proposal
        previous_streak = self._periodic_streak
        switch_count_before = self.periodic_reselection_count
        chosen = self._validate_periodic_sequence(
            self.periodic_stage_selector(np.asarray(q, dtype=float).copy())
        )
        if chosen == self.active_sequence:
            self._periodic_proposal = None
            self._periodic_streak = 0
        elif chosen == self._periodic_proposal:
            self._periodic_streak += 1
        else:
            self._periodic_proposal = chosen
            self._periodic_streak = 1
        confirmation_streak = self._periodic_streak
        confirmed = confirmation_streak >= self.periodic_reselection_confirmations
        limit_exhausted = (
            self.periodic_reselection_max_switches is not None
            and switch_count_before >= self.periodic_reselection_max_switches
        )
        accepted = confirmed and not limit_exhausted
        entry_stage = self.active_stage_name
        if accepted:
            # Compare whole actions, not the unfinished suffix. Re-entering an
            # earlier link is intentional; normal reached checks may skip it.
            self.active_sequence = chosen
            self._sequence_position = 0
            self.stage_index = (
                self.stage_frames.index(chosen[0]) if chosen else len(self.stage_frames)
            )
            self.periodic_reselection_count += 1
            self._periodic_proposal = None
            self._periodic_streak = 0
            self._stage_transition_events.append(
                (previous_stage, self.active_stage_name, "sequence_switch")
            )
            entry_stage = self.active_stage_name
            self._reset_stall_history()
            self._advance_reached_stages(q)
            self._lock_periodic_ee("accepted_sequence_ee")
        self._periodic_stage_selection_events.append(
            dict(
                control_call_index=self._control_call_index,
                simulated_time=(
                    self._control_call_index * self.config.trajectory.integration_period
                ),
                q=np.asarray(q, dtype=float).tolist(),
                current_sequence_before=list(previous_sequence),
                current_sequence_after=list(self.active_sequence),
                proposed_sequence=list(chosen),
                candidate_before=(
                    None if previous_candidate is None else list(previous_candidate)
                ),
                candidate_after=(
                    None
                    if self._periodic_proposal is None
                    else list(self._periodic_proposal)
                ),
                streak_before=previous_streak,
                streak_after=self._periodic_streak,
                confirmation_streak=confirmation_streak,
                accepted=accepted,
                max_switches=self.periodic_reselection_max_switches,
                switch_count_before=switch_count_before,
                switch_count_after=self.periodic_reselection_count,
                blocked_by_switch_limit=confirmed and limit_exhausted,
                lock_on_ee=self.periodic_reselection_lock_on_ee,
                ee_locked_after=self.periodic_ee_locked,
                previous_stage=previous_stage,
                entry_stage=entry_stage,
                next_stage=self.active_stage_name,
            )
        )

    def _select_next_stage(self, q, reason: str) -> None:
        chosen_frames = tuple(self.stage_selector(np.asarray(q, dtype=float).copy()))
        unknown = set(chosen_frames).difference(self._stage_link_numbers)
        if unknown:
            raise ValueError(
                f"stage selector returned unknown link frames: {sorted(unknown)}"
            )
        remaining = sorted(
            {
                frame
                for frame in chosen_frames
                if self._stage_link_numbers[frame] > self.last_completed_link
            },
            key=self._stage_link_numbers.__getitem__,
        )
        next_frame = remaining[0] if remaining else None
        self.stage_index = (
            self.stage_frames.index(next_frame)
            if next_frame is not None
            else len(self.stage_frames)
        )
        self._stage_selector_initialized = True
        self._stage_selection_events.append(
            {
                "reason": reason,
                "q": np.asarray(q, dtype=float).tolist(),
                "last_completed_link": self.last_completed_link,
                "chosen_frames": list(chosen_frames),
                "discarded_frames": [
                    frame
                    for frame in chosen_frames
                    if self._stage_link_numbers[frame] <= self.last_completed_link
                ],
                "next_frame": next_frame,
                "next_stage": self.active_stage_name,
            }
        )

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
        """Return active CLF value, joint gradient, and active error norm."""
        if self.config.trajectory.mode == "joint_clf":
            # Panda's bounded revolute joints use ordinary angle differences,
            # not wrapping across a physical joint limit. V has no 1/2 factor.
            error = np.asarray(q, dtype=float) - self.target_q
            return float(error @ error), 2.0 * error, float(np.linalg.norm(error))
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
        if self.stage_selector is not None and not self._stage_selector_initialized:
            self._select_next_stage(q, "initial_selection")
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
            if self.periodic_stage_selector is not None:
                self._sequence_position += 1
                self.stage_index = (
                    self.stage_frames.index(
                        self.active_sequence[self._sequence_position]
                    )
                    if self._sequence_position < len(self.active_sequence)
                    else len(self.stage_frames)
                )
            elif self.stage_selector is None:
                self.stage_index += 1
            else:
                self.last_completed_link = self._stage_link_numbers[
                    self.stage_frames[self.stage_index]
                ]
                self._select_next_stage(q, "target_reached")
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

        Command generation alone is not evidence of motion. Stall detection
        uses this feedback rather than the QP's requested velocity.
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
        self._advance_reached_stages(q)
        self._record_task_progress(q)
        self.last_status = self.status(q)
        return True

    def _build_qp_problem(self, q) -> QPProblem:
        """Build the unchanged CLF/hard-CBF problem at the cached safety state."""
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

        # Bound joint speed only; consecutive commands have no slew-rate limit.
        for joint in range(self.config.dof):
            row = np.zeros(self.config.dof + 1)
            row[joint] = 1.0
            rows.append(row)
            lower.append(-trajectory.velocity_limits[joint])
            upper.append(trajectory.velocity_limits[joint])

        return QPProblem(
            block_diag(
                self.config.qp.joint_velocity_weight * np.eye(self.config.dof),
                np.array([[trajectory.clf_slack_weight]]),
            ),
            np.zeros(self.config.dof + 1),
            np.vstack(rows),
            np.asarray(lower),
            np.asarray(upper),
        )

    def _advance_collision_stalled_stage(self, q) -> bool:
        """Skip one fixed intermediate target only when a collision CBF binds.

        This associates a stall with an active collision constraint; it does not
        establish that the constraint alone caused the stall. The diagnostic
        command is never applied and its solver cannot affect control warm starts.
        """
        if (
            not self.advance_on_collision_stall
            or self.qp_infeasible_stopped
            or self.stage_index >= len(self.stage_frames)
            or not self.status(q).stalled
        ):
            return False
        event = dict(
            control_call_index=self._control_call_index,
            q=q.tolist(),
            previous_stage=self.active_stage_name,
            next_stage=self.active_stage_name,
            skipped=False,
            diagnostic_status="not_run",
            diagnostic_message="",
            diagnostic_solve_time=0.0,
            diagnostic_valid=False,
            dual_threshold=1e-6,
            stall_statistics=dict(
                mean_applied_velocity_norm=float(np.mean(self.applied_velocity_norms)),
                velocity_samples=len(self.applied_velocity_norms),
                velocity_threshold=self.config.trajectory.stop_velocity_tolerance,
                best_error_at_window_start=float(self._progress_best_errors[0]),
                best_error_at_window_end=float(self._progress_best_errors[-1]),
                error_improvement=float(
                    self._progress_best_errors[0] - self._progress_best_errors[-1]
                ),
                progress_samples=len(self._progress_best_errors),
                progress_threshold=self.config.trajectory.stop_progress_tolerance,
            ),
            collision_constraints=[],
            active_collision_labels=[],
        )
        self._collision_stall_events.append(event)
        try:
            problem = self._build_qp_problem(q)
            diagnostic = self.collision_stall_solver.solve(problem)
            event["diagnostic_status"] = diagnostic.status.value
            event["diagnostic_message"] = diagnostic.message
            event["diagnostic_solve_time"] = float(diagnostic.solve_time)
            if diagnostic.status != QPStatus.SOLVED:
                return False
            x = np.asarray(diagnostic.x, dtype=float)
            dual = (
                None
                if diagnostic.dual is None
                else np.asarray(diagnostic.dual, dtype=float)
            )
            if x.shape != (self.config.dof + 1,) or not np.isfinite(x).all():
                event["diagnostic_message"] = "invalid diagnostic primal solution"
                return False
            if (
                dual is None
                or dual.shape != problem.lower.shape
                or not np.isfinite(dual).all()
            ):
                event["diagnostic_message"] = (
                    "missing or invalid diagnostic dual solution"
                )
                return False
            event["diagnostic_valid"] = True
            # Row zero is CLF. Only the following collision-CBF rows may
            # trigger advancement; joint limits and speed limits never do.
            for i, (label, h) in enumerate(
                zip(self.last_barrier.labels, self.last_barrier.values), start=1
            ):
                derivative = float(problem.A[i] @ x)
                gain_h = float(-problem.lower[i])
                residual = derivative + gain_h
                tolerance = (
                    self.config.qp.absolute_tolerance
                    + self.config.qp.relative_tolerance
                    * max(abs(derivative), abs(gain_h))
                )
                multiplier = float(-dual[i])
                active = (
                    abs(residual) <= tolerance and multiplier > event["dual_threshold"]
                )
                event["collision_constraints"].append(
                    dict(
                        label=label,
                        kind="self" if label.startswith("self_") else "obstacle",
                        h=float(h),
                        directional_derivative=derivative,
                        gain_times_h=gain_h,
                        residual=residual,
                        tolerance=tolerance,
                        lower_multiplier=multiplier,
                        active=bool(active),
                    )
                )
                if active:
                    event["active_collision_labels"].append(label)
        except Exception as error:
            event["diagnostic_status"] = "diagnostic_exception"
            event["diagnostic_message"] = f"{type(error).__name__}: {error}"
            event["diagnostic_valid"] = False
            return False
        if not event["active_collision_labels"]:
            return False
        previous_name = self.active_stage_name
        self.stage_index += 1
        self.stage_transition_count += 1
        event["skipped"] = True
        event["next_stage"] = self.active_stage_name
        self._stage_transition_events.append(
            (previous_name, self.active_stage_name, "collision_constrained_stall")
        )
        self._reset_stall_history()
        self._advance_reached_stages(q)
        self._record_task_progress(q)
        return True

    def get_velocity(self, q) -> np.ndarray:
        self._control_call_index += 1
        q = np.asarray(q, dtype=float)
        if self.qp_infeasible_stopped:
            return np.zeros(self.config.dof)
        self.last_barrier = self.safety.evaluate(q)
        self.last_barrier_q = q.copy()
        self.last_qp_status = "not_run"
        self.last_qp_solve_time = 0.0
        self._advance_reached_stages(q)
        # A normal reach wins over a proposal due at this same command state.
        self._lock_periodic_ee("reached_ee")
        self._poll_periodic_sequence(q)
        self._record_task_progress(q)
        self._advance_collision_stalled_stage(q)
        self._switch_candidate_if_stalled(q)
        if self.last_status is not None and (
            self.last_status.success or self.last_status.candidates_exhausted
        ):
            return np.zeros(self.config.dof)

        problem = self._build_qp_problem(q)
        result = self.solver.solve(problem)
        self.last_qp_solve_time = result.solve_time
        self.last_qp_message = result.message
        solver_status = result.status
        if solver_status == QPStatus.SOLVED and (
            np.asarray(result.x).shape != (self.config.dof + 1,)
            or not np.isfinite(result.x).all()
        ):
            solver_status = QPStatus.FAILED
            self.last_qp_message = "invalid_qp_solution: wrong shape or nonfinite x"
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

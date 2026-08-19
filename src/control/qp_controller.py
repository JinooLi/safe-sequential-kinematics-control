import numpy as np
from scipy.linalg import block_diag

from ..application.configuration import ControllerConfig
from ..domain.goal_configuration import GoalConfiguration, solve_goal_configurations
from ..domain.rdf import PlanarBernsteinRobot
from ..domain.safety import PlanarCapsuleRobot, obstacles_from_config
from ..domain.types import QPProblem, SafetyDiagnostics
from .solver import QPSolver


def wrap_to_pi(angle):
    """Wrap an angle to [-pi, pi) so the CLF uses the shortest yaw error."""
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def planar_task_jacobian(spatial_jacobian):
    """Extract [x, y, yaw] rows from Pinocchio's 6D spatial Jacobian.

    Pinocchio orders a LOCAL_WORLD_ALIGNED spatial Jacobian as
    [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z].
    A planar end-effector task therefore uses rows [0, 1, 5], not J[:3].
    """
    spatial_jacobian = np.asarray(spatial_jacobian, dtype=float)
    if spatial_jacobian.ndim != 2 or spatial_jacobian.shape[0] < 6:
        raise ValueError("spatial_jacobian must have shape (6, dof)")
    return spatial_jacobian[[0, 1, 5], :]


class QPController:
    def __init__(self, config: ControllerConfig, solver: QPSolver, verbose=True):
        """Initialize control logic from an already validated typed config."""
        self._solver = solver
        self.verbose = bool(verbose)
        self.dof = config.dof
        self.ee_obj = config.ee_target
        self.ee_orientation = config.ee_orientation
        self.control_mode = config.control_mode
        self.P = np.array(block_diag(config.qp.R, config.qp.Q))
        self.clf_slack_ratio = config.qp.d
        safety_config = config.safety
        self.safety_enabled = safety_config.enabled
        self.obstacles = obstacles_from_config(
            [
                {"pos": obstacle.position, "radius": obstacle.radius}
                for obstacle in config.obstacles
                if obstacle.enabled
            ]
        )
        self.safety_model = None
        if self.safety_enabled:
            lengths = safety_config.link_lengths
            representation = safety_config.representation
            if representation == "capsule":
                self.safety_model = PlanarCapsuleRobot(
                    lengths, safety_config.link_radius
                )
            elif representation == "bernstein":
                self.safety_model = PlanarBernsteinRobot.load(
                    safety_config.rdf_model_path
                )
            else:
                raise ValueError("safety representation must be 'capsule' or 'bernstein'")
            if self.safety_model.dof != self.dof:
                raise ValueError("safety link count must match controller dof")
        self.safe_distance = safety_config.safe_distance
        self.cbf_alpha = safety_config.alpha
        self.joint_limits = safety_config.joint_limits
        self.joint_limit_alpha = safety_config.joint_limit_alpha
        self.velocity_limit = safety_config.velocity_limit
        self.clf_rate = safety_config.clf_rate
        self.clf_weight = safety_config.clf_weight
        self.barrier_labels = [
            f"obs{obstacle_index + 1}_link{link_index + 1}"
            for obstacle_index, obstacle in enumerate(config.obstacles)
            if obstacle.enabled
            for link_index in range(self.dof)
        ]
        sequence = config.sequential_end_effector
        self.sequential_end_effector_enabled = sequence.enabled
        self.end_effector_index = 1 if sequence.enabled else self.dof
        self.goal_tolerance = sequence.goal_tolerance
        self.velocity_tolerance = sequence.velocity_tolerance
        self.barrier_tolerance = sequence.barrier_tolerance
        self.final_yaw_weight = sequence.final_yaw_weight
        self.link1_reverse_enabled = sequence.link1_reverse_enabled
        self.link1_stuck_cycles = sequence.link1_stuck_cycles
        self.link1_reverse_max_steps = sequence.link1_reverse_max_steps
        self.link1_reverse_weight = sequence.link1_reverse_weight
        self.link1_stuck_count = 0
        self.link1_reverse_active = False
        self.link1_reverse_target: float | None = None
        self.link1_reverse_steps = 0
        self.last_end_effector_switch_reason = None
        self.goal_configurations: tuple[GoalConfiguration, ...] = ()
        self.feasible_goal_configurations: tuple[GoalConfiguration, ...] = ()
        self.selected_goal_configuration: GoalConfiguration | None = None
        if self.sequential_end_effector_enabled:
            self.goal_configurations = solve_goal_configurations(
                self.ee_obj,
                self.ee_orientation,
                safety_config.link_lengths,
                self.obstacles,
                link_radius=safety_config.link_radius,
                safe_distance=safety_config.safe_distance,
                joint_limits=self.joint_limits,
            )
            self.feasible_goal_configurations = tuple(
                candidate for candidate in self.goal_configurations if not candidate.colliding
            )
            if not self.feasible_goal_configurations:
                raise ValueError("goal is infeasible: no collision-free IK configuration")

    def get_velocity(self, q, J, p_ee, ee_yaw):
        """작업공간 명령을 관절 속도로 바꾸기 위한 자리.

        Args:
            x: 작업공간에서 원하는 속도 또는 위치 오차.
            J: 현재 자세에서의 Jacobian.

        Notes:
            아직 구현되지 않았다. 보통 pseudo-inverse나 QP를 사용해
            q_dot을 계산하는 함수로 채우면 된다.
        """
        self._select_goal_configuration(q)
        if self._goal_reached(p_ee):
            reason = "link1 reverse target reached" if self.link1_reverse_active else "goal reached"
            self._clear_link1_reverse()
            self._advance_end_effector(reason)
            return np.zeros(self.dof)
        task_jacobian = planar_task_jacobian(J)
        task_state = np.array([p_ee[0], p_ee[1], ee_yaw], dtype=float)
        if self.link1_reverse_active:
            clf_value, clf_gradient_q = self._link1_reverse_clf_terms(q)
        else:
            clf_value, clf_gradient_q = self._task_clf_terms(task_state, task_jacobian)
        cbf_A, cbf_lower, _ = self._cbf_constraints(q)
        joint_limit_A, joint_limit_lower = self._joint_limit_cbf_constraints(q)
        # Vdot - relaxation <= -clf_rate * V
        rows = [np.concatenate((clf_gradient_q, [-1.0]))]
        lower = [-np.inf]
        upper = [-self.clf_rate * clf_value]

        if self.safety_enabled and self.safety_model is not None:
            for row, bound in zip(cbf_A, cbf_lower):
                rows.append(np.concatenate((row, [0.0])))
                lower.append(bound)
                upper.append(np.inf)

        # Keep q_i inside [q_min, q_max] with two zeroing CBFs per bounded
        # joint: h_lower = q_i-q_min and h_upper = q_max-q_i.  joint1 has a
        # null limit entry, so it deliberately contributes no constraint.
        for row, bound in zip(joint_limit_A, joint_limit_lower):
            rows.append(np.concatenate((row, [0.0])))
            lower.append(bound)
            upper.append(np.inf)

        # Bounded commanded joint velocity makes the safety QP well posed.
        for joint in range(self.dof):
            row = np.zeros(self.dof + 1)
            row[joint] = 1.0
            rows.append(row)
            lower.append(-self.velocity_limit)
            upper.append(self.velocity_limit)

        slack_row = np.zeros(self.dof + 1)
        slack_row[-1] = 1.0
        rows.append(slack_row)
        lower.append(0.0)
        upper.append(self.clf_slack_ratio * clf_value)

        A = np.vstack(rows)
        l = np.asarray(lower)
        u = np.asarray(upper)
        result = self._solver.solve(
            QPProblem(self.P, np.zeros(self.dof + 1), A, l, u)
        )
        if result is not None:
            input = result[: self.dof]
            self._handle_barrier_stop(q, input)
        else:
            if self.link1_reverse_active:
                self._clear_link1_reverse()
                self._advance_end_effector("link1 reverse QP infeasible")
            else:
                self._advance_end_effector("QP infeasible")
            input = np.zeros(self.dof)

        return input

    def _select_goal_configuration(self, q):
        """Choose the safe IK branch nearest the initial configuration once."""
        if self.selected_goal_configuration is not None or not self.sequential_end_effector_enabled:
            return
        q = np.asarray(q, dtype=float)
        self.selected_goal_configuration = min(
            self.feasible_goal_configurations,
            key=lambda candidate: np.linalg.norm(
                np.array([wrap_to_pi(angle - target) for angle, target in zip(q, candidate.q)])
            ),
        )
        if self.verbose:
            print(f"Selected sequential goal configuration: {self.selected_goal_configuration.name}")

    def active_target_position(self):
        """Return the position target for the currently active task endpoint."""
        if not self.sequential_end_effector_enabled:
            return self.ee_obj
        if self.selected_goal_configuration is None:
            raise RuntimeError("goal configuration has not been selected")
        return self.selected_goal_configuration.joint_positions[self.end_effector_index - 1]

    def _goal_reached(self, p_ee):
        return (
            self.sequential_end_effector_enabled
            and self.end_effector_index < self.dof
            and np.linalg.norm(np.asarray(p_ee, dtype=float)[:2] - self.active_target_position())
            <= self.goal_tolerance
        )

    def _uses_end_effector_orientation(self):
        """Use yaw throughout the final end-effector stage."""
        return not self.sequential_end_effector_enabled or self.end_effector_index >= self.dof

    def _task_clf_terms(self, task_state, task_jacobian):
        """Return CLF value and joint gradient for the active link endpoint."""
        task_state = np.asarray(task_state, dtype=float)
        position_error = task_state[:2] - self.active_target_position()
        if not self._uses_end_effector_orientation():
            # Intermediate sequential task points use position-only control.
            value = float(2.0 * (position_error @ position_error))
            gradient = 4.0 * position_error @ task_jacobian[:2]
        else:
            yaw_error = wrap_to_pi(task_state[2] - self.ee_orientation)
            yaw_weight = (
                self.final_yaw_weight if self.sequential_end_effector_enabled else 1.0
            )
            value = float(2.0 * (position_error @ position_error + yaw_weight * yaw_error**2))
            gradient = 4.0 * np.array(
                [position_error[0], position_error[1], yaw_weight * yaw_error]
            ) @ task_jacobian
        return self.clf_weight * value, self.clf_weight * gradient

    def _link1_reverse_clf_terms(self, q):
        """Return a lifted-angle CLF that drives joint1 around the other way."""
        if self.link1_reverse_target is None:
            raise RuntimeError("link1 reverse target has not been initialized")
        error = float(np.asarray(q, dtype=float)[0] - self.link1_reverse_target)
        value = self.clf_weight * self.link1_reverse_weight * error**2
        gradient = np.zeros(self.dof)
        gradient[0] = 2.0 * self.clf_weight * self.link1_reverse_weight * error
        return value, gradient

    def _start_link1_reverse(self, q):
        """Use the q1* ± 2π representation that starts in the opposite direction."""
        if self.selected_goal_configuration is None:
            raise RuntimeError("goal configuration has not been selected")
        q1 = float(np.asarray(q, dtype=float)[0])
        goal_q1 = float(self.selected_goal_configuration.q[0])
        normal_target = goal_q1 + 2.0 * np.pi * round((q1 - goal_q1) / (2.0 * np.pi))
        normal_direction = np.sign(normal_target - q1)
        if normal_direction == 0.0:
            normal_direction = 1.0
        self.link1_reverse_target = normal_target - 2.0 * np.pi * normal_direction
        self.link1_reverse_active = True
        self.link1_reverse_steps = 0
        self.link1_stuck_count = 0
        if self.verbose:
            print(f"Trying reverse link1 winding toward q1={self.link1_reverse_target:.3f}")

    def _clear_link1_reverse(self):
        self.link1_reverse_active = False
        self.link1_reverse_target = None
        self.link1_reverse_steps = 0
        self.link1_stuck_count = 0

    def _handle_barrier_stop(self, q, input):
        """Start or abandon the link1 reverse path while retaining hard CBFs."""
        if self.link1_reverse_active:
            self.link1_reverse_steps += 1
            if self._link1_joint_stopped_by_barrier(q, input):
                self._clear_link1_reverse()
                self._advance_end_effector("link1 reverse blocked by CBF")
            elif self.link1_reverse_steps >= self.link1_reverse_max_steps:
                self._clear_link1_reverse()
                self._advance_end_effector("link1 reverse timeout")
            return
        stopped = (
            self._link1_joint_stopped_by_barrier(q, input)
            if self.end_effector_index == 1 and self.link1_reverse_enabled
            else self._stopped_by_barrier(q, input)
        )
        if not stopped:
            self.link1_stuck_count = 0
            return
        if self.end_effector_index != 1 or not self.link1_reverse_enabled:
            self._advance_end_effector("stopped at CBF boundary")
            return
        self.link1_stuck_count += 1
        if self.link1_stuck_count >= self.link1_stuck_cycles:
            self._start_link1_reverse(q)

    def _link1_joint_stopped_by_barrier(self, q, input):
        return (
            abs(float(np.asarray(input, dtype=float)[0])) <= self.velocity_tolerance
            and self._barrier_is_active(q)
        )

    def _barrier_is_active(self, q):
        barrier_values = self._cbf_constraints(q)[2]
        return len(barrier_values) > 0 and np.min(barrier_values) <= self.barrier_tolerance

    def _stopped_by_barrier(self, q, input):
        """Detect a near-zero command caused at an active hard CBF boundary."""
        if not self.sequential_end_effector_enabled:
            return False
        if np.linalg.norm(input) > self.velocity_tolerance:
            return False
        return self._barrier_is_active(q)

    def _advance_end_effector(self, reason):
        """Advance once, unless the final link is already the active task point."""
        if not self.sequential_end_effector_enabled or self.end_effector_index >= self.dof:
            return False
        previous = self.end_effector_index
        self.end_effector_index += 1
        self.last_end_effector_switch_reason = reason
        if self.verbose:
            print(
                f"Switching task end-effector: link {previous} -> link "
                f"{self.end_effector_index} ({reason})"
            )
        return True

    def _cbf_constraints(self, q):
        """Return CBF rows, lower bounds, and barrier values for the current q."""
        if not self.safety_enabled or self.safety_model is None:
            return np.empty((0, self.dof)), np.empty(0), np.empty(0)
        return self.safety_model.cbf_constraints(
            np.asarray(q, dtype=float),
            self.obstacles,
            self.safe_distance,
            self.cbf_alpha,
        )

    def _joint_limit_cbf_constraints(self, q):
        """Return hard zeroing-CBF rows for every finite joint range.

        For ``h_lower = q_i-q_min`` and ``h_upper = q_max-q_i``, enforce
        ``h_dot + alpha*h >= 0``.  The returned rows have the common form
        ``A q_dot >= lower`` and remain active even if obstacle safety is off.
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (self.dof,):
            raise ValueError(f"q must have shape ({self.dof},)")
        rows, lower = [], []
        for joint, limit in enumerate(self.joint_limits):
            if limit is None:
                continue
            q_min, q_max = limit
            lower_row = np.zeros(self.dof)
            lower_row[joint] = 1.0
            rows.append(lower_row)
            lower.append(-self.joint_limit_alpha * (q[joint] - q_min))

            upper_row = np.zeros(self.dof)
            upper_row[joint] = -1.0
            rows.append(upper_row)
            lower.append(-self.joint_limit_alpha * (q_max - q[joint]))
        if not rows:
            return np.empty((0, self.dof)), np.empty(0)
        return np.vstack(rows), np.asarray(lower)

    def safety_diagnostics(self, q, J=None, p_ee=None, ee_yaw=None) -> SafetyDiagnostics:
        """Return current named barrier values for recording and display.

        Each value is ``h = f_link(p_obstacle, q) - r_obstacle -
        safe_distance``.  Positive values are safe; zero is the configured
        safety boundary; negative values mean that the margin is violated.
        """
        barrier_values = self._cbf_constraints(q)[2]
        if J is None or p_ee is None or ee_yaw is None:
            return SafetyDiagnostics(tuple(self.barrier_labels), barrier_values)
        self._select_goal_configuration(q)
        task_state = np.array([p_ee[0], p_ee[1], ee_yaw], dtype=float)
        task_jacobian = planar_task_jacobian(J)
        if self.link1_reverse_active:
            clf_value, _ = self._link1_reverse_clf_terms(q)
        else:
            clf_value, _ = self._task_clf_terms(task_state, task_jacobian)
        return SafetyDiagnostics(
            tuple(self.barrier_labels),
            barrier_values,
            clf_value=clf_value,
        )

    def zero_input(self):
        """모든 관절 actuator에 0 명령을 보낸다.

        Returns:
            np.ndarray: shape (dof,)의 zero vector.

        Usage:
            시뮬레이션 연결과 viewer 동작만 확인할 때 안전한 기본 입력으로 쓴다.
            현재 MuJoCo scene의 actuator가 velocity actuator이므로 이 값은
            목표 관절 속도 0 rad/s를 의미한다.
        """
        return np.zeros(self.dof)

    def clf(self, task_state):
        position_error = np.asarray(task_state[:2], dtype=float) - self.ee_obj
        yaw_error = wrap_to_pi(task_state[2] - self.ee_orientation)
        return float(2 * (position_error @ position_error + yaw_error * yaw_error))

    def dclf(self, task_state):
        position_error = np.asarray(task_state[:2], dtype=float) - self.ee_obj
        yaw_error = wrap_to_pi(task_state[2] - self.ee_orientation)
        return 4 * np.array([position_error[0], position_error[1], yaw_error])

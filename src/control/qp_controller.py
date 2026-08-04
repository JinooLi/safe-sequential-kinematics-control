import numpy as np
from scipy.linalg import block_diag

from ..application.configuration import ControllerConfig
from ..domain.clbf import (
    CLBFModule,
    boundary_distance_squared,
    build_cumulative_clbfs,
    end_effector_barrier,
)
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
    def __init__(self, config: ControllerConfig, solver: QPSolver):
        """Initialize control logic from an already validated typed config."""
        self._solver = solver
        self.dof = config.dof
        self.ee_obj = config.ee_target
        self.ee_orientation = config.ee_orientation
        self.control_mode = config.control_mode
        self.P = np.array(block_diag(config.qp.R, config.qp.Q))
        self.clf_slack_ratio = config.qp.d
        safety_config = config.safety
        self.safety_enabled = safety_config.enabled
        self.active_obstacle_indices = tuple(
            index
            for index, obstacle in enumerate(config.obstacles, start=1)
            if obstacle.enabled
        )
        self.obstacles = obstacles_from_config(
            [
                {"pos": obstacle.position, "radius": obstacle.radius}
                for obstacle in config.obstacles
                if obstacle.enabled
            ]
        )
        self.obstacles_by_index = dict(zip(self.active_obstacle_indices, self.obstacles))
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
        self.velocity_limit = safety_config.velocity_limit
        self.clf_rate = safety_config.clf_rate
        self.barrier_labels = [
            f"obs{obstacle_index + 1}_link{link_index + 1}"
            for obstacle_index, obstacle in enumerate(config.obstacles)
            if obstacle.enabled
            for link_index in range(self.dof)
        ]
        self.clbf_gamma = config.clbf.gamma
        self.clbf_end_effector_radius = config.clbf.end_effector_radius
        self.clbf_modules = self._resolve_clbf_modules(config)
        self.clbf_labels = tuple(f"W_obs{module.obstacle_index}" for module in self.clbf_modules)

    def get_velocity(self, q, J, p_ee, ee_yaw):
        """작업공간 명령을 관절 속도로 바꾸기 위한 자리.

        Args:
            x: 작업공간에서 원하는 속도 또는 위치 오차.
            J: 현재 자세에서의 Jacobian.

        Notes:
            아직 구현되지 않았다. 보통 pseudo-inverse나 QP를 사용해
            q_dot을 계산하는 함수로 채우면 된다.
        """
        task_jacobian = planar_task_jacobian(J)
        task_state = np.array([p_ee[0], p_ee[1], ee_yaw], dtype=float)

        # Decision variable: [q_dot (dof), relaxation].
        clf_value = self.clf(task_state)
        clf_gradient_q = self.dclf(task_state) @ task_jacobian
        cbf_A, cbf_lower, _ = self._cbf_constraints(q)
        if self.control_mode == "clf":
            # Vdot - relaxation <= -clf_rate * V
            rows = [np.concatenate((clf_gradient_q, [-1.0]))]
            lower = [-np.inf]
            upper = [-self.clf_rate * clf_value]
        else:
            # Use only the final cumulative CLBF: W_M_dot <= -gamma * W_M.
            clbf = self._clbf_results(clf_value, clf_gradient_q, p_ee, task_jacobian)[-1]
            rows = [np.concatenate((clbf.gradient_q, [0.0]))]
            lower = [-np.inf]
            upper = [-clbf.gamma * clbf.value]

        if self.safety_enabled and self.safety_model is not None:
            for row, bound in zip(cbf_A, cbf_lower):
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

        # CLBF rows are never relaxed. The legacy CLF uses its configured slack.
        slack_row = np.zeros(self.dof + 1)
        slack_row[-1] = 1.0
        rows.append(slack_row)
        if self.control_mode == "clf":
            lower.append(0.0)
            upper.append(self.clf_slack_ratio * clf_value)
        else:
            lower.append(0.0)
            upper.append(0.0)

        A = np.vstack(rows)
        l = np.asarray(lower)
        u = np.asarray(upper)
        result = self._solver.solve(
            QPProblem(self.P, np.zeros(self.dof + 1), A, l, u)
        )
        if result is not None:
            input = result[: self.dof]
        else:
            print("QP failed")
            input = np.zeros(self.dof)

        return input

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

    def _resolve_clbf_modules(self, config):
        """Resolve recursive CLBF parameters in obstacle priority order."""
        if self.control_mode != "clbf":
            return ()
        modules = []
        cumulative_s = 0.0
        # V = 2 ||e||^2, hence c1 = 2 in the paper's notation.
        c1 = 2.0
        for barrier_config in config.clbf.barriers:
            obstacle = self.obstacles_by_index[barrier_config.obstacle_index]
            epsilon = barrier_config.epsilon
            lambda_ = barrier_config.lambda_
            kappa = barrier_config.kappa
            if barrier_config.auto_from_previous:
                # The composed barrier has B >= -epsilon by construction.
                # Reuse epsilon_1, so this second term needs no separately
                # tuned epsilon value in controller.yaml.
                epsilon = modules[0].epsilon
                c_i3 = boundary_distance_squared(
                    self.ee_obj,
                    obstacle,
                    self.clbf_end_effector_radius,
                    self.safe_distance,
                    epsilon,
                )
                kappa = cumulative_s - c1 * c_i3
                if kappa <= 0.0:
                    raise ValueError(
                        "automatic CLBF parameter construction produced kappa <= 0; "
                        "increase the preceding epsilon*lambda-kappa margin or change barrier order"
                    )
                lambda_ = kappa / epsilon
            module = CLBFModule(
                barrier_config.obstacle_index,
                float(epsilon),
                float(lambda_),
                float(kappa),
            )
            cumulative_s += module.epsilon * module.lambda_ - module.kappa
            modules.append(module)
        return tuple(modules)

    def _clbf_results(self, clf_value, clf_gradient_q, p_ee, task_jacobian):
        if self.control_mode != "clbf":
            return ()
        barriers = []
        for module in self.clbf_modules:
            raw_barrier = end_effector_barrier(
                p_ee,
                task_jacobian,
                self.obstacles_by_index[module.obstacle_index],
                self.clbf_end_effector_radius,
                self.safe_distance,
                module.epsilon,
            )
            barriers.append(
                type(raw_barrier)(
                    label=f"obs{module.obstacle_index}",
                    value=raw_barrier.value,
                    gradient_q=raw_barrier.gradient_q,
                )
            )
        return build_cumulative_clbfs(
            clf_value,
            clf_gradient_q,
            barriers,
            self.clbf_gamma,
            self.clbf_modules,
        )

    def safety_diagnostics(self, q, J=None, p_ee=None, ee_yaw=None) -> SafetyDiagnostics:
        """Return current named barrier values for recording and display.

        Each value is ``h = f_link(p_obstacle, q) - r_obstacle -
        safe_distance``.  Positive values are safe; zero is the configured
        safety boundary; negative values mean that the margin is violated.
        """
        barrier_values = self._cbf_constraints(q)[2]
        if J is None or p_ee is None or ee_yaw is None:
            return SafetyDiagnostics(tuple(self.barrier_labels), barrier_values)
        task_state = np.array([p_ee[0], p_ee[1], ee_yaw], dtype=float)
        clf_value = self.clf(task_state)
        clf_gradient_q = self.dclf(task_state) @ planar_task_jacobian(J)
        clbfs = self._clbf_results(clf_value, clf_gradient_q, p_ee, planar_task_jacobian(J))
        return SafetyDiagnostics(
            tuple(self.barrier_labels),
            barrier_values,
            clf_value=clf_value,
            clbf_labels=tuple(module.label for module in clbfs),
            clbf_values=np.asarray([module.value for module in clbfs]),
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

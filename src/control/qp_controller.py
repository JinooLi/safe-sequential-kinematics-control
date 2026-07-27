import numpy as np
from scipy.linalg import block_diag

from ..application.configuration import ControllerConfig
from ..domain.rdf import PlanarBernsteinRobot
from ..domain.safety import PlanarCapsuleRobot, obstacles_from_config
from ..domain.types import QPProblem, SafetyDiagnostics
from .solver import QPSolver


class QPController:
    def __init__(self, config: ControllerConfig, solver: QPSolver):
        """Initialize control logic from an already validated typed config."""
        self._solver = solver
        self.dof = config.dof
        self.ee_obj = config.ee_target
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
        self.velocity_limit = safety_config.velocity_limit
        self.clf_rate = safety_config.clf_rate
        self.barrier_labels = [
            f"obs{obstacle_index + 1}_link{link_index + 1}"
            for obstacle_index in range(len(self.obstacles))
            for link_index in range(self.dof)
        ]

    def get_velocity(self, q, J, p_ee):
        """작업공간 명령을 관절 속도로 바꾸기 위한 자리.

        Args:
            x: 작업공간에서 원하는 속도 또는 위치 오차.
            J: 현재 자세에서의 Jacobian.

        Notes:
            아직 구현되지 않았다. 보통 pseudo-inverse나 QP를 사용해
            q_dot을 계산하는 함수로 채우면 된다.
        """
        J = J[:2]  # orientation과 z축 방향까지 자른다.
        p_ee = p_ee[:2]  # 어차피 2차원이니까 z축은 자른다.

        # Decision variable: [q_dot (dof), CLF relaxation].
        # Vdot - relaxation <= -clf_rate * V
        clf_row = np.concatenate((self.dclf(p_ee) @ J, [-1.0]))
        rows = [clf_row]
        lower = [-np.inf]
        upper = [-self.clf_rate * self.clf(p_ee)]

        if self.safety_enabled and self.safety_model is not None:
            cbf_A, cbf_lower, _ = self._cbf_constraints(q)
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

        # The relaxation is non-negative and at most d times the current CLF
        # value. CBF rows remain hard constraints and are never relaxed.
        slack_row = np.zeros(self.dof + 1)
        slack_row[-1] = 1.0
        rows.append(slack_row)
        lower.append(0.0)
        upper.append(self.clf_slack_ratio * self.clf(p_ee))

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

    def safety_diagnostics(self, q) -> SafetyDiagnostics:
        """Return current named barrier values for recording and display.

        Each value is ``h = f_link(p_obstacle, q) - r_obstacle -
        safe_distance``.  Positive values are safe; zero is the configured
        safety boundary; negative values mean that the margin is violated.
        """
        return SafetyDiagnostics(tuple(self.barrier_labels), self._cbf_constraints(q)[2])

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

    def clf(self, ee):
        return float(2 * (ee - self.ee_obj) @ (ee - self.ee_obj))

    def dclf(self, ee):
        return 4 * np.array(ee - self.ee_obj).T

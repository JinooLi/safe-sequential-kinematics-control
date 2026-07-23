import os

import numpy as np
import osqp
import scipy.sparse as sparse
import yaml


class Controller:
    def __init__(self, setting_addr):
        """controller.yaml에서 제어기 설정과 OSQP solver pool을 초기화한다.

        Args:
            setting_addr: solver_num, dof 값을 가진 YAML 파일 경로.
        """
        with open(setting_addr, "r") as f:
            data = yaml.safe_load(f)

        self.solver_num: int = data["solver_num"]
        self.dof: int = data["dof"]
        self.solver: list[osqp.OSQP] = []
        self.is_initialized: list[bool] = []
        for i in range(self.solver_num):
            self.solver.append(osqp.OSQP())
            self.is_initialized.append(False)

    def QP(
        self,
        i: int,
        P: np.ndarray,
        q: np.ndarray,
        A: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
    ):
        """OSQP로 quadratic program을 푼다.

        문제 형태:
            minimize 0.5 * x.T @ P @ x + q.T @ x
            subject to l <= A @ x <= u

        Args:
            i: 사용할 OSQP solver index. 0 <= i < solver_num 이어야 한다.
            P: 목적함수의 quadratic 행렬. shape은 (n, n).
            q: 목적함수의 linear 벡터. shape은 (n,) 또는 (n, 1).
            A: 제약 행렬. shape은 (m, n).
            l: 제약 하한 벡터. shape은 (m,) 또는 (m, 1).
            u: 제약 상한 벡터. shape은 (m,) 또는 (m, 1).

        Returns:
            tuple[bool, np.ndarray]: 성공 여부와 해 벡터. setup/update 단계에서
            예외가 나면 False와 zero vector를 반환한다.

        Notes:
            OSQP는 P/A는 sparse matrix를 받을 수 있지만 q/l/u는 1차원 dense
            vector를 기대한다. 같은 solver index를 반복 호출할 때는 기존 문제
            구조를 재사용하고 q/l/u/A.data만 갱신한다.
        """

        A = sparse.csc_matrix(A)  # type: ignore
        q = np.asarray(q).ravel()
        l = np.asarray(l).ravel()
        u = np.asarray(u).ravel()

        # OSQP 솔버 세팅 및 업데이트
        try:
            if not self.is_initialized[i]:
                P = sparse.csc_matrix(P)  # type: ignore
                self.solver[i].setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
                self.is_initialized[i] = True
            else:
                # 이 함수는 제어주기마다 값이 바뀔 때 사용한다.
                self.solver[i].update(q=q, l=l, u=u, Ax=A.data)
        except:
            return False, np.zeros(np.shape(q))

        # 연산 및 결과 반환

        result = self.solver[i].solve()

        if result.info.status == "solved":
            status: bool = True
        else:
            status: bool = False

        out = result.x

        return status, out

    def get_velocity(self, q, J, p_ee):
        """작업공간 명령을 관절 속도로 바꾸기 위한 자리.

        Args:
            x: 작업공간에서 원하는 속도 또는 위치 오차.
            J: 현재 자세에서의 Jacobian.

        Notes:
            아직 구현되지 않았다. 보통 pseudo-inverse나 QP를 사용해
            q_dot을 계산하는 함수로 채우면 된다.
        """
        dest = np.array([1, 2])
        J = J[:2]  # orientation과 z축 방향까지 자른다.
        p_ee = p_ee[:2]  # 어차피 2차원이니까 z축은 자른다.

        err = dest - p_ee
        input = J.T @ np.linalg.inv(J @ J.T + np.eye(2) * 0.01) @ err

        return input

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


if __name__ == "__main__":
    cur_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(cur_file_path)
    yaml_path = os.path.join(cur_dir, "../settings/controller.yaml")
    c = Controller(yaml_path)
    P = np.eye(3)
    q = np.zeros(3)
    A = np.eye(3)
    l = np.array([0, 1, 2])
    u = np.array([2, 2, 2])

    print(c.QP(1, P, q, A, l, u))

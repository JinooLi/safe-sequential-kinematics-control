import numpy as np
import pinocchio as pin


class RobotInterface:
    def __init__(self, urdf_path, ee_frame_names):
        """Pinocchio 로봇 모델과 end-effector frame을 초기화한다.

        Args:
            urdf_path: Pinocchio가 읽을 URDF 파일 경로.
            ee_frame_names: 위치와 Jacobian을 계산할 end-effector frame 이름 또는
                root에서 먼 순서의 frame 이름들.

        Notes:
            로봇공학용 Pinocchio 패키지가 설치되어 있어야 한다. Pinocchio는 URDF를 읽고 MuJoCo 모델과의 기하 일치는 application에서 검증한다.
        """
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        if isinstance(ee_frame_names, str):
            ee_frame_names = (ee_frame_names,)
        self.ee_frame_names = tuple(ee_frame_names)
        if not self.ee_frame_names:
            raise ValueError("at least one end-effector frame is required")
        self.ee_frame_ids = tuple(
            self.model.getFrameId(frame_name) for frame_name in self.ee_frame_names
        )
        missing = [
            name
            for name, frame_id in zip(self.ee_frame_names, self.ee_frame_ids)
            if frame_id >= self.model.nframes
        ]
        if missing:
            raise ValueError(f"end-effector frames not found in the URDF: {missing}")
        self._end_effector_index = len(self.ee_frame_ids) - 1
        self.nv = self.model.nv
        self._ee_yaw = 0.0

    def set_end_effector(self, link_index: int):
        """Select the 1-based link-end frame used for the next task Jacobian."""
        if not 1 <= link_index <= len(self.ee_frame_ids):
            raise ValueError(
                f"end-effector link index must be in [1, {len(self.ee_frame_ids)}]"
            )
        self._end_effector_index = link_index - 1

    def update_kinematics(self, q):
        """현재 관절 위치 q에서 end-effector 위치와 Jacobian을 계산한다.

        Args:
            q: Pinocchio model.nq 길이의 관절 위치 벡터.

        Returns:
            tuple[np.ndarray, np.ndarray]:
            end-effector의 world 위치 p_ee와 LOCAL_WORLD_ALIGNED 기준 6 x nv
            Jacobian J.

        Usage:
            MuJoCo의 mj_data.qpos.copy()를 넘겨 현재 자세의 kinematics를 업데이트한다.
            이 planar arm에서 위치 제어만 할 때는 J[:2, :]처럼 x/y 행만 사용할 수 있다.
        """
        # 순기구학 연산 (속도 dq 불필요)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # End-effector 위치 및 자코비안
        ee_frame_id = self.ee_frame_ids[self._end_effector_index]
        p_ee = self.data.oMf[ee_frame_id].translation
        rotation = self.data.oMf[ee_frame_id].rotation
        self._ee_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        J = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        return p_ee, J

    def end_effector_yaw(self):
        """Return the world-frame planar yaw from the latest kinematics update."""
        return self._ee_yaw

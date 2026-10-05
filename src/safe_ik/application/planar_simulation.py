import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from ..domain.interfaces import (
    KinematicsProvider,
    SimulationRecorder,
    VelocityController,
)
from ..domain.types import SimulationStep
from ..visualization.recording import plot, show_gui


def load_mujoco_model(mujoco_xml_path):
    """완전한 MuJoCo MJCF 파일을 로드한다.

    Args:
        mujoco_xml_path: 로봇, actuator, scene, camera, option을 모두 포함한
            MuJoCo XML 파일 경로.

    Returns:
        mujoco.MjModel: mujoco.MjData와 함께 mj_step()에 사용할 compiled model.

    Usage:
        model = load_mujoco_model("assets/robots/planar.xml")
        data = mujoco.MjData(model)
    """
    return mujoco.MjModel.from_xml_path(str(mujoco_xml_path))


def _use_fixed_camera_if_available(viewer, mj_model, camera_name):
    """viewer가 지정한 fixed camera를 사용하도록 설정한다.

    Args:
        viewer: mujoco.viewer.launch_passive()가 반환한 viewer.
        mj_model: camera가 포함된 MuJoCo model.
        camera_name: 사용할 camera 이름. 없으면 아무 것도 하지 않는다.
    """
    cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id == -1:
        return

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = cam_id


def set_goal_marker_position(mj_model, target_position):
    """Set the visual-only ``goal_marker`` geom to a planar target position."""
    target_position = np.asarray(target_position, dtype=float)
    if target_position.shape != (2,):
        raise ValueError("target_position must contain exactly [x, y]")

    marker_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker")
    if marker_id == -1:
        raise ValueError("MuJoCo model must define a geom named 'goal_marker'")

    mj_model.geom_pos[marker_id, :2] = target_position


def set_obstacle_visibility(mj_model, obstacle_enabled):
    """Show or hide numbered MuJoCo obstacle geoms without changing geometry."""
    for index, enabled in enumerate(obstacle_enabled, start=1):
        geom_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_geom{index}"
        )
        if geom_id == -1:
            raise ValueError(
                f"MuJoCo model must define a geom named 'obstacle_geom{index}'"
            )
        if not enabled:
            mj_model.geom_rgba[geom_id, 3] = 0.0


def run_simulation(
    mujoco_xml_path,
    controller: VelocityController,
    robot: KinematicsProvider,
    target_position=None,
    obstacle_enabled=None,
    duration=20.0,
    visualize=True,
    camera_name="top_down",
    viewer_sync_interval_steps=10,
    recorder: SimulationRecorder | None = None,
    data_path="outputs/sim_data.npz",
    csv_path="outputs/sim_data.csv",
    plot_path="outputs/sim_plot.png",
    show_recording_gui=True,
    verbose=True,
    save_recording=True,
    initial_q=None,
    kinematic=True,
):
    """MuJoCo 시뮬레이션을 지정한 시간 동안 실행한다.

    Args:
        mujoco_xml_path: 완전한 MuJoCo MJCF 모델 파일 경로.
        controller: 속도 명령과 barrier를 제공하는 제어기 구현체.
        robot: end-effector 위치/Jacobian을 제공하는 기구학 구현체.
        target_position: [x, y] 목표 좌표. 지정하면 visual-only ``goal_marker``를
            이 위치에 표시한다.
        obstacle_enabled: 장애물별 표시/CBF 활성화 여부. False인 장애물은 화면에서
            숨긴다. CBF 제외는 controller가 같은 설정을 사용해 처리한다.
        duration: simulated time 기준 실행 시간, 단위는 초.
        visualize: True이면 MuJoCo viewer 창을 띄워 실시간에 가깝게 보여준다.
        camera_name: viewer에서 사용할 MuJoCo fixed camera 이름. 없으면 기본 viewer
            camera를 그대로 사용한다.
        viewer_sync_interval_steps: viewer 화면을 갱신할 physics step 간격. 10이면
            timestep 0.002 s에서 약 50 FPS로 보이며, QP/physics는 매 step 수행한다.
        recorder: 지정하면 t, q, dq, ee, u, barrier를 기록한다. None이면 기록하지
            않는다.
        data_path: recorder가 있을 때 저장할 NPZ 파일 경로.
        csv_path: recorder가 있을 때 저장할 CSV 파일 경로.
        plot_path: recorder가 있을 때 저장할 matplotlib plot 이미지 경로.
        show_recording_gui: True이면 종료 후 Recorder GUI를 연다.
        verbose: False이면 batch experiment용 console output을 생략한다.
        save_recording: False이면 recorder에는 기록하되 NPZ/CSV/plot 파일은 저장하지
            않는다. Batch experiment가 자체 compact trajectory를 저장할 때 사용한다.
        initial_q: 지정하면 시뮬레이션을 이 관절 자세에서 시작한다.
        kinematic: True이면 QP 출력 ``u``를 실제 관절 속도로 간주해
            ``q ← integrate(q, u * dt)``로 직접 적분한다. 따라서 CBF가
            가정하는 ``q_dot = u``와 시뮬레이션이 일치한다. False이면
            기존 MuJoCo velocity actuator 동역학을 사용한다.

    Usage:
        run_simulation(
            "assets/robots/planar.xml",
            controller,
            robot,
            duration=5.0,
            visualize=True,
        )
    """
    if viewer_sync_interval_steps < 1:
        raise ValueError("viewer_sync_interval_steps must be at least 1")

    mj_model = load_mujoco_model(mujoco_xml_path)
    if target_position is not None:
        set_goal_marker_position(mj_model, target_position)
    if obstacle_enabled is not None:
        set_obstacle_visibility(mj_model, obstacle_enabled)
    mj_data = mujoco.MjData(mj_model)
    if initial_q is not None:
        initial_q = np.asarray(initial_q, dtype=float)
        if initial_q.shape != (mj_model.nq,):
            raise ValueError(f"initial_q must have shape ({mj_model.nq},)")
        mj_data.qpos[:] = initial_q
    dt = mj_model.opt.timestep
    mujoco.mj_forward(mj_model, mj_data)

    steps = math.ceil(duration / dt)
    if verbose:
        print(f"MuJoCo timestep: {dt}")
        print(f"Running for {duration:.3f} s ({steps} steps)")

    def step_once():
        q = mj_data.qpos.copy()
        dq = mj_data.qvel.copy()
        # A switch is evaluated at the same robot state, then the QP is solved
        # again for the next link endpoint.  At most dof-1 switches can occur.
        for _ in range(getattr(controller, "dof", 1)):
            if hasattr(robot, "set_end_effector") and getattr(
                controller, "sequential_end_effector_enabled", False
            ):
                robot.set_end_effector(controller.end_effector_index)
            p_ee, J = robot.update_kinematics(q)
            ee_yaw = robot.end_effector_yaw()
            active_index = getattr(controller, "end_effector_index", None)
            u = controller.get_velocity(q, J, p_ee, ee_yaw)
            if active_index == getattr(controller, "end_effector_index", None):
                break
        safety = controller.safety_diagnostics(q, J, p_ee, ee_yaw)

        u = np.asarray(u, dtype=float)
        if u.shape != (mj_model.nv,):
            raise ValueError(f"controller velocity must have shape ({mj_model.nv},)")
        if recorder is not None:
            recorder.record(
                SimulationStep(
                    mj_data.time,
                    q,
                    dq,
                    p_ee,
                    u,
                    safety,
                    end_effector_yaw=ee_yaw,
                    active_link=getattr(controller, "end_effector_index", None),
                    clf_slack=getattr(controller, "last_clf_slack", None),
                    qp_solve_time=getattr(controller, "last_qp_solve_time", None),
                )
            )
        if kinematic:
            # The CBF-QP is a kinematic controller: its decision variable is
            # q_dot itself.  Bypassing the velocity-servo actuator prevents
            # tracking lag from violating a constraint that the QP satisfied.
            mj_data.qvel[:] = u
            mujoco.mj_integratePos(mj_model, mj_data.qpos, mj_data.qvel, dt)
            mj_data.time += dt
            mujoco.mj_forward(mj_model, mj_data)
        else:
            mj_data.ctrl[:] = u
            mujoco.mj_step(mj_model, mj_data)

        return p_ee, ee_yaw

    if hasattr(robot, "set_end_effector") and getattr(
        controller, "sequential_end_effector_enabled", False
    ):
        robot.set_end_effector(controller.end_effector_index)
    p_ee, _ = robot.update_kinematics(mj_data.qpos.copy())
    ee_yaw = robot.end_effector_yaw()
    if visualize:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            _use_fixed_camera_if_available(viewer, mj_model, camera_name)

            start = time.time()
            step = 0
            while viewer.is_running() and step < steps:
                p_ee, ee_yaw = step_once()
                if step % viewer_sync_interval_steps == 0 or step + 1 == steps:
                    viewer.sync()

                next_time = start + (step + 1) * dt
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                step += 1
    else:
        for _ in range(steps):
            p_ee, ee_yaw = step_once()

    if verbose:
        print(f"Final time: {mj_data.time:.3f}")
        print(f"Final qpos: {mj_data.qpos}")
        print(f"Final qvel: {mj_data.qvel}")
        print(f"Final ee position: {p_ee}")
        print(f"Final ee yaw: {ee_yaw:.6f} rad")

    if recorder is not None and save_recording:
        recorder.save_npz(data_path)
        recorder.save_csv(csv_path)
        plot(recorder.arrays(), recorder.barrier_labels, path=plot_path)
        if verbose:
            print(f"Saved data: {data_path}")
            print(f"Saved CSV: {csv_path}")
            print(f"Saved plot: {plot_path}")
        if show_recording_gui:
            show_gui(recorder.arrays(), recorder.barrier_labels)

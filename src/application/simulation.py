import math
import time

import mujoco
import mujoco.viewer

from ..domain.interfaces import KinematicsProvider, SimulationRecorder, VelocityController
from ..domain.types import SimulationStep


def load_mujoco_model(mujoco_xml_path):
    """완전한 MuJoCo MJCF 파일을 로드한다.

    Args:
        mujoco_xml_path: 로봇, actuator, scene, camera, option을 모두 포함한
            MuJoCo XML 파일 경로.

    Returns:
        mujoco.MjModel: mujoco.MjData와 함께 mj_step()에 사용할 compiled model.

    Usage:
        model = load_mujoco_model("settings/mujoco.xml")
        data = mujoco.MjData(model)
    """
    return mujoco.MjModel.from_xml_path(mujoco_xml_path)


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


def run_simulation(
    mujoco_xml_path,
    controller: VelocityController,
    robot: KinematicsProvider,
    duration=20.0,
    visualize=True,
    camera_name="top_down",
    recorder: SimulationRecorder | None = None,
    data_path="outputs/sim_data.npz",
    csv_path="outputs/sim_data.csv",
    plot_path="outputs/sim_plot.png",
    show_recording_gui=True,
):
    """MuJoCo 시뮬레이션을 지정한 시간 동안 실행한다.

    Args:
        mujoco_xml_path: 완전한 MuJoCo MJCF 모델 파일 경로.
        controller: 속도 명령과 barrier를 제공하는 제어기 구현체.
        robot: end-effector 위치/Jacobian을 제공하는 기구학 구현체.
        duration: simulated time 기준 실행 시간, 단위는 초.
        visualize: True이면 MuJoCo viewer 창을 띄워 실시간에 가깝게 보여준다.
        camera_name: viewer에서 사용할 MuJoCo fixed camera 이름. 없으면 기본 viewer
            camera를 그대로 사용한다.
        recorder: 지정하면 t, q, dq, ee, u, barrier를 기록한다. None이면 기록하지
            않는다.
        data_path: recorder가 있을 때 저장할 NPZ 파일 경로.
        csv_path: recorder가 있을 때 저장할 CSV 파일 경로.
        plot_path: recorder가 있을 때 저장할 matplotlib plot 이미지 경로.
        show_recording_gui: True이면 종료 후 Recorder GUI를 연다.

    Usage:
        run_simulation(
            "settings/mujoco.xml",
            controller,
            robot,
            duration=5.0,
            visualize=True,
        )
    """
    mj_model = load_mujoco_model(mujoco_xml_path)
    mj_data = mujoco.MjData(mj_model)
    dt = mj_model.opt.timestep
    mujoco.mj_forward(mj_model, mj_data)

    ee_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
    steps = math.ceil(duration / dt)
    print(f"MuJoCo timestep: {dt}")
    print(f"Running for {duration:.3f} s ({steps} steps)")

    def step_once():
        q = mj_data.qpos.copy()
        dq = mj_data.qvel.copy()
        p_ee, J = robot.update_kinematics(q)
        safety = controller.safety_diagnostics(q)
        u = controller.get_velocity(q, J, p_ee)

        mj_data.ctrl[:] = u
        if recorder is not None:
            recorder.record(SimulationStep(mj_data.time, q, dq, p_ee, u, safety))
        mujoco.mj_step(mj_model, mj_data)

        return p_ee

    p_ee = mj_data.xpos[ee_body_id].copy()
    if visualize:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            _use_fixed_camera_if_available(viewer, mj_model, camera_name)

            start = time.time()
            step = 0
            while viewer.is_running() and step < steps:
                p_ee = step_once()
                viewer.sync()

                next_time = start + (step + 1) * dt
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                step += 1
    else:
        for _ in range(steps):
            p_ee = step_once()

    print(f"Final time: {mj_data.time:.3f}")
    print(f"Final qpos: {mj_data.qpos}")
    print(f"Final qvel: {mj_data.qvel}")
    print(f"Final ee position: {p_ee}")

    if recorder is not None:
        recorder.save_npz(data_path)
        recorder.save_csv(csv_path)
        recorder.plot(plot_path)
        print(f"Saved data: {data_path}")
        print(f"Saved CSV: {csv_path}")
        print(f"Saved plot: {plot_path}")
        if show_recording_gui:
            recorder.show_gui()

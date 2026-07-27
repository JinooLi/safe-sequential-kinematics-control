import os

from src.adapters.osqp_solver import OSQPSolver
from src.adapters.pinocchio_kinematics import RobotInterface
from src.application.configuration import load_controller_config, validate_geometry_consistency
from src.application.simulation import run_simulation
from src.control.qp_controller import QPController
from src.logging.recorder import Recorder

if __name__ == "__main__":
    cur_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(cur_file_path)
    mjc_path = os.path.join(cur_dir, "settings/mujoco.xml")
    urdf_path = os.path.join(cur_dir, "settings/urdf.xml")
    ctrl_path = os.path.join(cur_dir, "settings/controller.yaml")

    config = load_controller_config(ctrl_path)
    validate_geometry_consistency(config, urdf_path, mjc_path)
    controller = QPController(config, OSQPSolver())
    robot = RobotInterface(urdf_path, "ee_joint")
    recorder = Recorder()
    run_simulation(
        mjc_path,
        controller,
        robot,
        target_position=config.ee_target,
        obstacle_enabled=[obstacle.enabled for obstacle in config.obstacles],
        recorder=recorder,
        duration=20,
    )

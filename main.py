import os
import src.SimulationEnv as sim

if __name__ == "__main__":
    cur_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(cur_file_path)
    mjc_path = os.path.join(cur_dir, "settings/mujoco.xml")
    urdf_path = os.path.join(cur_dir, "settings/urdf.xml")
    ctrl_path = os.path.join(cur_dir, "settings/controller.yaml")

    sim.run_simulation(mjc_path, urdf_path, "ee_joint", ctrl_path, duration=20)

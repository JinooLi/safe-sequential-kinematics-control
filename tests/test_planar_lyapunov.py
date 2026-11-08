import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from safe_ik.adapters.planar_kinematics import RobotInterface
from safe_ik.application.planar_configuration import load_controller_config
from safe_ik.control.planar_controller import QPController
from safe_ik.domain.types import SafetyDiagnostics, SimulationStep
from safe_ik.logging.recorder import Recorder
from safe_ik.visualization.planar_lyapunov import plot_planar_lyapunov

ROOT = Path(__file__).resolve().parents[1]


class _UnusedSolver:
    def solve(self, problem):
        raise AssertionError("Evaluating Lyapunov values must not solve a QP")


class PlanarLyapunovTest(unittest.TestCase):
    def test_values_match_actual_controller_for_each_pinocchio_frame(self):
        config = load_controller_config(ROOT / "configs/planar.yaml")
        controller = QPController(config, _UnusedSolver(), verbose=False)
        robot = RobotInterface(
            ROOT / "assets/robots/planar.urdf",
            config.sequential_end_effector.frame_names,
        )
        controller._select_goal_configuration(np.zeros(3))
        selected = controller.selected_goal_configuration
        samples = [np.zeros(3), np.array([0.4, -0.7, 1.2]), selected.q]
        for q in samples:
            before = controller.end_effector_index
            values = controller.task_clf_values(q)
            self.assertEqual(controller.end_effector_index, before)
            self.assertIs(controller.selected_goal_configuration, selected)
            for stage in range(1, 4):
                controller.end_effector_index = stage
                robot.set_end_effector(stage)
                position, jacobian = robot.update_kinematics(q)
                diagnostic = controller.safety_diagnostics(
                    q, jacobian, position, robot.end_effector_yaw()
                )
                self.assertAlmostEqual(values[stage - 1], diagnostic.clf_value)
                np.testing.assert_allclose(values, diagnostic.clf_by_link)
        np.testing.assert_allclose(
            controller.task_clf_values(selected.q), 0, atol=1e-12
        )

    def test_roundtrip_keeps_all_values_and_old_traces_load(self):
        recorder = Recorder()
        values = np.array([3.0, 5.0, 9.0])
        diagnostic = SafetyDiagnostics((), np.empty(0), 3.0, values)
        values[:] = 100  # The diagnostic owns its sample.
        recorder.record(
            SimulationStep(
                0.0,
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                diagnostic,
                active_link=1,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            recorder.save_npz(path)
            restored = Recorder.from_npz(path)
            np.testing.assert_allclose(restored.arrays()["clf_by_link"], [[3, 5, 9]])
            csv_path = Path(directory) / "trace.csv"
            restored.save_csv(csv_path)
            with csv_path.open() as file:
                header = next(csv.reader(file))
            self.assertTrue({"V_link1", "V_link2", "V_link3"} <= set(header))
            np.savez(path, t=[0.0], q=[[0.0, 0.0, 0.0]], clf=[3.0])
            old = Recorder.from_npz(path).arrays()
            self.assertNotIn("clf_by_link", old)
            np.testing.assert_allclose(old["clf"], [3.0])

    def test_plot_preserves_inactive_values_and_masks_active_segments(self):
        arrays = {
            "t": np.arange(6, dtype=float),
            "clf_by_link": np.arange(18, dtype=float).reshape(6, 3),
            "active_link": np.array([1, 1, 2, 2, 3, 3]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lyapunov.svg"
            figure = plot_planar_lyapunov(arrays, path)
            self.assertTrue(path.is_file())
            for index, axis in enumerate(figure.axes[:3]):
                np.testing.assert_allclose(
                    axis.lines[0].get_ydata(), arrays["clf_by_link"][:, index]
                )
                active_values = axis.lines[1].get_ydata()
                np.testing.assert_array_equal(
                    np.isfinite(active_values), arrays["active_link"] == index + 1
                )
                self.assertEqual(len(axis.lines), 4)  # two curves, two switches
            self.assertEqual(figure.axes[-1].lines[0].get_drawstyle(), "steps-post")

    def test_missing_and_misaligned_data_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Record a new"):
            plot_planar_lyapunov({"t": [0, 1], "clf": [4, 3]})
        with self.assertRaisesRegex(ValueError, "valid time/stages"):
            plot_planar_lyapunov(
                {"t": [0, 1], "clf_by_link": np.ones((2, 3)), "active_link": [0, 1]}
            )

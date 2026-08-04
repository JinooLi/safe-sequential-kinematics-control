import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.domain.types import SafetyDiagnostics, SimulationStep
from src.logging.recorder import Recorder


class RecorderBarrierTest(unittest.TestCase):
    def test_barriers_are_saved_with_link_labels(self):
        recorder = Recorder()
        recorder.record(
            SimulationStep(
                time=0.0,
                q=np.zeros(2),
                dq=np.zeros(2),
                end_effector=np.zeros(2),
                command=np.zeros(2),
                safety=SafetyDiagnostics(
                    ("obs1_link1", "obs1_link2"),
                    np.array([0.12, -0.03]),
                    clf_value=0.5,
                    clbf_labels=("W_obs1", "W_obs2"),
                    clbf_values=np.array([-0.2, 0.3]),
                ),
                end_effector_yaw=0.42,
            )
        )
        arrays = recorder.arrays()
        np.testing.assert_allclose(arrays["barrier"], [[0.12, -0.03]])
        np.testing.assert_allclose(arrays["ee_yaw"], [0.42])
        np.testing.assert_allclose(arrays["clf"], [0.5])
        np.testing.assert_allclose(arrays["clbf"], [[-0.2, 0.3]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.csv"
            recorder.save_csv(path)
            with path.open() as file:
                header = next(csv.reader(file))
            npz_path = Path(directory) / "record.npz"
            recorder.save_npz(npz_path)
            restored = Recorder.from_npz(npz_path)
        self.assertIn("obs1_link1", header)
        self.assertIn("obs1_link2", header)
        self.assertIn("ee_yaw [rad]", header)
        self.assertIn("W_obs1", header)
        self.assertIn("W_obs2", header)
        self.assertEqual(restored.barrier_labels, ("obs1_link1", "obs1_link2"))
        self.assertEqual(restored.clbf_labels, ("W_obs1", "W_obs2"))
        np.testing.assert_allclose(restored.arrays()["ee_yaw"], [0.42])


if __name__ == "__main__":
    unittest.main()

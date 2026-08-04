from pathlib import Path
import unittest

import numpy as np

from src.adapters.osqp_solver import OSQPSolver
from src.application.configuration import load_controller_config
from src.application.visualize_clbf import evaluate_final_clbf_in_task_space
from src.control.qp_controller import QPController


ROOT = Path(__file__).resolve().parents[1]


class VisualizeCLBFTest(unittest.TestCase):
    def test_task_space_evaluation_accepts_scalar_and_grid_states(self):
        config = load_controller_config(ROOT / "settings/controller.yaml")
        controller = QPController(config, OSQPSolver())
        target = np.array([*config.ee_target, config.ee_orientation])
        scalar = evaluate_final_clbf_in_task_space(target, controller)
        grid = evaluate_final_clbf_in_task_space(np.stack((target, target)), controller)

        self.assertTrue(np.isfinite(scalar))
        self.assertEqual(grid.shape, (2,))
        np.testing.assert_allclose(grid, [scalar, scalar])


if __name__ == "__main__":
    unittest.main()

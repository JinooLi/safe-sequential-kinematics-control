"""Completed-run plots include recorded joint angles without recomputation."""

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from safe_ik.domain.panda_types import PandaControlTrace
from safe_ik.visualization.panda import save_panda_control_plot


class PandaPlotTest(unittest.TestCase):
    def setUp(self):
        self.trace = PandaControlTrace(
            time=np.array([0.0, 0.1, 0.2]),
            clf_value=np.array([1.0, 0.5, 0.2]),
            clf_slack=np.array([0.1, 0.05, 0.0]),
            stage_index=np.array([0, 0, 1]),
            stage_names=("approach", "goal"),
            barrier_labels=("self_pair",),
            barrier_values=np.array([[0.03], [0.02], [0.01]]),
            q=np.deg2rad(np.arange(21, dtype=float).reshape(3, 7) - 10),
        )
        self.addCleanup(plt.close, "all")

    def test_joint_angles_share_time_axis_and_are_displayed_in_degrees(self):
        original_q = self.trace.q.copy()
        with TemporaryDirectory() as directory, patch.object(plt, "show") as show:
            path = Path(directory) / "plots/control.png"
            self.assertTrue(save_panda_control_plot(self.trace, path, show=True))
            self.assertGreater(path.stat().st_size, 0)
            show.assert_called_once_with()
            axes = plt.gcf().axes
            self.assertEqual(len(axes), 4)
            angle_axis = axes[-1]
            self.assertEqual(angle_axis.get_ylabel(), "Joint angle [deg]")
            self.assertEqual(angle_axis.get_xlabel(), "time [s]")
            self.assertTrue(angle_axis.get_shared_x_axes().joined(axes[0], angle_axis))
            self.assertEqual(
                [text.get_text() for text in angle_axis.get_legend().get_texts()],
                [f"q{joint}" for joint in range(1, 8)],
            )
            for joint, line in enumerate(angle_axis.lines[:7]):
                np.testing.assert_array_equal(line.get_xdata(), self.trace.time)
                np.testing.assert_allclose(
                    line.get_ydata(), np.rad2deg(original_q[:, joint])
                )
            # Stage-change markers should align on every panel, including q.
            for axis in axes:
                np.testing.assert_array_equal(axis.lines[-1].get_xdata(), [0.2, 0.2])
        np.testing.assert_array_equal(self.trace.q, original_q)

    def test_legacy_trace_without_joint_angles_keeps_three_panels(self):
        with TemporaryDirectory() as directory, patch.object(plt, "show"):
            self.assertTrue(
                save_panda_control_plot(
                    replace(self.trace, q=None), Path(directory) / "plot.png", show=True
                )
            )
            self.assertEqual(len(plt.gcf().axes), 3)
            self.assertEqual(plt.gcf().axes[-1].get_xlabel(), "time [s]")

    def test_save_only_closes_figure_without_opening_window(self):
        figures_before = plt.get_fignums()
        with TemporaryDirectory() as directory, patch.object(plt, "show") as show:
            self.assertTrue(
                save_panda_control_plot(self.trace, Path(directory) / "plot.png")
            )
            show.assert_not_called()
        self.assertEqual(plt.get_fignums(), figures_before)

    def test_missing_or_empty_trace_does_not_create_plot(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "plot.png"
            for trace in (None, replace(self.trace, time=np.empty(0))):
                with self.subTest(trace=trace):
                    self.assertFalse(save_panda_control_plot(trace, path))
                    self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()

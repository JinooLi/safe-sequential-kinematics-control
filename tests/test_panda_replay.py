"""Compute/save before GUI; replay never reintegrates or invokes control."""

import io
import json
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import test_panda_runtime as runtime_fixture

from safe_ik.adapters.mujoco_plant import MujocoPlant
from safe_ik.application import panda_run
from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.application.panda_rollout import run_panda
from safe_ik.visualization.panda_replay import replay_panda_result

FIXTURE = runtime_fixture.ROOT / "tests/fixtures/panda.yaml"


class PandaReplayTest(unittest.TestCase):
    def setUp(self):
        self.config = load_panda_config(FIXTURE)

    def test_replay_uses_recorded_times_includes_final_frame_and_restores_state(self):
        plant = MujocoPlant(self.config)
        saved_q = plant.q
        plant.data.qvel[:] = 0.25
        plant.data.time = 7.0
        samples = np.vstack([saved_q, saved_q + 0.01, saved_q + 0.02])
        window = Mock()
        window.lock.side_effect = lambda: nullcontext()
        window.is_running.side_effect = [True, True, True, True, False]
        displayed = []
        window.sync.side_effect = lambda: displayed.append((plant.time, plant.q))
        with (
            patch.object(plant, "viewer", return_value=nullcontext(window)) as viewer,
            patch.object(
                plant, "advance", side_effect=AssertionError("must not integrate")
            ),
            patch(
                "safe_ik.adapters.mujoco_plant.time.monotonic",
                side_effect=[100.0, 100.0, 100.051, 100.101, 100.5],
            ),
            patch("safe_ik.adapters.mujoco_plant.time.sleep"),
        ):
            plant.replay(np.array([1.0, 1.05, 1.1]), samples)
        viewer.assert_called_once_with(hold=False)
        np.testing.assert_allclose([t for t, q in displayed], [1.0, 1.05, 1.1, 1.1])
        np.testing.assert_array_equal(
            np.stack([q for t, q in displayed]), samples[[0, 1, 2, 2]]
        )
        np.testing.assert_array_equal(plant.q, saved_q)
        np.testing.assert_array_equal(plant.data.qvel, np.full(7, 0.25))
        self.assertEqual(plant.time, 7.0)

    def test_close_during_playback_stops_immediately_without_finishing_trajectory(self):
        plant = MujocoPlant(self.config)
        window = Mock()
        window.is_running.side_effect = [True, False]
        with (
            patch.object(plant, "viewer", return_value=nullcontext(window)),
            patch.object(plant, "advance") as advance,
            patch(
                "safe_ik.adapters.mujoco_plant.time.monotonic", side_effect=[0.0, 0.0]
            ),
            patch("safe_ik.adapters.mujoco_plant.time.sleep") as sleep,
        ):
            plant.replay(np.array([0.0, 100.0]), np.vstack([plant.q, plant.q + 0.01]))
        window.sync.assert_called_once()
        sleep.assert_called_once()
        advance.assert_not_called()

    def test_invalid_replay_arrays_rejected_before_viewer(self):
        plant = MujocoPlant(self.config)
        for times, poses in (
            ([], np.empty((0, 7))),
            ([1.0, 0.0], np.zeros((2, 7))),
            ([0.0], np.zeros((1, 2))),
            ([float("nan")], np.zeros((1, 7))),
        ):
            with self.subTest(times=times), patch.object(plant, "viewer") as viewer:
                with self.assertRaisesRegex(ValueError, "replay needs"):
                    plant.replay(times, poses)
                viewer.assert_not_called()

    def test_final_integrated_pose_is_appended_to_pre_step_trace(self):
        plant = Mock(time=0.2)
        trace = SimpleNamespace(time=np.array([0.0, 0.1]), q=np.zeros((2, 7)))
        result = SimpleNamespace(trace=trace, final_q=np.ones(7))
        replay_panda_result(plant, result)
        times, poses = plant.replay.call_args.args
        np.testing.assert_array_equal(times, [0.0, 0.1, 0.2])
        np.testing.assert_array_equal(poses[-1], result.final_q)
        np.testing.assert_array_equal(trace.q, np.zeros((2, 7)))

    def test_failed_ik_without_trace_shows_stopped_pose_only(self):
        plant = Mock(time=0.0)
        result = SimpleNamespace(trace=None, final_q=self.config.initial_q.copy())
        replay_panda_result(plant, result)
        times, poses = plant.replay.call_args.args
        np.testing.assert_array_equal(times, [0.0])
        np.testing.assert_array_equal(poses, result.final_q[None])

    def test_cli_computes_and_saves_before_replay_and_shows_plot_after_close(self):
        fixture = runtime_fixture.PandaRuntimeTest()
        fixture.setUp()
        runtime = fixture.runtime()
        result = run_panda(fixture.config, runtime)
        events = []
        with TemporaryDirectory() as directory:
            output = Path(directory) / "test"

            def compute(*args, **kwargs):
                self.assertNotIn("observe", kwargs)
                self.assertNotIn("should_continue", kwargs)
                events.append("compute")
                return result

            def plot(trace, path, *, show):
                events.append("show plot" if show else "save plot")
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).touch()
                return True

            def replay(*args):
                self.assertTrue((output / "summary.json").is_file())
                self.assertTrue((output / "trajectory.npz").is_file())
                self.assertTrue((output / "plots/control.png").is_file())
                events.append("replay")

            with (
                patch(
                    "sys.argv",
                    [
                        "safe-ik",
                        "--config",
                        str(FIXTURE),
                        "--visualize",
                        "--show-plot",
                        "--output-dir",
                        directory,
                        "--run-name",
                        "test",
                    ],
                ),
                patch.object(panda_run, "build_panda_runtime", return_value=runtime),
                patch.object(panda_run, "save_run_provenance"),
                patch.object(panda_run, "run_panda", side_effect=compute) as calculate,
                patch.object(panda_run, "save_panda_control_plot", side_effect=plot),
                patch.object(panda_run, "replay_panda_result", side_effect=replay),
                redirect_stdout(io.StringIO()),
            ):
                panda_run.main()
            self.assertEqual(events, ["compute", "save plot", "replay", "show plot"])
            calculate.assert_called_once()
            self.assertIsNotNone(calculate.call_args.kwargs["progress"])

    def test_headless_cli_can_disable_progress_and_never_opens_viewer(self):
        fixture = runtime_fixture.PandaRuntimeTest()
        fixture.setUp()
        runtime = fixture.runtime()
        result = run_panda(fixture.config, runtime)
        with TemporaryDirectory() as directory:
            with (
                patch(
                    "sys.argv",
                    [
                        "safe-ik",
                        "--config",
                        str(FIXTURE),
                        "--no-progress",
                        "--rdf-device",
                        "cuda",
                        "--output-dir",
                        directory,
                    ],
                ),
                patch.object(
                    panda_run, "build_panda_runtime", return_value=runtime
                ) as build,
                patch.object(panda_run, "save_run_provenance") as provenance,
                patch.object(panda_run, "run_panda", return_value=result) as calculate,
                patch.object(panda_run, "save_panda_control_plot"),
                patch.object(panda_run, "replay_panda_result") as replay,
                redirect_stdout(io.StringIO()),
            ):
                panda_run.main()
                self.assertIsNone(calculate.call_args.kwargs["progress"])
                replay.assert_not_called()
                self.assertEqual(
                    build.call_args.args[0].safety.rdf.runtime_device, "cuda"
                )
                self.assertEqual(
                    provenance.call_args.kwargs["effective_yaml"]["safety"]["rdf"][
                        "runtime_device"
                    ],
                    "cuda",
                )

    def test_cuda_initialization_failure_saved_without_running(self):
        with TemporaryDirectory() as directory:
            with (
                patch(
                    "sys.argv",
                    [
                        "safe-ik",
                        "--config",
                        str(FIXTURE),
                        "--rdf-device",
                        "cuda",
                        "--output-dir",
                        directory,
                        "--run-name",
                        "cuda_failure",
                    ],
                ),
                patch.object(
                    panda_run,
                    "build_panda_runtime",
                    side_effect=RuntimeError("CUDA allocation failure"),
                ),
                patch.object(panda_run, "run_panda") as calculate,
                patch.object(panda_run, "save_run_provenance"),
                redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                panda_run.main()
            self.assertEqual(error.exception.code, 2)
            calculate.assert_not_called()
            failure = json.loads(
                (Path(directory) / "cuda_failure/summary.json").read_text()
            )["failure"]
            self.assertEqual(failure["reason"], "runtime_initialization_failed")
            self.assertIn("CUDA allocation failure", failure["detail"])

"""Saved-case playback restores the scene and never recomputes a trajectory."""

import hashlib
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mujoco
import numpy as np
import yaml

from safe_ik.adapters.mujoco_plant import MujocoPlant
from safe_ik.application.panda_comparison_replay import load_saved_replay, main

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PandaComparisonReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / "main"
        self.directory.mkdir()
        config = yaml.safe_load((ROOT / "configs/panda.yaml").read_text())
        self.recorded_model = "/recorded-server/project/assets/robots/panda.xml"
        config["robot"]["model_path"] = self.recorded_model
        config["safety"]["rdf"]["artifact"] = "/unavailable/rdf.npz"
        (self.directory / "config.yaml").write_text(yaml.safe_dump(config))
        self.case = {
            "case_id": "seed19_0000",
            "initial_q": config["robot"]["initial_q"],
            "target_position": [0.3, -0.2, 0.5],
            "target_quaternion": [1, 0, 0, 0],
            "ik_seed": 123,
            "obstacles": [
                {"name": "obstacle1", "position": [0.2, -0.3, 0.4], "radius": 0.07},
                {"name": "obstacle2", "position": [-0.2, 0.3, 0.6], "radius": 0.09},
                {"name": "obstacle3", "position": [0.4, 0.1, 0.5], "radius": 0.05},
                {"name": "obstacle4", "position": [-0.3, -0.2, 0.7], "radius": 0.06},
                {"name": "obstacle5", "position": [0.1, 0.4, 0.8], "radius": 0.08},
            ],
        }
        (self.directory / "cases.jsonl").write_text(json.dumps(self.case) + "\n")
        model_digest = digest(ROOT / "assets/robots/panda.xml")
        (self.directory / "manifest.json").write_text(
            json.dumps(
                {
                    "identity": {
                        "model_sha256": model_digest,
                        "model_resource_sha256": {self.recorded_model: model_digest},
                    },
                    "data_sha256": {
                        name: digest(self.directory / name)
                        for name in ("config.yaml", "cases.jsonl")
                    },
                }
            )
        )
        self.positions = np.array([self.case["initial_q"], self.case["initial_q"]])
        self.positions[-1, 0] += 0.01
        self.times = np.array([0.0, 0.02])
        for method in ("direct", "sequential"):
            folder = self.directory / self.case["case_id"] / method
            folder.mkdir(parents=True)
            np.savez(folder / "trajectory.npz", time=self.times, q=self.positions)
            (folder / "summary.json").write_text(
                json.dumps(
                    {
                        "reason": "timeout",
                        "success": False,
                        "final_q": self.positions[-1].tolist(),
                        "trace_sha256": digest(folder / "trajectory.npz"),
                    }
                )
            )

    def test_relocated_model_and_saved_scene_are_used(self):
        config, times, positions, _ = load_saved_replay(
            self.root, self.case["case_id"], "direct"
        )
        self.assertEqual(config.model_path, ROOT / "assets/robots/panda.xml")
        self.assertEqual(config.trajectory.mode, "direct")
        np.testing.assert_array_equal(
            config.target.position, self.case["target_position"]
        )
        np.testing.assert_array_equal(times, self.times)
        np.testing.assert_array_equal(positions, self.positions)
        plant = MujocoPlant(config)
        goal = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_SITE, "goal_marker")
        np.testing.assert_array_equal(
            plant.model.site_pos[goal], self.case["target_position"]
        )
        for index, obstacle in enumerate(self.case["obstacles"], 1):
            geom = mujoco.mj_name2id(
                plant.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_geom{index}"
            )
            self.assertGreaterEqual(geom, 0)
            np.testing.assert_array_equal(
                plant.model.geom_pos[geom], obstacle["position"]
            )
            self.assertEqual(plant.model.geom_size[geom, 0], obstacle["radius"])
            self.assertGreater(plant.model.geom_rgba[geom, 3], 0.0)

    def test_cli_passes_all_recorded_states_including_final_state_without_writing(self):
        before = {p: digest(p) for p in self.root.rglob("*") if p.is_file()}
        with (
            patch.dict(os.environ, {"DISPLAY": ":test"}),
            patch.object(MujocoPlant, "replay") as replay,
            patch.object(
                MujocoPlant, "advance", side_effect=AssertionError("must not integrate")
            ),
            redirect_stdout(StringIO()),
        ):
            main(
                [
                    "--results-dir",
                    str(self.root),
                    "--case-id",
                    self.case["case_id"],
                    "--method",
                    "direct",
                ]
            )
        replay.assert_called_once()
        times, positions = replay.call_args.args
        np.testing.assert_array_equal(times, self.times)
        np.testing.assert_array_equal(positions, self.positions)
        self.assertEqual(
            {p: digest(p) for p in self.root.rglob("*") if p.is_file()}, before
        )

    def test_list_and_dry_run_do_not_open_a_viewer(self):
        for args in (
            ["--list"],
            ["--case-id", self.case["case_id"], "--dry-run"],
        ):
            output = StringIO()
            with patch.object(MujocoPlant, "replay") as replay, redirect_stdout(output):
                main(["--results-dir", str(self.directory), *args])
            replay.assert_not_called()
            self.assertIn(self.case["case_id"], output.getvalue())

    def test_unavailable_ik_trajectory_is_reported_with_its_failure_reason(self):
        folder = self.directory / self.case["case_id"] / "sequential"
        (folder / "trajectory.npz").unlink()
        (folder / "summary.json").write_text(
            json.dumps({"reason": "ik_budget_exhausted", "success": False})
        )
        with self.assertRaisesRegex(
            ValueError, "no saved trajectory.*ik_budget_exhausted"
        ):
            load_saved_replay(self.directory, self.case["case_id"])

    def test_changed_model_or_recorded_trace_is_rejected(self):
        model = self.root / "different.xml"
        model.write_text("<mujoco/>")
        with self.assertRaisesRegex(ValueError, "model differs"):
            load_saved_replay(self.directory, self.case["case_id"], model_path=model)
        trace = self.directory / self.case["case_id"] / "sequential/trajectory.npz"
        trace.write_bytes(trace.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "trajectory hash mismatch"):
            load_saved_replay(self.directory, self.case["case_id"])

    def test_make_replay_dry_run_works_without_cuda_or_controller_modules(self):
        code = """
import sys
from safe_ik.application.panda_comparison_replay import main
main(sys.argv[1:])
assert not {'torch', 'pinocchio', 'osqp'} & sys.modules.keys()
assert 'safe_ik.application.panda_comparison' not in sys.modules
assert 'safe_ik.control.panda_controller' not in sys.modules
"""
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                "--results-dir",
                str(self.root),
                "--case-id",
                self.case["case_id"],
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            [
                "make",
                "-n",
                "panda-replay",
                f"RESULTS_DIR={self.root}",
                f"CASE_ID={self.case['case_id']}",
                "METHOD=direct",
                "ARGS=--dry-run",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("safe_ik.application.panda_comparison_replay", result.stdout)
        self.assertIn('--method "direct"', result.stdout)
        self.assertNotIn("--extra cuda", result.stdout)


if __name__ == "__main__":
    unittest.main()

"""Experiment settings, exact input snapshots, and resume guarantees."""

import json
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from safe_ik.application.artifacts import json_value
from safe_ik.application.panda_comparison import (
    atomic_json,
    main,
    verify_resume,
)
from safe_ik.application.panda_experiment_configuration import load_panda_experiment
from safe_ik.domain.panda_sampling_config import PandaSamplingConfig
from safe_ik.domain.safety_contracts import BarrierEvaluation
from safe_ik.experiments.panda_comparison import generate_cases

ROOT = Path(__file__).resolve().parents[1]


class PandaExperimentConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "controller.yaml"
        raw = yaml.safe_load((ROOT / "configs/panda.yaml").read_text())
        raw["robot"]["model_path"] = str(ROOT / "assets/robots/panda.xml")
        raw["safety"]["rdf"]["artifact"] = str(ROOT / "models/panda_rdf.npz")
        self.base.write_text(
            "# Base controller, preserved verbatim.\n" + yaml.safe_dump(raw)
        )
        (self.root / "experiments").mkdir()
        self.path = self.root / "experiments/test.yaml"
        self.data = {
            "controller_config": "../controller.yaml",
            "rdf_device": "cpu",
            "output_root": "../results",
            "output": None,
            "main": {"seeds": [19], "cases_per_seed": 3},
            "controller_overrides": {
                "trajectory": {
                    "clf_rate": 23.0,
                    "integration_period": 0.01,
                    "duration": 1.0,
                },
                "qp": {"absolute_tolerance": 1e-5},
            },
            "validation": {"subdivisions": 3},
            "sampling": {"obstacle_count": 3, "minimum_target_distance": 0.2},
        }
        self.write()

    def write(self):
        self.path.write_bytes(
            (
                "# Preserve this experiment comment.\r\n" + yaml.safe_dump(self.data)
            ).encode()
        )

    def test_paths_overrides_and_batches_are_loaded_from_experiment(self):
        settings = load_panda_experiment(self.path)
        self.assertEqual(settings.config_path, self.base)
        self.assertEqual(settings.output_root, self.root / "results")
        self.assertEqual(settings.config.trajectory.clf_rate, 23)
        self.assertEqual(settings.config.trajectory.integration_period, 0.01)
        self.assertEqual(settings.config.qp.absolute_tolerance, 1e-5)
        self.assertEqual(settings.config.safety.rdf.runtime_device, "cpu")
        self.assertEqual(settings.main, self.data["main"])
        self.assertEqual(settings.sampling.obstacle_count, 3)
        self.assertEqual(settings.subdivisions, 3)

    def test_cli_overrides_are_saved_and_resolved_yaml_can_be_loaded(self):
        settings = load_panda_experiment(
            self.path,
            device="cuda",
            duration=2.0,
            seeds=[21],
            cases_per_seed=4,
        )
        destination = self.root / "saved"
        destination.mkdir()
        hashes = settings.snapshot(destination)
        self.assertEqual(
            (destination / "experiment.yaml").read_bytes(), self.path.read_bytes()
        )
        self.assertEqual(
            (destination / "controller_source.yaml").read_bytes(),
            self.base.read_bytes(),
        )
        resolved = yaml.safe_load(
            (destination / "experiment.resolved.yaml").read_text()
        )
        self.assertEqual(resolved["main"], {"seeds": [21], "cases_per_seed": 4})
        self.assertEqual(resolved["rdf_device"], "cuda")
        reloaded = load_panda_experiment(destination / "experiment.resolved.yaml")
        self.assertEqual(json_value(reloaded.config), json_value(settings.config))
        self.assertEqual(
            set(hashes),
            {
                "experiment.yaml",
                "controller_source.yaml",
                "experiment.resolved.yaml",
                "config.yaml",
            },
        )

    def test_yaml_output_is_relative_to_experiment_and_cli_output_to_cwd(self):
        self.data["output"] = "../resumable"
        self.write()
        self.assertEqual(
            load_panda_experiment(self.path).output, self.root / "resumable"
        )
        self.assertEqual(
            load_panda_experiment(self.path, output="somewhere").output,
            Path("somewhere").resolve(),
        )

    def test_typo_in_experiment_or_controller_override_is_rejected(self):
        for mutate in (
            lambda d: d.update(cases_per_sead=3),
            lambda d: d["controller_overrides"]["trajectory"].update(clf_gain=10),
            lambda d: d["sampling"].update(obstacle_cout=3),
        ):
            data = deepcopy(self.data)
            mutate(self.data)
            self.write()
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_panda_experiment(self.path)
            self.data = data

    def test_invalid_batches_sampling_and_validation_are_rejected(self):
        for section, key, value in (
            ("main", "seeds", []),
            ("main", "seeds", [1, 1]),
            ("main", "seeds", [True]),
            ("main", "cases_per_seed", 0),
            ("validation", "subdivisions", True),
            ("sampling", "obstacle_count", 0),
            ("sampling", "minimum_barrier", -1),
        ):
            data = deepcopy(self.data)
            self.data[section][key] = value
            self.write()
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                load_panda_experiment(self.path)
            self.data = data

    def test_live_input_edits_and_saved_yaml_tampering_are_rejected(self):
        settings = load_panda_experiment(self.path)
        directory = self.root / "saved"
        directory.mkdir()
        identity = settings.identity()
        atomic_json(
            directory / "manifest.json",
            {
                "identity": identity,
                "data_sha256": settings.snapshot(directory),
            },
        )
        verify_resume(directory, identity)
        (directory / "experiment.yaml").write_text("changed: true\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_resume(directory, identity)
        self.path.write_text(self.path.read_text() + "# later edit\n")
        with self.assertRaisesRegex(ValueError, "YAML changed"):
            settings.identity()

    def test_dry_run_prints_effective_settings_without_creating_output(self):
        destination = self.root / "not-created"
        output = StringIO()
        with (
            patch("safe_ik.application.panda_comparison.run_experiment") as run,
            redirect_stdout(output),
        ):
            main(
                [
                    "--experiment-config",
                    str(self.path),
                    "--output",
                    str(destination),
                    "--duration",
                    "2",
                    "--dry-run",
                ]
            )
        run.assert_not_called()
        self.assertFalse(destination.exists())
        self.assertEqual(
            json.loads(output.getvalue())["controller"]["trajectory"]["duration"], 2
        )

    def test_cli_runs_requested_cases_even_without_ik_and_resumes_saved_results(self):
        self.data["main"] = {"seeds": [19, 20], "cases_per_seed": 1}
        self.write()
        destination = self.root / "results"
        args = ["--experiment-config", str(self.path), "--output", str(destination)]
        kin = SimpleNamespace(
            lower_limits=np.full(7, -10.0),
            upper_limits=np.full(7, 10.0),
            pose_and_jacobian=lambda q: (q[:3], np.array([1, 0, 0, 0]), np.eye(6, 7)),
        )
        safety = SimpleNamespace(
            evaluate=lambda q: BarrierEvaluation(
                ("self_test",), np.ones(1), np.zeros((1, 7))
            )
        )

        def cases(config, kin, safety, seeds, cases_per_seed, sampling):
            return [
                {
                    "case_id": f"seed{seed}_{i:04d}",
                    "initial_q": config.initial_q.tolist(),
                    "target_position": config.initial_q[:3].tolist(),
                    "target_quaternion": [1, 0, 0, 0],
                    "obstacles": [],
                    "ik_seed": seed + i,
                }
                for seed in seeds
                for i in range(cases_per_seed)
            ], {}

        module = "safe_ik.application.panda_comparison"
        with (
            patch(f"{module}.build_panda_components", return_value=(kin, safety)),
            patch(f"{module}.generate_cases", side_effect=cases),
            patch(f"{module}.scene_safety", return_value=safety),
            patch(
                f"{module}.MujocoPlant",
                side_effect=lambda config: SimpleNamespace(
                    q=config.initial_q, time=0.0
                ),
            ),
            patch(
                f"{module}.prepare_ik",
                return_value={
                    "selected": None,
                    "reason": "ik_budget_exhausted",
                    "wall_time": 0.0,
                },
            ) as prepare,
            redirect_stdout(StringIO()),
        ):
            main(args)
            analysis = json.loads((destination / "analysis.json").read_text())
            self.assertTrue(analysis["complete"])
            self.assertEqual(analysis["completed_pairs"], 2)
            self.assertEqual(analysis["ik_available_pairs"], 0)
            self.assertEqual(analysis["all_scenes"]["direct"]["success_count"], 2)
            self.assertTrue((destination / "complete.json").exists())
            self.assertEqual(
                {p.name for p in destination.iterdir() if p.is_dir()},
                {"seed19_0000", "seed20_0000"},
            )
            self.assertEqual(
                (destination / "experiment.yaml").read_bytes(), self.path.read_bytes()
            )
            trace = destination / "seed19_0000/direct/trajectory.npz"
            saved_time = trace.stat().st_mtime_ns
            main(args)
            self.assertEqual(trace.stat().st_mtime_ns, saved_time)
            self.assertEqual(prepare.call_count, 2)
            manifest = json.loads((destination / "manifest.json").read_text())
            verify_resume(destination, manifest["identity"])
            self.assertEqual(
                manifest["identity"]["experiment"]["main"], self.data["main"]
            )
            self.data["main"]["cases_per_seed"] = 4
            self.write()
            with self.assertRaisesRegex(ValueError, "settings changed"):
                main(args)
            self.assertEqual(prepare.call_count, 2)

    def test_sampler_uses_configured_bounds_radii_and_obstacle_count(self):
        class Kinematics:
            lower_limits = np.full(7, -1.0)
            upper_limits = np.full(7, 1.0)

            def pose_and_jacobian(self, q):
                return q[:3], np.array([1, 0, 0, 0]), np.eye(6, 7)

        class Safety:
            def evaluate(self, q):
                return BarrierEvaluation(("test",), np.ones(1), np.zeros((1, 7)))

        config = load_panda_experiment(self.path).config
        sampling = PandaSamplingConfig(
            workspace_lower=(-1, -1, -1),
            workspace_upper=(1, 1, 1),
            obstacle_count=3,
            obstacle_radius_range=(0.02, 0.03),
        )
        with patch(
            "safe_ik.experiments.panda_comparison.PandaCapsuleSafety",
            return_value=Safety(),
        ):
            cases, _ = generate_cases(config, Kinematics(), Safety(), [7], 2, sampling)
        for case in cases:
            self.assertEqual(len(case["obstacles"]), 3)
            for obstacle in case["obstacles"]:
                self.assertTrue(0.02 <= obstacle["radius"] <= 0.03)
                self.assertTrue(np.all(np.abs(obstacle["position"]) <= 1))


if __name__ == "__main__":
    unittest.main()

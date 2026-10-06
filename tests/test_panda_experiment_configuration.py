"""Experiment settings, exact input snapshots, and campaign resume guarantees."""

import json
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import yaml

from safe_ik.application.artifacts import json_value
from safe_ik.application.panda_comparison import (
    atomic_json,
    main,
    run_campaign,
    verify_resume,
)
from safe_ik.application.panda_experiment_configuration import load_panda_experiment
from safe_ik.domain.panda_sampling_config import PandaSamplingConfig
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
            "command": "campaign",
            "output_root": "../results",
            "output": None,
            "pilot": {"seeds": [7, 8], "cases_per_seed": 2},
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
        self.assertEqual(settings.pilot, self.data["pilot"])
        self.assertEqual(settings.main, self.data["main"])
        self.assertEqual(settings.sampling.obstacle_count, 3)
        self.assertEqual(settings.subdivisions, 3)

    def test_cli_overrides_are_saved_and_resolved_yaml_can_be_loaded(self):
        settings = load_panda_experiment(
            self.path,
            command="pilot",
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
        self.assertEqual(resolved["pilot"], {"seeds": [21], "cases_per_seed": 4})
        self.assertEqual(resolved["main"], self.data["main"])
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
            ("pilot", "seeds", []),
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
            patch("safe_ik.application.panda_comparison.run_campaign") as campaign,
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
        campaign.assert_not_called()
        self.assertFalse(destination.exists())
        self.assertEqual(
            json.loads(output.getvalue())["controller"]["trajectory"]["duration"], 2
        )

    def test_campaign_uses_yaml_batches_and_seals_root_snapshots(self):
        settings = load_panda_experiment(self.path)
        destination = self.root / "campaign"

        def fingerprint(config, config_path, seeds, cases_per_seed, *, experiment):
            return {
                "settings": experiment.identity(),
                "seeds": seeds,
                "count": cases_per_seed,
            }

        def run(directory, config_path, config, seeds, cases_per_seed, *, experiment):
            directory.mkdir()
            atomic_json(
                directory / "manifest.json",
                {
                    "identity": fingerprint(
                        config,
                        config_path,
                        seeds,
                        cases_per_seed,
                        experiment=experiment,
                    ),
                    "data_sha256": experiment.snapshot(directory),
                },
            )
            return {"complete": True, "ik_available_pairs": 1}

        with (
            patch(
                "safe_ik.application.panda_comparison.fingerprint",
                side_effect=fingerprint,
            ),
            patch(
                "safe_ik.application.panda_comparison.run_experiment", side_effect=run
            ) as runner,
        ):
            run_campaign(destination, settings)
            self.assertEqual(
                [c.kwargs["seeds"] for c in runner.call_args_list], [[7, 8], [19]]
            )
            self.assertEqual(
                [c.kwargs["cases_per_seed"] for c in runner.call_args_list], [2, 3]
            )
            self.assertEqual(
                (destination / "experiment.yaml").read_bytes(), self.path.read_bytes()
            )
            self.assertEqual(
                json.loads((destination / "status.json").read_text())["phase"],
                "complete",
            )
            self.data["main"]["cases_per_seed"] = 4
            self.write()
            with self.assertRaisesRegex(ValueError, "settings changed"):
                run_campaign(destination, load_panda_experiment(self.path))
            self.assertEqual(runner.call_count, 2)

    def test_sampler_uses_configured_bounds_radii_and_obstacle_count(self):
        class Kinematics:
            lower_limits = np.full(7, -1.0)
            upper_limits = np.full(7, 1.0)

            def pose_and_jacobian(self, q):
                return q[:3], np.array([1, 0, 0, 0]), np.eye(6, 7)

        from safe_ik.domain.safety_contracts import BarrierEvaluation

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

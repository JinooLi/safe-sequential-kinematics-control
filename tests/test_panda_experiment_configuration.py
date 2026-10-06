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
    fingerprint,
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
                    "success_barrier_tolerance": 5e-4,
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
        self.assertEqual(settings.config.trajectory.success_barrier_tolerance, 5e-4)
        self.assertEqual(settings.config.qp.absolute_tolerance, 1e-5)
        self.assertEqual(settings.config.safety.rdf.runtime_device, "cpu")
        self.assertEqual(settings.main, self.data["main"])
        self.assertEqual(settings.sampling.obstacle_count, 3)
        self.assertEqual(settings.subdivisions, 3)
        self.assertFalse(settings.use_all_ik_candidates)
        self.assertEqual(settings.candidate_policy, "one_nearest_no_switch")

    def test_cli_overrides_are_saved_and_resolved_yaml_can_be_loaded(self):
        self.data["use_all_ik_candidates"] = True
        self.data["controller_overrides"]["trajectory"].update(
            stage_task="pose",
            stage_orientation_weight=0.2,
            stage_orientation_tolerance=0.03,
        )
        self.write()
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
        self.assertTrue(resolved["use_all_ik_candidates"])
        reloaded = load_panda_experiment(destination / "experiment.resolved.yaml")
        self.assertTrue(reloaded.use_all_ik_candidates)
        self.assertEqual(reloaded.config.trajectory.stage_task, "pose")
        self.assertEqual(reloaded.config.trajectory.stage_orientation_weight, 0.2)
        self.assertEqual(reloaded.config.trajectory.stage_orientation_tolerance, 0.03)
        self.assertEqual(reloaded.config.trajectory.success_barrier_tolerance, 5e-4)
        self.assertEqual(reloaded.candidate_policy, "all_nearest_switch_on_stall")
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

    def test_candidate_policy_requires_boolean_and_prevents_changed_policy_resume(self):
        settings = load_panda_experiment(self.path)
        original = fingerprint(settings.config, self.base, [19], 3, experiment=settings)
        directory = self.root / "saved"
        directory.mkdir()
        atomic_json(directory / "manifest.json", {"identity": original})
        for value in ("false", 0, None, "all"):
            self.data["use_all_ik_candidates"] = value
            self.write()
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "true or false"),
            ):
                load_panda_experiment(self.path)
        self.data["use_all_ik_candidates"] = True
        self.write()
        settings = load_panda_experiment(self.path)
        changed = fingerprint(settings.config, self.base, [19], 3, experiment=settings)
        self.assertEqual(changed["candidate_policy"], "all_nearest_switch_on_stall")
        self.assertTrue(changed["experiment"]["use_all_ik_candidates"])
        with self.assertRaisesRegex(ValueError, "settings changed"):
            verify_resume(directory, changed)

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
                    "accepted_candidates": [],
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

    def test_cli_passes_only_selected_or_all_accepted_candidates(self):
        self.data["main"] = {"seeds": [19], "cases_per_seed": 1}
        kin = SimpleNamespace(
            lower_limits=np.full(7, -10.0),
            upper_limits=np.full(7, 10.0),
            pose_and_jacobian=lambda q: (q[:3], np.array([1, 0, 0, 0]), np.eye(6, 7)),
            frame_point_and_jacobian=lambda f, q: (q[:3], np.eye(6, 7)[:3]),
        )
        safety = SimpleNamespace(
            evaluate=lambda q: BarrierEvaluation(
                ("self_test",), np.ones(1), np.zeros((1, 7))
            )
        )

        def cases(config, *args):
            return [
                {
                    "case_id": "seed19_0000",
                    "initial_q": config.initial_q.tolist(),
                    "target_position": config.initial_q[:3].tolist(),
                    "target_quaternion": [1, 0, 0, 0],
                    "obstacles": [],
                    "ik_seed": 19,
                }
            ], {}

        def prepared(config, *args):
            candidates = [
                {
                    "q": (config.initial_q + 0.1 * i).tolist(),
                    "seed_index": seed,
                    "iterations": 1,
                    "position_error": 0,
                    "orientation_error": 0,
                    "minimum_barrier": 1,
                }
                for i, seed in enumerate((4, 8, 2), 1)
            ]
            return {
                "selected": candidates[0],
                "accepted_candidates": candidates[:2],
                "raw_candidates": candidates,
                "reason": None,
                "wall_time": 0,
            }

        module = "safe_ik.application.panda_comparison"
        for use_all in (False, True):
            self.data["use_all_ik_candidates"] = use_all
            self.write()
            destination = self.root / f"results-{use_all}"
            with (
                self.subTest(use_all=use_all),
                patch(f"{module}.build_panda_components", return_value=(kin, safety)),
                patch(f"{module}.generate_cases", side_effect=cases),
                patch(f"{module}.scene_safety", return_value=safety),
                patch(
                    f"{module}.MujocoPlant",
                    side_effect=lambda c: SimpleNamespace(q=c.initial_q, time=0.0),
                ),
                patch(f"{module}.prepare_ik", side_effect=prepared) as prepare,
                redirect_stdout(StringIO()),
            ):
                main(
                    [
                        "--experiment-config",
                        str(self.path),
                        "--output",
                        str(destination),
                    ]
                )
                prepare.assert_called_once()
                for method in ("direct", "sequential"):
                    folder = destination / "seed19_0000" / method
                    result = json.loads((folder / "summary.json").read_text())
                    expected = [] if method == "direct" else [4, 8] if use_all else [4]
                    self.assertEqual(result["candidate_count"], len(expected))
                    self.assertEqual(result["candidate_switch_count"], 0)
                    self.assertEqual(
                        result["candidate_policy"],
                        (
                            "not_used"
                            if method == "direct"
                            else load_panda_experiment(self.path).candidate_policy
                        ),
                    )
                    with np.load(folder / "trajectory.npz") as trace:
                        np.testing.assert_array_equal(
                            trace["candidate_seed_indices"], expected
                        )


if __name__ == "__main__":
    unittest.main()

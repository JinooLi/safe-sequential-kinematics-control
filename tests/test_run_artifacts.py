"""Provenance captures effective inputs and is safe to reload from a run directory."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from safe_ik.application.artifacts import create_run_directory, save_run_provenance
from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.application.planar_configuration import load_controller_config

ROOT = Path(__file__).resolve().parents[1]


class RunArtifactsTest(unittest.TestCase):
    def test_run_directories_never_reuse_an_existing_name(self):
        with TemporaryDirectory() as directory:
            self.assertNotEqual(
                create_run_directory(directory), create_run_directory(directory)
            )
            run = create_run_directory(directory, "saved")
            (run / "keep.txt").write_text("original")
            with self.assertRaises(FileExistsError):
                create_run_directory(directory, "saved")
            self.assertEqual((run / "keep.txt").read_text(), "original")
            with self.assertRaises(ValueError):
                create_run_directory(directory, "../escape")

    def test_panda_archived_yaml_reloads_with_overrides_and_mesh_hashes(self):
        source = ROOT / "tests/fixtures/panda.yaml"
        config = load_panda_config(source)
        config = replace(
            config,
            trajectory=replace(config.trajectory, mode="direct", duration=0.1),
            safety=replace(
                config.safety, rdf=replace(config.safety.rdf, runtime_device="cuda")
            ),
        )
        raw = yaml.safe_load(source.read_text())
        raw["trajectory"].update(mode="direct", duration=0.1)
        raw["safety"].setdefault("rdf", {})["runtime_device"] = "cuda"
        with TemporaryDirectory() as directory:
            run = create_run_directory(directory)
            save_run_provenance(
                run,
                config_path=source,
                config=config,
                root=ROOT,
                resources=[config.model_path],
                effective_yaml=raw,
            )
            loaded = load_panda_config(run / "config.yaml")
            self.assertEqual(loaded.model_path, config.model_path)
            self.assertEqual(loaded.trajectory.mode, "direct")
            self.assertEqual(loaded.trajectory.duration, 0.1)
            self.assertEqual(loaded.safety.rdf.runtime_device, "cuda")
            metadata = json.loads((run / "metadata.json").read_text())
            self.assertGreater(len(metadata["resource_sha256"]), 1)
            self.assertIn("git", metadata)
            self.assertIn("control/panda_controller.py", metadata["source_sha256"])
            resolved = json.loads((run / "resolved_config.json").read_text())
            self.assertEqual(
                resolved["safe_ik"]["random_seed"], config.safe_ik.random_seed
            )

    def test_planar_archived_yaml_resolves_rdf_independent_of_working_directory(self):
        source = ROOT / "tests/fixtures/planar.yaml"
        config = load_controller_config(source)
        with TemporaryDirectory() as directory:
            run = create_run_directory(directory)
            save_run_provenance(
                run,
                config_path=source,
                config=config,
                root=Path(directory),
                resources=[Path(config.safety.rdf_model_path)],
            )
            loaded = load_controller_config(run / "config.yaml")
            self.assertEqual(loaded.safety.rdf_model_path, config.safety.rdf_model_path)
            self.assertTrue(Path(loaded.safety.rdf_model_path).is_file())
            metadata = json.loads((run / "metadata.json").read_text())
            self.assertIsNone(metadata["git"]["commit"])

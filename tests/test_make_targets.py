"""Panda Make commands select a locked CUDA environment without probing GPUs."""

import os
import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("make"), "make is unavailable")
class PandaMakeTargetsTest(unittest.TestCase):
    def command(self, target, *overrides):
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        for key in ("MAKEFLAGS", "MAKEOVERRIDES", "PANDA_RDF_DEVICE"):
            environment.pop(key, None)
        return subprocess.check_output(
            ["make", "--no-print-directory", "-n", target, "UV=uv", *overrides],
            cwd=root,
            env=environment,
            text=True,
        )

    def test_normal_run_targets_select_locked_cuda(self):
        for target in (
            "panda-run",
            "panda-view",
            "panda-direct",
            "panda-direct-view",
            "panda-sequential",
            "panda-sequential-view",
        ):
            with self.subTest(target=target):
                command = self.command(target)
                self.assertIn("uv run --locked --extra cuda safe-ik", command)
                self.assertIn('--rdf-device "cuda"', command)
        command = self.command("panda-view")
        self.assertIn("--visualize --show-plot", command)

    def test_explicit_cpu_has_no_cuda_extra(self):
        command = self.command(
            "panda-view", "PANDA_RDF_DEVICE=cpu", "ARGS=--mode direct --duration 0.1"
        )
        self.assertIn('--rdf-device "cpu"', command)
        self.assertNotIn("--extra cuda", command)
        self.assertIn("--locked", command)
        self.assertIn("--mode direct --duration 0.1", command)

    def test_explicit_gpu_targets_override_cpu_choice(self):
        for target in ("panda-gpu-run", "panda-gpu-view"):
            command = self.command(target, "PANDA_RDF_DEVICE=cpu")
            self.assertIn("--extra cuda", command)
            self.assertIn("--rdf-device cuda", command)

    def test_planar_command_does_not_install_cuda(self):
        command = self.command("planar-run")
        self.assertNotIn("--extra cuda", command)
        self.assertNotIn("--rdf-device", command)


if __name__ == "__main__":
    unittest.main()

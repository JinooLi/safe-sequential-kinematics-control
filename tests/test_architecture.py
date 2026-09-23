"""Keep dependency boundaries and command discovery intact."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/safe_ik"


class ArchitectureTest(unittest.TestCase):
    def test_core_has_no_application_or_backend_dependencies(self):
        for layer in ("domain", "control"):
            for path in (PACKAGE / layer).glob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    modules = (
                        [node.module or ""]
                        if isinstance(node, ast.ImportFrom)
                        else (
                            [item.name for item in node.names]
                            if isinstance(node, ast.Import)
                            else []
                        )
                    )
                    for module in modules:
                        self.assertTrue(
                            not set(module.split("."))
                            & {
                                "application",
                                "adapters",
                                "mujoco",
                                "pinocchio",
                                "osqp",
                                "yaml",
                            },
                            f"{path.name} imports {module}",
                        )

    def test_pure_imports_do_not_load_native_backends_or_plotting(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
import safe_ik.domain.safe_pose_ik
import safe_ik.control.panda_controller
import safe_ik.experiments.analysis
import safe_ik.logging.recorder
assert not {'pinocchio', 'mujoco', 'osqp', 'matplotlib.pyplot'} & sys.modules.keys()
""",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_installed_entry_points_work_outside_checkout(self):
        with TemporaryDirectory() as directory:
            for command in (
                "safe-ik",
                "safe-ik-compare",
                "safe-ik-train-rdf",
                "safe-ik-train-panda-rdf",
                "safe-ik-visualize-rdf",
            ):
                executable = Path(sys.executable).parent / command
                result = subprocess.run(
                    [str(executable), "--help"],
                    cwd=directory,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertIn("usage:", result.stdout)

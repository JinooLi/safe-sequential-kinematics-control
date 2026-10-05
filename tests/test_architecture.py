"""Keep dependency boundaries and command discovery intact."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/safe_ik"

# A -> B means that A may import B. Same-layer imports must also be acyclic.
ALLOWED_LAYERS = {
    "domain": set(),
    "control": {"domain"},
    "adapters": {"domain"},
    "logging": {"domain"},
    "experiments": {"domain"},
    "visualization": {"domain", "logging", "experiments"},
    "application": {
        "domain",
        "control",
        "adapters",
        "logging",
        "experiments",
        "visualization",
    },
}


def imports_in(source, module, *, is_package=False):
    """Resolve absolute/relative imports, including local and TYPE_CHECKING ones."""
    package = module.split(".") if is_package else module.split(".")[:-1]
    result = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            result.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parent = package[: len(package) - node.level + 1]
                base = ".".join(
                    parent + (node.module.split(".") if node.module else [])
                )
            else:
                base = node.module or ""
            result.add(base)
            result.update(f"{base}.{item.name}" for item in node.names)
    return result


def dependency_cycle(graph):
    """Return an import cycle as a closed path, or None for a DAG."""
    finished, active, path = set(), set(), []

    def visit(module):
        if module in active:
            return path[path.index(module) :] + [module]
        if module in finished:
            return None
        active.add(module)
        path.append(module)
        for target in sorted(graph[module]):
            cycle = visit(target)
            if cycle:
                return cycle
        path.pop()
        active.remove(module)
        finished.add(module)
        return None

    for module in sorted(graph):
        cycle = visit(module)
        if cycle:
            return cycle
    return None


class ArchitectureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = {
            ".".join(
                path.relative_to(PACKAGE.parent).with_suffix("").parts
            ).removesuffix(".__init__"): path
            for path in PACKAGE.rglob("*.py")
        }
        cls.imports = {
            module: imports_in(
                path.read_text(), module, is_package=path.name == "__init__.py"
            )
            for module, path in cls.modules.items()
        }
        cls.graph = {
            module: {target for target in targets if target in cls.modules}
            for module, targets in cls.imports.items()
        }

    def test_all_layers_follow_declared_dependency_direction(self):
        for module, targets in self.graph.items():
            layer = module.split(".")[1] if "." in module else None
            if layer is not None:
                self.assertIn(layer, ALLOWED_LAYERS, f"Undeclared layer: {module}")
            for target in targets:
                target_layer = target.split(".")[1] if "." in target else None
                if layer != target_layer:
                    self.assertIn(
                        target_layer,
                        ALLOWED_LAYERS.get(layer, set()),
                        f"Forbidden layer dependency: {module} -> {target}",
                    )

    def test_all_modules_are_acyclic_including_local_imports(self):
        cycle = dependency_cycle(self.graph)
        self.assertIsNone(cycle, "Import cycle: " + " -> ".join(cycle or []))

    def test_shared_contracts_do_not_depend_on_algorithms(self):
        contracts = {
            f"safe_ik.domain.{name}"
            for name in (
                "types",
                "interfaces",
                "safety_contracts",
                "ik_contracts",
                "panda_types",
                "panda_config",
                "planar_config",
            )
        }
        for module in contracts:
            self.assertLessEqual(self.graph[module], contracts, module)

    def test_core_has_no_application_or_backend_dependencies(self):
        backends = {
            "mujoco",
            "pinocchio",
            "osqp",
            "yaml",
            "torch",
            "trimesh",
            "matplotlib",
        }
        for module, imports in self.imports.items():
            if module.startswith(("safe_ik.domain.", "safe_ik.control.")):
                for target in imports:
                    self.assertNotIn(
                        target.split(".")[0], backends, f"{module} -> {target}"
                    )

    def test_data_consumers_do_not_load_control_or_execution(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
import safe_ik.domain.panda_types
import safe_ik.logging.panda_results
import safe_ik.visualization.panda
import safe_ik.visualization.panda_replay
assert not any(name.startswith(('safe_ik.control', 'safe_ik.application'))
               for name in sys.modules)
assert 'safe_ik.domain.safe_pose_ik' not in sys.modules
assert not {'pinocchio', 'mujoco', 'osqp', 'matplotlib.pyplot'} & sys.modules.keys()
""",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_rdf_and_ik_do_not_load_the_capsule_implementation(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
import safe_ik.domain.panda_rdf_safety
import safe_ik.domain.safe_pose_ik
assert 'safe_ik.domain.spatial_safety' not in sys.modules
""",
            ],
            check=True,
            capture_output=True,
            text=True,
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


class DependencyInspectionTest(unittest.TestCase):
    def test_resolves_absolute_relative_and_function_local_imports(self):
        imports = imports_in(
            """
import safe_ik.domain.types as values
from ..domain import interfaces
from .panda_runtime import PandaRuntime
def initialize():
    from . import panda_components
""",
            "safe_ik.application.parallel_ik",
        )
        self.assertTrue(
            {
                "safe_ik.domain.types",
                "safe_ik.domain.interfaces",
                "safe_ik.application.panda_runtime",
                "safe_ik.application.panda_components",
            }
            <= imports
        )
        self.assertIn(
            "safe_ik.domain.types",
            imports_in("from . import types", "safe_ik.domain", is_package=True),
        )

    def test_cycle_detection_accepts_shared_leaves_and_reports_closed_path(self):
        graph = {
            "app": {"left", "right"},
            "left": {"data"},
            "right": {"data"},
            "data": set(),
        }
        self.assertIsNone(dependency_cycle(graph))
        graph["data"].add("left")
        self.assertEqual(dependency_cycle(graph), ["left", "data", "left"])

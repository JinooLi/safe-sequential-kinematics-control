import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import yaml

from safe_ik.adapters.osqp_solver import OSQPSolver
from safe_ik.application.panda_comparison import atomic_json, digest
from safe_ik.application.panda_comparison_rollout import comparison_rollout
from safe_ik.application.panda_configuration import (
    load_panda_config,
    parse_panda_config,
)
from safe_ik.application.panda_experiment_configuration import load_panda_experiment
from safe_ik.application.panda_joint_clf_comparison import (
    METHODS,
    analyze,
    fixed_candidate,
    method_config,
    run,
    saved_inputs,
    verify_trace,
)
from safe_ik.application.panda_runtime import PandaRuntime
from safe_ik.control.panda_controller import PandaTrajectoryController
from safe_ik.domain.ik_contracts import SafeIKCandidate
from safe_ik.domain.safety_contracts import BarrierEvaluation
from safe_ik.domain.types import QPSolution, QPStatus

ROOT = Path(__file__).resolve().parents[1]


class Kinematics:
    lower_limits = np.full(7, -10.0)
    upper_limits = np.full(7, 10.0)

    def pose_and_jacobian(self, q):
        return q[:3].copy(), np.array([1.0, 0, 0, 0]), np.eye(6, 7)

    def frame_point_and_jacobian(self, name, q):
        return q[:3].copy(), np.eye(3, 7)


class Safety:
    def evaluate(self, q):
        return BarrierEvaluation(("obstacle_test",), np.array([1.0]), np.zeros((1, 7)))


class Plant:
    def __init__(self, config):
        self.q, self.time = config.initial_q.copy(), 0.0

    def advance(self, velocity, period):
        self.q += velocity * period
        self.time += period


class CaptureSolver:
    def solve(self, problem):
        self.problem = problem
        return QPSolution(QPStatus.SOLVED, np.zeros(8), 0.0)


class JointCLFTest(unittest.TestCase):
    def setUp(self):
        cfg = load_panda_config(ROOT / "tests/fixtures/panda.yaml")
        self.config = replace(
            cfg,
            initial_q=np.zeros(7),
            target=replace(
                cfg.target,
                position=np.array([0.02, 0, 0]),
                quaternion=np.array([1.0, 0, 0, 0]),
                position_tolerance=0.001,
            ),
            trajectory=replace(
                cfg.trajectory,
                mode="joint_clf",
                stage_task="position",
                duration=2.0,
                integration_period=0.02,
                clf_rate=20.0,
                clf_slack_weight=5000.0,
            ),
        )
        self.target = np.array([0.02, 0, 0, 0.1, 0, 0, 0])
        self.candidate = SafeIKCandidate(self.target, 3, 1, 0.0, 0.0, 1.0)

    def controller(self, config=None, candidates=None, solver=None):
        return PandaTrajectoryController(
            config or self.config,
            Kinematics(),
            Safety(),
            (self.candidate,) if candidates is None else candidates,
            solver or CaptureSolver(),
        )

    def test_joint_clf_gradient_and_positive_definite_nullspace_error(self):
        controller = self.controller()
        q = np.array([0.01, -0.03, 0.01, 0.06, 0.01, -0.02, 0.01])
        value, gradient, error = controller._active_clf(q)
        self.assertAlmostEqual(value, np.sum((q - self.target) ** 2))
        self.assertAlmostEqual(error, np.linalg.norm(q - self.target))
        epsilon = 1e-7
        numerical = [
            (
                controller._active_clf(q + epsilon * e)[0]
                - controller._active_clf(q - epsilon * e)[0]
            )
            / (2 * epsilon)
            for e in np.eye(7)
        ]
        np.testing.assert_allclose(gradient, numerical, atol=1e-9)
        self.assertEqual(controller._active_clf(self.target)[0], 0)
        q = self.target.copy()
        q[3] += 0.5  # Same EE pose in this toy kinematics, distinct configuration.
        self.assertAlmostEqual(controller._active_clf(q)[0], 0.25)
        self.assertEqual(controller.stage_names, ("joint_configuration",))
        # Primary comparison acceptance is still EE pose, not an added q tolerance.
        self.assertTrue(controller.status(q).success)

    def test_only_clf_qp_row_changes_and_candidate_is_required(self):
        joint_solver, direct_solver = CaptureSolver(), CaptureSolver()
        joint = self.controller(solver=joint_solver)
        direct = self.controller(
            replace(
                self.config, trajectory=replace(self.config.trajectory, mode="direct")
            ),
            candidates=(),
            solver=direct_solver,
        )
        joint.get_velocity(self.config.initial_q)
        direct.get_velocity(self.config.initial_q)
        a, b = joint_solver.problem, direct_solver.problem
        np.testing.assert_array_equal(
            a.A[0], np.r_[2 * (self.config.initial_q - self.target), -1]
        )
        self.assertAlmostEqual(a.upper[0], -20 * np.sum(self.target**2))
        for first, second in [
            (a.P, b.P),
            (a.q, b.q),
            (a.A[1:], b.A[1:]),
            (a.lower, b.lower),
            (a.upper[1:], b.upper[1:]),
        ]:
            np.testing.assert_array_equal(first, second)
        with self.assertRaisesRegex(ValueError, "safe IK candidate"):
            self.controller(candidates=())

    def test_joint_mode_yaml_and_fixed_rollout_logs(self):
        path = ROOT / "tests/fixtures/panda.yaml"
        raw = yaml.safe_load(path.read_text())
        raw["trajectory"].update(mode="joint_clf", stage_frames=[])
        self.assertEqual(parse_panda_config(raw, path).trajectory.mode, "joint_clf")
        runtime = PandaRuntime(Kinematics(), Safety(), OSQPSolver(), Plant(self.config))
        result, trace = comparison_rollout(
            self.config, runtime, (self.candidate,), subdivisions=1
        )
        self.assertTrue(result["goal_reached"])
        self.assertEqual(result["candidate_switch_count"], 0)
        result["method"] = "joint_clf"
        trace["reference_joint_error"] = np.linalg.norm(
            trace["q"] - self.target, axis=1
        )
        verify_trace(
            result, trace, {"initial_q": self.config.initial_q}, self.candidate, 1
        )
        trace["clf_value"][0] *= 0.5
        with self.assertRaises(AssertionError):
            verify_trace(
                result, trace, {"initial_q": self.config.initial_q}, self.candidate, 1
            )

    def test_slack_override_changes_only_joint_clf_objective_weight(self):
        for method in METHODS:
            original = method_config(self.config, method)
            changed = method_config(self.config, method, 1000)
            first, second = CaptureSolver(), CaptureSolver()
            self.controller(original, solver=first).get_velocity(self.config.initial_q)
            self.controller(changed, solver=second).get_velocity(self.config.initial_q)
            expected = first.problem.P.copy()
            if method == "joint_clf":
                expected[-1, -1] = 1000
            np.testing.assert_array_equal(second.problem.P, expected)
            for field in ("q", "A", "lower", "upper"):
                np.testing.assert_array_equal(
                    getattr(first.problem, field), getattr(second.problem, field)
                )
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                method_config(self.config, "joint_clf", invalid)

    def test_saved_ik_validation_rejects_different_or_unsafe_targets(self):
        item = dict(
            q=self.target.tolist(),
            seed_index=3,
            iterations=1,
            position_error=0.0,
            orientation_error=0.0,
            minimum_barrier=1.0,
        )
        ik = {"selected": item, "accepted_candidates": [item]}
        np.testing.assert_array_equal(
            fixed_candidate(ik, self.config, Kinematics(), Safety())[0].q, self.target
        )
        self.assertEqual(
            fixed_candidate({"selected": None}, self.config, Kinematics(), Safety()), ()
        )
        with self.assertRaisesRegex(ValueError, "absent"):
            fixed_candidate(
                dict(ik, accepted_candidates=[]), self.config, Kinematics(), Safety()
            )
        bad = replace(
            self.config, target=replace(self.config.target, position=np.ones(3))
        )
        with self.assertRaisesRegex(ValueError, "fails current pose"):
            fixed_candidate(ik, bad, Kinematics(), Safety())

    def test_three_method_export_keeps_missing_ik_and_seals_inputs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "reference"
            ref.mkdir()
            artifact = root / "rdf.npz"
            artifact.write_bytes(b"stub artifact; safety is mocked")
            path = ROOT / "tests/fixtures/panda.yaml"
            raw = yaml.safe_load(path.read_text())
            raw["robot"]["model_path"] = str(self.config.model_path)
            raw["safety"].setdefault("rdf", {})["artifact"] = str(artifact)
            raw["trajectory"].update(
                duration=0.04, integration_period=0.02, stage_task="position"
            )
            (root / "controller.yaml").write_text(yaml.safe_dump(raw))
            (ref / "config.yaml").write_text(yaml.safe_dump(raw))
            cases = [
                dict(
                    case_id=f"case{i}",
                    seed=2332,
                    ik_seed=i,
                    initial_q=np.zeros(7).tolist(),
                    target_position=[0.02, 0, 0],
                    target_quaternion=[1, 0, 0, 0],
                    obstacles=[],
                )
                for i in range(2)
            ]
            (ref / "cases.jsonl").write_text(
                "".join(json.dumps(c) + "\n" for c in cases)
            )
            atomic_json(
                ref / "manifest.json",
                {
                    "data_sha256": {
                        n: digest(ref / n) for n in ["cases.jsonl", "config.yaml"]
                    }
                },
            )
            for index, case in enumerate(cases):
                folder = ref / case["case_id"]
                folder.mkdir()
                item = dict(
                    q=self.target.tolist(),
                    seed_index=3,
                    iterations=1,
                    position_error=0.0,
                    orientation_error=0.0,
                    minimum_barrier=1.0,
                )
                atomic_json(
                    folder / "ik.json",
                    dict(
                        selected=item if index == 0 else None,
                        accepted_candidates=[item] if index == 0 else [],
                        reason=None if index == 0 else "ik_budget_exhausted",
                    ),
                )
                for method in ("direct", "sequential"):
                    atomic_json(
                        folder / method / "summary.json",
                        {"ik_sha256": digest(folder / "ik.json")},
                    )
            (root / "experiment.yaml").write_text(
                yaml.safe_dump(
                    dict(
                        controller_config="controller.yaml",
                        rdf_device="cpu",
                        use_all_ik_candidates=True,
                        main={"seeds": [2332], "cases_per_seed": 2},
                        validation={"subdivisions": 1},
                    )
                )
            )
            experiment = load_panda_experiment(root / "experiment.yaml")
            output = root / "new"
            with (
                patch(
                    "safe_ik.application.panda_joint_clf_comparison.build_panda_components",
                    return_value=(Kinematics(), Safety()),
                ),
                patch(
                    "safe_ik.application.panda_joint_clf_comparison.scene_safety",
                    return_value=Safety(),
                ),
                patch(
                    "safe_ik.application.panda_joint_clf_comparison.MujocoPlant",
                    side_effect=Plant,
                ) as plant,
            ):
                analysis = run(ref, experiment, output, 2)
                plant.reset_mock()
                joint_output = root / "joint_only"
                single = run(
                    ref,
                    experiment,
                    joint_output,
                    2,
                    methods=("joint_clf",),
                    joint_clf_slack_weight=1000,
                )
                self.assertTrue(single["complete"])
                self.assertEqual(set(single["methods"]), {"joint_clf"})
                self.assertEqual(plant.call_count, 1)  # Other scene lacks an IK.
                applied_config = plant.call_args.args[0]
                self.assertEqual(applied_config.trajectory.mode, "joint_clf")
                self.assertEqual(applied_config.trajectory.clf_slack_weight, 1000)
                joint_rows = json.loads((joint_output / "summary.json").read_text())
                self.assertEqual(len(joint_rows), 2)
                self.assertTrue(all(r["clf_slack_weight"] == 1000 for r in joint_rows))
                manifest = json.loads((joint_output / "manifest.json").read_text())
                self.assertEqual(manifest["joint_clf_slack_weight_override"], 1000)
                self.assertEqual(
                    manifest["method_configs"]["joint_clf"]["trajectory"][
                        "clf_slack_weight"
                    ],
                    1000,
                )
            self.assertTrue(analysis["complete"])
            self.assertEqual(analysis["ik_available_cases"], 1)
            self.assertTrue(all(x["cases"] == 2 for x in analysis["methods"].values()))
            rows = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(rows), 6)
            self.assertEqual(sum(r["reason"] == "ik_budget_exhausted" for r in rows), 2)
            resolved = yaml.safe_load((output / "experiment.resolved.yaml").read_text())
            self.assertFalse(resolved["use_all_ik_candidates"])
            for name, sha in json.loads((output / "complete.json").read_text())[
                "data_sha256"
            ].items():
                self.assertEqual(digest(output / name), sha)
            (ref / "case0/ik.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "IK hash mismatch"):
                saved_inputs(ref, 2)

    def test_analysis_requires_all_three_methods_and_rejects_duplicates(self):
        rows = [
            dict(
                case_id="a",
                method=m,
                success=m != "direct",
                goal_reached=True,
                safety_violation=m == "direct",
                reason="goal_reached",
                completion_time=1.0,
                ik_available=True,
            )
            for m in METHODS
        ]
        self.assertFalse(analyze(rows[:2], 1)["complete"])
        result = analyze(rows, 1)
        self.assertTrue(result["complete"])
        self.assertEqual(result["paired"]["direct_vs_joint_clf"]["second_only"], 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze(rows + rows[:1], 1)


if __name__ == "__main__":
    unittest.main()

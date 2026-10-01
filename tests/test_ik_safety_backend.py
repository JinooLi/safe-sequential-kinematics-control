"""Independent IK backends, worker parity and fail-closed trajectory handoff."""

import json
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import test_panda_rdf_integration as integration
import test_panda_runtime as runtime_fixture
import yaml

from safe_ik.application.artifacts import save_run_provenance
from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.application.panda_rdf_artifact import save_bundle
from safe_ik.application.panda_rollout import _validate_ik_candidates, run_panda
from safe_ik.application.panda_runtime import PandaRuntime
from safe_ik.application.panda_setup import build_panda_components, build_panda_runtime
from safe_ik.control.panda_controller import PandaTrajectoryController
from safe_ik.domain.panda_rdf_safety import PandaRDFSafety
from safe_ik.domain.safe_pose_ik import SafeIKCandidate, SafeIKResult
from safe_ik.domain.spatial_safety import BarrierEvaluation, PandaCapsuleSafety

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/panda.yaml"


class IKSafetyBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        integration.PandaRDFIntegrationTest.setUpClass()
        cls.fixture = integration.PandaRDFIntegrationTest()
        cls.base = cls.fixture.config

    def config(self, ik_model, control_model, **kwargs):
        return replace(
            self.base,
            safe_ik=replace(self.base.safe_ik, self_collision_model=ik_model, **kwargs),
            safety=replace(self.base.safety, self_collision_model=control_model),
        )

    def test_yaml_backend_selection_inheritance_and_invalid_values(self):
        raw = yaml.safe_load(FIXTURE.read_text())
        raw["robot"]["model_path"] = str(self.base.model_path)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for control in ("capsule", "rdf"):
                raw["safety"]["self_collision_model"] = control
                for ik in (None, "capsule", "rdf"):
                    raw["safe_ik"]["self_collision_model"] = ik
                    path.write_text(yaml.safe_dump(raw))
                    config = load_panda_config(path)
                    self.assertEqual(config.ik_self_collision_model, ik or control)
                    self.assertEqual(config.safety.self_collision_model, control)
                raw["safe_ik"].pop("self_collision_model")
                path.write_text(yaml.safe_dump(raw))
                self.assertEqual(
                    load_panda_config(path).ik_self_collision_model, control
                )
            for value in ("sphere", "RDF", "", 0, True, [], {}):
                with self.subTest(value=value):
                    raw["safe_ik"]["self_collision_model"] = value
                    path.write_text(yaml.safe_dump(raw))
                    with self.assertRaisesRegex(
                        ValueError, "safe_ik.self_collision_model"
                    ):
                        load_panda_config(path)

    def test_serial_and_spawned_workers_use_ik_not_trajectory_backend(self):
        bundle, options = self.fixture.bundle()
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "synthetic.npz"
            save_bundle(bundle, artifact)
            for ik, control in (("capsule", "rdf"), ("rdf", "capsule")):
                with self.subTest(ik=ik, control=control):
                    config = self.config(
                        ik,
                        control,
                        workers=1,
                        max_restarts=2,
                        iteration_number=3,
                        convergence_hold_duration=0.02,
                        safety_gradient_gain=1e-6,
                    )
                    config = replace(
                        config,
                        obstacles=(),
                        safety=replace(
                            config.safety, rdf=replace(options, artifact=artifact)
                        ),
                        target=replace(
                            config.target,
                            position_tolerance=100,
                            orientation_tolerance=100,
                        ),
                    )
                    with patch("safe_ik.application.panda_setup.MujocoPlant"):
                        serial = build_panda_runtime(config)
                        parallel = build_panda_runtime(
                            replace(config, safe_ik=replace(config.safe_ik, workers=2))
                        )
                    expected_type = (
                        PandaCapsuleSafety if ik == "capsule" else PandaRDFSafety
                    )
                    control_type = (
                        PandaCapsuleSafety if control == "capsule" else PandaRDFSafety
                    )
                    self.assertIsInstance(serial.ik_safety, expected_type)
                    self.assertIsInstance(serial.safety, control_type)
                    self.assertIsNot(serial.ik_safety, serial.safety)
                    with serial.ik_search_factory() as search:
                        self.assertIs(search.safety, serial.ik_safety)
                        expected = [
                            search.solve(require_safe=safe) for safe in (True, False)
                        ]
                    with parallel.ik_search_factory() as search:
                        for safe, reference in zip((True, False), expected):
                            result = search.solve(require_safe=safe)
                            self.assertGreater(len(reference.candidates), 0)
                            self.assertEqual(
                                result.failure_reason, reference.failure_reason
                            )
                            self.assertEqual(
                                result.selected.seed_index,
                                reference.selected.seed_index,
                            )
                            self.assertEqual(
                                [c.seed_index for c in result.candidates],
                                [c.seed_index for c in reference.candidates],
                            )
                            for actual, wanted in zip(
                                result.candidates, reference.candidates
                            ):
                                self.assertEqual(actual.iterations, wanted.iterations)
                                np.testing.assert_allclose(
                                    actual.q, wanted.q, atol=1e-12, rtol=0
                                )
                                self.assertAlmostEqual(
                                    actual.minimum_barrier,
                                    wanted.minimum_barrier,
                                    places=12,
                                )
                        if ik == "rdf":
                            self.assertGreater(
                                search.rdf_metrics["timings"]["distance"], 0
                            )
                        else:
                            self.assertEqual(search.rdf_metrics["timings"], {})

    def test_shared_backend_reuses_instance_and_direct_mode_skips_unused_rdf(self):
        with patch("safe_ik.application.panda_setup.MujocoPlant"):
            runtime = build_panda_runtime(self.config(None, "capsule"))
            self.assertIs(runtime.ik_safety, runtime.safety)
            config = self.config("rdf", "capsule")
            config = replace(
                config, trajectory=replace(config.trajectory, mode="direct")
            )
            with patch("safe_ik.application.panda_rdf_artifact.load_bundle") as load:
                runtime = build_panda_runtime(config)
                self.assertIs(runtime.ik_safety, runtime.safety)
                load.assert_not_called()

    def test_rdf_unavailable_fails_closed_for_either_selected_backend(self):
        for ik, control in (("rdf", "capsule"), ("capsule", "rdf")):
            with self.subTest(ik=ik), TemporaryDirectory() as directory:
                config = self.config(ik, control)
                config = replace(
                    config,
                    safety=replace(
                        config.safety,
                        rdf=replace(
                            config.safety.rdf, artifact=Path(directory) / "missing.npz"
                        ),
                    ),
                )
                with patch("safe_ik.application.panda_setup.MujocoPlant") as plant:
                    with self.assertRaisesRegex(ValueError, "Panda RDF unavailable"):
                        build_panda_runtime(config)
                    plant.assert_not_called()

    def test_capsule_components_do_not_load_rdf_even_with_trajectory_rdf(self):
        config = self.config("capsule", "rdf")
        with patch("safe_ik.application.panda_rdf_artifact.load_bundle") as load:
            _, safety = build_panda_components(
                config, self_collision_model=config.ik_self_collision_model
            )
            self.assertIsInstance(safety, PandaCapsuleSafety)
            load.assert_not_called()

    def test_ik_only_rdf_artifact_is_in_run_provenance(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            artifact = directory / "test.npz"
            artifact.write_bytes(b"provenance test only")
            config = self.config("rdf", "capsule")
            config = replace(
                config,
                safety=replace(
                    config.safety, rdf=replace(config.safety.rdf, artifact=artifact)
                ),
            )
            output = directory / "run"
            output.mkdir()
            raw = yaml.safe_load(FIXTURE.read_text())
            raw["safe_ik"]["self_collision_model"] = "rdf"
            save_run_provenance(
                output,
                config_path=FIXTURE,
                config=config,
                root=ROOT,
                resources=[],
                effective_yaml=raw,
            )
            archived = load_panda_config(output / "config.yaml")
            self.assertEqual(archived.ik_self_collision_model, "rdf")
            self.assertEqual(archived.safety.rdf.artifact, artifact)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertIn(str(artifact), metadata["resource_sha256"])


class IKCandidateHandoffTest(unittest.TestCase):
    def setUp(self):
        base = load_panda_config(FIXTURE)
        self.config = replace(
            base,
            safe_ik=replace(base.safe_ik, self_collision_model="capsule"),
            safety=replace(base.safety, self_collision_model="rdf"),
        )
        self.candidates = tuple(
            SafeIKCandidate(
                base.initial_q + np.eye(7)[0] * distance, index, 2, 0.0, 0.0, 0.5
            )
            for index, distance in enumerate((0.1, 0.2, 0.3))
        )
        self.search = Mock()
        self.search.rdf_metrics = None
        self.search.solve.return_value = SafeIKResult(
            self.candidates, self.candidates[0], None
        )
        self.safety = Mock(spec=runtime_fixture.ConstantSafety)
        self.ik_safety = runtime_fixture.ConstantSafety()
        self.runtime = PandaRuntime(
            runtime_fixture.LinearKinematics(),
            self.safety,
            runtime_fixture.FixedSolver(),
            runtime_fixture.EulerPlant(base.initial_q),
            lambda: nullcontext(self.search),
            self.ik_safety,
        )

    def evaluation(self, value, gradient=0.0):
        return BarrierEvaluation(
            ("self_pair",), np.array([value]), np.full((1, 7), gradient)
        )

    def test_filters_all_candidates_selects_nearest_remaining_and_keeps_cbf_backend(
        self,
    ):
        self.safety.evaluate.side_effect = lambda q: self.evaluation(
            -0.01 if q[0] in (0.1, 0.3) else 0.0
        )
        with patch(
            "safe_ik.application.panda_rollout.PandaTrajectoryController",
            wraps=PandaTrajectoryController,
        ) as controller:
            result = run_panda(self.config, self.runtime, should_continue=lambda: False)
        self.assertIs(result.ik.selected, self.candidates[1])
        self.assertEqual(len(result.ik.candidates), 1)
        self.assertIs(controller.call_args.args[2], self.safety)
        self.assertEqual(controller.call_args.args[3], (self.candidates[1],))
        validation = result.safety_metadata["ik_candidate_validation"]
        self.assertEqual((validation["checked"], validation["accepted"]), (3, 1))
        self.assertEqual(
            validation["candidates"][0]["violations"], {"self_pair": -0.01}
        )
        self.assertEqual(result.safety_metadata["safe_ik_model"], "capsule")
        self.assertNotIn("safe_ik_rdf", result.safety_metadata)
        self.assertEqual(result.ik.selected.minimum_barrier, 0.5)
        self.search.solve.assert_called_once()

    def test_all_rejected_stops_without_motion_or_pose_only_diagnosis(self):
        self.safety.evaluate.side_effect = lambda q: self.evaluation(
            0.1 if q[0] == 0 else -0.01
        )
        with patch(
            "safe_ik.application.panda_rollout.PandaTrajectoryController"
        ) as controller:
            result = run_panda(self.config, self.runtime)
        self.assertEqual(
            result.ik.failure_reason, "ik_candidates_unsafe_for_trajectory"
        )
        self.assertEqual(result.ik.candidates, ())
        self.assertEqual(result.steps, 0)
        self.assertEqual(self.runtime.plant.commands, [])
        self.search.solve.assert_called_once()
        controller.assert_not_called()

    def test_nonfinite_values_or_gradients_cannot_pass_handoff(self):
        for value, gradient in (
            (float("nan"), 0),
            (float("inf"), 0),
            (0.1, float("nan")),
        ):
            with self.subTest(value=value, gradient=gradient):
                self.safety.evaluate.return_value = self.evaluation(value, gradient)
                result, metadata = _validate_ik_candidates(
                    self.search.solve.return_value, self.safety, self.config
                )
                self.assertIsNone(result.selected)
                self.assertEqual(metadata["accepted"], 0)

    def test_initial_rdf_violation_still_stops_before_capsule_ik(self):
        self.safety.evaluate.return_value = self.evaluation(-0.01)
        result = run_panda(self.config, self.runtime)
        self.assertEqual(result.failure["reason"], "unsafe_initial_state")
        self.search.solve.assert_not_called()
        self.assertEqual(self.runtime.plant.commands, [])

    def test_separate_backend_cannot_silently_use_trajectory_in_injected_runtime(self):
        self.runtime.ik_safety = None
        with self.assertRaisesRegex(ValueError, "PandaRuntime.ik_safety"):
            run_panda(self.config, self.runtime)

    def test_rdf_ik_metrics_come_from_ik_backend_not_capsule_control(self):
        config = replace(
            self.config,
            safe_ik=replace(self.config.safe_ik, self_collision_model="rdf"),
            safety=replace(self.config.safety, self_collision_model="capsule"),
        )
        self.safety.evaluate.return_value = self.evaluation(0.1)
        self.ik_safety.total_timings = {"distance": 5.0}
        self.ik_safety.artifact_id = "ik-only-test"
        self.ik_safety.query_policy = PandaRDFSafety.query_policy
        self.ik_safety.gate_metadata = {"policy": "ik-test-gate"}
        self.ik_safety.runtime_metadata = {"device": "cuda", "dtype": "float64"}

        def solve(*args, **kwargs):
            self.ik_safety.total_timings["distance"] += 3.0
            return SafeIKResult(self.candidates, self.candidates[0], None)

        self.search.solve.side_effect = solve
        result = run_panda(config, self.runtime, should_continue=lambda: False)
        self.assertEqual(result.safety_metadata["model"], "capsule")
        self.assertEqual(result.safety_metadata["safe_ik_artifact_id"], "ik-only-test")
        self.assertIsNone(result.safety_metadata["rdf_query_policy"])
        self.assertIsNone(result.safety_metadata["rdf_sphere_gate"])
        self.assertIsNone(result.safety_metadata["rdf_runtime"])
        self.assertEqual(
            result.safety_metadata["safe_ik_rdf_runtime"],
            self.ik_safety.runtime_metadata,
        )
        self.assertEqual(
            result.safety_metadata["safe_ik_rdf_sphere_gate"],
            self.ik_safety.gate_metadata,
        )
        self.assertEqual(
            result.safety_metadata["safe_ik_rdf_query_policy"],
            PandaRDFSafety.query_policy,
        )
        self.assertEqual(
            result.safety_metadata["safe_ik_rdf"]["timings"], {"distance": 3.0}
        )
        self.assertEqual(
            result.safety_metadata["ik_candidate_validation"]["accepted"], 3
        )

    def test_injected_serial_fallback_uses_explicit_ik_safety(self):
        self.safety.evaluate.return_value = self.evaluation(0.1)
        self.runtime.ik_search_factory = None
        with patch(
            "safe_ik.application.panda_rollout.SafePoseIKSolver",
            return_value=self.search,
        ) as solver:
            run_panda(self.config, self.runtime, should_continue=lambda: False)
        self.assertIs(solver.call_args.args[2], self.ik_safety)

    def test_same_backend_does_not_add_handoff_evaluations(self):
        self.safety.evaluate.return_value = self.evaluation(0.1)
        config = replace(
            self.config, safe_ik=replace(self.config.safe_ik, self_collision_model=None)
        )
        self.runtime.ik_safety = None
        with patch(
            "safe_ik.application.panda_rollout._validate_ik_candidates"
        ) as validate:
            result = run_panda(config, self.runtime, should_continue=lambda: False)
        validate.assert_not_called()
        self.assertIsNone(result.safety_metadata["ik_candidate_validation"])

    def test_direct_mode_ignores_ik_backend_and_never_searches(self):
        self.safety.evaluate.return_value = self.evaluation(0.1)
        config = replace(
            self.config, trajectory=replace(self.config.trajectory, mode="direct")
        )
        self.runtime.ik_safety = None
        result = run_panda(config, self.runtime, should_continue=lambda: False)
        self.assertTrue(result.ik_skipped)
        self.assertIsNone(result.safety_metadata["safe_ik_model"])
        self.search.solve.assert_not_called()

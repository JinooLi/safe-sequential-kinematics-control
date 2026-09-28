"""Geometry, moving-frame derivatives, fail-closed loading and CBF integration."""

import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np
import trimesh

from safe_ik.adapters.panda_kinematics import PandaKinematics
from safe_ik.adapters.panda_rdf_geometry import extract_visual_geometry
from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.application.panda_mesh_reconstruction import REFERENCE, mesh_digest
from safe_ik.application.panda_rdf_artifact import (
    RDFBundle,
    geometry_settings,
    load_bundle,
    read_candidate_for_validation,
    save_bundle,
)
from safe_ik.application.panda_rdf_training import (
    TrainingOptions,
    estimate_maximum,
    mesh_truth,
    minimum_sphere,
    surface_queries,
    validate_joint_gradients,
)
from safe_ik.application.panda_rollout import run_panda
from safe_ik.application.panda_runtime import PandaRuntime
from safe_ik.application.panda_setup import build_panda_components
from safe_ik.application.parallel_ik import panda_ik_search
from safe_ik.control.panda_controller import PandaTrajectoryController
from safe_ik.domain.panda_config import PandaRDFConfig
from safe_ik.domain.panda_rdf import BernsteinRDF, ExtendedRDF
from safe_ik.domain.panda_rdf_safety import PandaRDFSafety
from safe_ik.domain.safe_pose_ik import SafeIKResult, SafePoseIKSolver
from safe_ik.domain.spatial_safety import BarrierEvaluation, PandaCapsuleSafety
from safe_ik.domain.types import QPSolution, QPStatus

ROOT = Path(__file__).resolve().parents[1]


class PandaRDFIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_panda_config(ROOT / "tests/fixtures/panda.yaml")
        cls.geometry = extract_visual_geometry(cls.config.model_path)
        cls.kinematics = PandaKinematics(
            cls.config.model_path, cls.config.ee_frame, cls.geometry.vertices
        )

    def bundle(self):
        options = PandaRDFConfig(query_density=10)
        # Deliberately synthetic *test* artifact, never used as mesh approval.
        models, queries, spheres, validation = {}, {}, {}, {}
        rng = np.random.default_rng(4)
        r0, r1 = 0.2 + options.effective_margin, 1.5 * (0.2 + options.effective_margin)
        count = int(np.ceil(options.query_density * 4 * np.pi * r0 * r0))
        for name in self.geometry.vertices:
            weights = rng.normal(0.1, 0.01, (3, 3, 3))
            models[name] = ExtendedRDF(BernsteinRDF(np.zeros(3), r1, weights), r0, 0.2)
            queries[name] = rng.uniform(-0.05, 0.05, (count, 3))
            spheres[name] = {"mesh_radius": 0.2}
            validation[name] = {
                "passed": True,
                "sample_counts": [3000, 2000],
                "near_max_error": 0.001,
                "derivatives_passed": True,
                "reference_mesh_digest": mesh_digest(trimesh.creation.box()),
            }
        metadata = {
            "complete": True,
            "geometry_fingerprint": self.geometry.fingerprint,
            "settings": geometry_settings(options),
            "spheres": spheres,
            "validation": validation,
            "joint_validation": {"passed": True},
            "validation_reference": REFERENCE,
            "reconstruction": {
                name: {
                    "reference": REFERENCE,
                    "integrity_passed": True,
                    "mesh_digest": mesh_digest(trimesh.creation.box()),
                }
                for name in models
            },
        }
        return (
            RDFBundle(
                models,
                queries,
                self.geometry.pairs,
                metadata,
                reconstructed={name: trimesh.creation.box() for name in models},
            ),
            options,
        )

    def test_geometry_uses_parent_adjacency_and_closed_fingers(self):
        g = self.geometry
        self.assertEqual(len(g.vertices), 9)
        self.assertEqual(len(g.pairs), 28)
        self.assertNotIn(("link7", "hand"), g.pairs)
        self.assertIn(("link6", "hand"), g.pairs)
        self.assertIn(("link0", "hand"), g.pairs)
        sources = [v for v in g.provenance["visual_meshes"] if v["link"] == "hand"]
        self.assertEqual(
            {v["body"] for v in sources}, {"hand", "left_finger", "right_finger"}
        )

    def test_minimum_sphere_and_query_surface(self):
        mesh = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
        mesh.apply_translation([0.2, -0.1, 0.3])
        center, radius = minimum_sphere(mesh.vertices)
        np.testing.assert_allclose(center, [0.2, -0.1, 0.3], atol=1e-8)
        self.assertAlmostEqual(radius, np.linalg.norm([0.05, 0.1, 0.15]), places=8)
        points = surface_queries(mesh, 24, np.random.default_rng(0))
        self.assertLess(trimesh.proximity.closest_point(mesh, points)[1].max(), 1e-10)
        np.testing.assert_array_equal(
            points, surface_queries(mesh, 24, np.random.default_rng(0))
        )

    def test_maximum_linear_field_has_known_sphere_solution(self):
        weights = np.broadcast_to(
            np.array([-1.0, 1.0])[:, None, None], (2, 2, 2)
        ).copy()
        rdf = BernsteinRDF(np.zeros(3), 1.0, weights)
        maximum, report = estimate_maximum(
            rdf, 0.5, TrainingOptions(maximum_samples=100, maximum_starts=4)
        )
        self.assertAlmostEqual(maximum, 0.5, places=7)
        self.assertEqual(len(report["runs"]), 4)

    def test_real_panda_relative_and_aggregated_joint_gradients(self):
        bundle, options = self.bundle()
        report = validate_joint_gradients(
            self.kinematics, bundle, options, np.random.default_rng(0)
        )
        self.assertTrue(report["passed"], report)

    def test_obstacles_remain_exact_capsules_and_cache_reuses_evaluation(self):
        bundle, options = self.bundle()
        cfg = self.config
        capsule = PandaCapsuleSafety(
            self.kinematics,
            cfg.safety.capsules,
            cfg.obstacles,
            cfg.safety.margin,
            include_self=False,
        )
        rdf = PandaRDFSafety(
            self.kinematics,
            bundle.models,
            bundle.queries,
            bundle.pairs,
            options,
            capsule,
        )
        q = cfg.initial_q
        evaluated = rdf.evaluate(q)
        self.assertIs(evaluated, rdf.evaluate(q.copy()))
        exact = capsule.evaluate(q)
        self.assertEqual(evaluated.labels[28:], exact.labels)
        np.testing.assert_array_equal(evaluated.values[28:], exact.values)
        np.testing.assert_array_equal(evaluated.gradients[28:], exact.gradients)

    def test_artifact_roundtrip_and_dependency_compatibility(self):
        bundle, options = self.bundle()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rdf.npz"
            identifier = save_bundle(bundle, path)
            loaded = load_bundle(path, self.geometry, options)
            self.assertEqual(identifier, loaded.identifier)
            load_bundle(
                path,
                self.geometry,
                replace(options, top_k=None, cbf_gain=2, softmin_error=0.002),
            )
            for key, value in (
                ("physical_margin", 0.04),
                ("query_density", 30),
                ("seed", 1),
                ("transition_ratio", 0.7),
            ):
                with self.assertRaisesRegex(ValueError, key):
                    load_bundle(path, self.geometry, replace(options, **{key: value}))
            bundle.metadata["validation"]["link0"]["near_max_error"] = 0.006
            save_bundle(bundle, path)
            with self.assertRaisesRegex(ValueError, "link0.*validation failed"):
                load_bundle(path, self.geometry, options)
            with self.assertRaisesRegex(ValueError, "Regenerate"):
                load_bundle(path.with_name("missing.npz"), self.geometry, options)

    def test_default_rdf_missing_artifact_never_falls_back(self):
        config = replace(
            self.config,
            safety=replace(
                self.config.safety,
                self_collision_model="rdf",
                rdf=PandaRDFConfig(
                    artifact=Path("/tmp/nonexistent-panda-test-rdf.npz")
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "Panda RDF unavailable"):
            build_panda_components(config)

    def test_triangle_oracle_known_box_distances_and_signs(self):
        mesh = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
        points = np.array([[0.08, 0.01, 0.01], [0.04, 0.01, 0.01], [-0.09, 0.01, 0.01]])
        distance, gradient, ambiguous = mesh_truth(mesh, points)
        np.testing.assert_allclose(distance, [0.03, -0.01, 0.04], atol=1e-12)
        np.testing.assert_allclose(
            gradient, [[1, 0, 0], [1, 0, 0], [-1, 0, 0]], atol=1e-12
        )
        self.assertFalse(ambiguous.any())

    def test_query_refresh_keeps_coefficients_but_runtime_requires_new_samples(self):
        bundle, options = self.bundle()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.npz"
            save_bundle(bundle, path)
            changed = replace(options, query_density=20, seed=2)
            refreshed = read_candidate_for_validation(
                path, self.geometry, changed, refresh_queries=True
            )
            np.testing.assert_array_equal(
                refreshed.models["link0"].rdf.weights,
                bundle.models["link0"].rdf.weights,
            )
            save_bundle(refreshed, path)  # not resampled yet: still must be rejected
            with self.assertRaisesRegex(ValueError, "query count mismatch"):
                load_bundle(path, self.geometry, changed)

    def test_worker_loads_same_artifact_and_preserves_ik_candidates(self):
        bundle, options = self.bundle()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            save_bundle(bundle, path)
            config = replace(
                self.runtime_config(),
                obstacles=(),
                safety=replace(
                    self.config.safety,
                    self_collision_model="rdf",
                    rdf=replace(options, artifact=path),
                ),
                safe_ik=replace(
                    self.config.safe_ik,
                    workers=2,
                    max_restarts=2,
                    duration_per_seed=0.03,
                    convergence_hold_duration=0.02,
                    safety_gradient_gain=1e-6,
                ),
                target=replace(
                    self.config.target,
                    position_tolerance=100,
                    orientation_tolerance=100,
                ),
            )
            kinematics, safety = build_panda_components(config)
            serial = SafePoseIKSolver(config, kinematics, safety).solve()
            with panda_ik_search(config, kinematics, safety) as parallel:
                result = parallel.solve()
                self.assertGreater(parallel.rdf_metrics["timings"]["distance"], 0)
            self.assertEqual(len(serial.candidates), 2)
            self.assertEqual(
                [c.seed_index for c in result.candidates],
                [c.seed_index for c in serial.candidates],
            )
            for actual, expected in zip(result.candidates, serial.candidates):
                np.testing.assert_allclose(actual.q, expected.q, atol=1e-12, rtol=0)

    def test_nonfinite_solved_qp_is_stopped_and_diagnostics_are_saved(self):
        config = self.runtime_config()
        barrier = BarrierEvaluation(
            ("self_link0_hand",), np.array([0.1]), np.ones((1, 7))
        )
        safety, plant, solver = Mock(), Mock(), Mock()
        safety.evaluate.return_value = barrier
        safety.artifact_id, safety.last_timings = "synthetic", {}
        plant.q, plant.time = config.initial_q.copy(), 0.0
        solver.solve.return_value = QPSolution(
            QPStatus.SOLVED, np.full(8, np.nan), 0.001
        )
        result = run_panda(config, PandaRuntime(self.kinematics, safety, solver, plant))
        plant.advance.assert_not_called()
        self.assertEqual(result.failure["solver_status"], "failed")
        self.assertEqual(result.failure["q"], config.initial_q.tolist())
        self.assertEqual(result.failure["barriers"], {"self_link0_hand": 0.1})
        self.assertEqual(result.trace.minimum_self_pair, ("self_link0_hand",))

    def runtime_config(self):
        return replace(
            self.config,
            safety=replace(
                self.config.safety,
                self_collision_model="rdf",
                rdf=replace(self.config.safety.rdf, cbf_gain=2),
            ),
            trajectory=replace(self.config.trajectory, mode="direct", duration=0.05),
        )

    def test_initial_violation_prevents_ik_solver_and_plant_advance(self):
        config = self.runtime_config()
        barrier = BarrierEvaluation(
            ("self_link0_hand",), np.array([-0.01]), np.ones((1, 7))
        )
        safety, plant, solver = Mock(), Mock(), Mock()
        safety.evaluate.return_value = barrier
        safety.artifact_id = "test"
        safety.last_timings = {}
        plant.time, plant.q = 0.0, config.initial_q.copy()
        result = run_panda(config, PandaRuntime(self.kinematics, safety, solver, plant))
        self.assertEqual(result.failure["reason"], "unsafe_initial_state")
        self.assertEqual(result.failure["violations"], {"self_link0_hand": -0.01})
        solver.solve.assert_not_called()
        plant.advance.assert_not_called()

    def test_failed_ik_keeps_result_fields_and_safety_metadata(self):
        config = replace(self.runtime_config(), trajectory=self.config.trajectory)
        safety, plant, solver, search = Mock(), Mock(), Mock(), Mock()
        safety.evaluate.return_value = BarrierEvaluation(
            ("self_link0_hand",), np.array([0.1]), np.ones((1, 7))
        )
        safety.artifact_id, safety.total_timings = "synthetic", {}
        safety.last_timings = {}
        safety.minimum_self_barrier, safety.minimum_self_pair = 0.1, "self_link0_hand"
        plant.q, plant.time = config.initial_q.copy(), 0.0
        search.solve.return_value = SafeIKResult((), None, "not_found")
        search.rdf_metrics = None
        runtime = PandaRuntime(
            self.kinematics, safety, solver, plant, lambda: nullcontext(search)
        )
        result = run_panda(config, runtime)
        self.assertEqual(result.ik.failure_reason, "pose_ik_unreachable")
        self.assertEqual(result.elapsed, 0.0)
        self.assertEqual(result.steps, 0)
        np.testing.assert_array_equal(result.final_q, config.initial_q)
        self.assertEqual(result.safety_metadata["artifact_id"], "synthetic")
        plant.advance.assert_not_called()

    def test_self_cbf_has_own_gain_and_no_clf_slack(self):
        config = self.runtime_config()
        barrier = BarrierEvaluation(
            ("self_link0_hand", "obstacle1_base"), np.array([0.1, 0.2]), np.ones((2, 7))
        )
        safety, solver = Mock(), Mock()
        safety.evaluate.return_value = barrier
        solver.solve.return_value = QPSolution(QPStatus.SOLVED, np.zeros(8), 0.01)
        controller = PandaTrajectoryController(
            config, self.kinematics, safety, (), solver
        )
        controller.get_velocity(config.initial_q)
        problem = solver.solve.call_args.args[0]
        np.testing.assert_allclose(
            problem.lower[1:3], [-0.2, -config.safety.alpha * 0.2]
        )
        np.testing.assert_array_equal(problem.A[1:3, -1], 0)
        calls = safety.evaluate.call_count
        controller.safety_diagnostics(config.initial_q)
        controller.status(config.initial_q + 0.001)
        self.assertEqual(
            safety.evaluate.call_count, calls
        )  # no post-integration RDF check


if __name__ == "__main__":
    unittest.main()

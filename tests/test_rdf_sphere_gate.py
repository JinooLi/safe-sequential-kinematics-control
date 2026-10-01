"""Safe transition geometry, full chain rule, branch skipping and compatibility."""

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_panda_rdf_integration as integration
from scipy.spatial.transform import Rotation

from safe_ik.application.panda_rdf_artifact import geometry_settings, save_bundle
from safe_ik.application.panda_rdf_training import validate_joint_gradients
from safe_ik.application.panda_setup import build_panda_components, build_panda_runtime
from safe_ik.domain.panda_config import PandaRDFConfig
from safe_ik.domain.panda_rdf_safety import PandaRDFSafety
from safe_ik.domain.spatial_safety import BarrierEvaluation


class MovingCenters:
    def frame_pose_and_jacobian(self, name, q):
        jac = np.zeros((6, 2))
        if name == "link1":
            rotation = Rotation.from_rotvec([0, 0, 0.2 * q[1]]).as_matrix()
            origin = np.array([0, 0.3 * q[1], 0])
            jac[1, 1], jac[5, 1] = 0.3, 0.2
        else:
            rotation = Rotation.from_rotvec([0, 0.4 * q[1], 0]).as_matrix()
            origin = np.array([q[0], 0, 0])
            jac[0, 0], jac[4, 1] = 1, 0.4
        return rotation, origin, jac


class Field:
    def __init__(self, center):
        self.rdf = SimpleNamespace(center=np.asarray(center), radius=0.6)
        self.calls = 0

    def query(self, points, gradient=True):
        self.calls += 1
        p = points - self.rdf.center
        scales = np.array([0.4, 0.6, 0.8])
        value = 0.1 + np.sum(p**2 * scales, axis=1) + 0.02 * p[:, 1]
        grad = 2 * p * scales + [0, 0.02, 0]
        return value, grad if gradient else np.empty((0, 3))


class Obstacles:
    def evaluate(self, q):
        return BarrierEvaluation(
            ("obstacle",),
            np.array([0.2 + 0.3 * q[0] - 0.4 * q[1]]),
            np.array([[0.3, -0.4]]),
        )


class SphereGateTest(unittest.TestCase):
    def fixture(self, *, centered=False, reverse=False, fraction=0.5):
        models = {
            "link1": Field([0, 0, 0] if centered else [0.05, 0.01, 0.02]),
            "link4": Field([0, 0, 0] if centered else [-0.03, 0.02, 0.01]),
        }
        points = np.array(
            [
                [0.11, 0.01, 0],
                [-0.1, 0.03, 0.02],
                [0.02, 0.08, -0.01],
                [0.03, -0.09, 0.01],
                [0.01, 0.03, 0.1],
            ]
        )
        queries = {n: f.rdf.center + points for n, f in models.items()}
        vertices = {
            n: f.rdf.center + np.vstack((np.eye(3), -np.eye(3))) * 0.12
            for n, f in models.items()
        }
        options = PandaRDFConfig(
            top_k=2,
            physical_margin=0.01,
            sphere_gate=True,
            sphere_gate_transition_fraction=fraction,
        )
        pairs = (("link4", "link1"),) if reverse else (("link1", "link4"),)
        kin = MovingCenters()
        gated = PandaRDFSafety(
            kin,
            models,
            queries,
            pairs,
            options,
            Obstacles(),
            bounding_vertices=vertices,
        )
        plain = PandaRDFSafety(
            kin,
            models,
            queries,
            pairs,
            replace(options, sphere_gate=False),
            Obstacles(),
        )
        return gated, plain, vertices

    def q_for_length(self, length):
        return np.array([0.08 + np.sqrt(length**2 - 0.0002), 0.0])

    def gates(self, safety, q):
        poses = {
            n: safety.kinematics.frame_pose_and_jacobian(n, q) for n in safety.models
        }
        return safety.sphere_gate.evaluate(poses)

    def test_options_are_runtime_only_and_validate_strictly(self):
        plain = PandaRDFConfig()
        self.assertFalse(plain.sphere_gate)
        self.assertEqual(
            geometry_settings(plain),
            geometry_settings(replace(plain, sphere_gate=True)),
        )
        for bad in (0, 1, "true", None):
            with self.subTest(boolean=bad), self.assertRaises(ValueError):
                replace(plain, sphere_gate=bad)
        for bad in (0, 1, -0.2, 2, True, "0.5", float("nan"), float("inf")):
            with self.subTest(fraction=bad), self.assertRaises(ValueError):
                replace(plain, sphere_gate_transition_fraction=bad)

    def test_far_skips_every_rdf_call_but_keeps_row_and_obstacles(self):
        safety, _, _ = self.fixture()
        q = self.q_for_length(1.3)
        expected = self.gates(safety, q)
        result = safety.evaluate(q)
        self.assertEqual(result.labels, ("self_link1_link4", "obstacle"))
        self.assertEqual(sum(f.calls for f in safety.models.values()), 0)
        np.testing.assert_array_equal(result.values[:1], expected.values)
        np.testing.assert_array_equal(result.gradients[:1], expected.gradients)
        np.testing.assert_array_equal(result.values[1:], Obstacles().evaluate(q).values)
        np.testing.assert_array_equal(
            result.gradients[1:], Obstacles().evaluate(q).gradients
        )
        self.assertEqual(
            safety.last_gate_counts, dict(sphere_only=1, transition=0, rdf_only=0)
        )
        self.assertEqual(safety.last_timings["distance"], 0)
        self.assertEqual(safety.last_timings["selection"], 0)
        self.assertIs(safety.evaluate(q.copy()), result)
        self.assertEqual(
            safety.last_timings, dict(distance=0.0, gradient=0.0, selection=0.0)
        )

    def test_close_pairs_are_bitwise_unchanged_in_either_pair_order(self):
        for reverse in (False, True):
            safety, plain, _ = self.fixture(reverse=reverse)
            q = self.q_for_length(0.4)
            actual, expected = safety.evaluate(q), plain.evaluate(q)
            self.assertEqual(actual.labels, expected.labels)
            np.testing.assert_array_equal(actual.values, expected.values)
            np.testing.assert_array_equal(actual.gradients, expected.gradients)
            self.assertEqual(
                safety.last_gate_counts, dict(sphere_only=0, transition=0, rdf_only=1)
            )

    def test_every_transition_stays_in_certified_sphere_clearance(self):
        for fraction in (0.1, 0.5, 0.9):
            safety, _, _ = self.fixture(fraction=fraction)
            gate = safety.sphere_gate
            threshold = gate.outer_sums[0]
            width = gate.widths[0]
            for length in np.linspace(threshold - width, threshold, 101):
                value = self.gates(safety, self.q_for_length(length))
                self.assertGreater(value.values[0], 0)
                self.assertGreaterEqual(value.weights[0], 0)
                self.assertLessEqual(value.weights[0], 1)
            # Any possibly unsafe sphere clearance is inside the full-RDF region.
            inside = self.q_for_length(gate.clearance_sums[0] - 0.01)
            self.assertEqual(self.gates(safety, inside).weights[0], 1)

    def test_full_gradient_including_weight_and_rotating_centers_matches_fd(self):
        safety, plain, _ = self.fixture()
        step = 1e-6
        for length in (0.4, 0.9, 1.3):
            q = self.q_for_length(length) + [0, 0.17]
            analytic = safety.evaluate(q).gradients.copy()
            numeric = np.column_stack(
                [
                    (safety.evaluate(q + a).values - safety.evaluate(q - a).values)
                    / (2 * step)
                    for a in np.eye(2) * step
                ]
            )
            np.testing.assert_allclose(analytic, numeric, atol=1e-8, rtol=0)
        q = self.q_for_length(0.9) + [0, 0.17]
        gates = self.gates(safety, q)
        self.assertTrue(0 < gates.weights[0] < 1)
        incomplete = (1 - gates.weights[0]) * gates.gradients[0] + gates.weights[
            0
        ] * plain.evaluate(q).gradients[0]
        self.assertGreater(
            np.max(np.abs(safety.evaluate(q).gradients[0] - incomplete)), 0.01
        )

    def test_both_transition_boundaries_are_value_and_gradient_continuous(self):
        safety, _, _ = self.fixture()
        gate = safety.sphere_gate
        for length in (gate.outer_sums[0], gate.outer_sums[0] - gate.widths[0]):
            center = safety.evaluate(self.q_for_length(length))
            for offset in (-1e-7, 1e-7):
                neighbor = safety.evaluate(self.q_for_length(length + offset))
                np.testing.assert_allclose(
                    center.values, neighbor.values, atol=1e-6, rtol=0
                )
                np.testing.assert_allclose(
                    center.gradients, neighbor.gradients, atol=1e-5, rtol=0
                )

    def test_coincident_centers_are_finite_and_use_unmodified_rdf(self):
        safety, plain, _ = self.fixture(centered=True)
        actual, expected = safety.evaluate(np.zeros(2)), plain.evaluate(np.zeros(2))
        self.assertTrue(np.isfinite(actual.gradients).all())
        np.testing.assert_array_equal(actual.values, expected.values)
        np.testing.assert_array_equal(actual.gradients, expected.gradients)

    def test_missing_invalid_or_unbounded_geometry_fails_closed(self):
        safety, _, vertices = self.fixture()
        bad_vertices = [
            None,
            {},
            {**vertices, "link1": np.empty((0, 3))},
            {**vertices, "link1": np.full((2, 3), np.nan)},
            {**vertices, "link1": np.array([[2, 0, 0]])},
        ]
        for bad in bad_vertices:
            with self.subTest(vertices=bad), self.assertRaises(ValueError):
                PandaRDFSafety(
                    safety.kinematics,
                    safety.models,
                    safety.queries,
                    safety.pairs,
                    safety.options,
                    bounding_vertices=bad,
                )
        with self.assertRaisesRegex(ValueError, "safe transition reserve"):
            PandaRDFSafety(
                safety.kinematics,
                safety.models,
                safety.queries,
                safety.pairs,
                replace(safety.options, physical_margin=1),
                bounding_vertices=vertices,
            )


class SphereGateIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        integration.PandaRDFIntegrationTest.setUpClass()
        cls.fixture = integration.PandaRDFIntegrationTest()

    def bundle(self):
        options = PandaRDFConfig(query_density=10, sphere_gate=True)
        bundle, _ = self.fixture.bundle(options)
        # Large synthetic test spheres enclose the real visual geometry.
        for name, model in bundle.models.items():
            bundle.metadata["spheres"][name]["mesh_radius"] = 1.0
            model.inner_radius = 1 + options.effective_margin
            model.rdf.radius = 1.5 * model.inner_radius
            count = int(
                np.ceil(options.query_density * 4 * np.pi * model.inner_radius**2)
            )
            bundle.queries[name] = np.random.default_rng(4).uniform(
                -0.05, 0.05, (count, 3)
            )
        return bundle, options

    def test_loader_setup_validator_and_metadata_use_same_gate(self):
        bundle, options = self.bundle()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            save_bundle(bundle, path)
            config = replace(
                self.fixture.config,
                safety=replace(
                    self.fixture.config.safety,
                    self_collision_model="rdf",
                    rdf=replace(options, artifact=path),
                ),
            )
            kin, safety = build_panda_components(config)
            self.assertEqual(safety.gate_metadata["policy"], safety.sphere_gate.policy)
            self.assertEqual(len(safety.gate_metadata["pairs"]), len(safety.pairs))
            report = validate_joint_gradients(
                kin, bundle, options, np.random.default_rng(5)
            )
            self.assertTrue(report["passed"], report)
            self.assertEqual(
                report["sphere_gate"]["policy"], safety.gate_metadata["policy"]
            )

    def test_serial_and_spawn_workers_keep_gate_configuration(self):
        bundle, options = self.bundle()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            save_bundle(bundle, path)
            base = self.fixture.config
            config = replace(
                base,
                obstacles=(),
                safety=replace(
                    base.safety,
                    self_collision_model="rdf",
                    rdf=replace(options, artifact=path),
                ),
                safe_ik=replace(
                    base.safe_ik,
                    self_collision_model="rdf",
                    workers=1,
                    max_restarts=2,
                    iteration_number=3,
                    convergence_hold_duration=0.02,
                    safety_gradient_gain=1e-6,
                ),
                target=replace(
                    base.target, position_tolerance=100, orientation_tolerance=100
                ),
            )
            with patch("safe_ik.application.panda_setup.MujocoPlant"):
                serial = build_panda_runtime(config)
                parallel = build_panda_runtime(
                    replace(config, safe_ik=replace(config.safe_ik, workers=2))
                )
            with serial.ik_search_factory() as search:
                expected = search.solve(require_safe=True)
            with parallel.ik_search_factory() as search:
                result = search.solve(require_safe=True)
            self.assertGreater(len(expected.candidates), 0)
            self.assertEqual(len(result.candidates), len(expected.candidates))
            for actual, wanted in zip(result.candidates, expected.candidates):
                np.testing.assert_allclose(actual.q, wanted.q, atol=1e-12, rtol=0)
                self.assertAlmostEqual(
                    actual.minimum_barrier, wanted.minimum_barrier, places=12
                )


if __name__ == "__main__":
    unittest.main()

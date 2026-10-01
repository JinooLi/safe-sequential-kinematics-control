"""NumPy oracle parity for optional CPU-tensor and real CUDA inference."""

import importlib.util
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from safe_ik.adapters.rdf_torch_runtime import TorchRDFPairs
from safe_ik.application.panda_rdf_artifact import geometry_settings
from safe_ik.domain.panda_config import PandaRDFConfig
from safe_ik.domain.panda_rdf import BernsteinRDF, ExtendedRDF
from safe_ik.domain.panda_rdf_safety import PandaRDFSafety

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch


class Frames:
    def frame_pose_and_jacobian(self, name, q):
        i = {"link1": 0, "link4": 1, "hand": 2}[name]
        rotation = Rotation.from_rotvec([0, 0, (i + 1) * q[1]]).as_matrix()
        origin = np.array([i * q[0], 0.01 * i, 0])
        jac = np.zeros((6, 2))
        jac[0, 0], jac[5, 1] = i, i + 1
        return rotation, origin, jac


class RuntimeConfigTest(unittest.TestCase):
    def test_validation_and_no_artifact_mismatch(self):
        options = PandaRDFConfig()
        self.assertEqual(options.runtime_device, "cpu")
        self.assertEqual(
            geometry_settings(options),
            geometry_settings(replace(options, runtime_device="cuda")),
        )
        for bad in (None, True, "auto", "cuda:0", "mps", 0):
            with self.subTest(device=bad), self.assertRaises(ValueError):
                replace(options, runtime_device=bad)

    def test_missing_cuda_never_falls_back(self):
        if not HAS_TORCH:
            with self.assertRaisesRegex(ValueError, "PyTorch"):
                TorchRDFPairs({}, {}, (), PandaRDFConfig())
        else:
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(ValueError, "no CPU fallback"):
                    TorchRDFPairs({}, {}, (), PandaRDFConfig())


@unittest.skipUnless(HAS_TORCH, "optional torch is not installed")
class TensorParityTest(unittest.TestCase):
    device = "cpu"

    def fixture(self, k=2, gate=False, count=4):
        rng = np.random.default_rng(15)
        models, queries, vertices = {}, {}, {}
        for i, name in enumerate(("link1", "link4", "hand")):
            center = np.array([0.02, -0.01, 0.03]) * (i + 1)
            weights = rng.uniform(-0.02, 0.08, (count,) * 3)
            models[name] = ExtendedRDF(BernsteinRDF(center, 0.6, weights), 0.4, 0.05)
            queries[name] = center + rng.uniform(-0.08, 0.08, (i * 3 + 1, 3))
            vertices[name] = center + 0.1 * np.vstack((np.eye(3), -np.eye(3)))
        pairs = (("link1", "link4"), ("link4", "hand"), ("link1", "hand"))
        options = PandaRDFConfig(top_k=k, sphere_gate=gate)
        tensor = TorchRDFPairs(models, queries, pairs, options, device=self.device)
        plain = PandaRDFSafety(
            Frames(), models, queries, pairs, options, bounding_vertices=vertices
        )
        accelerated = PandaRDFSafety(
            Frames(),
            models,
            queries,
            pairs,
            options,
            bounding_vertices=vertices,
            pair_evaluator=tensor,
        )
        return plain, accelerated, tensor

    def test_fields_center_shells_boundaries_and_bp24(self):
        for count in (4, 24):
            plain, _, engine = self.fixture(count=count)
            queries = []
            for model in plain.models.values():
                radii = np.array(
                    [
                        0,
                        0.1,
                        0.4 - 1e-8,
                        0.4,
                        0.4 + 1e-8,
                        0.5,
                        0.6 - 1e-8,
                        0.6,
                        0.6 + 1e-8,
                        0.9,
                    ]
                )
                queries.append(model.rdf.center + radii[:, None] * [1, 0, 0])
            points = torch.tensor(
                np.stack(queries), dtype=torch.float64, device=self.device
            )
            targets = torch.arange(3, device=self.device)
            with torch.inference_mode():
                values, gradients = engine._field(points, targets, gradient=True)
            for i, model in enumerate(plain.models.values()):
                expected, derivative = model.query(queries[i])
                np.testing.assert_allclose(
                    values[i].cpu(), expected, atol=2e-12, rtol=0
                )
                np.testing.assert_allclose(
                    gradients[i].cpu(), derivative, atol=2e-11, rtol=0
                )

    def test_pair_values_gradients_masks_and_cache(self):
        for gate in (False, True):
            for k in (1, 2, 20, None):
                plain, accelerated, engine = self.fixture(k=k, gate=gate)
                for q in ([0.1, 0.13], [0.48, -0.05], [0.95, 0.12], [2, 0.3]):
                    expected, actual = plain.evaluate(q), accelerated.evaluate(q)
                    self.assertEqual(expected.labels, actual.labels)
                    np.testing.assert_allclose(
                        actual.values, expected.values, atol=2e-12, rtol=0
                    )
                    np.testing.assert_allclose(
                        actual.gradients, expected.gradients, atol=2e-11, rtol=0
                    )
                    self.assertIs(actual, accelerated.evaluate(q))
                    self.assertEqual(sum(accelerated.last_timings.values()), 0)
                if gate:
                    self.assertEqual(engine.last_active_pairs, 0)

    def test_gradient_fd_and_no_numpy_query_calls(self):
        plain, accelerated, _ = self.fixture(gate=True)
        q = np.array([0.95, 0.12])
        expected = plain.evaluate(q)
        with patch.object(
            ExtendedRDF,
            "query",
            side_effect=AssertionError("unexpected NumPy RDF query"),
        ):
            actual = accelerated.evaluate(q)
            np.testing.assert_allclose(
                actual.gradients, expected.gradients, atol=2e-11, rtol=0
            )
            step = 1e-6
            numeric = np.column_stack(
                [
                    (
                        accelerated.evaluate(q + a).values
                        - accelerated.evaluate(q - a).values
                    )
                    / (2 * step)
                    for a in np.eye(2) * step
                ]
            )
            np.testing.assert_allclose(actual.gradients, numeric, atol=1e-8, rtol=0)

    def test_stable_ties_padding_and_all_mode(self):
        plain, _, engine = self.fixture(k=2)
        for model in plain.models.values():
            model.rdf.center[:] = 0
        # Exact equal radii in the outer extension: different gradients but
        # identical values. Stable sort must retain +x,+y before -x,-y.
        plain.queries["link1"] = np.array([[0.9, 0, 0]])
        plain.queries["link4"] = 0.9 * np.array(
            [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
        )
        engine = TorchRDFPairs(
            plain.models,
            plain.queries,
            plain.query_pairs,
            plain.options,
            device=self.device,
        )
        poses = {n: (np.eye(3), np.zeros(3), np.zeros((6, 2))) for n in plain.models}
        for i, name in enumerate(plain.models):
            poses[name][2][0, 0] = i
        first, grad = engine.evaluate(poses, [1, 0])
        np.testing.assert_allclose(first, [0.505, 0.510], atol=1e-14, rtol=0)
        np.testing.assert_allclose(grad, [[-0.5, 0], [-1, 0]], atol=1e-14, rtol=0)

    def test_nonfinite_input_or_output_rejected(self):
        plain, _, engine = self.fixture()
        poses = {
            n: plain.kinematics.frame_pose_and_jacobian(n, [0.1, 0])
            for n in plain.models
        }
        poses["link1"][2][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            engine.evaluate(poses, [0])
        poses["link1"][2][0, 0] = 0
        with torch.no_grad():
            engine.weights[:] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            engine.evaluate(poses, [0])


@unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), "CUDA is unavailable")
class CudaParityTest(TensorParityTest):
    device = "cuda"


@unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), "CUDA is unavailable")
class CudaIntegrationTest(unittest.TestCase):
    def test_loader_validation_and_spawn_workers(self):
        from test_rdf_sphere_gate import SphereGateIntegrationTest

        SphereGateIntegrationTest.setUpClass()
        fixture = SphereGateIntegrationTest()
        original = fixture.bundle

        def cuda_bundle():
            bundle, options = original()
            return bundle, replace(options, runtime_device="cuda")

        with patch.object(fixture, "bundle", side_effect=cuda_bundle):
            fixture.test_loader_setup_validator_and_metadata_use_same_gate()
            fixture.test_serial_and_spawn_workers_keep_gate_configuration()


if __name__ == "__main__":
    unittest.main()

"""Optional backend parity; run these on the GPU host before fitting real data."""

import importlib.util
import unittest

import numpy as np

from safe_ik.adapters.rdf_torch_training import recursive_ridge_torch
from safe_ik.domain.panda_rdf import BernsteinRDF, recursive_ridge


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch not installed")
class TorchRDFTrainingTest(unittest.TestCase):
    def check_parity(self, device):
        rng = np.random.default_rng(5)
        points = rng.uniform(-0.3, 0.3, (100, 3))
        values = np.linalg.norm(points, axis=1) - 0.1
        batches = [(points[i : i + 20], values[i : i + 20]) for i in range(0, 100, 20)]
        cpu = BernsteinRDF(np.array([0.01, -0.02, 0.03]), 1.4, np.zeros((3,) * 3))
        gpu = BernsteinRDF(cpu.center.copy(), cpu.radius, np.zeros((3,) * 3))
        recursive_ridge(cpu, batches)
        steps = []
        recursive_ridge_torch(gpu, batches, device=device, progress=steps.append)
        self.assertEqual(steps, [1, 2, 3, 4, 5])
        self.assertEqual(gpu.weights.dtype, np.dtype("float64"))
        np.testing.assert_allclose(gpu.weights, cpu.weights, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(
            gpu.query(points)[0], cpu.query(points)[0], atol=1e-10, rtol=0
        )

    def test_torch_cpu_matches_scipy(self):
        self.check_parity("cpu")

    def test_cuda_matches_scipy(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        self.check_parity("cuda")

    def test_empty_and_nonfinite_input_rejected(self):
        model = BernsteinRDF(np.zeros(3), 1.0, np.zeros((2,) * 3))
        with self.assertRaisesRegex(ValueError, "empty"):
            recursive_ridge_torch(model, [], device="cpu")
        with self.assertRaisesRegex(ValueError, "invalid training"):
            recursive_ridge_torch(
                model, [(np.full((1, 3), np.nan), np.zeros(1))], device="cpu"
            )


if __name__ == "__main__":
    unittest.main()

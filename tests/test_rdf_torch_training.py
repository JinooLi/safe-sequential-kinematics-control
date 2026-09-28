"""Optional backend parity; run these on the GPU host before fitting real data."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from safe_ik.adapters.rdf_checkpoint import (
    RidgeState,
    load_checkpoint,
    model_signature,
    save_checkpoint,
)
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

    def check_resume(self, device):
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        rng = np.random.default_rng(8)
        p = rng.uniform(-0.3, 0.3, (100, 3))
        y = np.linalg.norm(p, axis=1) - 0.1
        batches = [(p[i : i + 20], y[i : i + 20]) for i in range(0, 100, 20)]
        model = lambda: BernsteinRDF(
            np.array([0.01, -0.02, 0.03]), 1.4, np.zeros((3,) * 3)
        )
        whole, first, second = model(), model(), model()
        final_states = []
        recursive_ridge_torch(
            whole,
            batches,
            device=device,
            checkpoint=lambda n, seen, w, b: final_states.append(
                (w.cpu().numpy().copy(), b.cpu().numpy().copy())
            ),
            checkpoint_interval=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context, prefix = {"pool": "fixed"}, lambda n: {"batches": str(n)}

            def save(n, seen, w, b):
                save_checkpoint(
                    root / str(n),
                    RidgeState(
                        w.cpu().numpy(),
                        b.cpu().numpy(),
                        {
                            "schema_version": 1,
                            "model": model_signature(first, 1e-4),
                            "completed_steps": n,
                            "consumed_samples": seen,
                            "context": context,
                            "batch_prefix_sha256": prefix(n),
                        },
                    ),
                    first,
                    1e-4,
                )

            recursive_ridge_torch(
                first,
                batches[:2],
                device=device,
                checkpoint=save,
                checkpoint_interval=2,
            )
            state = load_checkpoint(root / "2", first, 1e-4, context, prefix)
            progress, saved = [], []

            def resumed_checkpoint(n, seen, w, b):
                saved.append((n, seen))
                save(n, seen, w, b)

            recursive_ridge_torch(
                second,
                batches[2:],
                device=device,
                resume=state,
                progress=progress.append,
                checkpoint=resumed_checkpoint,
                checkpoint_interval=2,
            )
            self.assertEqual(progress, [3, 4, 5])
            self.assertEqual(saved, [(4, 80), (5, 100)])
            np.testing.assert_array_equal(second.weights, whole.weights)
            last = load_checkpoint(root / "5", second, 1e-4, context, prefix)
            np.testing.assert_array_equal(last.weights, final_states[0][0])
            np.testing.assert_array_equal(last.covariance, final_states[0][1])
            np.testing.assert_allclose(
                first.weights.ravel(), state.weights * first.radius, atol=0, rtol=0
            )
            with self.assertRaisesRegex(ValueError, "empty"):
                recursive_ridge_torch(second, [], device=device, resume=last)

    def test_cpu_checkpoint_resume_matches_uninterrupted(self):
        self.check_resume("cpu")

    def test_cuda_checkpoint_resume_matches_uninterrupted(self):
        self.check_resume("cuda")


if __name__ == "__main__":
    unittest.main()

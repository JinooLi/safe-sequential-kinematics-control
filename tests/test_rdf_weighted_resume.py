"""Explicit weighted continuation must equal an uninterrupted fixed data stream."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from safe_ik.adapters.rdf_checkpoint import (
    RidgeState,
    array_sha256,
    load_checkpoint,
    model_signature,
    save_checkpoint,
)
from safe_ik.adapters.rdf_torch_training import recursive_ridge_torch
from safe_ik.domain.panda_rdf import BernsteinRDF


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch not installed")
class WeightedResumeTest(unittest.TestCase):
    def check_resume(self, device, suffix_weight=1.0):
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        rng = np.random.default_rng(200)
        p = rng.uniform(-0.3, 0.3, (59, 3))
        y = np.linalg.norm(p, axis=1) - 0.1
        # Historical weights are retained; the suffix uses each new point once.
        q = np.r_[rng.uniform(1, 4, 36), np.full(23, suffix_weight)]
        batches = [
            (p[i:j], y[i:j], q[i:j]) for i, j in [(0, 18), (18, 36), (36, 54), (54, 59)]
        ]

        def model():
            return BernsteinRDF(np.zeros(3), 0.7, np.zeros((3,) * 3))

        whole, first, second = model(), model(), model()
        context = dict(objective="weighted", sample_weights_sha256=array_sha256(q))
        prefix = lambda n: {"data": array_sha256(p[: [0, 18, 36, 54, 59][n]])}
        final = []
        recursive_ridge_torch(
            whole,
            batches,
            0.013,
            device=device,
            weighted=True,
            checkpoint=lambda n, s, w, b: final.append(
                (w.cpu().numpy().copy(), b.cpu().numpy().copy())
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def save(n, s, w, b):
                save_checkpoint(
                    root / str(n),
                    RidgeState(
                        w.cpu().numpy(),
                        b.cpu().numpy(),
                        dict(
                            schema_version=1,
                            model=model_signature(first, 0.013),
                            completed_steps=n,
                            consumed_samples=s,
                            objective="weighted",
                            context=context,
                            batch_prefix_sha256=prefix(n),
                            sample_weight_prefix_sha256={
                                "data": array_sha256(q[: [0, 18, 36, 54, 59][n]])
                            },
                        ),
                    ),
                    first,
                    0.013,
                )

            recursive_ridge_torch(
                first, batches[:2], 0.013, device=device, weighted=True, checkpoint=save
            )
            parent = load_checkpoint(root / "2", first, 0.013, context, prefix)
            before_w, before_b = parent.weights.copy(), parent.covariance.copy()
            progress = []
            recursive_ridge_torch(
                second,
                batches[2:],
                0.013,
                device=device,
                weighted=True,
                allow_weighted_resume=True,
                resume=parent,
                checkpoint=save,
                progress=progress.append,
            )
            last = load_checkpoint(root / "4", second, 0.013, context, prefix)
            self.assertEqual(progress, [3, 4])
            self.assertEqual(last.metadata["consumed_samples"], 59)
            np.testing.assert_array_equal(second.weights, whole.weights)
            np.testing.assert_array_equal(last.weights, final[0][0])
            np.testing.assert_array_equal(last.covariance, final[0][1])
            np.testing.assert_array_equal(parent.weights, before_w)
            np.testing.assert_array_equal(parent.covariance, before_b)
            phi = whole.features(p)
            expected_b = np.linalg.inv(0.013 * np.eye(27) + phi.T @ (q[:, None] * phi))
            expected_w = expected_b @ (phi.T @ (q * y / 0.7))
            np.testing.assert_allclose(last.weights, expected_w, atol=1e-10, rtol=1e-9)
            np.testing.assert_allclose(
                last.covariance, expected_b, atol=1e-10, rtol=1e-9
            )

    def test_cpu_split_equals_whole_and_direct(self):
        self.check_resume("cpu")

    def test_cuda_split_equals_whole_and_direct(self):
        self.check_resume("cuda")

    def test_cpu_4x_suffix_split_equals_whole_and_direct(self):
        self.check_resume("cpu", 4.0)

    def test_cuda_4x_suffix_split_equals_whole_and_direct(self):
        self.check_resume("cuda", 4.0)

    def test_missing_weight_provenance_rejected(self):
        model = BernsteinRDF(np.zeros(3), 1.0, np.zeros((2,) * 3))
        state = RidgeState(np.zeros(8), np.eye(8), dict(objective="weighted"))
        with self.assertRaisesRegex(ValueError, "weight provenance"):
            recursive_ridge_torch(
                model,
                [],
                device="cpu",
                weighted=True,
                allow_weighted_resume=True,
                resume=state,
            )
        with self.assertRaisesRegex(ValueError, "requires a weighted parent"):
            recursive_ridge_torch(
                model, [], device="cpu", weighted=True, allow_weighted_resume=True
            )


if __name__ == "__main__":
    unittest.main()

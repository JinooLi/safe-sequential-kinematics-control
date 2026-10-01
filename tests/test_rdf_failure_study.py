"""Single-factor offline experiments, including weighted normal-equation parity."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import trimesh

from safe_ik.adapters.rdf_checkpoint import RidgeState, model_signature
from safe_ik.adapters.rdf_torch_training import recursive_ridge_torch
from safe_ik.application.panda_rdf_failure_study import (
    draw_mining_points,
    local_weights,
    rng_for,
    run_link,
    save_arrays,
    summarize_labels,
)
from safe_ik.application.panda_rdf_training import TrainingOptions
from safe_ik.application.panda_rdf_validation import distance_accuracy_report
from safe_ik.domain.panda_rdf import BernsteinRDF, ExtendedRDF


class LocalWeightTest(unittest.TestCase):
    def test_spatial_taper_reliable_exterior_only(self):
        points = np.array(
            [[0, 0, 0], [0.005, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0, 0, 0], [0, 0, 0]]
        )
        truth = np.array([0.01, 0.01, 0.01, 0.01, 0.01, -0.01])
        labels = np.array([0.01, 0.01, 0.01, 0.01, -0.01, -0.01])
        actual = local_weights(
            points, labels, truth, np.zeros((1, 3)), 0.01, 4.0, 0.001
        )
        np.testing.assert_allclose(actual, [4, 1.75, 1, 1, 1, 1], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(labels, [0.01, 0.01, 0.01, 0.01, -0.01, -0.01])
        np.testing.assert_array_equal(
            local_weights(points, labels, truth, np.empty((0, 3)), 0.01, 4.0, 0.001),
            np.ones(6),
        )

    def test_invalid_weight_inputs(self):
        p = np.zeros((2, 3))
        for radius, maximum, tolerance in [
            (0, 4, 0.001),
            (0.01, 1, 0.001),
            (0.01, 4, -1),
            (np.nan, 4, 0.001),
        ]:
            with self.assertRaises(ValueError):
                local_weights(
                    p, np.zeros(2), np.zeros(2), p, radius, maximum, tolerance
                )

    def test_mining_is_reproducible_and_separate_from_holdout(self):
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.1)
        model = BernsteinRDF(np.zeros(3), 0.3, np.zeros((2,) * 3))
        first = draw_mining_points(mesh, model, rng_for(20260929, 6, 101), 100)
        second = draw_mining_points(mesh, model, rng_for(20260929, 6, 101), 100)
        held = draw_mining_points(mesh, model, rng_for(20260929, 6, 102), 100)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, held))
        self.assertEqual(first.shape, (100, 3))

    def test_label_audit_counts(self):
        report = summarize_labels(np.array([-0.01, 0.01]), np.array([0.002, 0.01]))
        self.assertEqual(report["exterior_negative_labels"], 1)
        self.assertEqual(report["above_10mm"], 1)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch not installed")
class WeightedRidgeTest(unittest.TestCase):
    def test_link_workflow_preserves_inputs_and_publishes_complete_evidence(self):
        import torch

        module = "safe_ik.application.panda_rdf_failure_study."
        rng = np.random.default_rng(82)
        points = {
            k: rng.uniform(-0.01, 0.01, (10, 3)) for k in ("near", "random", "exterior")
        }
        labels = {k: np.full(10, 0.020, dtype=np.float32) for k in points}
        indices = {k: np.tile(np.arange(2), (800, 1)) for k in points}
        truth = np.full(10, 0.020)
        held_points, held_truth = rng.uniform(-0.01, 0.01, (5000, 3)), np.full(
            5000, 0.02
        )
        rdf = BernsteinRDF(np.zeros(3), 0.1, np.zeros((2,) * 3))
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.03)
        for name in ("link6", "link0", "hand"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source_root = root / "source"
                source_root.mkdir()
                save_arrays(
                    source_root / "validation.npz", points=held_points, truth=held_truth
                )
                model = ExtendedRDF(copy.deepcopy(rdf), 0.07, 0.0)
                meta = dict(
                    links=[name],
                    settings=dict(seed=0),
                    reconstruction={name: dict(mesh_digest="fixture")},
                    validation={},
                    maxima={},
                    link_training={name: dict(compute={"gpu": "fixture"})},
                )
                baseline = SimpleNamespace(
                    models={name: model}, metadata=meta, reconstructed={name: mesh}
                )
                bundle = copy.deepcopy(baseline)
                protocol = dict(
                    data_workers=1,
                    seed=20260929,
                    source_candidate_sha256="fixture",
                    input_sha256={name: {}},
                    local_weighting=dict(
                        mining_count=20,
                        threshold=0.008,
                        radius=0.010,
                        maximum=4.0,
                        label_tolerance=0.001,
                        pool="exterior",
                    ),
                )

                def fake_fit(
                    model, batches, regularization, *, weighted, checkpoint, **kwargs
                ):
                    batch = next(iter(batches))
                    self.assertEqual(len(batch), 3 if weighted else 2)
                    np.testing.assert_array_equal(
                        batch[1], np.full(6, 0.02, dtype=np.float32)
                    )
                    if weighted:
                        np.testing.assert_array_equal(batch[2][:4], np.ones(4))
                    w, b = torch.zeros(8, dtype=torch.float64), torch.eye(
                        8, dtype=torch.float64
                    )
                    for step in (400, 800):
                        checkpoint(step, step * 1280, w, b)
                    return model

                def fake_validation(mesh, model, rng, margin, *, evidence_path):
                    pred = model.rdf.query(held_points, False)[0]
                    save_arrays(
                        evidence_path,
                        points=held_points,
                        truth=held_truth,
                        prediction=pred,
                    )
                    return dict(
                        distance_accuracy_report(held_points, held_truth, pred, margin),
                        passed=False,
                        derivatives_passed=True,
                    )

                with (
                    patch(
                        module + "read_inputs",
                        return_value=(source_root, points, labels, truth, indices),
                    ),
                    patch(
                        module + "draw_mining_points", return_value=points["exterior"]
                    ),
                    patch(
                        module + "query_truth",
                        side_effect=lambda m, p, w, d: np.full(len(p), 0.02),
                    ),
                    patch(module + "recursive_ridge_torch", side_effect=fake_fit),
                    patch(
                        module + "torch_training_environment",
                        return_value={"device": "fixture"},
                    ),
                    patch(module + "estimate_maximum", return_value=(0.0, {})),
                    patch(module + "validate_model", side_effect=fake_validation),
                    patch("torch.cuda.synchronize"),
                    patch("torch.cuda.reset_peak_memory_stats"),
                    patch("torch.cuda.max_memory_allocated", return_value=0),
                    patch("torch.cuda.empty_cache"),
                ):
                    result = run_link(
                        root,
                        root,
                        name,
                        baseline,
                        bundle,
                        protocol,
                        TrainingOptions(basis_count=2),
                    )
                self.assertTrue(result["complete"])
                self.assertFalse(result["passed"])
                self.assertTrue((root / name / "comparison.json").exists())
                self.assertFalse(bundle.metadata["validation"][name]["passed"])
                with np.load(root / name / "targets.npz") as a:
                    for key in labels:
                        np.testing.assert_array_equal(a[key], labels[key])
                with np.load(root / name / "sample_weights.npz") as a:
                    np.testing.assert_array_equal(a["near"], np.ones(10))
                    np.testing.assert_array_equal(a["random"], np.ones(10))

    def check_weighted(self, device):
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        rng = np.random.default_rng(24)
        p = rng.uniform(-0.2, 0.2, (60, 3))
        y = np.linalg.norm(p, axis=1) - 0.1
        sample_weights = rng.uniform(1, 4, len(p))
        model = BernsteinRDF(np.array([0.01, -0.02, 0.03]), 0.7, np.zeros((3,) * 3))
        phi = model.features(p)
        reg = 0.013
        expected_b = np.linalg.inv(
            reg * np.eye(27) + phi.T @ (sample_weights[:, None] * phi)
        )
        expected_w = expected_b @ (phi.T @ (sample_weights * y / model.radius))
        final = []
        batches = [
            (p[i : i + 12], y[i : i + 12], sample_weights[i : i + 12])
            for i in range(0, len(p), 12)
        ]
        recursive_ridge_torch(
            model,
            batches,
            reg,
            device=device,
            weighted=True,
            checkpoint_interval=5,
            checkpoint=lambda n, seen, w, b: final.append(
                (n, seen, w.cpu().numpy().copy(), b.cpu().numpy().copy())
            ),
        )
        self.assertEqual(final[0][:2], (5, 60))
        np.testing.assert_allclose(final[0][2], expected_w, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(final[0][3], expected_b, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            model.weights.ravel(), expected_w * model.radius, atol=1e-10
        )

    def test_weighted_cpu_matches_direct_solution(self):
        self.check_weighted("cpu")

    def test_weighted_cuda_matches_direct_solution(self):
        self.check_weighted("cuda")

    def test_unit_weights_match_unweighted_exactly(self):
        p = np.random.default_rng(8).uniform(-0.2, 0.2, (20, 3))
        y = p[:, 0] ** 2 - p[:, 1]
        a = BernsteinRDF(np.zeros(3), 1.0, np.zeros((2,) * 3))
        b = BernsteinRDF(np.zeros(3), 1.0, np.zeros((2,) * 3))
        recursive_ridge_torch(a, [(p, y)], device="cpu")
        recursive_ridge_torch(b, [(p, y, np.ones(len(p)))], device="cpu", weighted=True)
        np.testing.assert_array_equal(a.weights, b.weights)

    def test_bad_weights_and_resume_rejected(self):
        p, y = np.zeros((2, 3)), np.zeros(2)
        model = BernsteinRDF(np.zeros(3), 1.0, np.zeros((2,) * 3))
        for w in (
            np.array([0, 1]),
            np.array([-1, 1]),
            np.array([np.nan, 1]),
            np.ones(3),
        ):
            with self.assertRaisesRegex(ValueError, "sample weights"):
                recursive_ridge_torch(model, [(p, y, w)], device="cpu", weighted=True)
        meta = dict(
            schema_version=1,
            model=model_signature(model, 1e-4),
            completed_steps=1,
            consumed_samples=2,
            objective="weighted",
        )
        state = RidgeState(np.zeros(8), np.eye(8), meta)
        for weighted in (False, True):
            with self.assertRaisesRegex(ValueError, "cannot resume"):
                recursive_ridge_torch(
                    model, [], device="cpu", weighted=weighted, resume=state
                )


if __name__ == "__main__":
    unittest.main()

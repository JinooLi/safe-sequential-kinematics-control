"""Full-link batch streams, archived validation and enriched checkpoint parity."""

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import trimesh

from safe_ik.adapters.rdf_checkpoint import array_sha256, load_checkpoint
from safe_ik.adapters.rdf_torch_training import recursive_ridge_torch
from safe_ik.application.panda_mesh_reconstruction import REFERENCE, mesh_digest
from safe_ik.application.panda_rdf_enriched_training import (
    prepare_batches,
    train_enriched_link,
)
from safe_ik.application.panda_rdf_training import TrainingOptions, validate_model
from safe_ik.application.rdf_exterior_sampling import (
    ExteriorSamplingOptions,
    sample_exterior_points,
)
from safe_ik.domain.panda_rdf import BernsteinRDF, ExtendedRDF


class EnrichedTrainingTest(unittest.TestCase):
    def test_full_builder_visits_all_links_even_if_distance_check_fails(self):
        from safe_ik.application import train_panda_rdf as cli

        root = Path(__file__).resolve().parents[1]
        mesh = trimesh.creation.box(extents=[0.01] * 3)
        reconstruction = {
            "reference": REFERENCE,
            "integrity_passed": True,
            "mesh_digest": mesh_digest(mesh),
        }
        fitted = Mock(
            return_value=(BernsteinRDF(np.zeros(3), 0.5, np.zeros((2,) * 3)), {})
        )
        failed = {
            "passed": False,
            "distance_passed": False,
            "derivatives_passed": True,
            "near_max_error": 0.011,
        }
        with (
            TemporaryDirectory() as directory,
            patch.multiple(
                cli,
                reconstruct_mesh=Mock(return_value=(mesh, reconstruction)),
                estimate_maximum=Mock(return_value=(0.1, {})),
                surface_queries=Mock(return_value=np.zeros((2, 3))),
                validate_model=Mock(return_value=failed.copy()),
                validate_joint_gradients=Mock(return_value={"passed": True}),
                benchmark_candidate=Mock(return_value={}),
                write_inspection_html=Mock(),
            ),
            patch(
                "safe_ik.adapters.rdf_torch_training.torch_training_environment",
                return_value={"device": "cuda"},
            ),
            patch(
                "safe_ik.application.panda_rdf_enriched_training.train_enriched_link",
                fitted,
            ),
            patch(
                "sys.argv",
                [
                    "train",
                    "--config",
                    str(root / "configs/panda.yaml"),
                    "--device",
                    "cuda",
                    "--data-workers",
                    "2",
                    "--training-config",
                    str(root / "configs/training/panda_rdf_training_full_bp24.yaml"),
                    "--exterior-sampling-config",
                    str(root / "configs/training/panda_rdf_exterior_sampling.yaml"),
                    "--output",
                    str(Path(directory) / "candidate.npz"),
                ],
            ),
        ):
            with self.assertRaises(SystemExit) as stopped:
                cli.main()
            self.assertEqual(stopped.exception.code, 2)
            self.assertEqual(fitted.call_count, 9)
            self.assertEqual(
                [c.kwargs["link_index"] for c in fitted.call_args_list], list(range(9))
            )
            for call in fitted.call_args_list:
                self.assertEqual(call.kwargs["data_workers"], 2)
                self.assertEqual(
                    call.kwargs["context"]["settings"]["error_margin"], 0.010
                )
                self.assertEqual(call.args[4].basis_count, 24)
                self.assertEqual(call.args[4].updates, 800)
            report = json.loads((Path(directory) / "candidate.report.json").read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(report["data_generation_execution"]["workers"], 2)
            self.assertEqual(len(report["validation"]), 9)
            self.assertEqual(report["training"]["options"]["near_batch"], 512)
            self.assertEqual(
                report["training"]["exterior_sampling"]["exterior_batch"], 640
            )
            with self.assertRaises(SystemExit):
                cli.main()  # Existing evidence cannot be overwritten.
            self.assertEqual(fitted.call_count, 9)

    def options(self):
        return (
            TrainingOptions(
                basis_count=3, updates=4, near_batch=4, random_batch=2, pool_size=20
            ),
            ExteriorSamplingOptions(
                pool_size=20, near_batch=2, random_batch=1, exterior_batch=3
            ),
        )

    def test_per_link_random_streams_are_independent_and_reproducible(self):
        training, exterior = self.options()
        mesh = trimesh.creation.box(extents=[0.1] * 3)
        results = []
        for index in (0, 1, 0):
            a, plan = prepare_batches(
                np.random.default_rng(3), training, exterior, index
            )
            p, _, _ = sample_exterior_points(
                mesh, np.zeros(3), 0.3, exterior, link_index=index
            )
            results.append((a, p))
            self.assertEqual(plan["exterior_batch_spawn_key"], [index, 8])
            self.assertEqual(plan["batch_sizes"], dict(near=2, random=1, exterior=3))
            self.assertEqual(plan["consumed_samples"], 24)
        for key in results[0][0]:
            np.testing.assert_array_equal(results[0][0][key], results[2][0][key])
        np.testing.assert_array_equal(results[0][1], results[2][1])
        self.assertFalse(
            np.array_equal(results[0][0]["exterior"], results[1][0]["exterior"])
        )
        self.assertFalse(np.array_equal(results[0][1], results[1][1]))

    def test_validation_evidence_keeps_interior_points_and_never_overwrites(self):
        mesh = trimesh.creation.box(extents=[0.1] * 3)
        model = ExtendedRDF(BernsteinRDF(np.zeros(3), 0.3, np.zeros((2,) * 3)), 0.2, 0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "validation.npz"
            report = validate_model(
                mesh, model, np.random.default_rng(4), 0.010, evidence_path=path
            )
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["points"].shape, (5000, 3))
                self.assertEqual(
                    int((data["truth"] < 0).sum()), report["interior_excluded_count"]
                )
                np.testing.assert_array_equal(data["exterior"], data["truth"] >= 0)
                self.assertEqual(float(data["error_margin"]), 0.010)
            with self.assertRaises(FileExistsError):
                validate_model(
                    mesh, model, np.random.default_rng(4), 0.010, evidence_path=path
                )

    def check_fit(self, device):
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        training, exterior = self.options()
        rng = np.random.default_rng(8)
        near, random, points = [rng.uniform(-0.1, 0.1, (20, 3)) for _ in range(3)]
        targets = [np.linalg.norm(p, axis=1) - 0.05 for p in (near, random, points)]
        mesh = trimesh.creation.box(extents=[0.1] * 3)
        with TemporaryDirectory() as parent:
            directory = Path(parent) / "link2"
            with (
                patch(
                    "safe_ik.application.panda_rdf_enriched_training.training_pools",
                    return_value=(near, targets[0], random, targets[1]),
                ),
                patch(
                    "safe_ik.application.panda_rdf_enriched_training.generate_exterior_pool",
                    return_value=(
                        points,
                        targets[2],
                        targets[2].copy(),
                        {"generation_seconds": 0.1},
                    ),
                ),
            ):
                model, report = train_enriched_link(
                    mesh,
                    np.zeros(3),
                    0.3,
                    np.random.default_rng(7),
                    training,
                    exterior,
                    link_index=2,
                    directory=directory,
                    context={"test": True},
                    checkpoint_interval=2,
                    device=device,
                    progress=lambda _: None,
                )
                with self.assertRaises(FileExistsError):
                    train_enriched_link(
                        mesh,
                        np.zeros(3),
                        0.3,
                        np.random.default_rng(7),
                        training,
                        exterior,
                        link_index=2,
                        directory=directory,
                        context={},
                        device=device,
                    )
            with np.load(directory / "batch_indices.npz") as data:
                indices = {key: data[key].copy() for key in data.files}
            context = json.loads((directory / "context.json").read_text())
            prefix = lambda step: {
                k: array_sha256(v[:step]) for k, v in indices.items()
            }
            state = load_checkpoint(
                directory / "checkpoints/step_000002",
                model,
                training.regularization,
                context,
                prefix,
            )
            final = load_checkpoint(
                directory / "checkpoints/step_000004",
                model,
                training.regularization,
                context,
                prefix,
            )
            resumed = BernsteinRDF(model.center, model.radius, np.zeros((3,) * 3))
            batches = [
                (
                    np.vstack(
                        (
                            near[indices["near"][s]],
                            random[indices["random"][s]],
                            points[indices["exterior"][s]],
                        )
                    ),
                    np.concatenate(
                        (
                            targets[0][indices["near"][s]],
                            targets[1][indices["random"][s]],
                            targets[2][indices["exterior"][s]],
                        )
                    ),
                )
                for s in range(2, 4)
            ]
            recursive_ridge_torch(
                resumed, batches, training.regularization, device=device, resume=state
            )
            np.testing.assert_array_equal(resumed.weights, model.weights)
            np.testing.assert_array_equal(
                (final.weights * model.radius).reshape(model.weights.shape),
                model.weights,
            )
            self.assertEqual(
                report["batch_plan"]["batch_sizes"], dict(near=2, random=1, exterior=3)
            )
            self.assertEqual(final.metadata["consumed_samples"], 24)
            self.assertEqual(
                [record["step"] for record in report["checkpoints"]], [2, 4]
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "optional torch not installed"
    )
    def test_enriched_cpu_checkpoint_replays_actual_batches(self):
        self.check_fit("cpu")

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "optional torch not installed"
    )
    def test_enriched_cuda_checkpoint_replays_actual_batches(self):
        self.check_fit("cuda")

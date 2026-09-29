"""Completed-link handoff keeps failures and requires complete, intact evidence."""

import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import trimesh
import yaml

from safe_ik.adapters.rdf_checkpoint import (
    RidgeState,
    array_sha256,
    file_sha256,
    model_signature,
    save_checkpoint,
)
from safe_ik.application.panda_mesh_reconstruction import REFERENCE, mesh_digest
from safe_ik.application.panda_rdf_artifact import (
    RDFBundle,
    geometry_settings,
    load_bundle,
    save_bundle,
)
from safe_ik.application.panda_rdf_handoff import (
    copy_prefix_evidence,
    read_completed_prefix,
)
from safe_ik.application.panda_rdf_training import TrainingOptions
from safe_ik.application.panda_rdf_validation import (
    accuracy_masks,
    distance_accuracy_report,
)
from safe_ik.application.rdf_exterior_sampling import ExteriorSamplingOptions
from safe_ik.domain.panda_config import PandaRDFConfig
from safe_ik.domain.panda_rdf import BernsteinRDF, ExtendedRDF


class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "source.npz"
        self.directory = self.path.with_suffix(".training") / "link0"
        self.directory.mkdir(parents=True)
        self.options = PandaRDFConfig()
        self.training = TrainingOptions(
            basis_count=2, updates=4, near_batch=4, random_batch=2, pool_size=10
        )
        self.exterior = ExteriorSamplingOptions(
            pool_size=10, near_batch=2, random_batch=1, exterior_batch=3
        )
        self.geometry = SimpleNamespace(
            vertices=dict.fromkeys(("link0", "link1", "link2")),
            pairs=(("link0", "link2"),),
            fingerprint="test",
            provenance={},
        )
        mesh = trimesh.creation.box(extents=[0.1] * 3)
        r0, r1 = 0.14, 0.21
        model = ExtendedRDF(BernsteinRDF(np.zeros(3), r1, np.zeros((2,) * 3)), r0, 0.1)
        sphere = dict(mesh_radius=0.1, r0=r0, r1=r1, center=[0.0, 0.0, 0.0])
        points, truth, prediction = (
            np.zeros((5000, 3)),
            np.full(5000, 0.02),
            np.zeros(5000),
        )
        report = distance_accuracy_report(points, truth, prediction, 0.010)
        report.update(
            passed=False,
            derivatives_passed=True,
            reference_mesh_digest=mesh_digest(mesh),
        )
        metadata = dict(
            complete=False,
            geometry_fingerprint="test",
            settings=geometry_settings(self.options),
            training=dict(
                baseline_options=asdict(self.training),
                exterior_sampling=asdict(self.exterior),
            ),
            training_compute={"device": "cuda", "gpu": "source"},
            validation_reference=REFERENCE,
            validation={"link0": report},
            reconstruction={
                "link0": dict(
                    reference=REFERENCE,
                    integrity_passed=True,
                    mesh_digest=mesh_digest(mesh),
                )
            },
            spheres={"link0": sphere},
            maxima={"link0": {}},
            joint_validation={"passed": False},
        )
        context = dict(
            link="link0",
            link_index=0,
            geometry_fingerprint="test",
            reference_mesh_digest=mesh_digest(mesh),
            settings=metadata["settings"],
            sphere=sphere,
            training_options={
                k: v for k, v in asdict(self.training).items() if k != "updates"
            },
            exterior_options=asdict(self.exterior),
        )
        for name in ("training_pool.npz", "exterior_pool.npz", "fit.npz"):
            np.savez(self.directory / name, test=np.arange(3))
        context["pool_sha256"] = {
            k: file_sha256(self.directory / k)
            for k in ("training_pool.npz", "exterior_pool.npz")
        }
        self.write("context.json", context)
        indices = {
            k: np.tile(np.arange(n), (4, 1))
            for k, n in dict(near=2, random=1, exterior=3).items()
        }
        np.savez(self.directory / "batch_indices.npz", **indices)
        plan = dict(array_sha256={k: array_sha256(v) for k, v in indices.items()})
        self.write("batch_plan.json", plan)
        records = []
        for step in (2, 4):
            state = RidgeState(
                np.zeros(8),
                np.eye(8),
                dict(
                    schema_version=1,
                    model=model_signature(model.rdf, self.training.regularization),
                    context=context,
                    batch_prefix_sha256={
                        k: array_sha256(v[:step]) for k, v in indices.items()
                    },
                    completed_steps=step,
                    consumed_samples=6 * step,
                ),
            )
            meta = save_checkpoint(
                self.directory / f"checkpoints/step_{step:06d}",
                state,
                model.rdf,
                self.training.regularization,
            )
            records.append(dict(step=step, sha256=meta["sha256"]))
        fit = dict(
            context=context,
            batch_plan=plan,
            checkpoints=records,
            fit_sha256=file_sha256(self.directory / "fit.npz"),
        )
        self.write("fit.report.json", fit)
        metadata["link_training"] = {"link0": fit}
        outside, near = accuracy_masks(truth)
        np.savez(
            self.directory / "validation.npz",
            points=points,
            truth=truth,
            prediction=prediction,
            exterior=outside,
            near=near,
            error_margin=0.010,
        )
        count = int(np.ceil(self.options.query_density * 4 * np.pi * r0**2))
        self.bundle = RDFBundle(
            {"link0": model},
            {"link0": np.zeros((count, 3))},
            self.geometry.pairs,
            metadata,
            reconstructed={"link0": mesh},
        )
        save_bundle(self.bundle, self.path)

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value))

    def read(self, **kwargs):
        args = dict(
            source=self.path,
            geometry=self.geometry,
            options=self.options,
            training=self.training,
            exterior=self.exterior,
        )
        return read_completed_prefix(**(args | kwargs))

    def test_roundtrip_preserves_failed_accuracy_and_runtime_rejection(self):
        bundle, audit = self.read()
        self.assertEqual(list(bundle.models), ["link0"])
        self.assertFalse(bundle.metadata["validation"]["link0"]["distance_passed"])
        self.assertEqual(audit["link0"]["checkpoints"], [2, 4])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_bundle(self.path, self.geometry, self.options)

    def test_reject_settings_updates_prefix_and_corrupt_matrix(self):
        for change in (
            {"options": replace(self.options, error_margin=0.005)},
            {"training": replace(self.training, updates=8)},
            {
                "geometry": SimpleNamespace(
                    vertices={"link1": None, "link0": None, "link2": None},
                    pairs=self.geometry.pairs,
                    fingerprint="test",
                )
            },
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read(**change)
        with (self.directory / "checkpoints/step_000004/inverse_information.npy").open(
            "r+b"
        ) as stream:
            stream.seek(-1, 2)
            stream.write(b"\xff")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.read()

    def test_missing_manifest_entry_and_mask_corruption_rejected(self):
        with np.load(self.path) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        meta = json.loads(str(arrays["metadata"]))
        meta["array_hashes"].pop("link0_weights")
        arrays["metadata"] = np.array(json.dumps(meta))
        np.savez(self.path, **arrays)
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.read()
        save_bundle(self.bundle, self.path)
        file = self.directory / "validation.npz"
        with np.load(file) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        arrays["near"][0] = False
        np.savez(file, **arrays)
        with self.assertRaisesRegex(ValueError, "mask"):
            self.read()

    def test_evidence_copy_refuses_overwrite(self):
        destination = self.root / "copy"
        copy_prefix_evidence(self.path, destination, ["link0"])
        self.assertEqual(
            file_sha256(self.directory / "validation.npz"),
            file_sha256(destination / "link0/validation.npz"),
        )
        with self.assertRaises(FileExistsError):
            copy_prefix_evidence(self.path, destination, ["link0"])

    def test_cli_skips_prefix_and_runs_full_joint_check_preserving_failure(self):
        from safe_ik.application import train_panda_rdf as cli

        mesh = self.bundle.reconstructed["link0"]
        self.geometry.vertices = {k: mesh.vertices for k in self.geometry.vertices}
        self.geometry.faces = {k: mesh.faces for k in self.geometry.vertices}
        self.geometry.provenance = {}
        config = SimpleNamespace(
            safety=SimpleNamespace(rdf=self.options, capsules=[]),
            model_path="test",
            ee_frame="test",
        )
        training_file, exterior_file = (
            self.root / "training.yaml",
            self.root / "exterior.yaml",
        )
        training_file.write_text(yaml.safe_dump(asdict(self.training)))
        exterior_file.write_text(yaml.safe_dump(asdict(self.exterior)))
        trained = Mock(return_value=(self.bundle.models["link0"].rdf, {}))
        argv = [
            "train",
            "--device",
            "cuda",
            "--training-config",
            str(training_file),
            "--exterior-sampling-config",
            str(exterior_file),
            "--output",
            str(self.root / "new.npz"),
            "--continue-from",
            str(self.path),
        ]
        with (
            patch.multiple(
                cli,
                load_panda_config=Mock(return_value=config),
                extract_visual_geometry=Mock(return_value=self.geometry),
                reconstruct_mesh=Mock(
                    return_value=(mesh, self.bundle.metadata["reconstruction"]["link0"])
                ),
                minimum_sphere=Mock(return_value=(np.zeros(3), 0.1)),
                estimate_maximum=Mock(return_value=(0.1, {})),
                surface_queries=Mock(return_value=np.zeros((1, 3))),
                validate_model=Mock(
                    return_value=self.bundle.metadata["validation"]["link0"].copy()
                ),
                PandaKinematics=Mock(),
                validate_joint_gradients=Mock(return_value={"passed": True}),
                benchmark_candidate=Mock(return_value={}),
                write_inspection_html=Mock(),
            ),
            patch(
                "safe_ik.adapters.rdf_torch_training.torch_training_environment",
                return_value={"device": "cuda", "gpu": "destination"},
            ),
            patch(
                "safe_ik.application.panda_rdf_enriched_training.train_enriched_link",
                trained,
            ),
            patch("sys.argv", argv),
        ):
            with self.assertRaises(SystemExit) as result:
                cli.main()
            self.assertEqual(result.exception.code, 2)
            self.assertEqual(
                [c.kwargs["link_index"] for c in trained.call_args_list], [1, 2]
            )
            cli.validate_joint_gradients.assert_called_once()
        metadata = json.loads((self.root / "new.report.json").read_text())
        self.assertTrue(metadata["complete"])
        self.assertFalse(metadata["validation"]["link0"]["passed"])
        self.assertEqual(metadata["handoffs"][0]["completed_links"], ["link0"])
        self.assertEqual(
            metadata["handoffs"][0]["destination_compute"]["gpu"], "destination"
        )

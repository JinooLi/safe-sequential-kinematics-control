"""Official voxel geometry, diagnostic-only discrepancy, reference persistence."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import trimesh

from safe_ik.application.panda_mesh_reconstruction import (
    REFERENCE,
    ReconstructionOptions,
    compare_surfaces,
    mesh_digest,
    mesh_from_voxels,
    reconstruct_mesh,
    require_valid_reconstruction,
)
from safe_ik.application.panda_rdf_artifact import (
    RDFBundle,
    read_reconstructed_meshes,
    save_bundle,
)
from safe_ik.application.panda_rdf_training import TrainingOptions, mesh_truth
from safe_ik.domain.panda_rdf import BernsteinRDF


class MeshReconstructionTest(unittest.TestCase):
    def test_training_uses_reference_but_queries_stay_on_original_visual(self):
        from safe_ik.application import train_panda_rdf as cli

        reference = trimesh.creation.box(extents=[0.01] * 3)
        report = {
            "reference": REFERENCE,
            "integrity_passed": True,
            "mesh_digest": mesh_digest(reference),
        }
        pools, queries, validation = (
            Mock(return_value=()),
            Mock(return_value=np.zeros((2, 3))),
            Mock(return_value={"passed": False, "near_max_error": 0.01}),
        )
        with (
            TemporaryDirectory() as directory,
            patch.multiple(
                cli,
                reconstruct_mesh=Mock(return_value=(reference, report)),
                training_pools=pools,
                surface_queries=queries,
                validate_model=validation,
                train_model=Mock(
                    return_value=BernsteinRDF(np.zeros(3), 0.5, np.zeros((2,) * 3))
                ),
                estimate_maximum=Mock(return_value=(0.1, {})),
                write_inspection_html=Mock(),
            ),
            patch(
                "sys.argv",
                [
                    "train",
                    "--config",
                    str(Path(__file__).parent / "fixtures/panda.yaml"),
                    "--links",
                    "link4",
                    "--output",
                    str(Path(directory) / "candidate.npz"),
                ],
            ),
        ):
            with self.assertRaises(SystemExit) as stopped:
                cli.main()
            self.assertEqual(stopped.exception.code, 2)
            self.assertIs(pools.call_args.args[0], reference)
            self.assertIs(validation.call_args.args[0], reference)
            self.assertIsNot(queries.call_args.args[0], reference)

    def test_official_spacing_and_bbox_restoration(self):
        n = 24
        axis = np.linspace(-1, 1, n)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
        sdf = np.max(np.abs(grid), axis=-1) - 0.5
        original = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        original.apply_translation([0.3, -0.7, 0.2])
        vertices = original.vertices.copy()
        mesh = mesh_from_voxels(np.pad(sdf, 1, constant_values=1), original, n)
        np.testing.assert_array_equal(original.vertices, vertices)
        np.testing.assert_allclose(
            mesh.bounding_box.centroid, original.bounding_box.centroid
        )
        # The official 2/N spacing shrinks the endpoint grid by (N-1)/N.
        np.testing.assert_allclose(
            mesh.extents, np.full(3, 0.1 * (n - 1) / n), atol=1e-8
        )
        truth, _, _ = mesh_truth(mesh, np.array([[0.3, -0.7, 0.2], [0.45, -0.7, 0.2]]))
        self.assertLess(truth[0], 0)
        self.assertGreater(truth[1], 0)

    def test_invalid_grids_and_open_result_are_rejected(self):
        box = trimesh.creation.box()
        with self.assertRaisesRegex(ValueError, "zero crossing"):
            mesh_from_voxels(np.ones((10, 10, 10)), box, 8)
        with self.assertRaisesRegex(ValueError, "padded voxel"):
            mesh_from_voxels(np.ones((8, 8, 8)), box, 8)
        box.update_faces(np.arange(1, len(box.faces)))
        with self.assertRaisesRegex(ValueError, "closed"):
            require_valid_reconstruction(box)

    def test_discrepancy_reports_large_changes_without_acceptance_threshold(self):
        original, shifted = trimesh.creation.box(), trimesh.creation.box()
        shifted.apply_translation([3, 0, 0])
        report = compare_surfaces(original, shifted, np.random.default_rng(0), 100)
        self.assertIsNone(report["acceptance_threshold"])
        self.assertNotIn("passed", report)
        self.assertGreater(report["original_to_reconstructed"]["max"], 2)
        self.assertGreater(report["reconstructed_to_original"]["max"], 2)
        self.assertEqual(
            report, compare_surfaces(original, shifted, np.random.default_rng(0), 100)
        )

    def test_reference_mesh_survives_incomplete_candidate_roundtrip(self):
        mesh = trimesh.creation.box()
        meta = {
            "validation_reference": REFERENCE,
            "reconstruction": {"link4": {"mesh_digest": mesh_digest(mesh)}},
        }
        bundle = RDFBundle({}, {}, (), meta, reconstructed={"link4": mesh})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.npz"
            save_bundle(bundle, path)
            meshes, restored = read_reconstructed_meshes(path)
            np.testing.assert_array_equal(meshes["link4"].vertices, mesh.vertices)
            np.testing.assert_array_equal(meshes["link4"].faces, mesh.faces)
            self.assertEqual(restored["links"], [])  # does not authorize runtime
            bundle.metadata["reconstruction"]["link4"]["mesh_digest"] = "wrong"
            save_bundle(bundle, path)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                read_reconstructed_meshes(path)

    def test_yaml_options_are_validated(self):
        self.assertEqual(
            TrainingOptions(
                reconstruction={"voxel_resolution": 64}
            ).reconstruction.voxel_resolution,
            64,
        )
        for kwargs in (
            {"voxel_resolution": 0},
            {"scan_count": 1},
            {"sdf_batch_size": 1.5},
        ):
            with self.assertRaises(ValueError):
                ReconstructionOptions(**kwargs)

    def test_chunked_normal_labels_use_official_grid_and_parameters(self):
        box = trimesh.creation.box()
        points = np.arange(8**3 * 3, dtype=np.float32).reshape(-1, 3)
        cloud = Mock()
        cloud.get_sdf.side_effect = lambda block, **kw: np.zeros(len(block))
        get_cloud = Mock(return_value=cloud)
        unit_cube = Mock(return_value=box)
        modules = {
            "mesh_to_sdf": SimpleNamespace(get_surface_point_cloud=get_cloud),
            "mesh_to_sdf.utils": SimpleNamespace(
                get_raster_points=lambda n: points, scale_to_unit_cube=unit_cube
            ),
        }
        with (
            patch.dict("sys.modules", modules),
            patch(
                "safe_ik.application.panda_mesh_reconstruction.mesh_from_voxels",
                return_value=box,
            ) as conversion,
        ):
            reconstruct_mesh(
                box,
                ReconstructionOptions(
                    voxel_resolution=8, sdf_batch_size=50, comparison_samples=10
                ),
                np.random.default_rng(0),
            )
        self.assertEqual(get_cloud.call_args.kwargs["bounding_radius"], 3**0.5)
        self.assertEqual(get_cloud.call_args.kwargs["scan_count"], 100)
        np.testing.assert_array_equal(
            np.vstack([c.args[0] for c in cloud.get_sdf.call_args_list]), points
        )
        self.assertTrue(
            all(
                len(c.args[0]) <= 50 and c.kwargs["sample_count"] == 100
                for c in cloud.get_sdf.call_args_list
            )
        )
        self.assertEqual(conversion.call_args.args[0].shape, (10, 10, 10))
        self.assertTrue(np.all(conversion.call_args.args[0][0] == 1))


if __name__ == "__main__":
    unittest.main()

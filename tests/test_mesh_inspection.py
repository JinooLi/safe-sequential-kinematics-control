"""Topology diagnostics do not confuse duplicated visual vertices with holes."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import trimesh

from safe_ik.visualization.mesh_inspection import inspect_mesh, write_inspection_html


class MeshInspectionTest(unittest.TestCase):
    def test_closed_box_with_split_vertices(self):
        box = trimesh.creation.box()
        vertices = box.triangles.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        original = vertices.copy()
        result = inspect_mesh(vertices, faces)
        self.assertTrue(result["watertight"])
        self.assertEqual(result["vertex_count"], 8)
        self.assertEqual(result["regions"], [])
        self.assertEqual(
            result["edge_counts"], dict(boundary=0, nonmanifold=0, winding=0)
        )
        np.testing.assert_array_equal(vertices, original)

    def test_removed_triangle_creates_one_boundary_loop(self):
        box = trimesh.creation.box()
        result = inspect_mesh(box.vertices, box.faces[1:])
        self.assertEqual(result["edge_counts"]["boundary"], 3)
        self.assertEqual(len(result["regions"]), 1)
        self.assertTrue(result["regions"][0]["closed_loop"])

    def test_reversed_face_is_not_a_hole(self):
        box = trimesh.creation.box()
        faces = box.faces.copy()
        faces[0] = faces[0][::-1]
        result = inspect_mesh(box.vertices, faces)
        self.assertEqual(result["edge_counts"]["boundary"], 0)
        self.assertEqual(result["edge_counts"]["winding"], 3)

    def test_three_faces_on_edge_are_nonmanifold(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]])
        faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
        result = inspect_mesh(vertices, faces)
        self.assertEqual(result["edge_counts"]["nonmanifold"], 1)
        self.assertEqual(result["edge_counts"]["boundary"], 6)

    def test_export_is_standalone_and_escapes_json(self):
        box = trimesh.creation.box()
        geometry = SimpleNamespace(
            vertices={"test": box.vertices},
            faces={"test": box.faces},
            fingerprint="</script>unsafe",
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "index.html"
            result = write_inspection_html(geometry, output)
            html = output.read_text()
        self.assertNotIn("__MESH_DATA__", html)
        self.assertNotIn("</script>unsafe", html)
        self.assertNotIn("<script src=", html)
        encoded = html.split('<script id="mesh-data" type="application/json">')[
            1
        ].split("</script>")[0]
        self.assertEqual(json.loads(encoded), result)

    def test_comparison_export_retains_both_geometries(self):
        box = trimesh.creation.box()
        geometry = SimpleNamespace(
            vertices={"test": box.vertices},
            faces={"test": box.faces},
            fingerprint="original",
        )
        reference = box.copy()
        reference.apply_scale(0.9)
        with TemporaryDirectory() as directory:
            result = write_inspection_html(
                geometry,
                Path(directory) / "compare.html",
                {"test": reference},
                {"test": {"comparison": {}}},
            )
        self.assertNotEqual(
            result["links"]["test"]["vertices"],
            result["reconstructed"]["test"]["vertices"],
        )
        self.assertEqual(result["fingerprint"], "original")


if __name__ == "__main__":
    unittest.main()

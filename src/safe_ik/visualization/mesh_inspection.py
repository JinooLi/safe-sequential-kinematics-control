"""Offline visual-mesh topology diagnostics and a self-contained HTML export.

Nothing here repairs the training geometry or approves an RDF artifact.
"""

import base64
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import trimesh


def _encoded(array, dtype):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode("ascii")


def _regions(edges, vertices, kind):
    """Connected edge components, not necessarily closed loops or real holes."""
    incident = {}
    for index, (a, b) in enumerate(edges):
        incident.setdefault(int(a), []).append(index)
        incident.setdefault(int(b), []).append(index)
    unseen = set(range(len(edges)))
    regions = []
    while unseen:
        pending = [min(unseen)]
        selected = []
        while pending:
            index = pending.pop()
            if index not in unseen:
                continue
            unseen.remove(index)
            selected.append(index)
            for vertex in edges[index]:
                pending.extend(incident[int(vertex)])
        component = edges[sorted(selected)]
        ids, degree = np.unique(component, return_counts=True)
        points = vertices[ids]
        low, high = points.min(axis=0), points.max(axis=0)
        regions.append(
            {
                "kind": kind,
                "edges": component.ravel().tolist(),
                "edge_count": len(component),
                "closed_loop": bool(np.all(degree == 2)),
                "center": ((low + high) / 2).tolist(),
                "span_mm": float(np.linalg.norm(high - low) * 1000),
                "length_mm": float(
                    np.linalg.norm(
                        vertices[component[:, 0]] - vertices[component[:, 1]], axis=1
                    ).sum()
                    * 1000
                ),
            }
        )
    return sorted(regions, key=lambda r: (-r["length_mm"], r["edges"][0]))


def inspect_mesh(vertices, faces):
    """Weld coincident positions on a copy; retain every face, including defects."""
    mesh = trimesh.Trimesh(
        vertices=np.array(vertices, copy=True),
        faces=np.array(faces, copy=True),
        process=False,
    )
    original_vertex_count = len(mesh.vertices)
    mesh.merge_vertices(digits_vertex=8)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    directed = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    balance = np.bincount(
        inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1)
    )
    edge_sets = {
        "boundary": edges[counts == 1],
        "nonmanifold": edges[counts > 2],
        "winding": edges[(counts == 2) & (np.abs(balance) == 2)],
    }
    regions = []
    for kind, selected in edge_sets.items():
        regions.extend(_regions(selected, vertices, kind))
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    return {
        "vertices": _encoded(vertices, "<f4"),
        "faces": _encoded(faces, "<u4"),
        "center": ((low + high) / 2).tolist(),
        "span_mm": float(np.linalg.norm(high - low) * 1000),
        "original_vertex_count": original_vertex_count,
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "edge_counts": {k: len(v) for k, v in edge_sets.items()},
        "degenerate_faces": int(np.sum(mesh.area_faces < 1e-18)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "regions": regions,
    }


def write_inspection_html(geometry, output, reconstructed=None, comparison=None):
    payload = {
        "fingerprint": geometry.fingerprint,
        "merge_digits": 8,
        "links": {
            name: inspect_mesh(vertices, geometry.faces[name])
            for name, vertices in geometry.vertices.items()
        },
        "reconstructed": {
            name: inspect_mesh(mesh.vertices, mesh.faces)
            for name, mesh in (reconstructed or {}).items()
        },
        "comparison": comparison or {},
    }
    template = (
        files("safe_ik.visualization")
        .joinpath("mesh_inspection.html")
        .read_text(encoding="utf-8")
    )
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    # Embedded JSON must not terminate its script element.
    html = template.replace("__MESH_DATA__", encoded.replace("<", "\\u003c"))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return payload

"""Visual geometry extraction only; mesh distance queries are offline operations."""

import json
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations

import mujoco
import numpy as np


@dataclass
class VisualGeometry:
    vertices: dict
    faces: dict
    pairs: tuple
    provenance: dict
    fingerprint: str


def extract_visual_geometry(model_path):
    """Recover authored visual meshes from MuJoCo's recentered mesh frames.

    Finger bodies are merged into hand coordinates. Adjacency comes from body
    parent IDs after that merge, never from a link-name numbering heuristic.
    """
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.nq != 7:
        raise ValueError("RDF requires seven arm joints and fixed closed fingers")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = tuple([f"link{i}" for i in range(8)] + ["hand"])
    ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}
    if min(ids.values()) < 0:
        raise ValueError("Panda RDF geometry requires link0..link7 and hand")
    owner = {v: k for k, v in ids.items()}
    fingers = {}
    for name in ("left_finger", "right_finger"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if (
            bid < 0
            or model.body_parentid[bid] != ids["hand"]
            or model.body_jntnum[bid]
            or not np.allclose(model.body_pos[bid], [0, 0, 0.0584], atol=1e-10)
        ):
            raise ValueError(f"{name}: expected fixed finger at closed limit 0 m")
        owner[bid] = "hand"
        fingers[name] = {
            "position": model.body_pos[bid].tolist(),
            "quaternion_wxyz": model.body_quat[bid].tolist(),
            "closed_joint_value": 0.0,
        }
    vertices, faces, sources = {n: [] for n in names}, {n: [] for n in names}, []
    for gid in range(model.ngeom):
        bid, mid = int(model.geom_bodyid[gid]), int(model.geom_dataid[gid])
        if bid not in owner or model.geom_group[gid] != 2:
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or mid < 0:
            raise ValueError("Panda visual geometry must consist of meshes")
        name = owner[bid]
        root = ids[name]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        world = (
            model.mesh_vert[va : va + vn] @ data.geom_xmat[gid].reshape(3, 3).T
            + data.geom_xpos[gid]
        )
        local = (world - data.xpos[root]) @ data.xmat[root].reshape(3, 3)
        offset = sum(len(v) for v in vertices[name])
        vertices[name].append(local)
        faces[name].append(model.mesh_face[fa : fa + fn].copy() + offset)
        sources.append(
            {
                "link": name,
                "body": model.body(bid).name,
                "mesh": model.mesh(mid).name,
                "mesh_path": (
                    model.mesh_path[model.mesh_pathadr[mid] :].split(b"\0")[0].decode()
                    if hasattr(model, "mesh_path")
                    else model.mesh(mid).name
                ),
            }
        )
    if any(not v for v in vertices.values()):
        raise ValueError("missing visual mesh for a Panda link")
    vertices = {n: np.vstack(v) for n, v in vertices.items()}
    faces = {n: np.vstack(v) for n, v in faces.items()}
    adjacency = set()
    for bid, name in owner.items():
        parent = int(model.body_parentid[bid])
        if parent in owner and owner[parent] != name:
            adjacency.add(frozenset((name, owner[parent])))
    pairs = tuple(p for p in combinations(names, 2) if frozenset(p) not in adjacency)
    provenance = {
        "frames": list(names),
        "pairs": pairs,
        "fingers": fingers,
        "visual_meshes": sources,
        "units": "m, rad, s",
        "coordinates": "body-local; local-to-world R p + t",
        "kinematics": {
            key: getattr(model, key).tolist()
            for key in (
                "body_parentid",
                "body_pos",
                "body_quat",
                "jnt_type",
                "jnt_bodyid",
                "jnt_pos",
                "jnt_axis",
                "jnt_range",
            )
        },
    }
    digest = sha256(json.dumps(provenance, sort_keys=True).encode())
    for name in names:
        digest.update(np.round(vertices[name], 10).astype("<f8").tobytes())
        digest.update(faces[name].astype("<i8").tobytes())
    return VisualGeometry(vertices, faces, pairs, provenance, digest.hexdigest())

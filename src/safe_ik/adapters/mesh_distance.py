"""Deterministic offline mesh oracle; positive outside, negative inside.

Keep the former closest-triangle normal and fixed primary ray decisions. Only
the stochastic conflicting-ray retry is replaced, by an integer winding-number
test on a closed, consistently oriented mesh. This is a versioned reference
change, not a physical ground-truth guarantee. No RNG or global patching is used.
"""

from dataclasses import dataclass

import numpy as np
import trimesh

SIGN_ORACLE = "triangle_normal_fixed_ray_integer_winding_v1"
WINDING_TOLERANCE = 1e-5
PRIMARY_DIRECTION = (0.4395064455, 0.617598629942, 0.652231566745)


class AmbiguousMeshSignError(ValueError):
    """The reference cannot assign a reliable deterministic sign."""


@dataclass(frozen=True)
class MeshDistance:
    signed: np.ndarray
    closest: np.ndarray
    unsigned: np.ndarray
    triangles: np.ndarray
    winding_fallback: np.ndarray


def winding_number(mesh, points):
    """Sum oriented triangle solid angles, per point in fixed face order.

    The solid-angle formula uses atan2(det(a,b,c), 1+a.b+b.c+c.a)
    for unit vectors to the three vertices. Keeping one point at a time bounds
    memory and makes reduction order independent of query chunks or workers.
    See Jacobson et al., Generalized Winding Numbers (SIGGRAPH 2013).
    """
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise AmbiguousMeshSignError("winding fallback needs a closed oriented mesh")
    result = []
    for point in np.asarray(points, dtype=np.float64):
        vectors = np.asarray(mesh.triangles) - point
        lengths = np.linalg.norm(vectors, axis=2)
        if not np.isfinite(lengths).all() or np.any(lengths == 0):
            raise AmbiguousMeshSignError("winding query is nonfinite or on a vertex")
        a, b, c = np.moveaxis(vectors / lengths[:, :, None], 1, 0)
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (
            1.0
            + np.einsum("ij,ij->i", a, b)
            + np.einsum("ij,ij->i", b, c)
            + np.einsum("ij,ij->i", c, a)
        )
        value = np.arctan2(numerator, denominator).sum(dtype=np.float64) / (2 * np.pi)
        if not np.isfinite(value) or abs(value - np.rint(value)) > WINDING_TOLERANCE:
            raise AmbiguousMeshSignError(f"noninteger closed-mesh winding: {value}")
        result.append(value)
    return np.asarray(result, dtype=np.float64)


def _fixed_ray_inside(mesh, points):
    """Return containment and which points required the nonrandom fallback."""
    inside = np.zeros(len(points), dtype=bool)
    fallback = np.zeros(len(points), dtype=bool)
    ids = np.flatnonzero(trimesh.bounds.contains(mesh.bounds, points))
    if not len(ids):
        return inside, fallback
    origins = points[ids]
    directions = np.tile(PRIMARY_DIRECTION, (len(origins), 1))
    _, ray_ids, _ = mesh.ray.intersects_location(
        np.vstack([origins, origins]),
        np.vstack([directions, -directions]),
        multiple_hits=True,
    )
    counts = np.bincount(ray_ids, minlength=2 * len(origins)).reshape(2, -1)
    parity = counts % 2 == 1
    agreed = parity[0] == parity[1]
    inside[ids[agreed]] = parity[0, agreed]
    # Preserve the former primary rule: a ray with zero intersections is outside.
    conflict = ~agreed & (counts != 0).all(axis=0)
    unresolved = ids[conflict]
    if len(unresolved):
        inside[unresolved] = np.abs(winding_number(mesh, points[unresolved])) > 0.5
        fallback[unresolved] = True
    return inside, fallback


def mesh_distance(mesh, points):
    """Exact triangle distances with deterministic, versioned sign decisions.

    The former surface tolerance convention is retained: unsigned <= tol.merge
    is nonpositive. Ambiguity is never converted to exclusion from validation.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("mesh queries must be finite (n,3) points")
    if not len(points):
        return MeshDistance(
            np.empty(0),
            np.empty((0, 3)),
            np.empty(0),
            np.empty(0, dtype=int),
            np.empty(0, dtype=bool),
        )
    closest, unsigned, triangles = trimesh.proximity.closest_point(mesh, points)
    if not np.isfinite(unsigned).all() or not np.isfinite(closest).all():
        raise AmbiguousMeshSignError("nonfinite mesh proximity")
    signed = -unsigned.copy()
    fallback = np.zeros(len(points), dtype=bool)
    ids = np.flatnonzero(unsigned > trimesh.constants.tol.merge)
    if len(ids):
        normals = mesh.face_normals[triangles[ids]]
        projection = (
            points[ids]
            - normals
            * np.einsum("ij,ij->i", points[ids] - closest[ids], normals)[:, None]
        )
        barycentric = trimesh.triangles.points_to_barycentric(
            mesh.triangles[triangles[ids]], projection
        )
        on_triangle = (
            (barycentric >= -trimesh.constants.tol.merge)
            & (barycentric <= 1 + trimesh.constants.tol.merge)
        ).all(axis=1)
        normal_sign = np.sign(
            np.einsum(
                "ij,ij->i",
                normals[on_triangle],
                points[ids[on_triangle]] - projection[on_triangle],
            )
        )
        if np.any(normal_sign == 0):
            raise AmbiguousMeshSignError("zero normal sign away from the surface")
        signed[ids[on_triangle]] = unsigned[ids[on_triangle]] * normal_sign
        ray_ids = ids[~on_triangle]
        if len(ray_ids):
            inside, used_winding = _fixed_ray_inside(mesh, points[ray_ids])
            signed[ray_ids] = unsigned[ray_ids] * np.where(inside, -1.0, 1.0)
            fallback[ray_ids] = used_winding
    return MeshDistance(signed, closest, unsigned, triangles, fallback)


def signed_mesh_distance(mesh, points):
    return mesh_distance(mesh, points).signed

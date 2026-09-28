"""Offline idiap/RDF voxel preprocessing; never mutates visual assets.

The upstream normal-vote SDF is evaluated in small batches to bound memory.
Grid order, padding, marching-cubes spacing and bbox restoration are unchanged.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version

import numpy as np
import trimesh

UPSTREAM = "550c13901ca62a659171b305b0d087d88c3fb580"
REFERENCE = "voxel_reconstructed_mesh"


@dataclass(frozen=True)
class ReconstructionOptions:
    voxel_resolution: int = 128
    scan_count: int = 100
    scan_resolution: int = 400
    normal_sample_count: int = 100
    sample_point_count: int = 10000000  # upstream argument; unused by scan mode
    sdf_batch_size: int = 2048
    comparison_samples: int = 5000  # per direction, diagnostic only

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"reconstruction.{key} must be a positive integer")
        if self.voxel_resolution < 4 or self.scan_count < 2:
            raise ValueError("voxel_resolution >= 4 and scan_count >= 2 required")


def mesh_digest(mesh):
    digest = sha256()
    for array, dtype in ((mesh.vertices, "<f8"), (mesh.faces, "<i8")):
        value = np.asarray(array, dtype=dtype)
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def require_valid_reconstruction(mesh):
    """Basic volume integrity, not a self-intersection or conservatism proof."""
    if (
        len(mesh.vertices) == 0
        or len(mesh.faces) == 0
        or not np.isfinite(mesh.vertices).all()
        or not mesh.is_watertight
        or not mesh.is_winding_consistent
        or not mesh.is_volume
    ):
        raise ValueError(
            "reconstructed mesh must be finite, closed and outward oriented"
        )


def mesh_from_voxels(voxels, original, resolution):
    """Use the official 2/N spacing, *not* an unrecorded 2/(N-1) correction."""
    from skimage.measure import marching_cubes

    if voxels.shape != (resolution + 2,) * 3 or not np.isfinite(voxels).all():
        raise ValueError("expected finite, one-layer-padded voxel grid")
    if not voxels.min() < 0 < voxels.max():
        raise ValueError("voxel SDF has no signed zero crossing")
    vertices, faces, _, _ = marching_cubes(
        voxels, level=0, spacing=(2 / resolution,) * 3
    )
    # Negative-inside SDF + skimage's default descent yields outward face winding.
    # Compute normals from faces rather than persisting skimage's opposite normals.
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    scale = 2 / np.max(original.bounding_box.extents)
    mesh.vertices /= scale
    mesh.vertices += original.bounding_box.centroid - mesh.bounding_box.centroid
    require_valid_reconstruction(mesh)
    return mesh


def compare_surfaces(original, reconstructed, rng, count):
    """Unsigned exact triangle distances from area-uniform samples, both ways."""
    result = {
        "sample_count_per_direction": count,
        "acceptance_threshold": None,
        "continuous_maximum_certificate": False,
        "units": "m",
    }
    for key, source, target in (
        ("original_to_reconstructed", original, reconstructed),
        ("reconstructed_to_original", reconstructed, original),
    ):
        points, _ = trimesh.sample.sample_surface(source, count, seed=rng)
        distances = np.concatenate(
            [
                trimesh.proximity.closest_point(target, points[i : i + 64])[1]
                for i in range(0, count, 64)
            ]
        )
        if not np.isfinite(distances).all():
            raise ValueError("nonfinite mesh comparison distance")
        worst = int(np.argmax(distances))
        result[key] = {
            "mean": float(distances.mean()),
            "p95": float(np.percentile(distances, 95)),
            "max": float(distances[worst]),
            "worst_sample": points[worst].tolist(),
        }
    return result


def reconstruct_mesh(original, options, rng):
    from mesh_to_sdf import get_surface_point_cloud
    from mesh_to_sdf.utils import get_raster_points, scale_to_unit_cube

    if not np.isfinite(original.vertices).all() or np.max(original.extents) <= 0:
        raise ValueError("invalid source mesh")
    cloud = get_surface_point_cloud(
        scale_to_unit_cube(original),
        surface_point_method="scan",
        bounding_radius=3**0.5,
        scan_count=options.scan_count,
        scan_resolution=options.scan_resolution,
        sample_point_count=options.sample_point_count,
        calculate_normals=True,
    )
    resolution = options.voxel_resolution
    points = get_raster_points(resolution)
    distances = np.concatenate(
        [
            cloud.get_sdf(
                points[i : i + options.sdf_batch_size],
                use_depth_buffer=False,
                sample_count=options.normal_sample_count,
            )
            for i in range(0, len(points), options.sdf_batch_size)
        ]
    )
    del cloud
    voxels = np.pad(distances.reshape((resolution,) * 3), 1, constant_values=1)
    mesh = mesh_from_voxels(voxels, original, resolution)
    return mesh, {
        "reference": REFERENCE,
        "upstream_commit": UPSTREAM,
        "options": asdict(options),
        "surface_point_method": "scan",
        "sign_method": "normal",
        "pad": True,
        "check_result": False,
        "spacing_normalized": [2 / resolution] * 3,
        "restore": "inverse unit-cube scale; reconstructed bbox center to original bbox center",
        "source_digest": mesh_digest(original),
        "mesh_digest": mesh_digest(mesh),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "volume": float(mesh.volume),
        "integrity_passed": True,
        "comparison": compare_surfaces(original, mesh, rng, options.comparison_samples),
        "versions": {
            name: version(name)
            for name in ("mesh-to-sdf", "scikit-image", "trimesh", "numpy")
        },
    }

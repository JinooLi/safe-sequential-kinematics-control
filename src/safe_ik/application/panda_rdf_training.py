"""Offline geometry, official RDF sampling, maximum estimation and validation.

No function in this module is called by the controller or IK workers.
"""

from dataclasses import asdict, dataclass
from importlib.metadata import version
from time import perf_counter

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from ..domain.panda_rdf import BernsteinRDF, ExtendedRDF, recursive_ridge
from .panda_mesh_reconstruction import UPSTREAM, ReconstructionOptions


@dataclass(frozen=True)
class TrainingOptions:
    basis_count: int = 8  # BP_8 means degree seven
    updates: int = 200
    near_batch: int = 1024
    random_batch: int = 256
    pool_size: int = 500000
    regularization: float = 1e-4
    scan_count: int = 100
    scan_resolution: int = 400
    normal_count: int = 100
    maximum_samples: int = 10000
    maximum_starts: int = 20
    optimizer_ftol: float = 1e-12
    reconstruction: ReconstructionOptions = ReconstructionOptions()

    def __post_init__(self):
        if isinstance(self.reconstruction, dict):
            object.__setattr__(
                self, "reconstruction", ReconstructionOptions(**self.reconstruction)
            )
        if not isinstance(self.reconstruction, ReconstructionOptions):
            raise ValueError("training.reconstruction must be a mapping")
        for key in (
            "basis_count",
            "updates",
            "near_batch",
            "random_batch",
            "pool_size",
            "scan_count",
            "scan_resolution",
            "normal_count",
            "maximum_samples",
            "maximum_starts",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"training.{key} must be a positive integer")
        if self.basis_count < 2 or self.scan_count < 2:
            raise ValueError("basis_count and scan_count must be at least two")
        if max(self.near_batch, self.random_batch) > self.pool_size:
            raise ValueError("training pool must contain each batch")
        if self.maximum_starts > self.maximum_samples:
            raise ValueError("maximum_starts exceeds maximum_samples")
        if any(
            not np.isfinite(v) or v <= 0
            for v in (self.regularization, self.optimizer_ftol)
        ):
            raise ValueError("regularization and optimizer tolerance must be positive")


def minimum_sphere(vertices):
    """Convex minimum enclosing ball on hull vertices; SLSQP, metric tolerance."""
    hull = np.asarray(vertices)[ConvexHull(vertices).vertices]
    origin = hull.mean(axis=0)
    scale = np.linalg.norm(hull - origin, axis=1).max()
    p = (hull - origin) / scale

    def constraint(x):
        return x[3] - np.sum((p - x[:3]) ** 2, axis=1)

    def jacobian(x):
        return np.column_stack((2 * (p - x[:3]), np.ones(len(p))))

    result = minimize(
        lambda x: x[3],
        [0, 0, 0, 1],
        jac=lambda x: np.array([0.0, 0.0, 0.0, 1.0]),
        constraints={"type": "ineq", "fun": constraint, "jac": jacobian},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success or constraint(result.x).min() < -1e-8:
        raise ValueError(f"minimum enclosing sphere failed: {result.message}")
    center = origin + scale * result.x[:3]
    radius = float(np.linalg.norm(vertices - center, axis=1).max())
    return center, radius


def sphere_directions(count):
    index = np.arange(count)
    z = 1 - 2 * (index + 0.5) / count
    angle = index * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(1 - z * z)
    return np.column_stack((r * np.cos(angle), r * np.sin(angle), z))


def dispersed_indices(points, count, initial=0):
    chosen = [int(initial)]
    squared = np.sum((points - points[initial]) ** 2, axis=1)
    for _ in range(1, count):
        squared[chosen] = -1
        index = int(np.argmax(squared))
        chosen.append(index)
        squared = np.minimum(squared, np.sum((points - points[index]) ** 2, axis=1))
    return np.asarray(chosen)


def surface_queries(mesh, count, rng):
    candidates, _ = trimesh.sample.sample_surface(mesh, max(count * 30, 1000), seed=rng)
    return candidates[dispersed_indices(candidates, count)]


def uniform_ball(rng, count, center, radius):
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return center + radius * rng.random(count)[:, None] ** (1 / 3) * directions


def training_pools(mesh, center, radius, rng, options):
    """Official scan/normal labels, normalized to the *outer* sphere.

    Reproduce mesh_to_sdf's 47% + 47% noisy surface / 6% uniform near pool,
    without its automatic re-normalization to a smaller mesh sphere. Restrict
    the uniform pool to the outer ball instead of the upstream cube. This is
    the specification's training-domain adaptation; regression is unchanged.
    Label in small chunks to avoid the upstream million-by-100 allocation.
    """
    from mesh_to_sdf import get_surface_point_cloud

    normalized = trimesh.Trimesh(
        (mesh.vertices - center) / radius, mesh.faces, process=False
    )
    cloud = get_surface_point_cloud(
        normalized,
        surface_point_method="scan",
        bounding_radius=1.0,
        scan_count=options.scan_count,
        scan_resolution=options.scan_resolution,
        calculate_normals=True,
    )
    count = options.pool_size
    half = int(count * 47 / 50) // 2
    surface = cloud.points[rng.integers(len(cloud.points), size=half)]
    near = np.vstack(
        (
            surface + rng.normal(scale=0.0025, size=(half, 3)),
            surface + rng.normal(scale=0.00025, size=(half, 3)),
            uniform_ball(rng, count - 2 * half, np.zeros(3), 1.0),
        )
    )
    invalid = np.linalg.norm(near, axis=1) > 1
    # Only the domain is altered: replace out-of-sphere draws with in-domain
    # surface-noise draws, without clipping points onto the sphere.
    while invalid.any():
        n = int(invalid.sum())
        near[invalid] = cloud.points[
            rng.integers(len(cloud.points), size=n)
        ] + rng.normal(scale=0.0025, size=(n, 3))
        invalid = np.linalg.norm(near, axis=1) > 1
    random = uniform_ball(rng, count, np.zeros(3), 1.0)

    def labels(points):
        return (
            np.concatenate(
                [
                    cloud.get_sdf(
                        points[i : i + 2048], sample_count=options.normal_count
                    )
                    for i in range(0, len(points), 2048)
                ]
            )
            * radius
        )

    return (
        near * radius + center,
        labels(near),
        random * radius + center,
        labels(random),
    )


def train_model(center, radius, pools, rng, options, *, device="cpu", progress=None):
    near, near_d, random, random_d = pools

    def batches():
        for _ in range(options.updates):
            a = rng.choice(len(near), options.near_batch, replace=False)
            b = rng.choice(len(random), options.random_batch, replace=False)
            yield np.vstack((near[a], random[b])), np.r_[near_d[a], random_d[b]]

    if device not in ("cpu", "cuda"):
        raise ValueError("training device must be cpu or cuda")
    fit = recursive_ridge
    kwargs = {}
    if device == "cuda":
        from ..adapters.rdf_torch_training import recursive_ridge_torch

        fit = recursive_ridge_torch
        kwargs = {"device": device, "progress": progress}
    return fit(
        BernsteinRDF(center, radius, np.zeros((options.basis_count,) * 3)),
        batches(),
        options.regularization,
        **kwargs,
    )


def estimate_maximum(rdf, inner_radius, options):
    directions = sphere_directions(options.maximum_samples)
    points = rdf.center + inner_radius * directions
    values, _ = rdf.query(points, gradient=False)
    order = np.argsort(-values, kind="stable")
    # Disperse among the highest decile, including the best sampled value.
    shortlist = order[: max(options.maximum_starts, len(order) // 10)]
    seeds = shortlist[dispersed_indices(directions[shortlist], options.maximum_starts)]
    maximum = float(values.max())
    runs = []
    for index in seeds:

        def objective(v):
            norm = np.linalg.norm(v)
            if norm < 1e-12:
                return 1e3, np.zeros(3)
            direction = v / norm
            f, grad = rdf.query((rdf.center + inner_radius * direction)[None])
            tangent_gradient = (grad[0] - direction * (grad[0] @ direction)) / norm
            return -f[0], -inner_radius * tangent_gradient

        result = minimize(
            objective,
            directions[index],
            jac=True,
            method="SLSQP",
            bounds=[(-1.0, 1.0)] * 3,
            constraints={
                "type": "eq",
                "fun": lambda v: v @ v - 1,
                "jac": lambda v: 2 * v,
            },
            options={"ftol": options.optimizer_ftol, "maxiter": 300},
        )
        norm = np.linalg.norm(result.x)
        if np.isfinite(norm) and norm > 0:
            value = float(
                rdf.query((rdf.center + inner_radius * result.x / norm)[None], False)[
                    0
                ][0]
            )
            maximum = max(maximum, value)
        else:
            raise ValueError("nonfinite maximum optimizer result")
        runs.append(
            {
                "success": bool(result.success),
                "message": str(result.message),
                "value": value,
                "iterations": int(result.nit),
            }
        )
    return maximum, {
        "method": "SLSQP equality constrained unit sphere",
        "ftol": options.optimizer_ftol,
        "sample_count": len(points),
        "starts": len(seeds),
        "sample_maximum": float(values.max()),
        "maximum": maximum,
        "runs": runs,
        "global_certificate": False,
    }


def mesh_truth(mesh, points):
    """Independent exact triangle proximity and trimesh signed-distance oracle.

    For non-watertight visual parts, sign is an empirical normal/ray convention;
    mark gradient ambiguity separately, never substitute scan training labels.
    """
    distances, gradients, ambiguous = [], [], []
    for block in np.array_split(points, max(1, int(np.ceil(len(points) / 64)))):
        closest, unsigned, triangles = trimesh.proximity.closest_point(mesh, block)
        signed = -trimesh.proximity.signed_distance(mesh, block)
        gradient = np.divide(
            block - closest,
            unsigned[:, None],
            out=np.zeros_like(block),
            where=unsigned[:, None] > 1e-10,
        )
        gradient *= np.where(signed < 0, -1, 1)[:, None]
        barycentric = trimesh.triangles.points_to_barycentric(
            mesh.triangles[triangles], closest
        )
        ambiguous.extend((unsigned < 1e-7) | (barycentric.min(axis=1) < 1e-5))
        distances.extend(signed)
        gradients.extend(gradient)
    return np.asarray(distances), np.asarray(gradients), np.asarray(ambiguous)


def numerical_gradient(model, points, step=1e-6):
    columns = []
    for axis in np.eye(3) * step:
        columns.append(
            (
                model.query(points + axis, False)[0]
                - model.query(points - axis, False)[0]
            )
            / (2 * step)
        )
    return np.column_stack(columns)


def validate_model(mesh, model, rng, error_margin):
    welded = mesh.copy()
    welded.merge_vertices()
    rdf = model.rdf
    global_points = uniform_ball(rng, 3000, rdf.center, rdf.radius)
    surface, ids = trimesh.sample.sample_surface(mesh, 2000, seed=rng)
    near_points = (
        surface + rng.uniform(-0.060, 0.060, (2000, 1)) * mesh.face_normals[ids]
    )
    points = np.vstack((global_points, near_points))
    truth, truth_gradient, ambiguous = mesh_truth(mesh, points)
    predicted, gradient = rdf.query(points)
    error = np.abs(predicted - truth)
    near = np.abs(truth) <= 0.060
    # Test f inside its ball and D in all three branches, independent of labels.
    check = uniform_ball(rng, 100, rdf.center, rdf.radius)
    rdf_difference = float(
        np.max(np.abs(rdf.query(check)[1] - numerical_gradient(rdf, check)))
    )
    # Mid-bin radii avoid placing a central-difference stencil exactly on the
    # merely C1 boundary (where its truncation error is O(step), not O(step^2)).
    extended_points = (
        rdf.center
        + sphere_directions(100)
        * ((np.arange(100) + 0.5) * 1.5 * rdf.radius / 100)[:, None]
    )
    extension_difference = float(
        np.max(
            np.abs(
                model.query(extended_points)[1]
                - numerical_gradient(model, extended_points)
            )
        )
    )
    continuity = []
    for radius in (model.inner_radius, rdf.radius):
        directions = sphere_directions(100)
        v0, g0 = model.query(rdf.center + directions * (radius - 1e-8))
        v1, g1 = model.query(rdf.center + directions * (radius + 1e-8))
        continuity.append(
            [float(np.max(np.abs(v1 - v0))), float(np.max(np.abs(g1 - g0)))]
        )
    derivative_pass = (
        rdf_difference <= 1e-5
        and extension_difference <= 1e-5
        and all(v < 1e-5 and g < 1e-4 for v, g in continuity)
    )
    norm = np.linalg.norm(gradient, axis=1)
    valid = ~ambiguous & (norm > 1e-8)
    angles = np.arccos(
        np.clip(
            np.sum(gradient[valid] * truth_gradient[valid], axis=1) / norm[valid], -1, 1
        )
    )
    worst = np.flatnonzero(near)[np.argsort(-error[near], kind="stable")[:5]]
    return {
        "passed": bool(error[near].max() <= error_margin and derivative_pass),
        "sample_counts": [3000, 2000],
        "near_count": int(near.sum()),
        "near_max_error": float(error[near].max()),
        "global_max_error": float(error[:3000].max()),
        "mean_error": float(error.mean()),
        "sign_errors": int(np.sum((predicted < 0) != (truth < 0))),
        "worst_near_points": [
            {
                "point": points[i].tolist(),
                "truth": float(truth[i]),
                "prediction": float(predicted[i]),
                "absolute_error": float(error[i]),
            }
            for i in worst
        ],
        "gradient_magnitude_mean_error": (
            float(np.abs(norm[valid] - 1).mean()) if valid.any() else None
        ),
        "gradient_angle_mean_rad": float(angles.mean()) if len(angles) else None,
        "ambiguous_gradient_count": int(ambiguous.sum()),
        "mesh_watertight": bool(mesh.is_watertight),
        "welded_mesh_watertight": bool(welded.is_watertight),
        "sign_oracle": "negative trimesh.proximity.signed_distance; exact triangles + normal/ray signs",
        "derivatives_passed": bool(derivative_pass),
        "rdf_fd_max": rdf_difference,
        "extension_fd_max": extension_difference,
        "boundary_value_gradient_jumps": continuity,
        "fd_step": 1e-6,
        "fd_tolerance": 1e-5,
        "boundary_step": 1e-8,
        "boundary_tolerances": [1e-5, 1e-4],
        "continuous_error_certificate": False,
    }


def validate_joint_gradients(kinematics, bundle, options, rng):
    from ..domain.panda_rdf import relative_point_jacobian
    from ..domain.panda_rdf_safety import PandaRDFSafety

    evaluator = PandaRDFSafety(
        kinematics, bundle.models, bundle.queries, bundle.pairs, options
    )
    maximum, relative_maximum, excluded = 0.0, 0.0, 0
    timings = []
    step = 1e-6
    for _ in range(5):
        q = rng.uniform(kinematics.lower_limits, kinematics.upper_limits)
        start = perf_counter()
        evaluation = evaluator.evaluate(q)
        timings.append(perf_counter() - start)
        numerical = np.column_stack(
            [
                (
                    evaluator.evaluate(q + axis).values
                    - evaluator.evaluate(q - axis).values
                )
                / (2 * step)
                for axis in np.eye(7) * step
            ]
        )
        # Selection changes are allowed but excluded from smooth-gradient tests.
        from ..domain.panda_rdf import top_indices

        def selection(state, target, source):
            R, t, _ = kinematics.frame_pose_and_jacobian(target, state)
            S, u, _ = kinematics.frame_pose_and_jacobian(source, state)
            local = (bundle.queries[source] @ S.T + u - t) @ R
            return top_indices(
                bundle.models[target].query(local, False)[0], options.top_k
            )

        for row, (left, right) in enumerate(bundle.pairs):
            stable = True
            for target, source in ((left, right), (right, left)):
                initial = np.sort(selection(q, target, source))
                for axis in np.eye(7) * step:
                    stable &= np.array_equal(
                        initial, np.sort(selection(q + axis, target, source))
                    )
                    stable &= np.array_equal(
                        initial, np.sort(selection(q - axis, target, source))
                    )
                R, t, jac = kinematics.frame_pose_and_jacobian(target, q)
                point, point_jac = kinematics.frame_point_and_jacobian(
                    source, q, bundle.queries[source][0]
                )
                analytic = relative_point_jacobian(
                    R, t, jac[:3], jac[3:], point[None], point_jac[None]
                )[0]

                def local(state):
                    rot, pos, _ = kinematics.frame_pose_and_jacobian(target, state)
                    x, _ = kinematics.frame_point_and_jacobian(
                        source, state, bundle.queries[source][0]
                    )
                    return rot.T @ (x - pos)

                numeric = np.column_stack(
                    [
                        (local(q + a) - local(q - a)) / (2 * step)
                        for a in np.eye(7) * step
                    ]
                )
                relative_maximum = max(
                    relative_maximum, float(np.max(np.abs(analytic - numeric)))
                )
            if stable:
                maximum = max(
                    maximum,
                    float(np.max(np.abs(evaluation.gradients[row] - numerical[row]))),
                )
            else:
                excluded += 1
    return {
        "top_k": options.top_k,
        "softmin_error": options.softmin_error,
        "passed": bool(
            maximum <= 1e-4
            and relative_maximum <= 1e-7
            and excluded < 5 * len(bundle.pairs)
        ),
        "joint_gradient_max_error": maximum,
        "relative_jacobian_max_error": relative_maximum,
        "switching_pairs_excluded": excluded,
        "configurations": 5,
        "step": step,
        "joint_tolerance": 1e-4,
        "relative_tolerance": 1e-7,
        "cpu_evaluation_seconds_mean": float(np.mean(timings)),
    }


def training_provenance(options):
    return {
        "options": asdict(options),
        "upstream_commit": UPSTREAM,
        "algorithm": "Li ICRA 2024 Algorithm 1; w0=0, B0=I/1e-4; float64 CPU solves",
        "normalization": "(p-c)/(2*r1)+0.5; targets/r1; metric weights",
        "sampling_adaptation": "outer-sphere normalization; uniform random ball replaces cube; shared scan cloud; chunked normal labels",
        "versions": {
            name: version(name)
            for name in ("numpy", "scipy", "trimesh", "mesh-to-sdf", "mujoco")
        },
    }


def benchmark_candidate(config, kinematics, bundle):
    """Offline timing only, including unapproved candidates; no trajectories."""
    from ..domain.panda_rdf_safety import PandaRDFSafety
    from ..domain.spatial_safety import PandaCapsuleSafety

    capsule = PandaCapsuleSafety(
        kinematics,
        config.safety.capsules,
        config.obstacles,
        config.safety.margin,
        config.safety.self_collision_exclusions,
    )
    obstacles = PandaCapsuleSafety(
        kinematics,
        config.safety.capsules,
        config.obstacles,
        config.safety.margin,
        include_self=False,
    )
    rdf = PandaRDFSafety(
        kinematics,
        bundle.models,
        bundle.queries,
        bundle.pairs,
        config.safety.rdf,
        obstacles,
    )
    rng = np.random.default_rng(
        np.random.SeedSequence(config.safety.rdf.seed, spawn_key=(0, 4))
    )
    states = rng.uniform(kinematics.lower_limits, kinematics.upper_limits, size=(21, 7))
    report = {
        "configurations": 20,
        "warmup": 1,
        "control_period": config.trajectory.integration_period,
        "purpose": "CPU timing only, not a safety or trajectory comparison",
    }
    for name, evaluator in (("rdf", rdf), ("capsule", capsule)):
        times = []
        for q in states:
            start = perf_counter()
            evaluation = evaluator.evaluate(q)
            times.append(perf_counter() - start)
        report[name] = {
            "mean_seconds": float(np.mean(times[1:])),
            "p95_seconds": float(np.percentile(times[1:], 95)),
            "barrier_count": len(evaluation.values),
        }
    report["rdf_components_total_seconds"] = rdf.total_timings
    return report

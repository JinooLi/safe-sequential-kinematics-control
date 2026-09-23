"""CPU 3D Bernstein RDF primitives for Panda self-collision.

Learning follows Li et al., ICRA 2024, Algorithm 1; this module does not
load geometry, approve artifacts, or replace the existing safety backend.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve
from scipy.special import comb, logsumexp


def _points(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite (N, 3) array")
    return points


def basis_1d(t, count):
    """Bernstein basis without clamping; derivative is valid at endpoints."""
    powers = np.arange(count)
    return (
        comb(count - 1, powers)
        * t[:, None] ** powers
        * (1 - t[:, None]) ** (count - 1 - powers)
    )


@dataclass
class BernsteinRDF:
    center: np.ndarray
    radius: float
    weights: np.ndarray

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        if (
            self.center.shape != (3,)
            or not np.isfinite(self.center).all()
            or not np.isfinite(self.radius)
            or self.radius <= 0
            or self.weights.ndim != 3
            or len(set(self.weights.shape)) != 1
            or self.weights.shape[0] < 2
            or not np.isfinite(self.weights).all()
        ):
            raise ValueError("invalid 3D Bernstein RDF")

    @property
    def count(self):
        """Number of basis functions per axis; degree is count - 1."""
        return self.weights.shape[0]

    def features(self, points):
        t = (_points(points) - self.center) / (2 * self.radius) + 0.5
        b = [basis_1d(t[:, axis], self.count) for axis in range(3)]
        return np.einsum("bi,bj,bk->bijk", *b).reshape(len(t), self.weights.size)

    def query(self, points, gradient=True):
        """Return metric distance and its unnormalized analytic gradient."""
        points = _points(points)
        values, gradients = [], []
        for start in range(0, len(points), 512):
            t = (points[start : start + 512] - self.center) / (2 * self.radius) + 0.5
            b = [basis_1d(t[:, axis], self.count) for axis in range(3)]
            values.append(np.einsum("bi,bj,bk,ijk->b", *b, self.weights, optimize=True))
            if gradient:
                g = []
                for axis in range(3):
                    low = basis_1d(t[:, axis], self.count - 1)
                    derivative = (
                        (self.count - 1)
                        * (
                            np.pad(low, ((0, 0), (1, 0)))
                            - np.pad(low, ((0, 0), (0, 1)))
                        )
                        / (2 * self.radius)
                    )
                    bases = list(b)
                    bases[axis] = derivative
                    g.append(
                        np.einsum(
                            "bi,bj,bk,ijk->b", *bases, self.weights, optimize=True
                        )
                    )
                gradients.append(np.column_stack(g))
        return (
            np.concatenate(values) if values else np.empty(0),
            np.vstack(gradients) if gradients else np.empty((0, 3)),
        )


def recursive_ridge(model, batches, regularization=1e-4):
    """Algorithm 1 with w0=0, B0=I/lambda as in official bf_sdf.py.

    Targets/coefficients are normalized by radius, as in unit-sphere training.
    Use a solve rather than an explicit inverse; run in float64 on CPU.
    The batches must be generated offline and inside the training sphere.
    """
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be positive")
    size = model.weights.size
    covariance = np.eye(size) / regularization
    weights = np.zeros(size)
    seen = 0
    for points, targets in batches:
        points = _points(points)
        targets = np.asarray(targets, dtype=float)
        if (
            not len(points)
            or targets.shape != (len(points),)
            or not np.isfinite(targets).all()
            or np.any(
                np.linalg.norm(points - model.center, axis=1) > model.radius + 1e-12
            )
        ):
            raise ValueError("invalid training batch or points outside training sphere")
        phi = model.features(points)
        cross = covariance @ phi.T
        gain = solve(np.eye(len(points)) + phi @ cross, cross.T, assume_a="pos").T
        weights += gain @ (targets / model.radius - phi @ weights)
        covariance -= gain @ cross.T
        covariance = (covariance + covariance.T) * 0.5
        if not np.isfinite(weights).all() or not np.isfinite(covariance).all():
            raise ValueError("nonfinite recursive regression state")
        seen += len(points)
    if not seen:
        raise ValueError("training batches must not be empty")
    model.weights = (weights * model.radius).reshape(model.weights.shape)
    return model


@dataclass
class ExtendedRDF:
    rdf: BernsteinRDF
    inner_radius: float
    surface_maximum: float

    def __post_init__(self):
        if not (0 < self.inner_radius < self.rdf.radius) or not np.isfinite(
            self.surface_maximum
        ):
            raise ValueError("invalid extension radii or maximum")

    def query(self, points, gradient=True):
        """C1 scalar blend; this surrogate is NOT a true-distance lower bound."""
        p = _points(points)
        offset = p - self.rdf.center
        rho = np.linalg.norm(offset, axis=1)
        inner = rho <= self.inner_radius
        outer = rho >= self.rdf.radius
        band = ~(inner | outer)
        value = self.surface_maximum + rho - self.inner_radius
        v = np.divide(
            offset, rho[:, None], out=np.zeros_like(offset), where=rho[:, None] > 0
        )
        grad = v.copy() if gradient else np.empty((0, 3))
        # Never evaluate the polynomial beyond its trained sphere.
        ids = np.flatnonzero(~outer)
        f, df = self.rdf.query(p[ids], gradient)
        value[ids] = f
        if gradient:
            grad[ids] = df
        band_ids = np.flatnonzero(band)
        if len(band_ids):
            width = self.rdf.radius - self.inner_radius
            t = (rho[band_ids] - self.inner_radius) / width
            sigma = t * t * (3 - 2 * t)
            dsigma = 6 * t * (1 - t) / width
            F = self.surface_maximum + rho[band_ids] - self.inner_radius
            f = value[band_ids].copy()
            value[band_ids] = (1 - sigma) * f + sigma * F
            if gradient:
                grad[band_ids] = (1 - sigma[:, None]) * grad[band_ids] + (
                    sigma + (F - f) * dsigma
                )[:, None] * v[band_ids]
        return value, grad


def top_indices(values, count=10):
    """Deterministic ordering on ties; selection is recomputed each call."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("distances must be a finite vector")
    if count is None:
        return np.arange(len(values))
    if type(count) is not int or count < 1:
        raise ValueError("top-k must be a positive integer or None")
    return np.argsort(values, kind="stable")[:count]


def soft_min(values, error=0.005):
    """Return unnormalized LSE and derivative weights; error is in metres."""
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or not np.isfinite(error)
        or error <= 0
    ):
        raise ValueError("soft-min requires finite distances and positive error")
    if len(values) == 1:
        return float(values[0]), np.ones(1)
    beta = np.log(len(values)) / error
    logits = -beta * (values - values.min())
    norm = logsumexp(logits)
    return float(values.min() - norm / beta), np.exp(logits - norm)


def relative_point_jacobian(
    rotation,
    origin,
    origin_jacobian,
    angular_jacobian,
    world_points,
    world_point_jacobians,
):
    """World-aligned Jacobians in; query-frame local point Jacobians out.

    Both frames move. For r=x-t, Jp=R.T @ (Jx-Jt+[r]x Jomega).
    """
    r = _points(world_points) - np.asarray(origin)
    cross = np.zeros((len(r), 3, 3))
    cross[:, 0, 1], cross[:, 0, 2] = -r[:, 2], r[:, 1]
    cross[:, 1, 0], cross[:, 1, 2] = r[:, 2], -r[:, 0]
    cross[:, 2, 0], cross[:, 2, 1] = -r[:, 1], r[:, 0]
    return np.einsum(
        "ij,bjk->bik",
        np.asarray(rotation).T,
        np.asarray(world_point_jacobians)
        - np.asarray(origin_jacobian)
        + cross @ np.asarray(angular_jacobian),
    )

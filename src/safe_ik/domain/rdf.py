"""Two-dimensional Bernstein-polynomial robot distance fields.

This is the 2D counterpart of the basis-function RDF in Li et al.  Each link
has a scalar field in its own coordinate frame.  Forward kinematics transforms
an obstacle query point into that frame, so the learned model is independent
of joint configuration during training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import comb

from .safety import PlanarCapsuleRobot


class BernsteinSDF2D:
    """SDF represented by a tensor-product Bernstein polynomial on a box."""

    def __init__(self, degree: int, bounds, weights=None):
        if degree < 1:
            raise ValueError("degree must be at least 1")
        self.degree = int(degree)
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.shape != (2, 2) or np.any(
            self.bounds[:, 1] <= self.bounds[:, 0]
        ):
            raise ValueError("bounds must have shape (2, 2) with [min, max] per axis")
        feature_count = (self.degree + 1) ** 2
        self.weights = (
            np.zeros(feature_count)
            if weights is None
            else np.asarray(weights, dtype=float).reshape(feature_count)
        )

    def _normalized(self, points):
        points = np.asarray(points, dtype=float)
        if points.shape[-1] != 2:
            raise ValueError("points must end in dimension 2")
        return (points - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

    def _basis_1d(self, t):
        """Return Bernstein values and t derivatives for a vector of inputs."""
        t = np.asarray(t, dtype=float).reshape(-1)
        indices = np.arange(self.degree + 1)
        values = (
            comb(self.degree, indices)
            * t[:, None] ** indices
            * (1.0 - t[:, None]) ** (self.degree - indices)
        )

        derivatives = np.zeros_like(values)
        # d B_i^n / dt = n (B_{i-1}^{n-1} - B_i^{n-1})
        lower_indices = np.arange(self.degree)
        lower = (
            comb(self.degree - 1, lower_indices)
            * t[:, None] ** lower_indices
            * (1.0 - t[:, None]) ** (self.degree - 1 - lower_indices)
        )
        derivatives[:, 0] = -self.degree * lower[:, 0]
        derivatives[:, -1] = self.degree * lower[:, -1]
        if self.degree > 1:
            derivatives[:, 1:-1] = self.degree * (lower[:, :-1] - lower[:, 1:])
        return values, derivatives

    def features_and_gradient(self, points):
        """Return basis values and physical-coordinate basis derivatives."""
        normalized = self._normalized(points)
        bx, dbx = self._basis_1d(normalized[..., 0])
        by, dby = self._basis_1d(normalized[..., 1])
        features = np.einsum("ni,nj->nij", bx, by).reshape(len(bx), -1)
        grad_x = np.einsum("ni,nj->nij", dbx, by).reshape(len(bx), -1)
        grad_y = np.einsum("ni,nj->nij", bx, dby).reshape(len(bx), -1)
        scale = self.bounds[:, 1] - self.bounds[:, 0]
        gradients = np.stack((grad_x / scale[0], grad_y / scale[1]), axis=-1)
        return features, gradients

    def fit(self, points, sdf_values, ridge: float = 1e-8):
        """Fit the coefficient vector with ridge least squares."""
        features, _ = self.features_and_gradient(points)
        values = np.asarray(sdf_values, dtype=float).reshape(-1)
        if len(features) != len(values):
            raise ValueError("points and sdf_values must have the same length")
        lhs = features.T @ features + ridge * np.eye(features.shape[1])
        self.weights = np.linalg.solve(lhs, features.T @ values)
        return self

    def query(self, points):
        """Evaluate the SDF and its analytic spatial gradient.

        Outside the fitted box, evaluate at the projected boundary point and
        add the Euclidean distance to that point.  This is the bounded-domain
        extension used by the RDF paper and prevents a polynomial extrapolation
        from producing unsafe, arbitrarily large values.
        """
        points = np.asarray(points, dtype=float)
        original_shape = points.shape[:-1]
        flat_points = points.reshape(-1, 2)
        projected = np.clip(flat_points, self.bounds[:, 0], self.bounds[:, 1])
        features, gradient_basis = self.features_and_gradient(projected)
        values = features @ self.weights
        gradients = np.einsum("nfi,f->ni", gradient_basis, self.weights)
        displacement = flat_points - projected
        displacement_norm = np.linalg.norm(displacement, axis=1)
        outside = displacement_norm > 0.0
        values[outside] += displacement_norm[outside]
        # Through the projection only a non-clamped coordinate varies.  The
        # remaining derivative comes from the added boundary distance.
        inside_coordinate = (flat_points >= self.bounds[:, 0]) & (
            flat_points <= self.bounds[:, 1]
        )
        gradients *= inside_coordinate
        gradients[outside] += displacement[outside] / displacement_norm[outside, None]
        return values.reshape(original_shape), gradients.reshape(*original_shape, 2)

    def project_to_level_set(
        self,
        seed,
        level: float = 0.0,
        max_iterations: int = 30,
        tolerance: float = 1e-6,
    ):
        """Project a seed to ``f(x)=level`` using Newton steps along gradient."""
        point = np.asarray(seed, dtype=float).copy()
        for _ in range(max_iterations):
            value, gradient = self.query(point)
            residual = float(value - level)
            if abs(residual) < tolerance:
                return point, True
            gradient = gradient.reshape(2)
            point -= residual * gradient / (gradient @ gradient + 1e-12)
            # The polynomial may be queried outside its training box, but a
            # projected level-set sample should remain inside it.
            point = np.clip(point, self.bounds[:, 0], self.bounds[:, 1])
        return point, False

    def sample_level_set(
        self, count: int, level: float = 0.0, rng=None, max_attempts: int | None = None
    ):
        """Randomly seed and project samples onto one SDF level set."""
        rng = np.random.default_rng() if rng is None else rng
        max_attempts = max_attempts or count * 50
        samples = []
        for _ in range(max_attempts):
            seed = rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            point, success = self.project_to_level_set(seed, level)
            if success:
                samples.append(point)
                if len(samples) == count:
                    break
        if len(samples) != count:
            raise RuntimeError("could not obtain enough level-set samples")
        return np.asarray(samples)


def capsule_sdf_local(points, length: float, radius: float):
    """Vectorized signed distance to the conservative local capsule."""
    points = np.asarray(points, dtype=float)
    nearest_x = np.clip(points[..., 0], 0.0, length)
    delta = points - np.stack((nearest_x, np.zeros_like(nearest_x)), axis=-1)
    return np.linalg.norm(delta, axis=-1) - radius


def training_samples(length: float, radius: float, bounds, count: int, rng=None):
    """Generate a uniform/near-surface mixture labelled by the reference SDF."""
    if count < 4:
        raise ValueError("count must be at least 4")
    rng = np.random.default_rng() if rng is None else rng
    bounds = np.asarray(bounds, dtype=float)
    uniform_count = count // 2
    points = [rng.uniform(bounds[:, 0], bounds[:, 1], size=(uniform_count, 2))]
    # Sample the capsule boundary, then perturb it.  This concentrates labels
    # near the CBF-relevant zero and safety-margin level sets.
    angles = rng.uniform(0.0, 2.0 * np.pi, size=count - uniform_count)
    cap = rng.integers(0, 2, size=len(angles)) * length
    boundary = np.column_stack((cap + radius * np.cos(angles), radius * np.sin(angles)))
    boundary += rng.normal(scale=2.0 * radius, size=boundary.shape)
    points.append(np.clip(boundary, bounds[:, 0], bounds[:, 1]))
    points = np.vstack(points)
    return points, capsule_sdf_local(points, length, radius)


class PlanarBernsteinRobot(PlanarCapsuleRobot):
    """A serial arm whose per-link SDFs are learned Bernstein fields."""

    def __init__(self, link_lengths, link_radius: float, models):
        super().__init__(link_lengths, link_radius)
        if len(models) != self.dof:
            raise ValueError("one Bernstein model is required per link")
        self.models = list(models)

    @classmethod
    def train_from_capsules(
        cls,
        link_lengths,
        link_radius: float,
        degree: int = 12,
        sample_count: int = 12000,
        padding: float = 0.3,
        ridge: float = 1e-8,
        rng=None,
    ):
        """Train per-link models from the conservative capsule reference."""
        rng = np.random.default_rng() if rng is None else rng
        models = []
        for length in link_lengths:
            bounds = np.array([[-padding, length + padding], [-padding, padding]])
            points, values = training_samples(
                length, link_radius, bounds, sample_count, rng
            )
            models.append(BernsteinSDF2D(degree, bounds).fit(points, values, ridge))
        return cls(link_lengths, link_radius, models)

    def link_sdf_and_gradient(self, point, q, link_index: int):
        point = np.asarray(point, dtype=float)
        q = np.asarray(q, dtype=float)
        origin, rotation, origin_jacobian = self._link_pose(q, link_index)
        local_point = rotation.T @ (point - origin)
        sdf, grad_local = self.models[link_index].query(local_point)
        local_point_jacobian = -rotation.T @ origin_jacobian
        local_point_jacobian[:, : link_index + 1] += (
            -np.array([[0.0, -1.0], [1.0, 0.0]]) @ local_point
        ).reshape(2, 1)
        return float(sdf), grad_local.reshape(2) @ local_point_jacobian

    def save(self, path):
        """Persist all per-link fields in one portable NPZ file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            link_lengths=self.link_lengths,
            link_radius=self.link_radius,
            degrees=np.array([model.degree for model in self.models]),
            bounds=np.stack([model.bounds for model in self.models]),
            weights=np.stack([model.weights for model in self.models]),
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            models = [
                BernsteinSDF2D(int(degree), bound, weight)
                for degree, bound, weight in zip(
                    data["degrees"], data["bounds"], data["weights"]
                )
            ]
            return cls(data["link_lengths"], float(data["link_radius"]), models)

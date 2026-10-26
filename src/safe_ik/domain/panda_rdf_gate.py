"""C1 pair-level sphere/RDF blending in a geometrically separated region."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SphereGateEvaluation:
    values: np.ndarray
    gradients: np.ndarray
    weights: np.ndarray
    weight_gradients: np.ndarray


class SpherePairGate:
    policy = "outer_sphere_safe_reserve_smoothstep_v1"

    def __init__(self, models, queries, pairs, vertices, margin, fraction):
        if vertices is None:
            raise ValueError("RDF sphere gate requires verified visual vertices")
        self.names = tuple(models)
        indices = {name: i for i, name in enumerate(self.names)}
        self.left = np.array([indices[a] for a, _ in pairs], dtype=int)
        self.right = np.array([indices[b] for _, b in pairs], dtype=int)
        centers, radii, outer = [], [], []
        for name in self.names:
            model = models[name]
            center = np.asarray(model.rdf.center, dtype=float)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError(f"{name}: invalid sphere center")
            radius = 0.0
            # The same center as the artifact, but bound the actual visual
            # triangles, not an unchecked radius from artifact metadata.
            for mapping in (vertices, queries):
                points = np.asarray(mapping.get(name, []), dtype=float)
                if (
                    points.ndim != 2
                    or points.shape[1] != 3
                    or not len(points)
                    or not np.isfinite(points).all()
                ):
                    raise ValueError(f"{name}: invalid sphere-bound vertices/queries")
                radius = max(
                    radius, float(np.linalg.norm(points - center, axis=1).max())
                )
            radius += 1e-9  # Outward roundoff guard in metres.
            if not np.isfinite(model.rdf.radius) or model.rdf.radius <= radius:
                raise ValueError(f"{name}: outer sphere does not contain geometry")
            centers.append(center.copy())
            radii.append(radius)
            outer.append(float(model.rdf.radius))
        self.centers = np.asarray(centers)
        radii, outer = np.asarray(radii), np.asarray(outer)
        self.outer_sums = outer[self.left] + outer[self.right]
        self.clearance_sums = radii[self.left] + radii[self.right] + margin
        reserve = self.outer_sums - self.clearance_sums
        self.widths = fraction * reserve
        if (
            not np.isfinite(reserve).all()
            or not np.isfinite(self.widths).all()
            or np.any(reserve <= 1e-9)
            or np.any(self.widths <= 0)
            or np.any(self.widths >= reserve)
        ):
            raise ValueError("RDF sphere gate has no strictly safe transition reserve")
        self.metadata = {
            "policy": self.policy,
            "bound": "visual mesh vertices and stored queries; artifact centers",
            "roundoff_padding_m": 1e-9,
            "bounding_radii_m": dict(zip(self.names, radii.tolist())),
            "outer_radii_m": dict(zip(self.names, outer.tolist())),
            "transition_fraction": fraction,
            "pairs": [
                {
                    "links": list(pair),
                    "outer_sum_m": float(self.outer_sums[i]),
                    "transition_width_m": float(self.widths[i]),
                    "full_rdf_distance_m": float(self.outer_sums[i] - self.widths[i]),
                    "minimum_transition_clearance_m": float(
                        reserve[i] - self.widths[i]
                    ),
                }
                for i, pair in enumerate(pairs)
            ],
        }

    def evaluate(self, poses):
        # Includes motion of the local sphere centers due to frame rotation.
        offsets = np.stack([poses[n][0] @ c for n, c in zip(self.names, self.centers)])
        centers = offsets + np.stack([poses[n][1] for n in self.names])
        jac = np.stack([poses[n][2] for n in self.names])
        center_jac = jac[:, :3] + np.cross(
            jac[:, 3:].transpose(0, 2, 1), offsets[:, None]
        ).transpose(0, 2, 1)
        delta = centers[self.right] - centers[self.left]
        length = np.linalg.norm(delta, axis=1)
        unit = np.divide(
            delta, length[:, None], out=np.zeros_like(delta), where=length[:, None] > 0
        )
        gradient = np.einsum(
            "bi,bij->bj", unit, center_jac[self.right] - center_jac[self.left]
        )
        t = np.clip((self.outer_sums - length) / self.widths, 0.0, 1.0)
        weight = t * t * (3 - 2 * t)
        slope = -6 * t * (1 - t) / self.widths
        # At coincident centers weight=1, slope=0: the sphere gradient is
        # unused. Returning finite zeros avoids an irrelevant 0/0.
        return SphereGateEvaluation(
            length - self.clearance_sums, gradient, weight, slope[:, None] * gradient
        )

    def evaluate_values(self, poses):
        offsets = np.stack([poses[n][0] @ c for n, c in zip(self.names, self.centers)])
        centers = offsets + np.stack([poses[n][1] for n in self.names])
        length = np.linalg.norm(centers[self.right] - centers[self.left], axis=1)
        t = np.clip((self.outer_sums - length) / self.widths, 0.0, 1.0)
        return length - self.clearance_sums, t * t * (3 - 2 * t)

"""Planar capsule geometry and CBF constraints for the robot controller.

The MuJoCo/URDF links are cylinders of radius ``link_radius`` whose axes lie
along each link's local x axis.  Their planar projection has flat end caps;
this module uses a capsule that *contains* that projection.  It is therefore
a conservative reference geometry for collision avoidance, and a useful
baseline against which a learned RDF can later be validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SKEW = np.array([[0.0, -1.0], [1.0, 0.0]])


@dataclass(frozen=True)
class CircleObstacle:
    """A circular planar obstacle."""

    center: np.ndarray
    radius: float


class PlanarCapsuleRobot:
    """Conservative 2D SDF model of a serial planar arm with capsule links.

    The query gradient is analytic except at the usual SDF non-smooth sets
    (capsule medial axis and equidistant links).  Each link is queried
    separately, so the controller does not need to differentiate a ``min``
    operation when the nearest link changes.
    """

    def __init__(self, link_lengths, link_radius: float):
        self.link_lengths = np.asarray(link_lengths, dtype=float)
        if self.link_lengths.ndim != 1 or len(self.link_lengths) == 0:
            raise ValueError("link_lengths must be a non-empty 1-D sequence")
        if link_radius <= 0.0:
            raise ValueError("link_radius must be positive")
        self.link_radius = float(link_radius)
        self.dof = len(self.link_lengths)

    def _link_pose(self, q: np.ndarray, link_index: int):
        """Return origin, rotation, orientation, and origin Jacobian."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.dof,):
            raise ValueError(f"q must have shape ({self.dof},)")
        if not 0 <= link_index < self.dof:
            raise IndexError("invalid link index")

        angles = np.cumsum(q)
        origin = np.zeros(2)
        joint_positions = [origin.copy()]
        for i in range(link_index):
            direction = np.array([np.cos(angles[i]), np.sin(angles[i])])
            origin = origin + self.link_lengths[i] * direction
            joint_positions.append(origin.copy())

        origin_jacobian = np.zeros((2, self.dof))
        for joint in range(link_index):
            displacement = origin - joint_positions[joint]
            origin_jacobian[:, joint] = _SKEW @ displacement

        theta = angles[link_index]
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        return origin, rotation, origin_jacobian

    def _link_sdf_gradients(self, point, q, link_index: int):
        """Return SDF derivatives for a fixed query point and its position."""
        point = np.asarray(point, dtype=float)
        if point.shape != (2,):
            raise ValueError("point must have shape (2,)")
        q = np.asarray(q, dtype=float)
        origin, rotation, origin_jacobian = self._link_pose(q, link_index)
        local_point = rotation.T @ (point - origin)
        length = self.link_lengths[link_index]

        closest_x = np.clip(local_point[0], 0.0, length)
        delta = local_point - np.array([closest_x, 0.0])
        norm_delta = np.linalg.norm(delta)
        # The gradient is undefined on the segment centreline.  The chosen
        # normal is deterministic and is only used for an already penetrating
        # point, where a conservative recovery direction is sufficient.
        if norm_delta < 1e-12:
            grad_local = np.array([0.0, 1.0])
            distance_to_axis = 0.0
        else:
            grad_local = delta / norm_delta
            distance_to_axis = norm_delta
        sdf = distance_to_axis - self.link_radius

        local_point_jacobian = -rotation.T @ origin_jacobian
        # theta_k = q_0 + ... + q_k.
        local_point_jacobian[:, : link_index + 1] += (-_SKEW @ local_point).reshape(
            2, 1
        )
        gradient_q = grad_local @ local_point_jacobian
        gradient_point = rotation @ grad_local
        return float(sdf), gradient_q, gradient_point

    def link_sdf_and_gradient(self, point, q, link_index: int):
        """Return capsule signed distance and analytic derivative w.r.t. q.

        ``point`` is treated as fixed.  Use ``self_collision_cbf_constraint``
        when the query point itself is attached to the robot.
        """
        sdf, gradient_q, _ = self._link_sdf_gradients(point, q, link_index)
        return sdf, gradient_q

    def end_effector_position_and_jacobian(self, q):
        """Return the final link-tip position and its planar Jacobian."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.dof,):
            raise ValueError(f"q must have shape ({self.dof},)")
        angles = np.cumsum(q)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        position = self.link_lengths @ directions
        jacobian = np.zeros((2, self.dof))
        for joint in range(self.dof):
            downstream = self.link_lengths[joint:] @ directions[joint:]
            jacobian[:, joint] = _SKEW @ downstream
        return position, jacobian

    def self_collision_cbf_constraint(self, q, minimum_distance: float, alpha: float):
        """Return a hard CBF keeping the final tip away from link 1.

        The barrier is ``h = f_link1(p_ee(q), q) - minimum_distance``.
        Its derivative includes both link-1 motion and the motion of the
        end-effector query point, so it can be used directly as
        ``gradient @ q_dot >= -alpha * h``.
        """
        if self.dof < 2:
            raise ValueError("self-collision CBF requires at least two links")
        if minimum_distance < 0.0 or alpha <= 0.0:
            raise ValueError("minimum_distance must be >= 0 and alpha must be > 0")
        q = np.asarray(q, dtype=float)
        position, position_jacobian = self.end_effector_position_and_jacobian(q)
        sdf, fixed_point_gradient, point_gradient = self._link_sdf_gradients(
            position, q, link_index=0
        )
        h = sdf - minimum_distance
        gradient = fixed_point_gradient + point_gradient @ position_jacobian
        return gradient, float(-alpha * h), float(h)

    def base_to_link_cbf_constraint(
        self, q, link_index: int, minimum_distance: float, alpha: float
    ):
        """Return a CBF keeping a fixed base point away from one link.

        The base point is fixed at the planar origin, so the derivative is
        simply the configuration derivative of that link's capsule SDF.
        """
        if minimum_distance < 0.0 or alpha <= 0.0:
            raise ValueError("minimum_distance must be >= 0 and alpha must be > 0")
        sdf, gradient = self.link_sdf_and_gradient(
            np.zeros(2), q, link_index=link_index
        )
        h = sdf - minimum_distance
        return gradient, float(-alpha * h), float(h)

    def cbf_constraints(self, q, obstacles, safe_distance: float, alpha: float):
        """Construct ``A q_dot >= lower`` for all link--circle pairs.

        For a circle, ``h = f_link(center, q) - obstacle_radius - d_safe``.
        The returned rows impose ``dh/dq q_dot >= -alpha h``.
        """
        if safe_distance < 0.0 or alpha <= 0.0:
            raise ValueError("safe_distance must be >= 0 and alpha must be > 0")
        rows, lower, distances = [], [], []
        for obstacle in obstacles:
            for link_index in range(self.dof):
                sdf, gradient_q = self.link_sdf_and_gradient(
                    obstacle.center, q, link_index
                )
                h = sdf - obstacle.radius - safe_distance
                rows.append(gradient_q)
                lower.append(-alpha * h)
                distances.append(h)
        if not rows:
            return (
                np.empty((0, self.dof)),
                np.empty(0),
                np.empty(0),
            )
        return np.vstack(rows), np.asarray(lower), np.asarray(distances)


def obstacles_from_config(obstacle_config) -> list[CircleObstacle]:
    """Parse the existing controller.yaml obstacle list."""
    obstacles = []
    for entry in obstacle_config or []:
        center = np.asarray(entry["pos"], dtype=float)
        if center.shape != (2,):
            raise ValueError("obstacle pos must contain [x, y]")
        obstacles.append(CircleObstacle(center=center, radius=float(entry["radius"])))
    return obstacles

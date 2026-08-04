"""Render the CLF=constant level surface in a 3-DOF joint space."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .configuration import load_controller_config


def end_effector_task_state(q, link_lengths):
    """Return planar end-effector ``[x, y, yaw]`` for a serial arm."""
    q = np.asarray(q, dtype=float)
    link_lengths = np.asarray(link_lengths, dtype=float)
    if q.shape[-1] != len(link_lengths):
        raise ValueError("last q dimension must match link_lengths")
    angles = np.cumsum(q, axis=-1)
    x = np.sum(link_lengths * np.cos(angles), axis=-1)
    y = np.sum(link_lengths * np.sin(angles), axis=-1)
    return np.stack((x, y, angles[..., -1]), axis=-1)


def evaluate_clf_in_q_space(q, link_lengths, ee_target, ee_orientation):
    """Evaluate the controller's CLF directly on one or more configurations."""
    state = end_effector_task_state(q, link_lengths)
    position_error = state[..., :2] - np.asarray(ee_target, dtype=float)
    yaw_error = (state[..., 2] - float(ee_orientation) + np.pi) % (2.0 * np.pi) - np.pi
    return 2.0 * (np.sum(position_error * position_error, axis=-1) + yaw_error * yaw_error)


_TETRAHEDRA = np.array(
    [
        [0, 5, 1, 6],
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
    ],
    dtype=int,
)


def _edge_crossing(point_a, value_a, point_b, value_b, level):
    """Linearly interpolate a level-set crossing on one tetrahedron edge."""
    fraction = (level - value_a) / (value_b - value_a)
    return point_a + fraction * (point_b - point_a)


def marching_tetrahedra(q_axes, values, level):
    """Extract triangular level-set faces from a scalar field on a 3D grid.

    This small dependency-free marching-tetrahedra implementation is used
    instead of requiring a separate 3D image-processing package.
    """
    if len(q_axes) != 3 or values.shape != tuple(len(axis) for axis in q_axes):
        raise ValueError("values must have one dimension for each of three q_axes")

    faces = []
    for index in np.ndindex(*(dimension - 1 for dimension in values.shape)):
        i, j, k = index
        corners = np.array(
            [
                [q_axes[0][i], q_axes[1][j], q_axes[2][k]],
                [q_axes[0][i + 1], q_axes[1][j], q_axes[2][k]],
                [q_axes[0][i + 1], q_axes[1][j + 1], q_axes[2][k]],
                [q_axes[0][i], q_axes[1][j + 1], q_axes[2][k]],
                [q_axes[0][i], q_axes[1][j], q_axes[2][k + 1]],
                [q_axes[0][i + 1], q_axes[1][j], q_axes[2][k + 1]],
                [q_axes[0][i + 1], q_axes[1][j + 1], q_axes[2][k + 1]],
                [q_axes[0][i], q_axes[1][j + 1], q_axes[2][k + 1]],
            ]
        )
        cube_values = np.array(
            [
                values[i, j, k], values[i + 1, j, k], values[i + 1, j + 1, k], values[i, j + 1, k],
                values[i, j, k + 1], values[i + 1, j, k + 1], values[i + 1, j + 1, k + 1], values[i, j + 1, k + 1],
            ]
        )
        for tetrahedron in _TETRAHEDRA:
            points = corners[tetrahedron]
            tetra_values = cube_values[tetrahedron]
            inside = tetra_values <= level
            count = int(np.sum(inside))
            if count == 0 or count == 4:
                continue
            crossings = []
            for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
                if inside[a] != inside[b]:
                    crossings.append(_edge_crossing(points[a], tetra_values[a], points[b], tetra_values[b], level))
            if count in (1, 3):
                faces.append(np.asarray(crossings))
            else:  # A four-sided intersection; split it into two triangles.
                faces.append(np.asarray([crossings[0], crossings[1], crossings[2]]))
                faces.append(np.asarray([crossings[0], crossings[2], crossings[3]]))
    return np.asarray(faces)


def render_clf_level_set(
    link_lengths,
    ee_target,
    ee_orientation,
    level: float = 1.0,
    q_min: float = -np.pi,
    q_max: float = np.pi,
    resolution: int = 36,
    output_path: str | Path | None = None,
    show: bool = False,
):
    """Render ``V(q)=level`` for a three-link planar arm and return its figure."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if len(link_lengths) != 3:
        raise ValueError("q-space surface rendering currently requires exactly 3 DOF")
    if level < 0.0 or q_min >= q_max or resolution < 2:
        raise ValueError("level must be non-negative, q_min < q_max, and resolution >= 2")

    q_axes = tuple(np.linspace(q_min, q_max, resolution) for _ in range(3))
    grid = np.stack(np.meshgrid(*q_axes, indexing="ij"), axis=-1)
    values = evaluate_clf_in_q_space(grid, link_lengths, ee_target, ee_orientation)
    faces = marching_tetrahedra(q_axes, values, level)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    if len(faces):
        surface = Poly3DCollection(faces, facecolor="tab:blue", edgecolor="none", alpha=0.50)
        ax.add_collection3d(surface)
    else:
        ax.text2D(0.05, 0.95, "No V(q) = level surface in this range", transform=ax.transAxes)
    q_reference = np.zeros(3)
    ax.scatter(*q_reference, color="black", s=28, label="q = 0 (reference)")
    ax.set(xlim=(q_min, q_max), ylim=(q_min, q_max), zlim=(q_min, q_max))
    ax.set_xlabel("q₁ [rad]")
    ax.set_ylabel("q₂ [rad]")
    ax.set_zlabel("q₃ [rad]")
    ax.set_title(f"CLF level surface in q-space: V(q) = {level:g}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    return fig, faces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="settings/controller.yaml")
    parser.add_argument("--level", type=float, default=1.0, help="CLF level V(q) to draw")
    parser.add_argument("--q-min", type=float, default=-np.pi, help="Lower joint-angle bound [rad]")
    parser.add_argument("--q-max", type=float, default=np.pi, help="Upper joint-angle bound [rad]")
    parser.add_argument("--resolution", type=int, default=36, help="Samples per joint axis")
    parser.add_argument("--output", default="outputs/clf_level_set.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    config = load_controller_config(args.config)
    _, faces = render_clf_level_set(
        config.safety.link_lengths,
        config.ee_target,
        config.ee_orientation,
        level=args.level,
        q_min=args.q_min,
        q_max=args.q_max,
        resolution=args.resolution,
        output_path=args.output,
        show=args.show,
    )
    print(f"Saved CLF level-set visualization ({len(faces)} triangles): {args.output}")


if __name__ == "__main__":
    main()

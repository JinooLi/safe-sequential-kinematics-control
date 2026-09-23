"""Distance field evaluation and rendering for planar safety models."""

from pathlib import Path

import numpy as np

from ..domain.rdf import PlanarBernsteinRobot, capsule_sdf_local


def evaluate_robot_distance_field(robot, q, points):
    """Evaluate ``min_k f_k(p, q)`` for a batch of planar world points."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    q = np.asarray(q, dtype=float)
    field = np.full(len(points), np.inf)
    for link_index in range(robot.dof):
        origin, rotation, _ = robot._link_pose(q, link_index)
        local_points = (points - origin) @ rotation
        if isinstance(robot, PlanarBernsteinRobot):
            values, _ = robot.models[link_index].query(local_points)
        else:
            values = capsule_sdf_local(
                local_points, robot.link_lengths[link_index], robot.link_radius
            )
        field = np.minimum(field, values)
    return field


def _link_endpoints(robot, q):
    endpoints = []
    for link_index, length in enumerate(robot.link_lengths):
        origin, rotation, _ = robot._link_pose(q, link_index)
        endpoints.append((origin, origin + rotation @ np.array([length, 0.0])))
    return endpoints


def render_distance_field(
    robot,
    obstacles,
    safe_distance: float,
    q,
    x_limits=(-1.0, 4.0),
    y_limits=(-2.0, 3.0),
    resolution: int = 220,
    output_path: str | Path | None = None,
    show: bool = False,
):
    """Create a figure with SDF values, zero contour, and safety contour."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Circle

    x = np.linspace(*x_limits, resolution)
    y = np.linspace(*y_limits, resolution)
    xx, yy = np.meshgrid(x, y)
    values = evaluate_robot_distance_field(
        robot, q, np.column_stack((xx.ravel(), yy.ravel()))
    )
    values = values.reshape(xx.shape)

    maximum = max(float(np.max(np.abs(values))), safe_distance, 0.1)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.pcolormesh(
        xx,
        yy,
        values,
        shading="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
    )
    fig.colorbar(image, ax=ax, label="signed distance to robot [m]")
    ax.contour(xx, yy, values, levels=[0.0], colors="black", linewidths=2)
    ax.contour(
        xx,
        yy,
        values,
        levels=[safe_distance],
        colors="black",
        linestyles="--",
        linewidths=1.5,
    )

    for origin, end in _link_endpoints(robot, q):
        ax.plot([origin[0], end[0]], [origin[1], end[1]], color="black", linewidth=2)
        ax.add_patch(
            Circle(origin, robot.link_radius, fill=False, color="black", linewidth=1)
        )
        ax.add_patch(
            Circle(end, robot.link_radius, fill=False, color="black", linewidth=1)
        )
    for obstacle in obstacles:
        ax.add_patch(
            Circle(obstacle.center, obstacle.radius, color="tab:red", alpha=0.65)
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Robot distance field: solid = surface, dashed = safety level")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    return fig

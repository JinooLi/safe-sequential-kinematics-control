"""Render 3D CLBF task-space slices for all pairs of [x, y, yaw]."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..control.qp_controller import QPController
from ..domain.clbf import build_cumulative_clbfs, end_effector_barrier
from .configuration import load_controller_config


_TASK_PAIRS = ((0, 1), (0, 2), (1, 2))
_TASK_LABELS = ("x [m]", "y [m]", "yaw [rad]")


def evaluate_final_clbf_in_task_space(task_state, controller: QPController):
    """Evaluate the final CLBF directly in a planar task state.

    The CLBF value depends on ``[x, y, yaw]`` through the existing CLF and
    on ``[x, y]`` through the end-effector obstacle barriers.  An identity
    task Jacobian is sufficient because this helper visualizes values only;
    it does not visualize a joint-space gradient.
    """
    task_state = np.asarray(task_state, dtype=float)
    if task_state.shape[-1] != 3:
        raise ValueError("task_state last dimension must be [x, y, yaw]")
    flat_states = task_state.reshape(-1, 3)
    values = np.empty(len(flat_states))
    identity_task_jacobian = np.eye(3)
    for index, state in enumerate(flat_states):
        barriers = tuple(
            end_effector_barrier(
                state[:2],
                identity_task_jacobian,
                controller.obstacles_by_index[module.obstacle_index],
                controller.clbf_end_effector_radius,
                controller.safe_distance,
                module.epsilon,
            )
            for module in controller.clbf_modules
        )
        values[index] = build_cumulative_clbfs(
            controller.clf(state),
            np.zeros(3),
            barriers,
            controller.clbf_gamma,
            controller.clbf_modules,
        )[-1].value
    return values.reshape(task_state.shape[:-1])


def render_clbf_task_slices(
    controller: QPController,
    xy_limit: float = 3.2,
    yaw_limit: float = np.pi,
    resolution: int = 100,
    output_path: str | Path | None = None,
    show: bool = False,
):
    """Plot final ``W_2`` surfaces over xy, x-yaw, and y-yaw task slices.

    The coordinate not displayed in each panel is fixed at its configured
    end-effector target.  The red curve denotes the CLBF zero level.
    """
    import matplotlib.pyplot as plt

    if controller.control_mode != "clbf":
        raise ValueError("CLBF slices require control_mode: clbf")
    if xy_limit <= 0.0 or yaw_limit <= 0.0 or resolution < 8:
        raise ValueError("limits must be positive and resolution must be at least 8")

    task_target = np.array(
        [*controller.ee_obj, controller.ee_orientation], dtype=float
    )
    axes = (
        np.linspace(-xy_limit, xy_limit, resolution),
        np.linspace(-xy_limit, xy_limit, resolution),
        np.linspace(-yaw_limit, yaw_limit, resolution),
    )
    slice_data = []
    for first, second in _TASK_PAIRS:
        horizontal, vertical = np.meshgrid(axes[first], axes[second], indexing="xy")
        task_grid = np.broadcast_to(task_target, horizontal.shape + (3,)).copy()
        task_grid[..., first] = horizontal
        task_grid[..., second] = vertical
        value = evaluate_final_clbf_in_task_space(task_grid, controller)
        slice_data.append((first, second, horizontal, vertical, value))

    value_min = min(float(np.nanmin(value)) for *_, value in slice_data)
    value_max = max(float(np.nanmax(value)) for *_, value in slice_data)
    figure = plt.figure(figsize=(16, 5.5), constrained_layout=True)
    surfaces = []
    for panel, (first, second, horizontal, vertical, value) in enumerate(slice_data, start=1):

        axis = figure.add_subplot(1, 3, panel, projection="3d")
        surface = axis.plot_surface(
            horizontal,
            vertical,
            value,
            cmap="viridis",
            vmin=value_min,
            vmax=value_max,
            linewidth=0,
            antialiased=True,
        )
        # Project the W=0 boundary onto the panel floor for easy reading.
        z_floor = float(np.nanmin(value))
        if np.nanmin(value) <= 0.0 <= np.nanmax(value):
            axis.contour(horizontal, vertical, value, levels=[0.0], colors="crimson", offset=z_floor)
        axis.scatter(
            task_target[first], task_target[second],
            evaluate_final_clbf_in_task_space(task_target, controller),
            color="crimson", s=35, label="target",
        )
        axis.set_xlabel(_TASK_LABELS[first])
        axis.set_ylabel(_TASK_LABELS[second])
        axis.set_zlabel("final CLBF W₂")
        fixed = ({0, 1, 2} - {first, second}).pop()
        axis.set_title(
            f"W₂({_TASK_LABELS[first]}, {_TASK_LABELS[second]})\n"
            f"{_TASK_LABELS[fixed]} = target"
        )
        axis.legend(loc="upper right")
        surfaces.append(surface)
    figure.colorbar(surfaces[-1], ax=figure.axes, shrink=0.62, pad=0.08, label="W₂")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
    if show:
        plt.show()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="settings/controller.yaml")
    parser.add_argument("--xy-limit", type=float, default=3.2)
    parser.add_argument("--yaw-limit", type=float, default=np.pi)
    parser.add_argument("--resolution", type=int, default=100)
    parser.add_argument("--output", default="outputs/clbf_task_slices.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    config = load_controller_config(args.config)
    controller = QPController(config, solver=None)
    render_clbf_task_slices(
        controller,
        xy_limit=args.xy_limit,
        yaw_limit=args.yaw_limit,
        resolution=args.resolution,
        output_path=args.output,
        show=args.show,
    )
    print(f"Saved CLBF task-space slices: {args.output}")


if __name__ == "__main__":
    main()

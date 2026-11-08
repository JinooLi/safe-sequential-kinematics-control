"""Per-link Lyapunov histories, including inactive sequential targets."""

from pathlib import Path

import numpy as np

from .recording import _configure_matplotlib_cache


def plot_planar_lyapunov(arrays, path=None, *, show=False):
    """Plot each task on its own axis and emphasize its active interval.

    All curves use the saved controller weights and selected IK. They are
    evaluated at every state; only the bold segments were active in the QP.
    """
    required = ("t", "clf_by_link", "active_link")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(
            "Trace lacks per-link Lyapunov data ("
            + ", ".join(missing)
            + "). Record a new sequential planar run."
        )
    t = np.asarray(arrays["t"], dtype=float)
    values = np.asarray(arrays["clf_by_link"], dtype=float)
    active = np.asarray(arrays["active_link"])
    if (
        t.ndim != 1
        or not t.size
        or values.ndim != 2
        or values.shape[0] != t.size
        or values.shape[1] != 3
        or active.shape != t.shape
        or not np.all(np.isfinite(t))
        or not np.all(np.isfinite(values))
        or np.any(np.diff(t) <= 0)
        or not np.all(np.isin(active, (1, 2, 3)))
    ):
        raise ValueError("Expected finite planar values (N, 3) and valid time/stages.")

    _configure_matplotlib_cache()
    if show:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(
            4,
            1,
            sharex=True,
            figsize=(10, 8),
            gridspec_kw={"height_ratios": [1, 1, 1, 0.5]},
            constrained_layout=True,
        )
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(10, 8), constrained_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.subplots(
            4, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 0.5]}
        )

    switch_times = t[np.flatnonzero(np.diff(active) != 0) + 1]
    for index, (axis, color) in enumerate(zip(axes[:3], ("C0", "C1", "C2"))):
        task = index + 1
        axis.plot(
            t,
            values[:, index],
            color=color,
            alpha=0.55,
            linestyle="--",
            linewidth=1.2,
            label="Evaluated at every state",
        )
        axis.plot(
            t,
            np.where(active == task, values[:, index], np.nan),
            color=color,
            linewidth=2.5,
            label="Active in the QP",
        )
        axis.set_ylabel(f"$V_{task}$")
        axis.set_title(
            f"Link {task}: " + ("position + yaw" if task == 3 else "position"),
            loc="left",
            fontsize=10,
        )
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)

    axes[-1].step(t, active, where="post", color="0.25", linewidth=1.8)
    axes[-1].set_yticks([1, 2, 3], ["link1", "link2", "link3 / EE"])
    axes[-1].set_ylim(0.7, 3.3)
    axes[-1].set_ylabel("Active task")
    axes[-1].set_xlabel("Simulation time [s]")
    axes[-1].grid(alpha=0.25)
    for axis in axes:
        for time in switch_times:
            axis.axvline(time, color="0.45", linestyle=":", linewidth=1)
    figure.suptitle("Planar 3-DOF: sequential Lyapunov functions")
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
    if show:
        plt.show()
    return figure

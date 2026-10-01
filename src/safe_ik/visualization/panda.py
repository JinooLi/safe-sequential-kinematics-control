"""Panda diagnostic plots from completed traces."""

from pathlib import Path

import numpy as np

from ..application.panda_results import PandaControlTrace


def save_panda_control_plot(
    trace: PandaControlTrace | None, path: str | Path, *, show: bool = False
) -> bool:
    """Save CLF, slack, CBF, and recorded joint angles; optionally show them."""
    if trace is None or trace.time.size == 0:
        return False

    import matplotlib.pyplot as plt

    rows = 4 if trace.q is not None else 3
    figure, axes = plt.subplots(
        rows, 1, sharex=True, figsize=(11, 3 * rows), layout="constrained"
    )
    axes[0].plot(trace.time, trace.clf_value, color="tab:blue")
    axes[0].set_ylabel("CLF  V(q)")
    axes[0].set_title("Panda CLF-CBF QP diagnostics")

    axes[1].plot(trace.time, trace.clf_slack, color="tab:orange")
    axes[1].set_ylabel("CLF slack  δ")

    for values in trace.barrier_values.T:
        axes[2].plot(trace.time, values, color="tab:gray", alpha=0.32, linewidth=0.8)
    if trace.barrier_values.size:
        axes[2].plot(
            trace.time,
            np.min(trace.barrier_values, axis=1),
            color="black",
            linewidth=2.0,
            label="minimum CBF",
        )
    axes[2].axhline(
        0.0, color="tab:red", linestyle="--", linewidth=1.0, label="safety boundary"
    )
    axes[2].set_ylabel("CBF  hᵢ(q)")
    axes[2].legend(loc="best")

    if trace.q is not None:
        # Convert only for display; recorded/controller angles remain radians.
        for joint, angles in enumerate(np.rad2deg(trace.q).T, start=1):
            axes[3].plot(trace.time, angles, label=f"q{joint}", linewidth=1.2)
        axes[3].set_ylabel("Joint angle [deg]")
        axes[3].legend(loc="best", ncol=min(7, trace.q.shape[1]))

    for axis in axes:
        axis.grid(True, alpha=0.3)
    transition_samples = np.flatnonzero(np.diff(trace.stage_index) != 0) + 1
    for sample in transition_samples:
        for axis in axes:
            axis.axvline(
                trace.time[sample], color="tab:purple", alpha=0.35, linewidth=0.8
            )
        stage = int(trace.stage_index[sample])
        if stage < len(trace.stage_names):
            axes[0].annotate(
                trace.stage_names[stage],
                (trace.time[sample], trace.clf_value[sample]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7,
                rotation=90,
            )
    axes[-1].set_xlabel("time [s]")
    plot_path = Path(path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return True

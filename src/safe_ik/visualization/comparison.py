"""Plots built from completed comparison artifacts."""

import json
from pathlib import Path

import numpy as np

from ..experiments.models import COMPARISONS, METHODS


def _write_paired_plot(output_dir: Path, records):
    """Render direct-vs-sequential scatter plots for task decomposition."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(20, 3.6), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 5)
    metrics = (
        ("completion_time", "Time to goal [s]"),
        ("control_effort", "Control effort [rad]"),
        ("min_barrier", "Minimum barrier [m]"),
        ("max_clf_slack", "Maximum CLF slack"),
        ("nonzero_clf_slack_ratio_to_goal", "Nonzero CLF slack ratio to goal"),
    )
    baseline, candidate = COMPARISONS["task_decomposition"]
    for axis, (metric, title) in zip(axes, metrics):
        x, y = [], []
        for record in records:
            if (
                record[baseline][metric] is not None
                and record[candidate][metric] is not None
            ):
                x.append(record[baseline][metric])
                y.append(record[candidate][metric])
        if x:
            lower, upper = min(x + y), max(x + y)
            padding = max((upper - lower) * 0.05, 1e-6)
            axis.scatter(x, y, alpha=0.75, s=22)
            axis.plot(
                [lower - padding, upper + padding],
                [lower - padding, upper + padding],
                "k--",
                linewidth=1,
            )
            axis.set_xlim(lower - padding, upper + padding)
            axis.set_ylim(lower - padding, upper + padding)
        axis.set_title(title)
        axis.set_xlabel(baseline)
        axis.set_ylabel(candidate)
        axis.grid(True)
    figure.savefig(output_dir / "paired_metrics.png", dpi=180)


def _joint_points(q, link_lengths):
    angles = np.cumsum(q)
    vectors = (
        np.column_stack((np.cos(angles), np.sin(angles)))
        * np.asarray(link_lengths)[:, None]
    )
    return np.vstack((np.zeros(2), np.cumsum(vectors, axis=0)))


def visualize_case(results_dir: Path, case_id: int, output_path: Path | None = None):
    """Render direct and sequential task-decomposition trajectories."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    record = next(
        (item for item in summary["cases"] if item["case_id"] == case_id), None
    )
    if record is None:
        raise ValueError(f"case_id {case_id} was not found in {results_dir}")
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True
    )
    colors = {"direct": "tab:orange", "sequential": "tab:blue"}
    titles = {"direct": "Direct EE CLF + CBF", "sequential": "Sequential CLF"}
    link_lengths = summary["settings"]["link_lengths"]
    axis_limit = (
        max(
            sum(link_lengths),
            max(abs(value) for value in summary["settings"]["obstacle_position_range"]),
            max(abs(value) for value in record["target"]),
        )
        + 0.2
    )
    for axis, mode in zip(axes, METHODS):
        title = titles[mode]
        result = record[mode]
        trace = np.load(results_dir / result["trajectory_file"])
        for obstacle in record["obstacles"]:
            axis.add_patch(
                Circle(
                    obstacle["position"],
                    obstacle["radius"],
                    color="crimson",
                    alpha=0.45,
                )
            )
        axis.plot(
            trace["final_xy"][:, 0],
            trace["final_xy"][:, 1],
            color=colors[mode],
            linewidth=1.5,
            label="final EE path",
        )
        axis.scatter(*trace["final_xy"][0], color="black", s=25, label="start")
        axis.scatter(*trace["final_xy"][-1], color=colors[mode], s=45, label="final")
        axis.scatter(
            *record["target"], marker="*", color="tab:green", s=150, label="goal"
        )
        arrow_scale = 0.30
        axis.arrow(
            *record["target"],
            arrow_scale * np.cos(record["target_yaw"]),
            arrow_scale * np.sin(record["target_yaw"]),
            color="tab:green",
            width=0.012,
            length_includes_head=True,
        )
        axis.arrow(
            *trace["final_xy"][-1],
            arrow_scale * np.cos(result["final_yaw"]),
            arrow_scale * np.sin(result["final_yaw"]),
            color=colors[mode],
            width=0.010,
            length_includes_head=True,
        )
        joints = _joint_points(trace["q"][-1], link_lengths)
        axis.plot(
            joints[:, 0],
            joints[:, 1],
            "o-",
            color="black",
            linewidth=2,
            label="final arm",
        )
        axis.set_aspect("equal")
        axis.set_xlim(-axis_limit, axis_limit)
        axis.set_ylim(-axis_limit, axis_limit)
        axis.grid(True)
        axis.set_title(
            f"{title}\nsuccess={result['success']}, pos err={result['position_error']:.3f} m, "
            f"yaw err={result['yaw_error']:.3f} rad"
        )
        axis.legend(loc="upper right", fontsize=8)
    figure.suptitle(f"Case {case_id}: {record['comparisons']}")
    output_path = output_path or results_dir / f"case_{case_id:03d}_comparison.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved visualization: {output_path}")


def visualize_all_cases(results_dir: Path, output_dir: Path | None = None):
    """Render one comparison PNG for every case in a completed experiment."""
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    output_dir = output_dir or results_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in summary["cases"]:
        case_id = record["case_id"]
        visualize_case(
            results_dir,
            case_id,
            output_dir / f"case_{case_id:03d}_comparison.png",
        )
    print(f"Saved {len(summary['cases'])} visualizations: {output_dir}")

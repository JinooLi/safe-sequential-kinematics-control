"""Build narrative figures from frozen evaluation records; no new rollout."""

from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/masters-defense-story-mpl")
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "evidence", ROOT / "Fig"
BLUE, GREEN, RED, GRAY = "#205493", "#28735e", "#a84439", "#77818b"
FONT_SIZE = 22  # One source size for all visible labels; scaled with the vector figure.
plt.rcParams.update({
    "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE, "figure.titlesize": FONT_SIZE,
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{kotex}\usepackage[english]{babel}\usepackage[T1]{fontenc}\usepackage{lmodern}\renewcommand{\familydefault}{\sfdefault}",
    "font.family": "sans-serif", "font.size": FONT_SIZE,
    "axes.unicode_minus": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelcolor": "#243447",
    "text.color": "#243447",
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=.08)
    subprocess.run([
        "pdftoppm", "-r", "150", "-singlefile", "-png",
        str(FIG / f"{name}.pdf"), str(FIG / name),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    plt.close(fig)


def rows(name):
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))


def read(name):
    return json.loads((DATA / name).read_text())


def decorate(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.15)


def fk(q):
    angles = np.cumsum(q)
    return np.vstack(([0, 0], np.cumsum(
        np.c_[np.cos(angles), np.sin(angles)], axis=0)))


source_names = [
    "case030_report.json", "case_030_direct.npz", "case_030_sequential.npz",
    "all_links_comparison.csv", "pair_outcomes.csv", "selector_choices.csv",
    "recomputed_summary.json",
]
audit = {
    "font_rendering": "pdfLaTeX + ko.TeX + Latin Modern; PNG rendered from PDF",
    "no_new_simulation": True,
    "no_new_training": True,
    "source_sha256": {
        name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        for name in source_names
    },
}

# Case 30: reproduce the initial and same-time Direct geometry from scene30.
# This precedes the IK-method explanation, so show only the EE goal here.
report = read("case030_report.json")
case = report["original_case"]
assert case["case_id"] == 30
assert case["sequential"]["success"] and not case["direct"]["success"]
assert case["direct"]["failure_reason"] == "qp_infeasible"
goals = np.asarray(report["lyapunov_reconstruction"]["targets"])
np.testing.assert_allclose(
    fk(report["lyapunov_reconstruction"]["selected_ik_q"])[1:],
    goals, atol=1e-12,
)
with np.load(DATA / "case_030_direct.npz", allow_pickle=False) as direct, \
        np.load(DATA / "case_030_sequential.npz", allow_pickle=False) as sequential:
    np.testing.assert_array_equal(direct["t"], sequential["t"])
    np.testing.assert_array_equal(direct["q"][0], sequential["q"][0])
    snapshot = report["snapshots"][-1]
    index = snapshot["sample_index"]
    snapshot_time = float(direct["t"][index])
    assert abs(snapshot_time - 18.226) < 1e-9
    states = [direct["q"][0].copy(), direct["q"][index].copy()]
    position_error = float(np.linalg.norm(direct["final_xy"][index] - case["target"]))
    yaw_delta = float(direct["final_yaw"][index] - case["target_yaw"])
    yaw_error = float(abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))))

margin = .08
cloud = [np.zeros((1, 2)), goals, *[fk(q) for q in states]]
for obstacle in case["obstacles"]:
    center = np.asarray(obstacle["position"])
    radius = obstacle["radius"] + margin
    cloud.append(np.asarray([center - radius, center + radius]))
cloud = np.concatenate(cloud)
lo, hi = cloud.min(axis=0) - .22, cloud.max(axis=0) + .22
linkcolors = ["#0072B2", "#E69F00", "#009E73"]


def scene(ax, q, active):
    points = fk(q)
    for obstacle in case["obstacles"]:
        ax.add_patch(Circle(
            obstacle["position"], obstacle["radius"] + margin,
            fill=False, edgecolor="#9b9b9b", ls=":", lw=1.5, zorder=1))
        ax.add_patch(Circle(
            obstacle["position"], obstacle["radius"], facecolor="#d0d0d0",
            edgecolor="#666666", lw=1.5, zorder=2))
    for j, color in enumerate(linkcolors):
        a, b = points[j:j + 2]
        delta = b - a
        normal = .05 * np.asarray([-delta[1], delta[0]]) / np.linalg.norm(delta)
        ax.add_patch(Polygon(
            [a + normal, b + normal, b - normal, a - normal],
            color=color, zorder=3))
        for endpoint in (a, b):
            ax.add_patch(Circle(endpoint, .05, color=color, zorder=3))
        if j == 2:
            ax.plot(*goals[j], marker="*", ms=12, color=color,
                    markeredgecolor="white", markeredgewidth=.5, zorder=6)
    ax.plot(*points.T, "o", ms=4, mfc="white", mec="#444444", mew=.7, zorder=5)
    ax.plot(0, 0, "s", ms=7, color="#333333", zorder=6)
    if active is not None:
        ax.plot(*points[active], "o", ms=14, mfc="none", mec="#333333", mew=1.4, zorder=7)
    headings = [(goals[-1], case["target_yaw"], "#555555")]
    if active == 3:
        headings.append((points[-1], sum(q), linkcolors[-1]))
    for point, yaw, color in headings:
        ax.annotate("", xy=point + .25 * np.asarray([np.cos(yaw), np.sin(yaw)]),
                    xytext=point, arrowprops={"arrowstyle": "->", "color": color,
                                             "lw": 1.3, "mutation_scale": 9})
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal")
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, q, active, title in zip(
        axes, states, [None, 3],
        ["Initial configuration\n$t=0.000$ s", "Direct: goal not reached\n$t=18.226$ s"]):
    scene(ax, q, active)
    ax.set_title(title, fontsize=FONT_SIZE, pad=8)
legend = [
    Patch(facecolor="#d0d0d0", edgecolor="#666666", label="Obstacle"),
    Line2D([], [], color=linkcolors[-1], marker="*", ls="none", ms=10, label="EE goal"),
]
fig.subplots_adjust(left=.01, right=.99, bottom=.17, top=.77, wspace=.13)
fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=FONT_SIZE,
           frameon=False, handlelength=1.7, columnspacing=1.1)
save(fig, "story_planar_direct")
audit["planar_direct"] = {
    "case_id": 30, "snapshot_seconds": [0, snapshot_time],
    "same_initial_state_verified": True,
    "displayed_goals": "EE target position and yaw only; intermediate IK targets omitted before method introduction",
    "same_snapshot_time_as_sequential_success": True,
    "direct_position_error_at_snapshot_m": position_error,
    "direct_yaw_error_at_snapshot_rad": yaw_error,
    "direct_final_failure_class": case["direct"]["failure_reason"],
    "direct_qp_infeasible_count": case["direct"]["qp_infeasible_count"],
    "failure_class_describes_full_35s_evaluation_not_snapshot": True,
    "obstacle_halo": "Nominal 0.08 m geometric margin; not exact RDF contours",
    "all_geometry_inside_common_axes": bool(np.all(cloud > lo) and np.all(cloud < hi)),
}

# Time comparison must use the intersection of successful scenes.
validation = rows("all_links_comparison.csv")
assert len(validation) == len({row["case_id"] for row in validation}) == 1000
shared = [row for row in validation
          if row["direct_120_success"] == "True"
          and row["sequential_120_success"] == "True"]
assert len(shared) == 330
times = [float(np.mean([float(row[f"{method}_120_completion_time"])
                       for row in shared])) for method in ("direct", "sequential")]
assert [round(value, 2) for value in times] == [24.24, 44.48]
fig, ax = plt.subplots(figsize=(8, 2.5))
bars = ax.barh(["Direct", "All-link Sequential"], times, color=[GRAY, BLUE], height=.5)
time_labels = ax.bar_label(bars, labels=[f"{value:.2f} s" for value in times], padding=7, fontsize=FONT_SIZE)
time_labels[1].set_color(RED)
time_labels[1].set_fontweight("bold")
ax.invert_yaxis()
ax.set(xlim=(0, 58), xticks=[0, 20, 40], xlabel="Mean completion time (s)")
decorate(ax)
fig.tight_layout()
save(fig, "story_time_cost")
audit["time_cost"] = {
    "split": "validation", "evaluation_seconds": 120,
    "shared_success_scenes": len(shared),
    "direct_mean_seconds": times[0], "all_link_sequential_mean_seconds": times[1],
    "all_link_over_direct_ratio": times[1] / times[0],
    "emphasis": "44.48 s label in red and bold; unchanged archived mean",
}

# Oracle is the post-hoc existence of a successful pair among all 10 pairs.
# It is not a selector that can be executed without knowing the outcomes.
pair_rows = rows("pair_outcomes.csv")
pairs = read("recomputed_summary.json")["action_pairs"]
validation_pairs = [row for row in pair_rows if row["split"] == "validation"]
assert len(validation_pairs) == 10000
scene_pairs = {}
for row in validation_pairs:
    scene_pairs.setdefault(row["case_id"], {})[int(row["pair_index"])] = int(row["success"])
assert len(scene_pairs) == 1000 and all(len(outcomes) == 10 for outcomes in scene_pairs.values())
pair_successes = [sum(outcomes[index] for outcomes in scene_pairs.values()) for index in range(10)]
best_index = int(np.argmax(pair_successes))
oracle = sum(max(outcomes.values()) for outcomes in scene_pairs.values())
assert pairs[5] == [3, 5] and pair_successes[5] == 576
assert pairs[best_index] == [2, 5] and pair_successes[best_index] == 581
assert oracle == 820
values = np.asarray([pair_successes[5], pair_successes[best_index], oracle]) / 10
fig, ax = plt.subplots(figsize=(8.5, 3))
bars = ax.barh(["Preset pair (3,5)", "Best pair (2,5)", "Oracle\n(post-hoc upper bound)"],
               values, color=[GRAY, BLUE, GREEN], height=.54)
ax.bar_label(bars, labels=[f"{value:.1f}\\%" for value in values], padding=6, fontsize=FONT_SIZE)
ax.invert_yaxis()
ax.set(xlim=(0, 103), xticks=[0, 25, 50, 75, 100], xlabel="Success rate (%)")
decorate(ax)
fig.tight_layout()
save(fig, "story_oracle")
audit["oracle"] = {
    "split": "validation", "scenes": len(scene_pairs), "pairs_per_scene": 10,
    "fixed_35_successes": pair_successes[5],
    "best_fixed_pair": pairs[best_index], "best_fixed_successes": pair_successes[best_index],
    "scene_wise_oracle_successes": oracle,
    "oracle_definition": "At least one of the ten candidate pairs succeeded in a scene",
    "oracle_is_post_hoc_upper_bound_not_learned_selector_performance": True,
    "pair_successes": pair_successes,
}

# Explain the observed test improvement as rescued scenes minus lost scenes.
test_pairs = {(row["case_id"], int(row["pair_index"])): row
              for row in pair_rows if row["split"] == "test"}
selected = [row for row in rows("selector_choices.csv")
            if row["split"] == "test" and row["family"] == "baseline" and row["seed"] == "11"]
assert len(selected) == len({row["case_id"] for row in selected}) == 1000
counts = {"both_success": 0, "learned_only": 0, "fixed_only": 0, "both_failure": 0}
for row in selected:
    learned = int(row["success"])
    fixed = int(test_pairs[(row["case_id"], 5)]["success"])
    assert learned == int(test_pairs[(row["case_id"], int(row["selected_pair_index"]))]["success"])
    key = ("both_success" if learned and fixed else "learned_only" if learned
           else "fixed_only" if fixed else "both_failure")
    counts[key] += 1
assert counts == {"both_success": 514, "learned_only": 95, "fixed_only": 73, "both_failure": 318}
fig, ax = plt.subplots(figsize=(8, 2.5))
bars = ax.barh(["Learned selection\nonly successes", "Preset pair (3,5)\nonly successes"],
              [counts["learned_only"], counts["fixed_only"]], color=[GREEN, GRAY], height=.5)
ax.bar_label(bars, labels=["95 episodes", "73 episodes"], padding=7, fontsize=FONT_SIZE)
ax.invert_yaxis()
ax.set(xlim=(0, 123), xticks=[0, 25, 50, 75, 100], xlabel="Number of test episodes")
decorate(ax)
fig.tight_layout()
save(fig, "story_selection_effect")
audit["selection_effect"] = {
    "split": "test", "scenes": 1000, "model_family": "baseline", "model_seed": 11,
    "reference_pair_index": 5, "reference_pair": [3, 5], **counts,
    "net_success_gain": counts["learned_only"] - counts["fixed_only"],
    "difference_percentage_points": (counts["learned_only"] - counts["fixed_only"]) / 10,
    "interpretation": "Paired descriptive result; does not establish a causal mechanism or significance",
}

# Two explicitly selected examples show that the successful pair can depend
# on the scene. These rows are not an aggregate performance statistic.
example_specs = [
    ("Episode A", "seed2026120009_0017", [2, 5], 65.72),
    ("Episode B", "seed2026120069_0014", [5, 6], 23.42),
]
selected_index = {row["case_id"]: row for row in selected}
example_audit = []
fig, ax = plt.subplots(figsize=(11, 4.4))
for row_index, (label, case_id, expected_pair, expected_time) in enumerate(example_specs):
    outcomes = [test_pairs[(case_id, index)] for index in range(10)]
    successes = [index for index, row in enumerate(outcomes) if int(row["success"])]
    choice = selected_index[case_id]
    chosen_index = int(choice["selected_pair_index"])
    assert len(successes) == 1 and successes[0] == chosen_index
    assert pairs[chosen_index] == expected_pair
    assert round(float(choice["completion_time_seconds"]), 2) == expected_time
    for pair_index, outcome in enumerate(outcomes):
        success = bool(int(outcome["success"]))
        ax.add_patch(Rectangle((pair_index - .39, row_index - .32), .78, .64,
                               facecolor=GREEN if success else "#eef0f2",
                               edgecolor="none"))
        if success:
            ax.text(pair_index, row_index, "Success",
                    ha="center", va="center", fontsize=FONT_SIZE, color="white")
        if pair_index == chosen_index:
            ax.add_patch(Rectangle((pair_index - .44, row_index - .37), .88, .74,
                                   facecolor="none", edgecolor=BLUE, lw=3))
    example_audit.append({
        "display_label": label, "case_id": case_id,
        "successful_pair_indices": successes, "learned_pair_index": chosen_index,
        "learned_pair": pairs[chosen_index], "completion_seconds": float(choice["completion_time_seconds"]),
        "all_pair_successes": [int(row["success"]) for row in outcomes],
    })
ax.set(xlim=(-.6, 9.6), ylim=(1.58, -.55))
ax.set_xticks(range(10), [f"({a},{b})" for a, b in pairs], fontsize=FONT_SIZE)
ax.set_yticks([0, 1], [row[0] for row in example_specs], fontsize=FONT_SIZE)
ax.tick_params(axis="both", length=0, pad=9)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.legend(handles=[
    Patch(facecolor=GREEN, edgecolor="none", label="Success"),
    Patch(facecolor="#eef0f2", edgecolor="none", label="Failure"),
    Patch(facecolor="none", edgecolor=BLUE, linewidth=3, label="Learned selection"),
], loc="lower center", bbox_to_anchor=(.55, .005), ncol=3,
          fontsize=FONT_SIZE, frameon=False, columnspacing=2)
fig.subplots_adjust(left=.10, right=.99, top=.82, bottom=.25)
save(fig, "story_selection_examples")
audit["selection_examples"] = {
    "split": "test", "model_family": "baseline", "model_seed": 11,
    "examples": example_audit,
    "selection_note": "Two selected successes illustrate scene-dependent pair choice; not representative sample statistics",
}

audit["figure_files"] = [
    "story_planar_direct", "story_time_cost", "story_oracle", "story_selection_effect",
    "story_selection_examples",
]
(DATA / "story_figure_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(audit, ensure_ascii=False, indent=2))

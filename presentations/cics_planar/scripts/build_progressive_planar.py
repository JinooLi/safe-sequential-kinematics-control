"""Render a fixed-view planar sequence from the archived case-30 states.

The first three images accumulate problem inputs with the same initial robot.
The last three images display saved Sequential transition/completion samples.
All six PDF media boxes and axes transforms are identical for page-flip continuity.
"""

from pathlib import Path
import hashlib
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/masters-defense-progressive-planar-mpl")
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "evidence", ROOT / "Fig"
FONT_SIZE = 22  # One source size for all visible labels; scaled with the vector figure.
plt.rcParams.update({
    "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE, "figure.titlesize": FONT_SIZE,
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{kotex}\usepackage[english]{babel}\usepackage[T1]{fontenc}\usepackage{lmodern}\renewcommand{\familydefault}{\sfdefault}",
    "font.family": "sans-serif", "font.size": FONT_SIZE,
    "axes.unicode_minus": False,
    "axes.labelcolor": "#243447", "text.color": "#243447",
})


def fk(q):
    angles = np.cumsum(q)
    return np.vstack(([0, 0], np.cumsum(
        np.c_[np.cos(angles), np.sin(angles)], axis=0)))


report = json.loads((DATA / "case030_report.json").read_text())
case = report["original_case"]
q_star = np.asarray(report["lyapunov_reconstruction"]["selected_ik_q"])
goals = np.asarray(report["lyapunov_reconstruction"]["targets"])
ik_points = fk(q_star)
target = np.asarray(case["target"])
target_yaw = float(case["target_yaw"])
snapshots = report["snapshots"]

with np.load(DATA / "case_030_direct.npz", allow_pickle=False) as direct, \
        np.load(DATA / "case_030_sequential.npz", allow_pickle=False) as sequential:
    q0 = sequential["q"][0].copy()
    np.testing.assert_array_equal(q0, direct["q"][0])
    saved_q, saved_times = [], []
    for snapshot in snapshots:
        index = snapshot["sample_index"]
        q = sequential["q"][index].copy()
        time = float(sequential["t"][index])
        np.testing.assert_allclose(q, snapshot["q"], atol=1e-12)
        np.testing.assert_allclose(time, snapshot["time"], atol=1e-12)
        np.testing.assert_allclose(fk(q)[snapshot["completed_link"]],
                                   snapshot["endpoint"], atol=1e-12)
        saved_q.append(q)
        saved_times.append(time)
np.testing.assert_allclose(saved_times, [2.128, 6.874, 18.226], atol=1e-9)
np.testing.assert_allclose(ik_points[1:], goals, atol=1e-12)
np.testing.assert_allclose(ik_points[-1], target, atol=1e-12)
yaw_residual = float(abs(np.arctan2(np.sin(np.sum(q_star) - target_yaw),
                                  np.cos(np.sum(q_star) - target_yaw))))
assert yaw_residual < 1e-12

COLORS = ["#0072B2", "#E69F00", "#009E73"]
MARGIN = .08
LINK_RADIUS = .05
HEADING_END = target + .27 * np.asarray([np.cos(target_yaw), np.sin(target_yaw)])

# A common extent includes every physical link, enlarged target, and obstacle halo.
cloud = [fk(q) for q in [q0, q_star, *saved_q]]
cloud += [HEADING_END.reshape(1, 2), goals - .15, goals + .15]
for obstacle in case["obstacles"]:
    center = np.asarray(obstacle["position"])
    radius = obstacle["radius"] + MARGIN
    cloud.append(np.asarray([center - radius, center + radius]))
cloud = np.concatenate(cloud)
lo, hi = cloud.min(axis=0) - .16, cloud.max(axis=0) + .16
span = hi - lo
# Physical PDF size follows the geometry. No variable tight-bbox cropping.
figure_height = 4.8
figure_width = figure_height * float(span[0] / span[1])


def draw_robot(ax, q):
    points = fk(q)
    for index, color in enumerate(COLORS):
        a, b = points[index:index + 2]
        delta = b - a
        normal = LINK_RADIUS * np.asarray([-delta[1], delta[0]]) / np.linalg.norm(delta)
        ax.add_patch(Polygon([a + normal, b + normal, b - normal, a - normal],
                             facecolor=color, edgecolor="none", zorder=4))
        for endpoint in (a, b):
            ax.add_patch(Circle(endpoint, LINK_RADIUS, facecolor=color,
                                edgecolor="none", zorder=4))
    ax.plot(*points.T, "o", ms=5.2, mfc="white", mec="#444444", mew=.85, zorder=5)
    ax.plot(0, 0, "s", ms=10, color="#333333", zorder=6)
    return points


def draw_scene(ax, show_ik, active):
    for obstacle in case["obstacles"]:
        ax.add_patch(Circle(obstacle["position"], obstacle["radius"] + MARGIN,
                            fill=False, edgecolor="#969696", ls=":", lw=1.5, zorder=1))
        ax.add_patch(Circle(obstacle["position"], obstacle["radius"],
                            facecolor="#d0d0d0", edgecolor="#666666", lw=1.5, zorder=2))
    if show_ik:
        ax.plot(*ik_points.T, color="#858d96", ls=(0, (4, 3)), lw=2.3, zorder=3)
    for index in ([0, 1, 2] if show_ik else [2]):
        color = COLORS[index]
        ax.plot(*goals[index], marker="*", ms=(28 if active == 3 else 24) if index == 2 else (21 if active == index + 1 else 12),
                color=color, markeredgecolor="white", markeredgewidth=.5, zorder=8)
    ax.annotate("", xy=HEADING_END, xytext=target,
                arrowprops={"arrowstyle": "->", "color": COLORS[2],
                            "lw": 1.7, "mutation_scale": 12}, zorder=7)


stages = [
    ("01_initial", q0, False, False, None, None),
    ("02_scene", q0, True, False, None, None),
    ("03_ik", q0, True, True, None, None),
    ("04_link1", saved_q[0], True, True, 1, snapshots[0]),
    ("05_link2", saved_q[1], True, True, 2, snapshots[1]),
    ("06_ee", saved_q[2], True, True, 3, snapshots[2]),
]
stage_audit = []
for name, q, show_scene, show_ik, active, snapshot in stages:
    fig = plt.figure(figsize=(figure_width, figure_height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal")
    ax.axis("off")
    if show_scene:
        draw_scene(ax, show_ik, active)
    points = draw_robot(ax, q)
    # The controlled endpoint is outlined in its link color. The active target
    # is only an enlarged star; no circle suggests a tolerance boundary.
    if active is not None:
        ax.plot(*points[active], "o", ms=9, mfc="white", mec=COLORS[active - 1],
                mew=2, zorder=9)
    stem = f"progressive_planar_{name}"
    output = FIG / f"{stem}.pdf"
    fig.savefig(output, bbox_inches=None, pad_inches=0)
    subprocess.run(["pdftoppm", "-r", "150", "-singlefile", "-png", str(output),
                    str(FIG / stem)], check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE)
    stage_audit.append({
        "name": name, "robot_q": q.tolist(), "robot_points": points.tolist(),
        "obstacles_visible": show_scene, "ee_target_pose_visible": show_scene,
        "selected_ik_and_intermediate_targets_visible": show_ik,
        "active_target": active,
        "sample_index": None if snapshot is None else snapshot["sample_index"],
        "time_s": 0.0 if snapshot is None else snapshot["time"],
        "selection": "Identical initial state; accumulating information" if snapshot is None else snapshot["selection"],
        "figure_files": [f"Fig/{stem}.pdf", f"Fig/{stem}.png"],
        "axes_position": ax.get_position().bounds,
    })
    plt.close(fig)

sources = ["case030_report.json", "case_030_direct.npz", "case_030_sequential.npz"]
audit = {
    "source_sha256": {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in sources},
    "case_id": case["case_id"], "stages": stage_audit,
    "selected_ik_q": q_star.tolist(), "selected_ik_points": ik_points.tolist(),
    "selected_ik_ee_yaw_residual_rad": yaw_residual,
    "max_selected_ik_target_fk_residual_m": float(np.max(np.abs(ik_points[1:] - goals))),
    "common_axis_bounds": {"lower": lo.tolist(), "upper": hi.tolist()},
    "common_pdf_size_inches": [figure_width, figure_height],
    "identical_geometry_transform": len({tuple(stage["axes_position"]) for stage in stage_audit}) == 1,
    "all_geometry_inside_axes": bool(np.all(cloud > lo) and np.all(cloud < hi)),
    "first_three_states_are_identical": bool(np.array_equal(stages[0][1], stages[1][1]) and np.array_equal(stages[1][1], stages[2][1])),
    "active_target_emphasis": "Active target is an enlarged saturated link-color star; no target circle or disk; small colored outline marks actual controlled endpoint",
    "goal_configuration_is_not_path": True,
    "ee_goal_marker_size_points": {"inactive": 24, "active": 28},
    "intermediate_goal_marker_size_points": {"inactive": 12, "active": 21},
    "last_state_is_not_forced_to_selected_ik": True,
    "obstacle_halo": "Nominal 0.08 m geometric margin, not an exact RDF contour",
    "link_geometry_radius_m": LINK_RADIUS,
    "font_rendering": "pdfLaTeX + ko.TeX + Latin Modern; PNG rendered from PDF; no in-figure text",
    "numeric_axes_or_time_labels": False,
    "new_simulation_or_training": False,
}
(DATA / "progressive_planar_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"stage_count": len(stages), "common_pdf_size_inches": audit["common_pdf_size_inches"],
                  "identical_geometry_transform": audit["identical_geometry_transform"],
                  "all_geometry_inside_axes": audit["all_geometry_inside_axes"]}, indent=2))

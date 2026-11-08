"""Show planar problem setup as accumulated information, not robot motion."""

from pathlib import Path
import hashlib
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/masters-defense-planar-setup-mpl")
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "evidence", ROOT / "Fig"
plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{kotex}\usepackage[english]{babel}\usepackage[T1]{fontenc}\usepackage{lmodern}\renewcommand{\familydefault}{\sfdefault}",
    "font.family": "sans-serif", "font.size": 22,
    "axes.unicode_minus": False,
    "axes.labelcolor": "#243447", "text.color": "#243447",
})


def fk(q):
    angles = np.cumsum(q)
    return np.vstack(([0, 0], np.cumsum(
        np.c_[np.cos(angles), np.sin(angles)], axis=0)))


report = json.loads((DATA / "case030_report.json").read_text())
case = report["original_case"]
reconstruction = report["lyapunov_reconstruction"]
q_star = np.asarray(reconstruction["selected_ik_q"])
ik_points = fk(q_star)
goals = np.asarray(reconstruction["targets"])
target = np.asarray(case["target"])
target_yaw = float(case["target_yaw"])
with np.load(DATA / "case_030_direct.npz", allow_pickle=False) as direct, \
        np.load(DATA / "case_030_sequential.npz", allow_pickle=False) as sequential:
    q0 = direct["q"][0].copy()
    np.testing.assert_array_equal(q0, sequential["q"][0])
initial_points = fk(q0)
goal_residual = float(np.max(np.abs(ik_points[1:] - goals)))
ee_position_residual = float(np.linalg.norm(ik_points[-1] - target))
yaw_difference = float(np.sum(q_star) - target_yaw)
ee_yaw_residual = float(abs(np.arctan2(np.sin(yaw_difference), np.cos(yaw_difference))))
np.testing.assert_allclose(ik_points[1:], goals, atol=1e-12)
assert ee_position_residual < 1e-12 and ee_yaw_residual < 1e-12

link_colors = ["#0072B2", "#E69F00", "#009E73"]
margin = .08
target_heading_end = target + .24 * np.asarray([np.cos(target_yaw), np.sin(target_yaw)])
cloud = [initial_points, ik_points, target_heading_end.reshape(1, 2)]
for obstacle in case["obstacles"]:
    center = np.asarray(obstacle["position"])
    radius = obstacle["radius"] + margin
    cloud.append(np.asarray([center - radius, center + radius]))
cloud = np.concatenate(cloud)
lo, hi = cloud.min(axis=0) - .20, cloud.max(axis=0) + .20


def draw_initial_robot(ax):
    for index, color in enumerate(link_colors):
        a, b = initial_points[index:index + 2]
        delta = b - a
        normal = .05 * np.asarray([-delta[1], delta[0]]) / np.linalg.norm(delta)
        ax.add_patch(Polygon([a + normal, b + normal, b - normal, a - normal],
                             color=color, zorder=4))
        for endpoint in (a, b):
            ax.add_patch(Circle(endpoint, .05, color=color, zorder=4))
    ax.plot(*initial_points.T, "o", ms=4.7, mfc="white", mec="#444444", mew=.7, zorder=5)
    ax.plot(0, 0, "s", ms=8, color="#333333", zorder=6)


fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.8))
titles = ["Initial configuration", "EE goal and obstacles", "Selected IK configuration"]
panel_audit = []
for panel, (ax, title) in enumerate(zip(axes, titles)):
    if panel >= 1:
        for obstacle in case["obstacles"]:
            ax.add_patch(Circle(
                obstacle["position"], obstacle["radius"] + margin,
                fill=False, edgecolor="#9b9b9b", ls=":", lw=1.5, zorder=1))
            ax.add_patch(Circle(
                obstacle["position"], obstacle["radius"], facecolor="#d0d0d0",
                edgecolor="#666666", lw=1.5, zorder=2))
        ax.plot(*target, marker="*", ms=14, color=link_colors[2],
                markeredgecolor="white", markeredgewidth=.5, zorder=7)
        ax.annotate("", xy=target_heading_end, xytext=target,
                    arrowprops={"arrowstyle": "->", "color": "#555555",
                                "lw": 1.5, "mutation_scale": 10})
    if panel == 2:
        ax.plot(*ik_points.T, color="#8a8f95", ls="--", lw=2.3, zorder=3)
        for index in (0, 1):
            ax.plot(*goals[index], marker="*", ms=14, color=link_colors[index],
                    markeredgecolor="white", markeredgewidth=.5, zorder=7)
    draw_initial_robot(ax)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal")
    ax.axis("off")
    ax.set_title(title, fontsize=23, pad=10)
    panel_audit.append({
        "title": title, "robot_q": q0.tolist(),
        "obstacles_visible": panel >= 1, "ee_target_pose_visible": panel >= 1,
        "ik_configuration_and_intermediate_targets_visible": panel == 2,
    })

fig.subplots_adjust(left=.005, right=.995, top=.86, bottom=.15, wspace=.17)
# These arrows connect information-setup panels, not robot states in time.
for x0, x1 in [(.321, .344), (.657, .680)]:
    fig.add_artist(FancyArrowPatch(
        (x0, .54), (x1, .54), transform=fig.transFigure,
        arrowstyle="->", mutation_scale=17, lw=1.6, color="#205493"))
fig.legend(handles=[
    Line2D([], [], color="#444444", lw=3.3, label="Initial robot"),
    Patch(facecolor="#d0d0d0", edgecolor="#666666", label="Obstacle"),
    Line2D([], [], color="#8a8f95", ls="--", lw=2.3, label="Selected IK"),
    Line2D([], [], color=link_colors[2], marker="*", ls="none", ms=11, label="Goal"),
], loc="lower center", ncol=4, fontsize=20, frameon=False,
          handlelength=1.5, columnspacing=1.5)
output = FIG / "planar_setup.pdf"
fig.savefig(output, bbox_inches="tight", pad_inches=.08)
subprocess.run(["pdftoppm", "-r", "150", "-singlefile", "-png",
                str(output), str(FIG / "planar_setup")], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
plt.close(fig)

sources = ["case030_report.json", "case_030_direct.npz", "case_030_sequential.npz"]
audit = {
    "source_sha256": {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in sources},
    "case_id": case["case_id"], "panels": panel_audit,
    "initial_q_identical_in_all_panels": True,
    "selected_ik_q": q_star.tolist(),
    "max_ik_link_target_fk_residual_m": goal_residual,
    "ik_ee_position_residual_m": ee_position_residual,
    "ik_ee_yaw_residual_rad": ee_yaw_residual,
    "ik_source": "Archived reconstructed branch; no new IK solve or rollout",
    "interpretation": "Information accumulates from left to right; dashed IK is a goal configuration, not an execution path",
    "axis_bounds": {"lower": lo.tolist(), "upper": hi.tolist()},
    "common_scale_and_axes": True,
    "all_geometry_inside_axes": bool(np.all(cloud > lo) and np.all(cloud < hi)),
    "obstacle_halo": "Nominal 0.08 m geometric margin, not an exact RDF contour",
    "font_rendering": "pdfLaTeX + ko.TeX + Latin Modern; PNG rendered from PDF",
    "numeric_axes_or_time_labels": False,
    "new_simulation_or_training": False,
    "figure_files": ["Fig/planar_setup.pdf", "Fig/planar_setup.png"],
}
(DATA / "planar_setup_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(audit, ensure_ascii=False, indent=2))

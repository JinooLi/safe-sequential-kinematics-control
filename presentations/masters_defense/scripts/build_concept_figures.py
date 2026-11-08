"""Render concept-first planar scenes and CLF surfaces from frozen evidence.

Only concept_* assets are generated. Recorded poses are reused without
simulation; identifying episode/time values remain in the audit, not figures.
"""

from pathlib import Path
import hashlib
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/masters-defense-concept-mpl")
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "evidence", ROOT / "Fig"
FONT_SIZE = 24  # One source size for all visible labels; scaled with the vector figure.
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


def fk(q):
    angles = np.cumsum(q)
    return np.vstack(([0, 0], np.cumsum(
        np.c_[np.cos(angles), np.sin(angles)], axis=0)))


names = ["case030_report.json", "case_030_direct.npz", "case_030_sequential.npz", "panda_example.json"]
audit = {
    "font_rendering": "pdfLaTeX + ko.TeX + Latin Modern; PNG rendered from PDF",
    "no_new_simulation": True,
    "no_new_training": True,
    "source_sha256": {
        name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in names
    },
    "screen_style": "No episode labels, timing values, numeric axis ticks or scale bars",
}

report = json.loads((DATA / "case030_report.json").read_text())
case = report["original_case"]
assert case["case_id"] == 30
assert case["sequential"]["success"] and not case["direct"]["success"]
goals = np.asarray(report["lyapunov_reconstruction"]["targets"])
np.testing.assert_allclose(
    fk(report["lyapunov_reconstruction"]["selected_ik_q"])[1:], goals, atol=1e-12)
with np.load(DATA / "case_030_direct.npz", allow_pickle=False) as direct, \
        np.load(DATA / "case_030_sequential.npz", allow_pickle=False) as sequential:
    np.testing.assert_array_equal(direct["t"], sequential["t"])
    np.testing.assert_array_equal(direct["q"][0], sequential["q"][0])
    snapshots = report["snapshots"]
    indices = [snapshot["sample_index"] for snapshot in snapshots]
    times = [float(sequential["t"][index]) for index in indices]
    np.testing.assert_allclose(times, [2.128, 6.874, 18.226], atol=1e-9)
    stage_poses = [sequential["q"][index].copy() for index in indices]
    for q, snapshot in zip(stage_poses, snapshots):
        np.testing.assert_allclose(q, snapshot["q"], atol=1e-12)
    initial_pose = direct["q"][0].copy()
    direct_pose = direct["q"][indices[-1]].copy()
    direct_position_error = float(np.linalg.norm(direct["final_xy"][indices[-1]] - case["target"]))

margin = .08
linkcolors = ["#0072B2", "#E69F00", "#009E73"]
cloud = [np.zeros((1, 2)), goals, *[fk(q) for q in [initial_pose, direct_pose, *stage_poses]]]
for obstacle in case["obstacles"]:
    center = np.asarray(obstacle["position"])
    radius = obstacle["radius"] + margin
    cloud.append(np.asarray([center - radius, center + radius]))
cloud = np.concatenate(cloud)
lo, hi = cloud.min(axis=0) - .22, cloud.max(axis=0) + .22


def scene(ax, q, active, show_intermediate):
    points = fk(q)
    if show_intermediate:
        ik = np.vstack(([0, 0], goals))
        ax.plot(*ik.T, color="#8a8f95", ls="--", lw=2, zorder=1)
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
        ax.add_patch(Polygon([a + normal, b + normal, b - normal, a - normal],
                             color=color, zorder=3))
        for endpoint in (a, b):
            ax.add_patch(Circle(endpoint, .05, color=color, zorder=3))
        if show_intermediate or j == 2:
            ax.plot(*goals[j], marker="*", ms=13, color=color,
                    markeredgecolor="white", markeredgewidth=.5, zorder=6)
    ax.plot(*points.T, "o", ms=4, mfc="white", mec="#444444", mew=.7, zorder=5)
    ax.plot(0, 0, "s", ms=8, color="#333333", zorder=6)
    if active is not None:
        ax.plot(*points[active], "o", ms=16, mfc="none", mec="#333333", mew=1.7, zorder=7)
    headings = [(goals[-1], case["target_yaw"], "#555555")]
    if active == 3:
        headings.append((points[-1], sum(q), linkcolors[-1]))
    for point, yaw, color in headings:
        ax.annotate("", xy=point + .25 * np.asarray([np.cos(yaw), np.sin(yaw)]),
                    xytext=point, arrowprops={"arrowstyle": "->", "color": color,
                                             "lw": 1.4, "mutation_scale": 10})
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal")
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
for ax, q, active, title in zip(
        axes, [initial_pose, direct_pose], [None, 3],
        ["Initial configuration", "Direct: goal not reached"]):
    scene(ax, q, active, show_intermediate=False)
    ax.set_title(title, fontsize=FONT_SIZE, pad=9)
fig.subplots_adjust(left=.01, right=.99, bottom=.17, top=.87, wspace=.12)
fig.legend(handles=[
    Patch(facecolor="#d0d0d0", edgecolor="#666666", label="Obstacle"),
    Line2D([], [], color=linkcolors[-1], marker="*", ls="none", ms=11, label="EE goal"),
], loc="lower center", ncol=2, fontsize=FONT_SIZE, frameon=False, columnspacing=2)
save(fig, "concept_planar_direct")

fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.2))
for ax, q, active, title in zip(axes, stage_poses, [1, 2, 3], ["link 1", "link 2", "EE"]):
    scene(ax, q, active, show_intermediate=True)
    ax.set_title(title, fontsize=FONT_SIZE, pad=9)
fig.subplots_adjust(left=.005, right=.995, bottom=.20, top=.88, wspace=.08)
fig.legend(handles=[
    Patch(facecolor="#d0d0d0", edgecolor="#666666", label="Obstacle"),
    Line2D([], [], color="#8a8f95", ls="--", label="IK configuration"),
    Line2D([], [], color="#444444", marker="*", ls="none", ms=11, label="Link targets"),
    Line2D([], [], color="#333333", marker="o", mfc="none", ls="none", ms=11, label="Active control point"),
], loc="lower center", ncol=4, fontsize=FONT_SIZE, frameon=False,
           handlelength=1.5, columnspacing=1.1)
save(fig, "concept_planar_stages")
audit["recorded_geometry"] = {
    "case_id": case["case_id"], "snapshots_seconds": times,
    "direct_snapshot_seconds": times[-1],
    "direct_position_error_m": direct_position_error,
    "direct_final_failure_reason": case["direct"]["failure_reason"],
    "selected_ik_targets_verified": True,
    "all_geometry_inside_common_axes": bool(np.all(cloud > lo) and np.all(cloud < hi)),
    "obstacle_halo": "Nominal 0.08 m geometric margin; not exact RDF contours",
}

# Surface families reproduce the existing planar CLF science. Positive
# weights set the illustrative shape only; no numerical height is labelled.
w_v, w_psi = 1.0, .5
for index, suffix, color in [(1, "1", linkcolors[0]), (2, "2", linkcolors[1]), (3, "ee", linkcolors[2])]:
    fig = plt.figure(figsize=(3.9, 3.4))
    ax = fig.add_subplot(111, projection="3d")
    if index < 3:
        error = np.linspace(-1, 1, 51)
        x, y = np.meshgrid(error, error)
        z = w_v * (x * x + y * y)
        xlabel = rf"$e_{{p,{index},x}}$"
        ylabel = rf"$e_{{p,{index},y}}$"
    else:
        x, y = np.meshgrid(np.linspace(0, 1, 51), np.linspace(-np.pi, np.pi, 65))
        z = w_v * (x * x + w_psi * (1 - np.cos(y)))
        xlabel, ylabel = r"$\|e_{p,\mathrm{ee}}\|$", r"$e_\psi$"
    assert np.all(z >= 0) and np.isclose(z.min(), 0)
    cmap = LinearSegmentedColormap.from_list(f"clf_{suffix}", ["#f7fafb", color])
    ax.plot_surface(x, y, z, cmap=cmap, alpha=.92, linewidth=.25,
                    edgecolor="white", rstride=3, cstride=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE, labelpad=-2)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE, labelpad=-2)
    ax.set_box_aspect((1, 1, .78))
    ax.view_init(elev=27, azim=-55)
    fig.subplots_adjust(left=.01, right=.99, top=.98, bottom=.10)
    save(fig, f"concept_clf_{suffix}")

audit["clf_shapes"] = {
    "symbolic_formula": "V_i=w_V||e_p,i||^2; Vee=w_V(||e_p,ee||^2+w_psi(1-cos(e_psi))), w_V,w_psi>0",
    "coordinates": "Intermediate tasks: planar position-error coordinates; EE: position-error norm and yaw error",
    "illustrative_weights": {"w_V": w_v, "w_psi": w_psi},
    "archived_trajectory_weights": {"w_V": 10, "w_psi": .5},
    "numeric_ticks": False,
    "titles": False,
    "task_colors": {"1": linkcolors[0], "2": linkcolors[1], "ee": linkcolors[2]},
    "constraint_and_formula_labels": "Added in native TeX by the calling slide, outside these surface assets",
}

# Static archived Panda geometry with a target pose triad. No numeric axes,
# scale bars or case identifiers are needed for the concept introduction.
panda = json.loads((DATA / "panda_example.json").read_text())
positions = np.asarray(panda["initial_frame_positions"])
target = np.asarray(panda["target_position"])
rotation = np.asarray(panda["target_rotation"])
obstacles = np.asarray(panda["obstacles"])
assert positions.shape == (9, 3) and obstacles.shape == (10, 4)
np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
assert np.isclose(np.linalg.det(rotation), 1)
fig = plt.figure(figsize=(6.0, 4.8))
ax = fig.add_subplot(projection="3d")
phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi, 32), np.linspace(0, np.pi, 20))
for x, y, z, radius in obstacles:
    ax.plot_surface(
        x + radius * np.cos(phi) * np.sin(theta),
        y + radius * np.sin(phi) * np.sin(theta),
        z + radius * np.cos(theta),
        color="#a84439", alpha=.17, linewidth=0, shade=False)
ax.plot(*positions.T, "o-", color="#205493", lw=6, ms=6)
ax.scatter(*positions[0], color="#333333", marker="s", s=75, depthshade=False)
ax.scatter(*target, color="#009E73", marker="*", s=230, depthshade=False)
for index, color in enumerate(["#a84439", "#009E73", "#205493"]):
    ax.quiver(*target, *(.16 * rotation[:, index]), color=color,
              linewidth=2.4, arrow_length_ratio=.25)
ax.set(xlim=(-.95, .95), ylim=(-.95, .95), zlim=(0, 1.30))
ax.set_box_aspect((1, 1, .8), zoom=1.5)
ax.view_init(elev=24, azim=35)
ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, bottom=.10, top=1)
fig.legend(handles=[
    Patch(facecolor="#a84439", alpha=.25, edgecolor="none", label="Obstacle"),
    Line2D([], [], color="#205493", lw=4, marker="o", ms=6, label="Initial configuration"),
    Line2D([], [], color="#009E73", marker="*", ls="none", ms=12, label="Goal pose"),
], loc="lower center", ncol=3, fontsize=FONT_SIZE, frameon=False,
          handlelength=1.4, columnspacing=1.1)
save(fig, "concept_panda_scene")
audit["panda_scene"] = {
    "case_id": panda["case_id"],
    "obstacle_count": len(obstacles),
    "saved_frame_names": panda["frame_names"],
    "source_dataset_sha256": panda["dataset_sha256"],
    "source_model_sha256": panda["model_sha256"],
    "target_rotation_orthogonality_verified": True,
    "rendering": "Saved initial frame connections and spherical obstacles; triad denotes target EE pose, not a goal robot configuration",
    "robot_lines_are_not_collision_geometry": True,
    "numeric_axes_ticks_or_case_labels": False,
}
audit["figure_files"] = [
    "concept_planar_direct", "concept_planar_stages",
    "concept_clf_1", "concept_clf_2", "concept_clf_ee",
    "concept_panda_scene",
]
(DATA / "concept_figure_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(audit, ensure_ascii=False, indent=2))

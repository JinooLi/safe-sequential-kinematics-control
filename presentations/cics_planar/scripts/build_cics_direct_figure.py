"""저장된 episode 30으로 CICS Direct 실패 그림의 한국어 표기 생성.

masters_defense/scripts/build_concept_figures.py의 기존 plot을 재사용.
데이터, 로봇 자세, 장애물, 축 범위와 색상은 기존 그림과 동일.
"""

from pathlib import Path
import argparse
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cics-planar-direct-mpl")
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "evidence", ROOT / "Fig"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--only", action="append", metavar="FIGURE",
                    help="Regenerate only this figure stem; repeat for multiple figures.")
requested_figures = set(parser.parse_args().only or [])
generated_figures = []
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
    if requested_figures and name not in requested_figures:
        plt.close(fig)
        return
    generated_figures.append(name)
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
        ["초기 자세", "Direct: 목표 미도달"]):
    scene(ax, q, active, show_intermediate=False)
    ax.set_title(title, fontsize=FONT_SIZE, pad=9)
fig.subplots_adjust(left=.01, right=.99, bottom=.17, top=.87, wspace=.12)
fig.legend(handles=[
    Patch(facecolor="#d0d0d0", edgecolor="#666666", label="장애물"),
    Line2D([], [], color=linkcolors[-1], marker="*", ls="none", ms=11, label="목표"),
], loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=2,
   fontsize=FONT_SIZE, frameon=False, columnspacing=2)
save(fig, "concept_planar_direct")

print("concept_planar_direct.pdf/png: Korean labels; original geometry and input data preserved")

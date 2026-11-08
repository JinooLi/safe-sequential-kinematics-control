"""Render slide 3 from archived case 180; no simulation or QP re-solve."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/masters-defense-boundary-mpl")
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
    "axes.unicode_minus": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelcolor": "#243447", "text.color": "#243447"})

case = next(c for c in json.loads((DATA / "planar_cases.json").read_text()) if c["case_id"] == 180)
assert not case["direct"]["success"]
assert case["direct"]["failure_reason"] == "timeout"
assert case["direct"]["qp_infeasible_count"] == 0
with np.load(DATA / "case_180_direct.npz", allow_pickle=False) as raw:
    tr = {k: raw[k].copy() for k in raw.files}
t, q = tr["t"], tr["q"]
angles = np.cumsum(q, axis=1)
points = np.concatenate((np.zeros((len(t), 1, 2)),
    np.cumsum(np.stack((np.cos(angles), np.sin(angles)), axis=-1), axis=1)), axis=1)
np.testing.assert_allclose(points[:, -1], tr["final_xy"], atol=1e-12)
ep = np.linalg.norm(tr["final_xy"] - case["target"], axis=1)
eyaw = np.abs(np.arctan2(np.sin(tr["final_yaw"] - case["target_yaw"]),
                       np.cos(tr["final_yaw"] - case["target_yaw"])))
v = 10 * ep**2 + 5 * (1 - np.cos(eyaw))
np.testing.assert_allclose(v, tr["clf"], atol=1e-10)
np.testing.assert_allclose([ep[-1], eyaw[-1]],
    [case["direct"]["position_error"], case["direct"]["yaw_error"]], atol=1e-12)
h = tr["barrier"].min(axis=1)
speed = np.linalg.norm(tr["u"], axis=1)
tail = t >= 15
assert speed[tail].max() < 1e-6 and h[tail].min() >= 0
assert h[tail].max() < 1e-6 and np.all(ep[tail] > .8)
assert np.all(np.argmin(tr["barrier"][tail], axis=1) == 2)
assert np.all(tr["clf_slack"][tail] > 9.5)

blue, red, green = "#205493", "#A84439", "#28735E"
fig = plt.figure(figsize=(10.5, 4.5))
grid = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=.45, wspace=.34)
scene = fig.add_subplot(grid[:, 0])
p = points[-1]
# A spatial close-up: the distant second obstacle lies below this viewport.
# No geometric halo is labelled as the learned RDF's exact zero contour.
obs = case["obstacles"][0]
center = np.array(obs["position"])
scene.add_patch(Circle(center, obs["radius"], facecolor="#d0d0d0", edgecolor="#666666", lw=1.5))
for j, color in enumerate(["#0072B2", "#E69F00", "#009E73"]):
    a, b = p[j:j+2]
    d = b - a
    normal = .05 * np.array([-d[1], d[0]]) / np.linalg.norm(d)
    scene.add_patch(Polygon([a+normal, b+normal, b-normal, a-normal], color=color, zorder=3))
    for end in [a, b]:
        scene.add_patch(Circle(end, .05, color=color, zorder=3))
scene.plot(*p.T, "o", ms=5, mfc="white", mec="#444444", mew=.8, zorder=4)
scene.plot(0, 0, "s", ms=8, color="#333333", zorder=5)
scene.plot(*tr["final_xy"].T, color="#676767", ls="--", lw=1.4, zorder=2)
goal = np.array(case["target"])
scene.plot(*goal, "*", ms=15, color=green, zorder=6)
scene.text(goal[0]-.16, goal[1]+.17, "Goal", color=green, ha="center", fontsize=FONT_SIZE)
scene.text(p[-1, 0]+.20, p[-1, 1]-.12, "EE", color=blue, fontsize=FONT_SIZE)
scene.text(center[0]-.14, center[1]+.36, "Obstacle", ha="right", fontsize=FONT_SIZE)
for xy, yaw, color in [(goal, case["target_yaw"], green), (p[-1], tr["final_yaw"][-1], blue)]:
    scene.annotate("", xy=xy+.32*np.array([np.cos(yaw), np.sin(yaw)]), xytext=xy,
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.6})
# Highlight the relevant obstacle/link region, using the recorded h value.
d = p[3] - p[2]
near = p[2] + np.clip((center-p[2]) @ d / (d @ d), 0, 1) * d
gap_direction = (near-center) / np.linalg.norm(near-center)
contact = center + gap_direction * (obs["radius"] + np.linalg.norm(near-center) - .05) / 2
scene.annotate("Near boundary", xy=contact, xytext=(1.18, .43), color=red, fontsize=FONT_SIZE,
    ha="center", arrowprops={"arrowstyle": "->", "color": red, "lw": 1.5})
scene.plot([.08, .58], [-.32, -.32], color="#333333", lw=2)
scene.text(.33, -.47, "0.5 m", ha="center", fontsize=FONT_SIZE)
scene.set(xlim=(-.12, 3.25), ylim=(-.55, 1.65), aspect="equal")
scene.axis("off")
scene.set_title("Direct endpoint (close-up)", fontsize=FONT_SIZE, pad=12)

for row, y, color, title, ylabel in [
    (0, h, red, "Safety-boundary approach", r"$h_{\min}$ (m)"),
    (1, ep, blue, "Residual goal error", r"$\|e_p\|$ (m)")]:
    ax = fig.add_subplot(grid[row, 1])
    ax.axvspan(15, 35, color="#dfe7f0", alpha=.55, lw=0)
    ax.plot(t, y, color=color, lw=2.3)
    ax.set_xlim(0, 35)
    ax.set_xticks([0, 15, 35])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE, pad=7)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.grid(axis="y", alpha=.17)
    if row == 0:
        ax.axhline(0, color="#444444", ls=":", lw=1)
        ax.set_ylim(-.015, .29)
        ax.set_yticks([0, .1, .2])
        ax.text(23, .065, r"$h_{\min}\approx 0$", color=red, ha="center", fontsize=FONT_SIZE)
        ax.tick_params(labelbottom=False)
    else:
        ax.set_ylim(0, 1.8)
        ax.set_yticks([0, .8, 1.6])
        ax.text(25, 1.05, "0.82 m", color=blue, ha="center", fontsize=FONT_SIZE)
        ax.set_xlabel("Time (s)", fontsize=FONT_SIZE)
fig.subplots_adjust(left=.01, right=.985, bottom=.14, top=.88)
fig.savefig(FIG / "case180_boundary_stall.pdf", bbox_inches="tight", pad_inches=.08)
plt.close(fig)
subprocess.run(["pdftoppm", "-r", "150", "-singlefile", "-png",
    str(FIG / "case180_boundary_stall.pdf"), str(FIG / "case180_boundary_stall")], check=True)

audit = {
    "case_id": 180, "source": "case_180_direct.npz",
    "source_sha256": hashlib.sha256((DATA / "case_180_direct.npz").read_bytes()).hexdigest(),
    "selection": "Sequential-only successes with no Direct QP infeasibility, obstacle barrier active at termination; smallest maximum speed over 25–35 s among seven matching cases.",
    "failure_reason": case["direct"]["failure_reason"], "qp_infeasible_count": 0,
    "raw_samples": len(t), "time_range_seconds": [float(t[0]), float(t[-1])],
    "snapshot_seconds": float(t[-1]), "tail_window_seconds": [15, 35],
    "tail_max_speed_rad_s": float(speed[tail].max()),
    "tail_min_barrier_m": float(h[tail].min()), "tail_max_barrier_m": float(h[tail].max()),
    "tail_max_position_drift_m": float(np.linalg.norm(tr["final_xy"][tail]-tr["final_xy"][-1], axis=1).max()),
    "active_barrier_column_zero_based": 2, "active_barrier_label": "obs1_link3",
    "final_position_error_m": float(ep[-1]), "final_yaw_error_rad": float(eyaw[-1]),
    "final_clf": float(v[-1]), "final_clf_slack": float(tr["clf_slack"][-1]),
    "max_reconstructed_clf_error": float(abs(v-tr["clf"]).max()),
    "geometry": "Recorded q, 1 m links and 0.05 m physical link radius; spatial close-up omits distant second obstacle. No exact RDF contour drawn.",
    "plotting": "Every stored sample; no smoothing, new simulation, or QP re-solve. Shaded interval 15–35 s.",
    "interpretation": "Observed boundary approach, vanishing command and residual task error; no claim of a universal cause for all failures or a proof of stable equilibrium."
}
(DATA / "boundary_case_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(audit, ensure_ascii=False, indent=2))

"""Publication-size vector figures from unchanged archived case-030 samples."""

import hashlib
import json

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon

import generate_report as source

MM = 1 / 25.4
OUT = source.OUTPUT / "paper"
LINKS = ("#0072B2", "#E69F00", "#009E73")
DIRECT = "#A74484"


def save(figure, name):
    for extension in ("pdf", "svg", "png"):
        figure.savefig(OUT / f"{name}.{extension}", dpi=600, facecolor="white")


def scene(axis, q, targets, case, active, limits, *, target_size=24,
          orientation_linewidth=0.65, orientation_head_size=5,
          ik_linewidth=0.65, clearance_linewidth=0.55,
          endpoint_ring_linewidth=0.6):
    points = np.vstack((np.zeros(2), source.endpoints(q, np.ones(3))))
    goal = np.vstack((np.zeros(2), targets))
    axis.plot(goal[:, 0], goal[:, 1], color="0.6", linestyle=(0, (3, 2)),
              linewidth=ik_linewidth, zorder=1)
    for obstacle in case["obstacles"]:
        axis.add_patch(Circle(obstacle["position"], obstacle["radius"] + 0.08,
                               fill=False, edgecolor="0.55", linestyle=(0, (1.5, 2)),
                               linewidth=clearance_linewidth, zorder=1))
        axis.add_patch(Circle(obstacle["position"], obstacle["radius"],
                               facecolor="0.8", edgecolor="0.35", linewidth=0.6, zorder=2))
    for i, color in enumerate(LINKS):
        start, end = points[i:i + 2]
        delta = end - start
        normal = 0.05 * np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        axis.add_patch(Polygon([start + normal, end + normal, end - normal, start - normal],
                                facecolor=color, edgecolor="none", zorder=3))
        for p in (start, end):
            axis.add_patch(Circle(p, 0.05, facecolor=color, edgecolor="none", zorder=3))
        axis.scatter(*targets[i], marker="*", s=target_size, color=color,
                     edgecolors="white", linewidths=0.25 * np.sqrt(target_size / 24), zorder=6)
    axis.scatter(points[:, 0], points[:, 1], s=6, facecolors="white",
                 edgecolors="0.2", linewidths=0.35, zorder=5)
    axis.scatter(0, 0, marker="s", s=18, color="0.2", zorder=6)
    if active is not None:
        # A fixed screen-size highlight of the actual controlled endpoint.
        # This is not a target-centered, metric goal-tolerance boundary.
        axis.scatter(*points[active], s=43, facecolors="none", edgecolors="0.2",
                     linewidths=endpoint_ring_linewidth, zorder=7,
                     gid="controlled-endpoint-ring", label="Controlled endpoint")
    target_yaw = case["target_yaw"]
    axis.annotate("", xy=targets[-1] + 0.29 * np.array([np.cos(target_yaw), np.sin(target_yaw)]),
                  xytext=targets[-1], arrowprops={"arrowstyle": "->", "lw": orientation_linewidth,
                                                  "color": "0.25", "mutation_scale": orientation_head_size})
    if active == 3:
        yaw = np.sum(q)
        axis.annotate("", xy=points[-1] + 0.29 * np.array([np.cos(yaw), np.sin(yaw)]),
                      xytext=points[-1], arrowprops={"arrowstyle": "->", "lw": orientation_linewidth,
                                                     "color": LINKS[-1], "mutation_scale": orientation_head_size})
    axis.set_xlim(limits[0]); axis.set_ylim(limits[1]); axis.set_aspect("equal")
    axis.set_axis_off()


def decorate_graph(axis, title):
    axis.set_title(title, loc="left", fontsize=9, pad=5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="0.9", linewidth=0.5)
    axis.set_xlabel("Time [s]", labelpad=2)
    axis.tick_params(length=2, width=0.6, pad=2)


def main():
    OUT.mkdir(exist_ok=True)
    try:
        font_manager.findfont("Liberation Sans", fallback_to_default=False)
        family = "Liberation Sans"
    except ValueError:
        family = "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": family, "font.size": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans", "path.simplify": False,
    })
    report = json.loads((source.OUTPUT / "report.json").read_text())
    for path, digest in report["source_sha256"].items():
        assert source.sha256(source.ROOT / path) == digest
    with np.load(source.OUTPUT / "lyapunov.npz", allow_pickle=False) as data:
        t = data["t"]; active = data["active_link"]
        values = data["clf_by_link"]; direct = data["direct_clf"]
        seq_q = data["sequential_q"]; direct_q = data["direct_q"]; targets = data["targets"]
    assert values.shape == (17500, 3)
    case = report["original_case"]
    snapshots = report["snapshots"]
    end_index = snapshots[-1]["sample_index"]
    end_time = snapshots[-1]["time"]
    np.testing.assert_array_equal(seq_q[0], direct_q[0])
    states = [seq_q[0]] + [np.asarray(item["q"]) for item in snapshots] + [direct_q[end_index]]
    times = [float(t[0])] + [item["time"] for item in snapshots] + [end_time]
    stages = [None, 1, 2, 3, 3]
    cloud = [np.zeros((1, 2)), targets]
    cloud += [source.endpoints(q, np.ones(3)) for q in states]
    for obstacle in case["obstacles"]:
        center = np.asarray(obstacle["position"])
        radius = obstacle["radius"] + 0.08
        cloud.extend([(center - radius)[None], (center + radius)[None]])
    cloud = np.concatenate(cloud)
    low, high = cloud.min(axis=0) - 0.15, cloud.max(axis=0) + 0.15
    limits = ((float(low[0]), float(high[0])), (float(low[1]), float(high[1])))

    fig = Figure(figsize=(180 * MM, 112 * MM)); FigureCanvasAgg(fig)
    layout = fig.add_gridspec(3, 1, left=0.075, right=0.99, bottom=0.13, top=0.93,
                             height_ratios=[1.12, 0.18, 1], hspace=0.08)
    upper = layout[0].subgridspec(1, 5, wspace=0.09)
    labels = ["(a) Initial pose", "(b) Sequential: link 1", "(c) Sequential: link 2",
              "(d) Sequential: EE", "(e) Direct: EE"]
    for i, (q, title, stage, time) in enumerate(zip(states, labels, stages, times)):
        axis = fig.add_subplot(upper[i])
        scene(axis, q, targets, case, stage, limits)
        axis.set_title(f"{title}\n$t={time:.3f}$ s", fontsize=8, pad=3)
        if i == 0:
            x, y = limits[0][0] + 0.15, limits[1][1] - 0.35
            axis.plot([x, x + 0.5], [y, y], color="0.2", linewidth=1.2)
            axis.text(x + 0.25, y + 0.10, "0.5 m", ha="center", fontsize=7)
    key = fig.add_subplot(layout[1]); key.axis("off")
    key.legend(handles=[
        Patch(facecolor="0.8", edgecolor="0.35", label="Obstacle"),
        Line2D([], [], color="0.55", linestyle=":", label="Clearance margin"),
        Line2D([], [], color="0.6", linestyle="--", label="Selected IK posture"),
        Line2D([], [], marker="*", linestyle="none", color="0.25", markersize=6, label="Selected IK targets"),
    ], loc="center", ncol=4, frameon=False, handlelength=1.6, columnspacing=1.1)

    lower = layout[2].subgridspec(1, 3, wspace=0.42)
    axes = [fig.add_subplot(lower[i]) for i in range(3)]
    for i, axis in enumerate(axes[:2]):
        axis.plot(t, values[:, i], color=LINKS[i], linestyle=(0, (3, 2)),
                  linewidth=0.8, alpha=0.7)
        axis.plot(t, np.where(active == i + 1, values[:, i], np.nan),
                  color=LINKS[i], linewidth=1.5)
        axis.axvspan(0 if i == 0 else snapshots[0]["time"], snapshots[i]["time"],
                    color=LINKS[i], alpha=0.07, linewidth=0)
        decorate_graph(axis, f"({'fg'[i]}) Sequential: $V_{i + 1}$")
        axis.set_ylabel(f"$V_{i + 1}$", labelpad=1)
        axis.set_ylim(bottom=0)
    axes[0].legend(handles=[
        Line2D([], [], color="0.25", lw=1.5, label="Active"),
        Line2D([], [], color="0.5", lw=0.8, ls="--", label="Inactive"),
    ], loc="upper right", frameon=False, handlelength=1.5, borderpad=0.1, labelspacing=0.25)
    axis = axes[2]
    axis.plot(t, direct, color=DIRECT, ls="-.", lw=1.1, label="Direct")
    axis.plot(t, values[:, 2], color=LINKS[-1], ls=(0, (3, 2)), lw=0.8, alpha=0.7)
    axis.plot(t, np.where(active == 3, values[:, 2], np.nan),
                  color=LINKS[-1], lw=1.5, label="Sequential")
    axis.axvline(end_time, color="0.3", ls="-.", lw=0.65)
    axis.annotate("EE reached", xy=(end_time, 0.28), xycoords=("data", "axes fraction"),
                  xytext=(3, 0), textcoords="offset points", fontsize=7, color="0.25")
    decorate_graph(axis, "(h) Common EE function")
    axis.set_ylabel("$V_{EE}$", labelpad=0)
    axis.set_ylim(0, 180)
    axis.set_yticks([0, 50, 100, 150])
    axis.legend(loc="upper right", frameon=False, handlelength=1.5, borderpad=0.1, labelspacing=0.25)
    for axis in axes:
        axis.set_xlim(0, 35); axis.set_xticks([0, 10, 20, 35])
        for time in (snapshots[0]["time"], snapshots[1]["time"]):
            axis.axvline(time, color="0.7", linestyle=":", linewidth=0.55)
    save(fig, "planar_case030_main")

    aux = Figure(figsize=(180 * MM, 64 * MM)); FigureCanvasAgg(aux)
    gs = aux.add_gridspec(1, 2, left=0.10, right=0.985, top=0.86, bottom=0.22, wspace=0.31)
    aux_axes = [aux.add_subplot(gs[i]) for i in range(2)]
    mask = t <= end_time
    for method, color, style in (("direct", DIRECT, "-"), ("sequential", LINKS[-1], "--")):
        with np.load(source.SOURCE / case[method]["trajectory_file"], allow_pickle=False) as trace:
            # Historical columns 0:6: two obstacles, three links each.
            h = trace["barrier"][:, :6].min(axis=1)
            command = np.linalg.norm(trace["u"], axis=1)
        aux_axes[0].plot(t[mask], h[mask], color=color, ls=style, lw=1.1, label=method.title())
        aux_axes[1].semilogy(t[mask], command[mask], color=color, ls=style, lw=1.1, label=method.title())
    decorate_graph(aux_axes[0], "(a) Obstacle safety boundary")
    decorate_graph(aux_axes[1], "(b) Joint velocity command")
    aux_axes[0].set_ylabel("Minimum obstacle $h$ [m]")
    aux_axes[1].set_ylabel(r"$\|\dot q_{cmd}\|_2$ [rad/s]")
    aux_axes[0].set_yscale("symlog", linthresh=1e-3)
    aux_axes[0].axhline(0, color="0.35", lw=0.65)
    aux_axes[0].axhline(1e-3, color="0.5", ls=":", lw=0.7)
    aux_axes[0].text(0.6, 1.4e-3, "1 mm", fontsize=7, color="0.4")
    for axis in aux_axes:
        axis.set_xlim(0, end_time); axis.set_xticks([0, 5, 10, 15, end_time])
        axis.legend(frameon=False, loc="lower left", handlelength=2)
    save(aux, "planar_case030_obstacles")

    direct_points = source.endpoints(direct_q[end_index], np.ones(3))
    direct_position_error = np.linalg.norm(direct_points[-1] - np.asarray(case["target"]))
    direct_yaw_error = abs(np.arctan2(
        np.sin(np.sum(direct_q[end_index]) - case["target_yaw"]),
        np.cos(np.sum(direct_q[end_index]) - case["target_yaw"]),
    ))
    caption = f"""Sequential-only success approaching the unselected IK branch (case 30).
(a) Common initial pose for both methods, at t={t[0]:.3f} s.
(b,c) Saved states at the goal-reached transitions for link 1 and link 2,
at t={snapshots[0]['time']:.3f} s and t={snapshots[1]['time']:.3f} s.
(d,e) Sequential and Direct states at the same time, t={end_time:.3f} s, when
Sequential first meets the final EE success tolerances. Colored capsules show
links 1-3; stars and the dashed posture show the fixed selected IK targets.
The colored arm in panel (d) reaches the EE target from the opposite elbow
branch to the selected dashed IK posture. Both exact IK branches share the
same EE position and orientation. No IK target change is introduced.
Open rings identify the task endpoint illustrated in each snapshot; arrows
show desired/current EE orientation where applicable. All five spatial panels
use the same scale. Gray disks show physical obstacle radii; dotted halos
indicate the nominal 0.08 m clearance margin. Link shapes use the RDF model's
0.05 m capsule radius. Intermediate-stage position errors are
{snapshots[0]['position_error']:.5f} m and {snapshots[1]['position_error']:.5f} m.
At first Sequential success, position/yaw errors are
{snapshots[-1]['position_error']:.5f} m/{snapshots[-1]['yaw_error']:.5f} rad;
Direct's same-time errors are {direct_position_error:.5f} m/{direct_yaw_error:.5f} rad.
(f,g) Sequential link-task Lyapunov functions; thick solid and thin dashed
segments denote active and inactive objectives. (h) The same final EE Lyapunov
function on both trajectories, shown on a linear axis; Direct uses a
dash-dotted curve. Vertical dotted lines mark link-task transitions and the
dash-dotted line marks first Sequential success. V1 and V2 are
2w||p_i-p_i*||^2; V_EE is 2w[||p_EE-p_EE*||^2+beta(1-cos(psi-psi*))], with
w=5 and beta=0.5. Both runs have a 35 s budget. Direct is unsuccessful and
reports three infeasible QPs in the archive. Full saved trajectories, including
Sequential motion after first success, are plotted without smoothing.
The supplementary obstacle barrier plot uses a signed symmetric logarithmic
axis (linear for |h| <= 1 mm) to retain Direct's negative barrier values.
This selected case illustrates behavior rather than population-level efficacy.
"""
    (OUT / "caption.txt").write_text(caption)
    latex = r"""\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{planar_case030_main.pdf}
  \caption{Sequential approaching an unselected IK branch (case 30).
  (a) Common initial pose at $t=0$~s.
  (b,c) Sequential intermediate goal-reaching states.
  (d,e) Sequential and Direct at the same time, $t=END_TIME$~s,
  when Sequential first meets the final end-effector success tolerances.
  Gray disks represent obstacles; dotted halos indicate the nominal 0.08~m
  clearance margin. Stars and the dashed arm show the fixed selected IK
  targets. The colored arm in (d) approaches the opposite elbow branch
  while reaching the same EE pose. Open rings identify the illustrated
  task endpoints. All spatial panels share the same scale.
  (f,g) Sequential link-task Lyapunov functions; thick solid and thin dashed
  portions indicate active and inactive objectives.
  (h) The common final EE Lyapunov function on both trajectories
  (linear scale), with Direct shown dash-dotted. Vertical dotted lines
  mark link-task transitions; the dash-dotted vertical line marks first
  Sequential success. Both methods have a 35~s budget; Direct is
  unsuccessful and reports QP infeasibility. This is an illustrative case.}
  \label{fig:planar-opposite-ik-case}
\end{figure*}
""".replace("END_TIME", f"{end_time:.3f}")
    (OUT / "figure.tex").write_text(latex)
    meta = {
        "case_id": 30, "main_size_mm": [180, 112], "supplementary_size_mm": [180, 64],
        "font_family": family, "font_size_pt": 8.5, "png_dpi": 600,
        "vector_outputs": ["PDF with embedded TrueType fonts", "SVG with editable text"],
        "raw_inputs_modified": False, "smoothing": False, "data_decimation": False,
        "initial_pose_q_rad": seq_q[0].tolist(), "spatial_panel_count": len(states),
        "first_success_time_s": end_time,
        "direct_same_time_position_error_m": float(direct_position_error),
        "direct_same_time_yaw_error_rad": float(direct_yaw_error),
        "source_report_sha256": hashlib.sha256((source.OUTPUT / "report.json").read_bytes()).hexdigest(),
        "notes": ["Target-journal dimensions and typography still need final adjustment.",
                  "All main-figure Lyapunov axes are linear.",
                  "Dotted obstacle halos are nominal geometric margins, not exact RDF level sets."],
    }
    (OUT / "figure_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

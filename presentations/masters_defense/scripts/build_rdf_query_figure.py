"""Illustrate evaluating link b's field at link a's surface samples.

This is an analytic capsule slice, not a fitted Panda RDF or a simulation.
The visual uses the source/target convention of PandaRDFSafety: surface
samples belong to a; the signed-distance field belongs to b.
"""

from pathlib import Path
import hashlib
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/masters-defense-rdf-query-mpl")
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
FIG = ROOT / "Fig"
EVIDENCE = ROOT / "evidence"
FONT_SIZE = 30  # One source size for all visible labels; scaled with the vector figure.
plt.rcParams.update({
    "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE, "figure.titlesize": FONT_SIZE,
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": (
        r"\usepackage[T1]{fontenc}\usepackage{lmodern}"
        r"\renewcommand{\familydefault}{\sfdefault}"
    ),
    "font.family": "sans-serif",
    "font.size": FONT_SIZE,
    "text.color": "#243447",
    "axes.unicode_minus": False,
})


def capsule_sdf(points, start, end, radius):
    """Exact signed distance to a capsule, for vectors of 2D points."""
    points = np.asarray(points)
    start, end = np.asarray(start), np.asarray(end)
    segment = end - start
    fraction = np.clip((points - start) @ segment / (segment @ segment), 0, 1)
    closest = start + fraction[..., None] * segment
    return np.linalg.norm(points - closest, axis=-1) - radius


def capsule_boundary(start, end, radius, points_per_cap=80):
    start, end = np.asarray(start), np.asarray(end)
    angle = np.arctan2(*(end - start)[::-1])
    angles_end = np.linspace(angle - np.pi / 2, angle + np.pi / 2, points_per_cap)
    angles_start = np.linspace(angle + np.pi / 2, angle + 3 * np.pi / 2,
                               points_per_cap)
    return np.concatenate((
        end + radius * np.c_[np.cos(angles_end), np.sin(angles_end)],
        start + radius * np.c_[np.cos(angles_start), np.sin(angles_start)],
    ))


target_start, target_end = np.array([-.69, -.40]), np.array([.69, -.40])
target_radius = .28
source_start, source_end = np.array([-.64, .62]), np.array([.52, .87])
source_radius = .15
direction = source_end - source_start
direction /= np.linalg.norm(direction)
normal = np.array([-direction[1], direction[0]])
sample_centers = source_start + np.linspace(0, 1, 4)[:, None] * (source_end - source_start)
queries = np.vstack((
    sample_centers + source_radius * normal,
    sample_centers - source_radius * normal,
    source_start - source_radius * direction,
    source_end + source_radius * direction,
))
highlight_index = 6
query = queries[highlight_index]
distances = capsule_sdf(queries, target_start, target_end, target_radius)
np.testing.assert_allclose(
    capsule_sdf(queries, source_start, source_end, source_radius), 0, atol=1e-12)
assert np.all(distances > 0)
assert capsule_sdf(np.array([[0, -.40]]), target_start, target_end, target_radius)[0] < 0
np.testing.assert_allclose(
    capsule_sdf(capsule_boundary(target_start, target_end, target_radius),
                target_start, target_end, target_radius), 0, atol=1e-12)

# A single target field fills the whole coordinate slice, including each
# source query location. Contours never represent the source link's field.
x = np.linspace(-1.40, 1.40, 360)
y = np.linspace(-.98, 1.30, 300)
xx, yy = np.meshgrid(x, y)
field = capsule_sdf(np.stack((xx, yy), axis=-1), target_start,
                    target_end, target_radius)
colors = LinearSegmentedColormap.from_list(
    "signed_capsule_field", [(0, "#e7a995"), (.5, "#fffaf3"),
                             (.66, "#dcecf3"), (.83, "#91c1d6"), (1, "#4e94b5")])
norm = TwoSlopeNorm(vmin=-target_radius, vcenter=0, vmax=1.8)
levels = [-.28, -.20, -.10, 0, .20, .40, .60, .80, 1.0, 1.2, 1.4, 1.6, 1.8]

fig = plt.figure(figsize=(6.5, 5.0))
ax = fig.add_axes([.025, .15, .95, .76])
ax.contourf(xx, yy, field, levels=levels, cmap=colors, norm=norm, extend="max", zorder=0)
ax.contour(xx, yy, field, levels=[.2, .4, .6, .8, 1.0, 1.2, 1.4],
           colors="#6d9eb4", linewidths=.65, zorder=1)
boundary = capsule_boundary(target_start, target_end, target_radius)
ax.add_patch(Polygon(boundary, closed=True, facecolor="none", edgecolor="#586776",
                     linewidth=2.0, zorder=2))
ax.text(0, -.40, r"Link $b$", ha="center", va="center", fontsize=FONT_SIZE, zorder=4)
ax.text(0, -.79, r"$\tilde d_b=0$", ha="center", va="center",
        fontsize=FONT_SIZE, zorder=4)

source_boundary = capsule_boundary(source_start, source_end, source_radius)
ax.add_patch(Polygon(source_boundary, closed=True, facecolor="white",
                     edgecolor="#475d70", linewidth=2, zorder=3))
ax.text(*(source_start + source_end) / 2, r"$a$", ha="center", va="center",
        fontsize=FONT_SIZE, rotation=np.degrees(np.arctan2(direction[1], direction[0])), zorder=4)
query_color = "#d65e00"
ax.scatter(*queries.T, s=61, c=query_color, edgecolors="white", linewidths=.9, zorder=5)
ax.scatter(*query, s=220, facecolors="none", edgecolors=query_color,
           linewidths=2.0, zorder=6)

# The leader identifies evaluation at the existing surface sample; it has
# no arrowhead and does not depict motion into the target link.
ax.annotate(r"$d_k=\tilde d_b(\xi_{ab,k})$", xy=query, xytext=(.50, .11),
            ha="center", va="center", fontsize=FONT_SIZE, color=query_color,
            bbox={"boxstyle": "round,pad=.23", "fc": "white", "ec": query_color,
                  "lw": 1.25},
            arrowprops={"arrowstyle": "-", "color": query_color,
                        "lw": 1.6, "shrinkA": 5, "shrinkB": 8}, zorder=7)
ax.text(-1.25, 1.17, r"Surface queries", fontsize=FONT_SIZE, va="center", zorder=8)
ax.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), aspect="equal")
ax.axis("off")

fig.text(.5, .955, r"Schematic field $\tilde d_b$", ha="center", va="center", fontsize=FONT_SIZE)
# The sign legend distinguishes a signed field from a clearance band.
legend_ax = fig.add_axes([.13, .087, .74, .026])
legend_values = np.linspace(-target_radius, 1.8, 256)[None, :]
legend_ax.imshow(legend_values, aspect="auto", cmap=colors, norm=norm,
                 extent=(-target_radius, 1.8, 0, 1), interpolation="nearest")
zero_x = .13 + .74 * target_radius / (target_radius + 1.8)
legend_ax.axvline(0, color="#586776", lw=1)
legend_ax.axis("off")
fig.text(.04, .055, r"$<0$", ha="left", va="center", fontsize=FONT_SIZE)
fig.text(zero_x, .055, r"$0$", ha="center", va="center", fontsize=FONT_SIZE)
fig.text(.87, .055, r"$>0$", ha="right", va="center", fontsize=FONT_SIZE)

FIG.mkdir(exist_ok=True)
path = FIG / "rdf_query_field.pdf"
fig.savefig(path, bbox_inches="tight", pad_inches=.05)
plt.close(fig)
subprocess.run(["pdftoppm", "-r", "180", "-singlefile", "-png", str(path),
                str(FIG / "rdf_query_field")], check=True, stdout=subprocess.DEVNULL,
               stderr=subprocess.PIPE)

sources = [
    REPO / "src/safe_ik/domain/panda_rdf_safety.py",
    Path("/home/jinoo/latex-ws/LaTeX-env/ws/masters_tethis/sections/Preliminaries/RDF.tex"),
    Path("/home/jinoo/latex-ws/LaTeX-env/ws/masters_tethis/sections/Method/RDF_CBF.tex"),
]
audit = {
    "representation": "Analytic signed-distance field of a capsule, schematic 2D slice",
    "actual_panda_rdf": False,
    "no_new_simulation": True,
    "no_new_training": True,
    "font_rendering": "pdfLaTeX + Latin Modern Sans; PNG rasterized from vector PDF",
    "size_inches": [6.5, 5.0],
    "recommended_slide_size_mm": [65, 50],
    "source_target_convention": "Source a surface samples, target b signed-distance field",
    "coordinate_convention": "All geometry shown in target link b frame",
    "query_coordinate_symbol": "xi_ab,k = R_b^T (R_a s_a,k + p_a - p_b)",
    "runtime_field_symbol": "tilde d_b",
    "target_capsule": {"start": target_start.tolist(), "end": target_end.tolist(),
                       "radius": target_radius},
    "source_capsule": {"start": source_start.tolist(), "end": source_end.tolist(),
                       "radius": source_radius},
    "query_coordinates": queries.tolist(),
    "query_distances_illustrative": distances.tolist(),
    "highlight_index": highlight_index,
    "surface_query_residual_max": float(np.abs(capsule_sdf(
        queries, source_start, source_end, source_radius)).max()),
    "queries_all_on_source_surface": True,
    "queries_all_outside_target": True,
    "field_negative_inside_target_and_zero_on_surface_verified": True,
    "positive_contours_increasing_with_distance": [.2, .4, .6, .8, 1.0, 1.2, 1.4],
    "no_query_motion_arrow": True,
    "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sources],
    "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [path, FIG / "rdf_query_field.png"]},
}
(EVIDENCE / "rdf_query_figure_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print("Rendered RDF query schematic and verified source-surface / target-field geometry.")

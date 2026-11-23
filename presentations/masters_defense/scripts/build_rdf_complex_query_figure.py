"""Draw a two-object query illustration using exact polygon signed distances.

The independently constructed concave shapes are a 2D teaching example, not
Panda geometry, a fitted RDF, or a new control experiment. All coordinates are
shown in target b's frame. Source a contributes surface query locations; only
target b contributes the displayed distance field.
"""

from pathlib import Path
import hashlib
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/masters-defense-rdf-complex-query-mpl")
import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "Fig"
EVIDENCE = ROOT / "evidence"
FONT_SIZE = 29.5
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": (
        r"\usepackage[T1]{fontenc}\usepackage{lmodern}"
        r"\renewcommand{\familydefault}{\sfdefault}"
    ),
    "font.family": "sans-serif",
    "font.size": FONT_SIZE,
    "text.color": "black",
    "axes.unicode_minus": False,
})


def polygon_sdf(points, vertices):
    """Exact Euclidean distance to polygon segments, signed by ray crossing."""
    points = np.asarray(points, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    squared = np.full(points.shape[:-1], np.inf)
    inside = np.zeros(points.shape[:-1], dtype=bool)
    px, py = points[..., 0], points[..., 1]
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        edge = end - start
        assert edge @ edge > 0
        fraction = np.clip((points - start) @ edge / (edge @ edge), 0, 1)
        closest = start + fraction[..., None] * edge
        squared = np.minimum(squared, np.sum((points - closest) ** 2, axis=-1))
        if start[1] != end[1]:
            crosses_y = (start[1] > py) != (end[1] > py)
            crossing_x = start[0] + (py - start[1]) * edge[0] / edge[1]
            inside ^= crosses_y & (px < crossing_x)
    return np.where(inside, -1, 1) * np.sqrt(squared)


def polygon_is_simple(vertices):
    """Reject intersections between any two nonadjacent boundary segments."""
    def orient(a, b, c):
        ab, ac = b - a, c - a
        return ab[0] * ac[1] - ab[1] * ac[0]

    n = len(vertices)
    for i in range(n):
        a, b = vertices[i], vertices[(i + 1) % n]
        for j in range(i + 1, n):
            if j in {(i + 1) % n, (i - 1) % n}:
                continue
            c, d = vertices[j], vertices[(j + 1) % n]
            values = (orient(a, b, c), orient(a, b, d),
                      orient(c, d, a), orient(c, d, b))
            if values[0] * values[1] <= 0 and values[2] * values[3] <= 0:
                if (max(min(a[0], b[0]), min(c[0], d[0])) <=
                        min(max(a[0], b[0]), max(c[0], d[0])) and
                        max(min(a[1], b[1]), min(c[1], d[1])) <=
                        min(max(a[1], b[1]), max(c[1], d[1]))):
                    return False
    return True


def polygon_is_concave(vertices):
    edges = np.roll(vertices, -1, axis=0) - vertices
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    return bool(np.any(turns > 1e-12) and np.any(turns < -1e-12))


source = np.array([
    [-1.15, .30], [-1.15, .95], [-.90, 1.10], [-.67, 1.10],
    [-.67, .83], [-.36, .83], [-.36, .52], [-.60, .52],
    [-.60, .28], [-.85, .28], [-.85, .47], [-1.00, .47],
    [-1.00, .30],
])
target = np.array([
    [.05, -.75], [.90, -.75], [1.13, -.53], [1.13, -.20],
    [.87, -.20], [.87, .15], [1.08, .15], [1.08, .55],
    [.80, .75], [.40, .75], [.40, .50], [.10, .50],
    [.10, .20], [.50, .20], [.50, -.30], [.05, -.30],
])
assert polygon_is_simple(source) and polygon_is_simple(target)
assert polygon_is_concave(source) and polygon_is_concave(target)

# Sample selected edges directly, so every query lies on the source boundary.
query_edges = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11])
query_fractions = np.array([.44, .45, .55, .48, .50, .60, .52, .55, .52, .5, .6])
queries = source[query_edges] + query_fractions[:, None] * (
    source[(query_edges + 1) % len(source)] - source[query_edges])
highlight_index = 5
highlight = queries[highlight_index]
source_residual = np.abs(polygon_sdf(queries, source))
query_distances = polygon_sdf(queries, target)
np.testing.assert_allclose(source_residual, 0, atol=1e-12)
assert np.all(query_distances > 0)

# Independent analytic reference values for a unit square, plus this target's
# boundary/interior and concave opening. These catch sign and projection errors.
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
np.testing.assert_allclose(
    polygon_sdf(np.array([[.5, .5], [2, .5], [2, 2], [0, .3]]), square),
    np.array([-.5, 1, np.sqrt(2), 0]), atol=1e-12)
fractions = np.linspace(0, 1, 17)
boundary_samples = (target[:, None, :] + fractions[None, :, None] *
                    (np.roll(target, -1, axis=0) - target)[:, None, :]).reshape(-1, 2)
target_surface_residual = np.abs(polygon_sdf(boundary_samples, target))
np.testing.assert_allclose(target_surface_residual, 0, atol=1e-12)
assert polygon_sdf(np.array([[.75, -.50]]), target)[0] < 0
assert polygon_sdf(np.array([[.30, 0.0]]), target)[0] > 0

x = np.linspace(-1.45, 1.45, 440)
y = np.linspace(-1.08, 1.43, 380)
xx, yy = np.meshgrid(x, y)
field = polygon_sdf(np.stack((xx, yy), axis=-1), target)
colors = LinearSegmentedColormap.from_list(
    "concave_target_signed_distance",
    [(0, "#e89981"), (.5, "#fffaf0"), (.63, "#dcecf5"),
     (.80, "#a5cce1"), (1, "#5596bf")],
)
norm = TwoSlopeNorm(vmin=-.40, vcenter=0, vmax=2.8)
levels = [-.4, -.3, -.2, -.1, 0, .2, .4, .6, .8, 1, 1.2,
          1.4, 1.6, 1.8, 2, 2.4, 2.8]

fig = plt.figure(figsize=(6.5, 5.5))
ax = fig.add_axes([.015, .015, .97, .97])
ax.contourf(xx, yy, field, levels=levels, cmap=colors, norm=norm,
            extend="both", zorder=0)
ax.contour(xx, yy, field, levels=[.2, .4, .6, .8, 1, 1.2, 1.4, 1.6, 1.8, 2],
           colors="#7d9fac", linewidths=.75, zorder=1)
ax.add_patch(Polygon(target, closed=True, facecolor="none", edgecolor="#39434d",
                     linewidth=2.2, zorder=2))
ax.add_patch(Polygon(source, closed=True, facecolor="white", edgecolor="#39434d",
                     linewidth=2.2, zorder=3))
ax.text(-.93, .72, r"$a$", ha="center", va="center", zorder=4)
ax.text(.75, -.50, r"$b$", ha="center", va="center", zorder=4)

red = "#ff0000"
ax.scatter(*queries.T, s=100, c=red, edgecolors="white", linewidths=1.2, zorder=5)
ax.scatter(*highlight, s=330, facecolors="none", edgecolors=red,
           linewidths=2.1, zorder=6)

# Identification leaders have no arrowheads: the query stays at its surface
# position, and the target field is evaluated at that existing coordinate.
ax.annotate("Surface queries", xy=queries[2], xytext=(-1.33, 1.28),
            ha="left", va="center", color="black",
            arrowprops={"arrowstyle": "-", "lw": 1.6, "color": red,
                        "shrinkA": 6, "shrinkB": 8}, zorder=7)
ax.annotate(r"$d_k=\tilde d_b(\xi_{ab,k})$", xy=highlight, xytext=(-.68, -.39),
            ha="center", va="center", color=red,
            bbox={"boxstyle": "round,pad=.22", "fc": "white", "ec": red,
                  "lw": 1.3},
            arrowprops={"arrowstyle": "-", "color": red, "lw": 1.6,
                        "shrinkA": 5, "shrinkB": 10}, zorder=8)
ax.annotate(r"$\tilde d_b=0$", xy=(.60, -.75), xytext=(.60, -.99),
            ha="center", va="center",
            arrowprops={"arrowstyle": "-", "lw": 1.4, "color": "#39434d",
                        "shrinkA": 4, "shrinkB": 3}, zorder=7)
ax.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), aspect="equal")
ax.axis("off")

FIG.mkdir(exist_ok=True)
EVIDENCE.mkdir(exist_ok=True)
pdf = FIG / "rdf_complex_query_field.pdf"
png = FIG / "rdf_complex_query_field.png"
fig.savefig(pdf, bbox_inches="tight", pad_inches=.035)
plt.close(fig)
subprocess.run(["pdftoppm", "-r", "180", "-singlefile", "-png", str(pdf),
                str(png.with_suffix(""))], check=True, stdout=subprocess.DEVNULL,
               stderr=subprocess.PIPE)

sources = [
    Path(__file__),
    ROOT / "scripts/build_rdf_query_figure.py",
    Path("/home/jinoo/latex-ws/LaTeX-env/ws/masters_tethis/sections/Method/RDF_CBF.tex"),
]
audit = {
    "representation": "Exact signed distance to a concave target polygon; original 2D teaching illustration",
    "actual_panda_rdf": False,
    "fitted_rdf": False,
    "capsule_approximation": False,
    "no_new_simulation": True,
    "no_new_training": True,
    "font_rendering": "pdfLaTeX + Latin Modern Sans; PNG rasterized from vector PDF",
    "font_size_pt": FONT_SIZE,
    "size_inches": [6.5, 5.5],
    "recommended_slide_size_mm": [56, 51],
    "source_target_convention": "Source a surface samples, target b signed-distance field",
    "coordinate_convention": "All geometry shown in the target b frame; no query motion",
    "runtime_field_symbol": "tilde d_b",
    "query_coordinate_symbol": "xi_ab,k",
    "source_polygon_vertices": source.tolist(),
    "target_polygon_vertices": target.tolist(),
    "both_polygons_simple": True,
    "both_polygons_concave": True,
    "query_count": len(queries),
    "query_edge_indices": query_edges.tolist(),
    "query_edge_fractions": query_fractions.tolist(),
    "query_coordinates": queries.tolist(),
    "query_distances_illustrative": query_distances.tolist(),
    "highlight_index": highlight_index,
    "surface_query_residual_max": float(source_residual.max()),
    "target_surface_residual_max": float(target_surface_residual.max()),
    "queries_all_on_source_surface": True,
    "queries_all_outside_target": True,
    "field_negative_inside_target_and_zero_on_surface_verified": True,
    "concave_opening_positive_verified": True,
    "analytic_square_reference_verified": True,
    "field_grid_includes_every_query": bool(np.all(
        (queries[:, 0] >= x[0]) & (queries[:, 0] <= x[-1]) &
        (queries[:, 1] >= y[0]) & (queries[:, 1] <= y[-1]))),
    "distance_computation": "Minimum Euclidean distance to exact boundary segments; ray-crossing inside test",
    "no_query_motion_arrow": True,
    "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sources],
    "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [pdf, png]},
}
(EVIDENCE / "rdf_complex_query_figure_audit.json").write_text(
    json.dumps(audit, indent=2) + "\n")
print("Rendered concave-object RDF query schematic; exact polygon-distance checks passed.")

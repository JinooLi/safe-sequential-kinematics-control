"""Label the current and goal end-effector poses without changing geometry.

The shared renderer predates a main guard. Execute only its setup and drawing
definitions, stopping before the stage-rendering loop, so its original figures
and audit files remain untouched. This annotated figure adds lower-left margin
so the goal-heading label can sit beyond the arrow tip without clipping.
"""

import ast
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
renderer = ROOT / "scripts" / "build_progressive_planar.py"
tree = ast.parse(renderer.read_text(), filename=str(renderer))
setup = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "stages"
        for target in node.targets
    ):
        break
    setup.append(node)
namespace = {"__file__": str(renderer), "__name__": "goal_pose_geometry"}
exec(compile(ast.Module(body=setup, type_ignores=[]), str(renderer), "exec"), namespace)

plt = namespace["plt"]
height = namespace["figure_height"]
lo, hi = namespace["lo"].copy(), namespace["hi"].copy()
lo -= [.30, .21]
span = hi - lo
width = height * float(span[0] / span[1])
target = namespace["target"]
heading_end = namespace["HEADING_END"]
heading_direction = (heading_end - target) / .27
heading_label = heading_end + .08 * heading_direction

# The scene occupies 43% of a 118-mm text area on slide 3. Compensate for
# includegraphics scaling so every symbol has an effective size of 10 pt.
display_width_mm = .43 * 118
source_font_pt = 10 * width * 25.4 / display_width_mm

fig = plt.figure(figsize=(width, height))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal")
ax.axis("off")
namespace["draw_scene"](ax, False, None)
points = namespace["draw_robot"](ax, namespace["q0"])

# The current end-effector lies near the right edge. Translate its heading
# arrow above the final link instead of extending it beyond the fixed viewport.
# The direction remains parallel to the end-effector's current x-axis.
current_ee = points[-1]
current_color = "#111111"
ax.annotate("", xy=(current_ee[0] + .05, current_ee[1] + .27),
            xytext=(current_ee[0] - .45, current_ee[1] + .27),
            arrowprops={"arrowstyle": "->", "color": current_color,
                        "lw": 1.7, "mutation_scale": 12}, zorder=10)
ax.text(current_ee[0] - .2, current_ee[1] + .57, r"$\psi_3$",
        fontsize=source_font_pt, ha="center", va="center",
        color=current_color, zorder=10)
ax.annotate(r"$p_3$", xy=current_ee,
            xytext=(current_ee[0] - .64, current_ee[1] - .35),
            fontsize=source_font_pt, ha="left", va="center",
            color=current_color,
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 1,
                        "shrinkA": 3, "shrinkB": 5}, zorder=10)

green = namespace["COLORS"][2]
ax.text(.02, -1.76, r"$p_d$", fontsize=source_font_pt,
        ha="left", va="center", color=green, zorder=10)
ax.text(*heading_label, r"$\psi_d$", fontsize=source_font_pt,
        ha="right", va="top", color=green, zorder=10)

stem = ROOT / "Fig" / "progressive_planar_02_scene_labeled"
fig.savefig(stem.with_suffix(".pdf"), bbox_inches=None, pad_inches=0)
subprocess.run(["pdftoppm", "-r", "150", "-singlefile", "-png",
                str(stem.with_suffix(".pdf")), str(stem)], check=True)
plt.close(fig)

print(json.dumps({
    "output": str(stem.with_suffix(".pdf")),
    "pdf_size_inches": [width, height],
    "goal_position": target.tolist(),
    "goal_heading_arrow_tip": heading_end.tolist(),
    "goal_heading_label_position": heading_label.tolist(),
    "current_end_effector_position": current_ee.tolist(),
    "current_heading_arrow_direction": [1, 0],
    "label_font_pt_at_50_74_mm_width": 10,
    "native_font_pt": source_font_pt,
}, indent=2))

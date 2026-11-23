"""Compare candidate-average, best-single-pair, and oracle Validation success.

The candidate mean weights the ten pairs equally. The best single pair and
episode-wise oracle are post-hoc summaries of the same frozen Validation data;
neither bar reports an executable learned selector. No new rollout or training.
"""

from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/masters-defense-two-link-average-mpl")
import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "evidence", ROOT / "Fig"
FONT_SIZE = 22
BLUE, GREEN, GRAY = "#205493", "#28735e", "#77818b"
plt.rcParams.update({
    "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE, "figure.titlesize": FONT_SIZE,
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": (
        r"\usepackage{kotex}\usepackage[english]{babel}\usepackage[T1]{fontenc}"
        r"\usepackage{lmodern}\renewcommand{\familydefault}{\sfdefault}"
    ),
    "font.family": "sans-serif", "font.size": FONT_SIZE,
    "axes.unicode_minus": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelcolor": "#243447",
    "text.color": "#243447",
})

with (DATA / "pair_outcomes.csv").open(newline="") as stream:
    records = [row for row in csv.DictReader(stream)
               if row["split"] == "validation"]
pairs = json.loads((DATA / "recomputed_summary.json").read_text())["action_pairs"]
assert len(pairs) == 10 and len(records) == 10000
episode_pairs = {}
for row in records:
    pair_index = int(row["pair_index"])
    assert 0 <= pair_index < len(pairs)
    assert json.loads(row["pair"]) == pairs[pair_index]
    success = int(row["success"])
    assert success in (0, 1)
    outcomes = episode_pairs.setdefault(row["case_id"], {})
    assert pair_index not in outcomes, "Duplicate episode/pair outcome"
    outcomes[pair_index] = success
assert len(episode_pairs) == 1000
assert all(set(outcomes) == set(range(10)) for outcomes in episode_pairs.values())

pair_successes = np.array([
    sum(outcomes[index] for outcomes in episode_pairs.values())
    for index in range(10)
], dtype=int)
total_successes = int(pair_successes.sum())
best_index = int(np.argmax(pair_successes))
oracle_successes = sum(max(outcomes.values()) for outcomes in episode_pairs.values())
assert total_successes == 5473
assert pairs[best_index] == [2, 5] and pair_successes[best_index] == 581
assert oracle_successes == 820
values = np.array([
    total_successes / len(records),
    pair_successes[best_index] / len(episode_pairs),
    oracle_successes / len(episode_pairs),
]) * 100
np.testing.assert_allclose(values, [54.73, 58.10, 82.00], atol=1e-12)

labels = ["10-pair mean", "Best single pair\n(2,5)",
          "Oracle\n(post-hoc upper bound)"]
fig, ax = plt.subplots(figsize=(8.5, 3))
bars = ax.barh(labels, values, color=[GRAY, BLUE, GREEN], height=.54)
plt.setp(ax.get_yticklabels(), linespacing=.85)
ax.bar_label(bars, labels=[f"{value:.2f}\\%" for value in values],
             padding=6, fontsize=FONT_SIZE)
ax.invert_yaxis()
ax.set(xlim=(0, 103), xticks=[0, 25, 50, 75, 100], xlabel="Success rate (%)")
ax.set_axisbelow(True)
ax.grid(axis="x", alpha=.15)
fig.tight_layout()
FIG.mkdir(exist_ok=True)
pdf = FIG / "story_oracle_average.pdf"
png = FIG / "story_oracle_average.png"
fig.savefig(pdf, bbox_inches="tight", pad_inches=.08)
plt.close(fig)
subprocess.run([
    "pdftoppm", "-r", "150", "-singlefile", "-png", str(pdf),
    str(png.with_suffix("")),
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

sources = [Path(__file__), DATA / "pair_outcomes.csv", DATA / "recomputed_summary.json"]
audit = {
    "split": "validation",
    "episodes": len(episode_pairs),
    "pairs_per_episode": len(pairs),
    "episode_pair_records": len(records),
    "candidate_pair_successes": pair_successes.tolist(),
    "candidate_pair_order": pairs,
    "candidate_average": {
        "successes": total_successes,
        "denominator": len(records),
        "success_rate_percent": float(values[0]),
        "definition": "Equal mean of all ten candidate-pair success rates on the same 1000 episodes",
    },
    "best_single_pair": {
        "pair_index": best_index,
        "pair": pairs[best_index],
        "successes": int(pair_successes[best_index]),
        "success_rate_percent": float(values[1]),
        "post_hoc_on_this_validation_set": True,
    },
    "oracle": {
        "successes": oracle_successes,
        "success_rate_percent": float(values[2]),
        "definition": "At least one of the ten pairs succeeds in an episode",
        "post_hoc_upper_bound_not_a_learned_selector": True,
    },
    "display_labels": labels,
    "font_rendering": "pdfLaTeX + ko.TeX + Latin Modern Sans; PNG rasterized from PDF",
    "font_size_pt": FONT_SIZE,
    "figure_size_inches": [8.5, 3],
    "no_new_simulation": True,
    "no_new_training": True,
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sources},
    "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [pdf, png]},
}
(DATA / "two_link_average_figure_audit.json").write_text(
    json.dumps(audit, indent=2) + "\n")
print("Validation figure verified: 10-pair mean 54.73%, best single pair 58.10%, oracle 82.00%.")

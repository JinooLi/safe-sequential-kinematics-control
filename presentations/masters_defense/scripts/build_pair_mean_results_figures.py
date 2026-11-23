"""Compare learned selection against the mean of ten frozen candidate pairs.

The baseline is the arithmetic mean of the ten success indicators in each
episode. The difference distribution therefore uses the same 1,000 Test
episodes and does not treat the 10,000 episode/pair records as independent.
No new rollout or training is performed.
"""

from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/masters-defense-pair-mean-results-mpl")
import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "evidence", ROOT / "Fig"
FONT_SIZE = 22
BLUE, GREEN, RED, GRAY = "#0000FF", "#008000", "#FF0000", "#77818b"
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


def rows(filename):
    with (DATA / filename).open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, stem):
    pdf, png = FIG / f"{stem}.pdf", FIG / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)
    subprocess.run([
        "pdftoppm", "-r", "150", "-singlefile", "-png", str(pdf),
        str(png.with_suffix("")),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return [pdf, png]


def main():
    pair_order = json.loads((DATA / "recomputed_summary.json").read_text())["action_pairs"]
    assert len(pair_order) == 10
    pairs = {}
    for row in rows("pair_outcomes.csv"):
        if row["split"] != "test":
            continue
        index = int(row["pair_index"])
        assert 0 <= index < 10 and json.loads(row["pair"]) == pair_order[index]
        outcome = int(row["success"])
        assert outcome in (0, 1)
        episode = pairs.setdefault(row["case_id"], {})
        assert index not in episode
        episode[index] = outcome
    assert len(pairs) == 1000
    assert all(set(outcomes) == set(range(10)) for outcomes in pairs.values())

    selected_rows = [row for row in rows("selector_choices.csv")
                     if row["split"] == "test" and row["family"] == "baseline"
                     and row["seed"] == "11"]
    selected = {row["case_id"]: row for row in selected_rows}
    baseline_rows = rows("test_baselines_comparison.csv")
    baselines = {row["case_id"]: row for row in baseline_rows}
    assert len(selected_rows) == len(baseline_rows) == len(selected) == len(baselines) == 1000
    assert set(selected) == set(baselines) == set(pairs)

    episode_ids = sorted(pairs)
    learned, pair_means, integer_differences = [], [], []
    for case_id in episode_ids:
        row = selected[case_id]
        choice = int(row["selected_pair_index"])
        success = int(row["success"])
        assert json.loads(row["selected_pair"]) == pair_order[choice]
        assert success == pairs[case_id][choice]
        pair_sum = sum(pairs[case_id].values())
        learned.append(success)
        pair_means.append(pair_sum / 10)
        # Integer arithmetic keeps exact 0.1-wide outcome bins deterministic.
        integer_differences.append(10 * success - pair_sum)

    direct = [row["direct_120_success"] == "True" for row in baselines.values()]
    all_links = [row["sequential_120_success"] == "True" for row in baselines.values()]
    pair_counts = np.array([sum(outcomes[k] for outcomes in pairs.values())
                            for k in range(len(pair_order))])
    best_index = int(pair_counts.argmax())
    assert np.count_nonzero(pair_counts == pair_counts[best_index]) == 1
    best_pair = pair_order[best_index]
    best_rate = float(pair_counts[best_index] / len(episode_ids))
    assert best_pair == [3, 5] and pair_counts[best_index] == 587
    values = np.array([np.mean(direct), np.mean(all_links),
                       np.mean(pair_means), best_rate, np.mean(learned)]) * 100
    np.testing.assert_allclose(values, [45.50, 55.20, 55.20, 58.70, 60.90], atol=1e-12)
    differences = np.asarray(integer_differences) / 10
    np.testing.assert_allclose(differences, np.asarray(learned) - pair_means, atol=1e-12)
    mean_difference = float(differences.mean())
    np.testing.assert_allclose(mean_difference, .057, atol=1e-12)
    centers_int = np.arange(-9, 10)
    counts = np.array([integer_differences.count(int(value)) for value in centers_int])
    assert int(counts.sum()) == 1000

    FIG.mkdir(exist_ok=True)
    labels = ["Direct", "All-link Sequential", "10-pair mean",
              f"Best single pair ({best_pair[0]},{best_pair[1]})", "Learned selection"]
    fig, ax = plt.subplots(figsize=(8.8, 3.7))
    bars = ax.barh(labels, values, color=[GRAY, BLUE, GRAY, BLUE, GREEN], height=.54)
    ax.bar_label(bars, labels=[f"{value:.2f}\\%" for value in values],
                 padding=6, fontsize=FONT_SIZE)
    ax.invert_yaxis()
    ax.set(xlim=(0, 77), xlabel="Test success rate (%)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.15)
    fig.tight_layout()
    outputs = save(fig, "test_success_pair_mean")

    fig, ax = plt.subplots(figsize=(8.8, 3.7))
    centers = centers_int / 10
    ax.bar(centers, counts, width=.075,
           color=[RED if value < 0 else BLUE if value > 0 else GRAY
                  for value in centers_int])
    ax.axvline(mean_difference, color=BLUE, linestyle="--", linewidth=1.8)
    ax.annotate(r"Mean: $+0.057$", xy=(mean_difference, 370),
                xytext=(.37, 440), color=BLUE, ha="center", va="center",
                arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 1.3})
    ax.set(xlim=(-.98, .98), ylim=(0, 505),
           xticks=[-.9, -.6, -.3, 0, .3, .6, .9], yticks=[0, 200, 400],
           xlabel="Episode-wise success difference", ylabel="Episodes")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    outputs += save(fig, "selection_gain_pair_mean")

    sources = [Path(__file__), DATA / "pair_outcomes.csv",
               DATA / "selector_choices.csv", DATA / "test_baselines_comparison.csv",
               DATA / "recomputed_summary.json"]
    audit = {
        "split": "test", "episodes": 1000, "pairs_per_episode": 10,
        "selector_family": "baseline", "selector_seed": 11,
        "baseline_definition": "Arithmetic mean of ten candidate-pair success indicators per episode",
        "success_plot": {"labels": labels, "rates_percent": values.tolist(),
                         "figure_size_inches": [8.8, 3.7]},
        "best_single_pair": {
            "definition": "Highest aggregate Test success among ten fixed candidate pairs; one same pair across all episodes",
            "pair": best_pair, "pair_index": best_index,
            "successes": int(pair_counts[best_index]), "episodes": len(episode_ids),
            "success_rate_percent": best_rate * 100,
            "all_pair_success_counts": pair_counts.tolist(),
            "unique_maximum": True, "selected_after_evaluation": True,
            "episode_wise_oracle": False,
        },
        "difference_plot": {
            "definition": "d_i = selected success - mean of ten pair successes in the same episode",
            "values": centers.tolist(), "counts": counts.tolist(),
            "mean": mean_difference, "mean_percentage_points": mean_difference * 100,
            "bin_width": .1, "rendered_bar_width": .075,
            "negative_episodes": int(counts[centers_int < 0].sum()),
            "zero_episodes": int(counts[centers_int == 0].sum()),
            "positive_episodes": int(counts[centers_int > 0].sum()),
            "figure_size_inches": [8.8, 3.7],
        },
        "font_rendering": "pdfLaTeX + ko.TeX + Latin Modern Sans; PNG rasterized from PDF",
        "font_size_pt": FONT_SIZE,
        "emphasis_colors": {"blue": BLUE, "red": RED, "green": GREEN},
        "no_new_simulation": True, "no_new_training": True,
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sources},
        "output_sha256": {path.name: sha(path) for path in outputs},
    }
    (DATA / "pair_mean_results_figure_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"rates_percent": values.tolist(),
                      "difference_counts": counts.tolist(), "mean_difference": mean_difference}))


if __name__ == "__main__":
    main()

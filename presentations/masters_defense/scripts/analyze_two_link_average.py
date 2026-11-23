"""Reaggregate archived Validation outcomes for the mean of all ten pairs.

Run from the repository with ``.venv/bin/python
presentations/masters_defense/scripts/analyze_two_link_average.py``.
This script performs no simulation, random pair sampling, or learning. It checks
the presentation snapshots against their originals and the recorded NPZ/CSV
hashes before writing only evidence/two_link_average_audit.json.
"""

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "evidence"
EVALUATION = Path(
    "outputs/panda_link_selection/fresh_evaluation/"
    "evaluation-2000-20261020T003931Z"
)
ALL_LINKS = Path(
    "docs/experiments/results/panda_all_links_validation_1000_20261023"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10),
            f"Mismatch in {label}: {actual} != {expected}")


def flag(value):
    require(value in {"True", "False", "1", "0"}, f"Invalid flag: {value}")
    return value in {"True", "1"}


def success_time(row, key):
    value = float(row[key])
    require(math.isfinite(value) and 0 <= value <= 120 + 1e-8,
            f"Invalid successful completion time: {row}")
    return value


def paired_statistics(records, all_link_rows):
    """Condition the given equally weighted episode--pair records on success."""
    common = [row for row in records if flag(row["success"]) and
              flag(all_link_rows[row["case_id"]]["sequential_120_success"])]
    reference = [success_time(all_link_rows[row["case_id"]],
                              "sequential_120_completion_time") for row in common]
    candidate = [success_time(row, "completion_time") for row in common]
    reference_mean, candidate_mean = mean(reference), mean(candidate)
    both_success = len(common)
    reference_success = sum(flag(all_link_rows[row["case_id"]]
                                 ["sequential_120_success"]) for row in records)
    candidate_success = sum(flag(row["success"]) for row in records)
    return {
        "both_success_records": both_success,
        "unique_common_success_episodes": len({row["case_id"] for row in common}),
        "reference_only_success_records": reference_success - both_success,
        "candidate_only_success_records": candidate_success - both_success,
        "both_failure_records": len(records) - reference_success -
            candidate_success + both_success,
        "all_links_mean_seconds": reference_mean,
        "two_link_mean_seconds": candidate_mean,
        "mean_paired_difference_seconds": mean(
            c - r for c, r in zip(candidate, reference)),
        "relative_reduction_percent": 100 * (1 - candidate_mean / reference_mean),
    }, common


def main():
    source_manifest = json.loads((EVIDENCE / "sources.json").read_text())
    source_records = {Path(item["copy"]).name: item for item in source_manifest}
    snapshot_names = ["pair_outcomes.csv", "all_links_comparison.csv",
                      "all_links_analysis.json", "recomputed_summary.json"]
    checked_sources = []
    for name in snapshot_names:
        snapshot, item = EVIDENCE / name, source_records[name]
        original = Path(item["source"])
        snapshot_hash = sha256(snapshot)
        require(snapshot_hash == item["sha256"], f"Changed snapshot: {snapshot}")
        require(sha256(original) == snapshot_hash, f"Changed original: {original}")
        checked_sources.append({
            "snapshot": str(snapshot.relative_to(ROOT)),
            "original": str(original), "sha256": snapshot_hash,
            "snapshot_matches_original": True,
            "matches_source_manifest": True,
        })

    original_evidence = Path(source_records["pair_outcomes.csv"]["source"]).parent
    provenance_path = original_evidence / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    raw_paths = [EVALUATION / "dataset/dataset.npz",
                 EVALUATION / "dataset/results.csv", EVALUATION / "split.csv",
                 ALL_LINKS / "comparison.csv", ALL_LINKS / "analysis.json"]
    raw_sources = []
    for relative_path in raw_paths:
        digest = sha256(REPOSITORY / relative_path)
        require(digest == provenance["source_sha256"][str(relative_path)],
                f"Raw source hash mismatch: {relative_path}")
        raw_sources.append({"path": str(relative_path), "sha256": digest,
                            "matches_archived_provenance": True})
    require(sha256(REPOSITORY / ALL_LINKS / "comparison.csv") ==
            sha256(EVIDENCE / "all_links_comparison.csv"),
            "All-link snapshot differs from raw comparison")

    pairs = json.loads((EVIDENCE / "recomputed_summary.json").read_text())["action_pairs"]
    summary = json.loads((EVIDENCE / "recomputed_summary.json").read_text())["splits"]["validation"]
    all_link_summary = json.loads((EVIDENCE / "all_links_analysis.json").read_text())[
        "duration_comparisons"]["sequential"]["reference"]
    records = [row for row in load_csv(EVIDENCE / "pair_outcomes.csv")
               if row["split"] == "validation"]
    all_link_list = load_csv(EVIDENCE / "all_links_comparison.csv")
    all_link_rows = {row["case_id"]: row for row in all_link_list}
    require(len(all_link_rows) == len(all_link_list), "Duplicate all-link episodes")
    episode_ids = {row["case_id"] for row in records}
    require(episode_ids == set(all_link_rows), "Validation/all-link episode mismatch")
    require(len(episode_ids) == 1000 and len(pairs) == 10 and len(records) == 10000,
            "Unexpected Validation population")
    record_counts = Counter((row["case_id"], int(row["pair_index"])) for row in records)
    require(all(record_counts[episode, index] == 1 for episode in episode_ids
                for index in range(len(pairs))), "Missing/duplicate candidate record")
    require(all(json.loads(row["pair"]) == pairs[int(row["pair_index"])]
                for row in records), "Inconsistent candidate pair indexing")

    # Verify every Validation outcome against the archived NPZ, rollout CSV,
    # and split assignment. An invalid numerical label counts as unsuccessful;
    # it is not dropped from the denominator.
    split_rows = load_csv(REPOSITORY / EVALUATION / "split.csv")
    validation_ids = {row["case_id"] for row in split_rows if row["split"] == "validation"}
    require(validation_ids == episode_ids, "Original split differs from snapshot")
    raw_result_list = load_csv(REPOSITORY / EVALUATION / "dataset/results.csv")
    raw_results = {(row["case_id"], tuple(json.loads(row["pair"]))): row
                   for row in raw_result_list}
    require(len(raw_results) == len(raw_result_list), "Duplicate raw result rows")
    invalid_labels = 0
    successful_times_checked = 0
    with np.load(REPOSITORY / EVALUATION / "dataset/dataset.npz",
                 allow_pickle=False) as dataset:
        require(dataset["link_pairs"].tolist() == pairs, "Raw NPZ pair order differs")
        npz_indexes = {str(case_id): i for i, case_id in enumerate(dataset["case_id"])}
        for row in records:
            episode, pair_index = row["case_id"], int(row["pair_index"])
            i = npz_indexes[episode]
            raw = raw_results[episode, tuple(pairs[pair_index])]
            valid = bool(dataset["label_valid"][i, pair_index])
            raw_success_label = int(dataset["success"][i, pair_index])
            require(flag(raw["label_valid"]) == valid, "CSV/NPZ validity mismatch")
            require(flag(row["success"]) == flag(raw["success"]) ==
                    (valid and raw_success_label == 1), "Raw success mismatch")
            if not valid:
                require(raw_success_label == -1 and not flag(row["success"]),
                        "Invalid numerical label was not counted as failure")
                invalid_labels += 1
            if flag(row["success"]):
                close(success_time(row, "completion_time"),
                      success_time(raw, "completion_time"), "raw successful time")
                successful_times_checked += 1

    successes = sum(flag(row["success"]) for row in records)
    all_link_successes = sum(flag(row["sequential_120_success"])
                            for row in all_link_list)
    require(all_link_successes == all_link_summary["successes"] == 590,
            "All-link 120-second success count changed")
    require(invalid_labels == summary["numerical_errors"] == 40,
            "Validation invalid-label count changed")
    close(successes / len(records), summary["uniform_expected_success_rate"],
          "existing uniform expected success rate")
    pooled, common = paired_statistics(records, all_link_rows)
    per_pair = []
    for index, pair in enumerate(pairs):
        subset = [row for row in records if int(row["pair_index"]) == index]
        pair_successes = sum(flag(row["success"]) for row in subset)
        require(pair_successes == summary["pair_successes"][index],
                "Existing per-pair success count changed")
        paired, _ = paired_statistics(subset, all_link_rows)
        per_pair.append({
            "pair_index": index, "pair": pair, "episodes": len(subset),
            "successes": pair_successes,
            "success_rate_percent": 100 * pair_successes / len(subset),
            "paired_common_success": paired,
            "pooled_common_time_weight": paired["both_success_records"] /
                pooled["both_success_records"],
        })

    macro_reference = mean(p["paired_common_success"]["all_links_mean_seconds"]
                           for p in per_pair)
    macro_candidate = mean(p["paired_common_success"]["two_link_mean_seconds"]
                           for p in per_pair)
    fixed = next(p for p in per_pair if p["pair"] == [3, 5])
    old_fixed = summary["fixed35_vs_all_links"]
    old_to_new = {
        "both_success": "both_success_records",
        "reference_only": "reference_only_success_records",
        "candidate_only": "candidate_only_success_records",
        "both_failure": "both_failure_records",
        "reference_common_mean": "all_links_mean_seconds",
        "candidate_common_mean": "two_link_mean_seconds",
        "mean_time_difference": "mean_paired_difference_seconds",
    }
    for old_key, new_key in old_to_new.items():
        close(fixed["paired_common_success"][new_key], old_fixed[old_key],
              f"existing (3,5) {old_key}")
    close(fixed["success_rate_percent"] - 100 * all_link_successes / len(episode_ids),
          old_fixed["difference_pp"], "existing (3,5) success difference")

    all_link_common_multiplicity = Counter(row["case_id"] for row in common)
    audit = {
        "description": "All ten two-link candidates on the archived Validation episodes",
        "split": "validation",
        "controller_horizon_seconds": 120,
        "no_new_simulation": True,
        "no_new_rollouts": True,
        "no_new_training": True,
        "no_random_pair_draws": True,
        "input_sources": checked_sources,
        "raw_sources": raw_sources,
        "provenance": {"path": str(provenance_path), "sha256": sha256(provenance_path)},
        "analysis_script": {"path": "scripts/analyze_two_link_average.py",
                            "sha256": sha256(Path(__file__))},
        "population": {
            "episodes": len(episode_ids), "candidate_pairs": len(pairs),
            "episode_pair_outcomes": len(records), "action_pairs": pairs,
            "numerical_errors_included_as_unsuccessful": invalid_labels,
        },
        "success": {
            "all_links_successes": all_link_successes,
            "all_links_episode_denominator": len(episode_ids),
            "all_links_success_rate_percent": 100 * all_link_successes / len(episode_ids),
            "two_link_successes": successes,
            "two_link_episode_pair_denominator": len(records),
            "two_link_success_rate_percent": 100 * successes / len(records),
            "two_link_minus_all_links_percentage_points":
                100 * (successes / len(records) - all_link_successes / len(episode_ids)),
            "estimand": "Mean success over uniformly weighted episodes and all ten pairs; "
                "the empirical expected success of choosing one candidate uniformly per episode.",
        },
        "pooled_common_success": pooled,
        "pooled_time_interpretation": {
            "prior": "Each of the 1000 episodes and each of its ten candidate pairs has "
                "equal weight before conditioning on joint success.",
            "conditioning": "Retain episode--pair records where both All-link Sequential "
                "and that two-link candidate succeed within 120 seconds.",
            "means": "E[T_all | S_all=1,S_pair=1] and E[T_pair | S_all=1,S_pair=1] "
                "under the uniform episode--pair prior.",
            "pair_weights_after_conditioning": "Proportional to each pair's common-success "
                "count, rather than equal across the ten pairs.",
            "episode_weights_after_conditioning": "Proportional to the number of a given "
                "episode's two-link candidates that jointly succeed with All-link; "
                "the same All-link completion time is repeated for each such pair.",
            "not_independent_episodes": True,
            "uncertainty_if_added": "Resample/cluster by episode, not by episode--pair record.",
            "not_an_unconditional_runtime_claim": "Failures are excluded from paired "
                "completion-time means. The time reduction does not imply a shorter mean "
                "runtime over all attempts or a higher success rate.",
            "common_episode_multiplicity_histogram": {
                str(count): number for count, number in
                sorted(Counter(all_link_common_multiplicity.values()).items())},
        },
        "per_pair": per_pair,
        "equal_pair_macro_common_success": {
            "all_links_mean_seconds": macro_reference,
            "two_link_mean_seconds": macro_candidate,
            "mean_paired_difference_seconds": macro_candidate - macro_reference,
            "relative_reduction_of_macro_means_percent":
                100 * (1 - macro_candidate / macro_reference),
            "mean_of_per_pair_relative_reductions_percent": mean(
                p["paired_common_success"]["relative_reduction_percent"] for p in per_pair),
            "definition": "Give each pair equal weight AFTER computing its own conditional "
                "common-success means. This differs from conditioning the uniform pair "
                "policy on common success and is not the selected slide estimand.",
        },
        "existing_fixed_3_5_reproduction": {
            "pair_index": fixed["pair_index"],
            "successes": fixed["successes"],
            "success_rate_percent": fixed["success_rate_percent"],
            "paired_common_success": fixed["paired_common_success"],
            "all_existing_counts_and_means_match": True,
        },
        "recommended_slide_labels": {
            "method": "Two-link average (all 10 pairs)",
            "success": "59.00% vs. 54.73%",
            "time": "Paired mean time: 46.99 s to 35.05 s (25.41% lower)",
            "time_sample": "4,545 common-success episode--pair outcomes (577 episodes)",
        },
        "verification": {
            "all_snapshot_and_original_sha256_checked": True,
            "all_raw_npz_csv_split_hashes_match_archived_provenance": True,
            "validation_episode_ids_match_all_links_and_original_split": True,
            "each_episode_has_each_candidate_exactly_once": True,
            "validation_outcomes_checked_against_raw_npz_and_csv": len(records),
            "successful_completion_times_checked_against_raw_csv": successful_times_checked,
            "existing_uniform_mean_reproduced": True,
            "existing_per_pair_success_counts_reproduced": True,
            "existing_fixed_3_5_counts_and_paired_means_reproduced": True,
            "all_links_120_second_source_used_not_150_second_results": True,
        },
    }
    destination = EVIDENCE / "two_link_average_audit.json"
    destination.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"audit": str(destination), "success": audit["success"],
                      "pooled_common_success": pooled,
                      "equal_pair_macro_common_success": audit[
                          "equal_pair_macro_common_success"]}, indent=2))


if __name__ == "__main__":
    main()

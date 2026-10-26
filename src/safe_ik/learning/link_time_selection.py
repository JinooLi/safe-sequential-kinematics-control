"""Softmax selection of successful, fast link pairs using saved simulation time."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .link_selection import (
    LINK_PAIRS,
    OBJECTIVE,
    SELECTION_METRIC,
    LinkTimeSelectionNet,
    load_dataset,
    load_time_checkpoint,
    sha256,
    validate_time_scale,
)


def discounted_rewards(labels, completion_times, time_scale_seconds):
    validate_time_scale(time_scale_seconds)
    labels, times = np.asarray(labels), np.asarray(completion_times)
    if labels.shape != times.shape or labels.ndim != 2 or labels.shape[1] != 10:
        raise ValueError("labels and completion times must have shape [scenes, 10]")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("time rewards require binary effective labels")
    success = labels.astype(bool)
    if not np.isfinite(times[success]).all() or np.any(times[success] < 0):
        raise ValueError("successful completion times must be finite and nonnegative")
    rewards = np.zeros(labels.shape, dtype=np.float64)
    rewards[success] = np.exp(-times[success] / time_scale_seconds)
    return rewards


def time_reward_loss(logits, labels, completion_times, time_scale_seconds=120.0):
    """-log(sum softmax(z)_j y_j exp(-t_j/tau)); all-failure rows have zero loss.

    Use logsumexp directly, never log(softmax()). Failed/unsafe/numerical-error
    times are masked before arithmetic. Average over all scenes, including
    zero-contribution all-failure scenes. Equal successful times reduce to
    successful-mass loss plus a constant; the optimum favors the fastest success.
    """
    validate_time_scale(time_scale_seconds)
    if logits.shape != labels.shape or logits.shape != completion_times.shape:
        raise ValueError(
            "logits, labels and completion times must have matching shapes"
        )
    positive = labels > 0
    has_success = positive.any(dim=1, keepdim=True)
    safe_times = torch.where(positive, completion_times, 0.0)
    log_rewards = (-safe_times / time_scale_seconds).masked_fill(
        ~(positive | ~has_success), -torch.inf
    )
    return (
        torch.logsumexp(logits, dim=1) - torch.logsumexp(logits + log_rewards, dim=1)
    ).mean()


def load_completion_times(data, dataset_directory):
    """Join hashed source CSVs by scene and pair; never use failed elapsed time.

    Combined historical datasets omit timing arrays. Traverse their existing
    provenance tree and validate the leaf arrays and CSV labels against the
    unchanged combined dataset. No dataset files are edited.
    """
    lookup = {str(case): i for i, case in enumerate(data["case_id"])}
    pair_lookup = {pair: i for i, pair in enumerate(LINK_PAIRS)}
    times = np.full(data["labels"].shape, np.nan, dtype=np.float64)
    seen = np.zeros_like(data["labels"], dtype=bool)
    provenance, visited = [], set()

    def visit(directory, expected_complete=None, expected_dataset=None):
        directory = Path(directory).resolve()
        complete_hash = sha256(directory / "complete.json")
        if expected_complete is not None and complete_hash != expected_complete:
            raise ValueError("timing source complete checksum mismatch")
        if directory in visited:
            raise ValueError("duplicate/cyclic timing source")
        visited.add(directory)
        source = load_dataset(directory, expected_dataset)
        complete = json.loads((directory / "complete.json").read_text())
        csv_path = directory / "results.csv"
        if not csv_path.exists():
            sources = complete.get("sources", [])
            if not sources:
                raise ValueError(f"no timing source for {directory}")
            for item in sources:
                visit(
                    item.get("path", item.get("directory")),
                    item.get("complete_sha256"),
                    item.get("dataset_sha256"),
                )
            return
        csv_hash = sha256(csv_path)
        if csv_hash != complete["data_sha256"].get("results.csv"):
            raise ValueError("timing CSV checksum mismatch")
        source_ids = source["case_id"].tolist()
        source_id_set = set(source_ids)
        if any(case not in lookup for case in source_ids):
            raise ValueError("timing source contains an unknown scene")
        positions = np.array([lookup[case] for case in source_ids])
        for key in (
            "initial_q",
            "selected_ik",
            "target_position",
            "target_quaternion",
            "obstacles",
            "success",
            "label_valid",
        ):
            if not np.array_equal(data[key][positions], source[key]):
                raise ValueError(f"timing source does not match input/labels: {key}")
        count = 0
        with csv_path.open() as handle:
            for row in csv.DictReader(handle):
                case = row["case_id"]
                if case not in source_id_set:
                    raise ValueError("timing CSV contains an unknown source scene")
                i, j = lookup[case], pair_lookup[tuple(json.loads(row["pair"]))]
                if seen[i, j]:
                    raise ValueError("duplicate scene/pair timing")
                if row["label_valid"] not in ("True", "False") or row[
                    "success"
                ] not in ("True", "False"):
                    raise ValueError("invalid timing CSV outcome encoding")
                valid, success = row["label_valid"] == "True", row["success"] == "True"
                if valid != bool(data["label_valid"][i, j]) or success != bool(
                    data["labels"][i, j]
                ):
                    raise ValueError("timing CSV outcome mismatch")
                if int(row["candidate_switch_count"]) != 0:
                    raise ValueError("timing labels must use one fixed IK")
                if success:
                    t = float(row["completion_time"])
                    if (
                        not np.isfinite(t)
                        or not 0 <= t <= 120.0 + 1e-6
                        or row["reason"] != "goal_reached"
                        or row["goal_reached"] != "True"
                        or row["interpolated_success"] != "True"
                        or not np.isclose(t, float(row["elapsed"]), atol=1e-8, rtol=0)
                    ):
                        raise ValueError(
                            "invalid successful simulation completion time"
                        )
                    times[i, j] = t
                seen[i, j] = True
                count += 1
        if count != len(source_ids) * 10:
            raise ValueError("incomplete timing CSV")
        provenance.append(
            dict(
                directory=str(directory),
                scenes=len(source_ids),
                complete_sha256=complete_hash,
                dataset_sha256=source["dataset_sha256"],
                results_csv_sha256=csv_hash,
            )
        )

    visit(dataset_directory, expected_dataset=data["dataset_sha256"])
    if not seen.all():
        raise ValueError("missing completion-time scene/pair records")
    discounted_rewards(data["labels"], times, 120.0)
    return times, provenance


def timing_metrics(labels, times, probabilities, time_scale_seconds=120.0):
    rewards = discounted_rewards(labels, times, time_scale_seconds)
    p = np.asarray(probabilities)
    if p.shape != rewards.shape or not np.isfinite(p).all():
        raise ValueError("invalid prediction array")
    choices = p.argmax(axis=1)
    row = np.arange(len(choices))
    successes = np.asarray(labels)[row, choices].astype(bool)
    chosen_times = np.asarray(times)[row, choices]
    oracle_times = np.where(labels, times, np.inf).min(axis=1)
    selected_times = chosen_times[successes]
    return dict(
        scenes=len(choices),
        selected_successes=int(successes.sum()),
        selected_success_rate=float(successes.mean()),
        selected_time_reward=float(rewards[row, choices].mean()),
        success_time_mean_seconds=(
            float(selected_times.mean()) if successes.any() else None
        ),
        success_time_median_seconds=(
            float(np.median(selected_times)) if successes.any() else None
        ),
        failure_penalized_120s_cost=float(
            np.where(successes, chosen_times, 120.0).mean()
        ),
        fastest_success_choices=int(
            (
                successes & np.isclose(chosen_times, oracle_times, atol=1e-6, rtol=0)
            ).sum()
        ),
        oracle_successes=int(np.isfinite(oracle_times).sum()),
        selection_counts=np.bincount(choices, minlength=10).tolist(),
    )

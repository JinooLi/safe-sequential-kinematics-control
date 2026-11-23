"""Train and evaluate the selected one-head Panda link-combination model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import yaml
from scipy.stats import binomtest
from torch.nn import functional as F

from safe_ik.domain.link_actions import (
    LINK_ACTIONS,
    LINK_PAIRS,
    action_id,
    catalog_arrays,
    catalog_metadata,
    validate_actions,
)
from safe_ik.learning.link_selection import (
    MODEL_SPEC,
    LinkSuccessNet,
    joint_statistics,
    load_dataset,
    load_model,
    load_scene_split,
    make_state,
    sha256,
)

SELECTION_METRIC = "validation_selected_success_rate"


def successful_mass_loss(logits, targets):
    """Mean -log softmax mass on successes; all-failure scenes contribute zero.

    Average over all scenes, including zero contributions, so lambda has a fixed
    per-scene meaning. A full mask for all-failure rows prevents logsumexp(-inf)
    and its undefined gradient. All-success rows also have exactly zero loss.
    """
    positive = targets > 0
    safe_mask = positive | ~positive.any(dim=1, keepdim=True)
    return (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(logits.masked_fill(~safe_mask, -torch.inf), dim=1)
    ).mean()


def loss_components(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    selection = successful_mass_loss(logits, targets)
    return bce + 3.0 * selection, bce, selection


def objective_metadata(settings):
    return {
        "optimization_loss": "BCE + 3 * negative_log_successful_softmax_mass",
        "bce_loss_weight": 1.0,
        "selection_loss_weight": 3.0,
        "selection_loss_reduction": "mean_over_all_scenes; all_failure_rows_zero",
        "optimizer": "adamw",
        "weight_decay": settings["weight_decay"],
    }


def make_optimizer(model, settings):
    validate_training_settings(settings, internal=True)
    return torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )


def checkpoint_rank(validation_bce, validation_successes):
    return (-validation_successes, validation_bce)


def select_run(runs):
    return min(
        runs,
        key=lambda run: (
            *checkpoint_rank(
                run["best_validation_bce"], run["best_validation_selected_successes"]
            ),
            run["seed"],
        ),
    )


def selection_description():
    return "maximum validation selection success; ties use minimum validation BCE"


def validate_training_settings(training, *, internal=False):
    allowed = {
        "seeds",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
    }
    if internal:
        allowed.add("dataset_sha256")
    unknown = set(training) - allowed
    if unknown:
        raise ValueError(
            f"unsupported training settings for the selected model: {sorted(unknown)}"
        )
    if not internal or "seeds" in training:
        seeds = training.get("seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("training seeds must be distinct integers in [0, 2**32)")
    for key in ("batch_size", "max_epochs", "patience"):
        if type(training.get(key)) is not int or training[key] <= 0:
            raise ValueError(f"invalid {key}")
    value = training.get("learning_rate")
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise ValueError("invalid learning_rate")
    value = training.get("weight_decay")
    if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
        raise ValueError("invalid weight_decay")


def validate_settings(settings):
    allowed = {"dataset_sha256", "numerical_error_policy", "split", "training"}
    if not isinstance(settings, dict) or set(settings) != allowed:
        raise ValueError(
            "config must contain only dataset_sha256, numerical_error_policy, split and training; model and objective are fixed"
        )
    digest = settings["dataset_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("invalid dataset_sha256")
    if settings["numerical_error_policy"] != "failure":
        raise ValueError("this model labels numerical errors as failures")
    split = settings["split"]
    if not isinstance(split, dict) or set(split) - {
        "train",
        "validation",
        "test",
        "seed",
        "file",
        "sha256",
    }:
        raise ValueError("invalid split settings")
    for key in ("train", "validation", "test"):
        if type(split.get(key)) is not int or split[key] < (0 if key == "test" else 1):
            raise ValueError(f"invalid split {key}")
    if type(split.get("seed")) is not int or split["seed"] < 0:
        raise ValueError("invalid split seed")
    if ("file" in split) != ("sha256" in split):
        raise ValueError("frozen split requires file and sha256 together")
    if not isinstance(settings["training"], dict):
        raise ValueError("invalid training settings")
    validate_training_settings(settings["training"])


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_metrics(
    labels, probabilities, fixed_pair, bootstrap_seed=0, *, actions=None
):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.shape != probabilities.shape or labels.ndim != 2 or labels.shape[1] == 0:
        raise ValueError("expected matching [scenes, actions] arrays")
    if actions is not None:
        actions = validate_actions(actions)
        if len(actions) != labels.shape[1]:
            raise ValueError("metric action catalog/label dimension mismatch")
    if (
        not np.isin(labels, [0, 1]).all()
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("invalid metric inputs")
    chosen = probabilities.argmax(axis=1)
    selected = labels[np.arange(len(labels)), chosen]
    baseline = labels[:, fixed_pair]
    differences = selected - baseline
    wins = int(((selected == 1) & (baseline == 0)).sum())
    losses = int(((selected == 0) & (baseline == 1)).sum())
    clipped = probabilities.astype(np.float64).clip(1e-7, 1 - 1e-7)
    rng = np.random.default_rng(bootstrap_seed)
    samples = rng.integers(len(labels), size=(5000, len(labels)))
    ci = np.quantile(differences[samples].mean(axis=1), [0.025, 0.975]).tolist()
    bins = []
    for low in np.linspace(0, 0.8, 5):
        high = float(low + 0.2)
        mask = (probabilities >= low) & (
            (probabilities < high) if high < 0.999 else (probabilities <= 1)
        )
        bins.append(
            {
                "lower": float(low),
                "upper": high,
                "count": int(mask.sum()),
                "mean_prediction": (
                    float(probabilities[mask].mean()) if mask.any() else None
                ),
                "observed_success_rate": (
                    float(labels[mask].mean()) if mask.any() else None
                ),
            }
        )
    metrics = {
        "scenes": len(labels),
        "selected_successes": int(selected.sum()),
        "selected_success_rate": float(selected.mean()),
        "fixed_action_successes": int(baseline.sum()),
        "fixed_action_success_rate": float(baseline.mean()),
        "random_selection_expected_success_rate": float(labels.mean()),
        "oracle_successes": int(labels.max(axis=1).sum()),
        "oracle_success_rate": float(labels.max(axis=1).mean()),
        "paired_wins": wins,
        "paired_losses": losses,
        "delta_vs_fixed": float(differences.mean()),
        "paired_bootstrap_95_ci": ci,
        "mcnemar_exact_p_exploratory": (
            float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
        ),
        "bce": float(
            -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean()
        ),
        "brier": float(((probabilities - labels) ** 2).mean()),
        "threshold_0_5_accuracy": float(((probabilities >= 0.5) == labels).mean()),
        "selection_counts": np.bincount(chosen, minlength=labels.shape[1]).tolist(),
        "calibration_bins": bins,
    }
    if actions is not None:
        counts = np.asarray([len(links) for links in actions])[chosen]
        metrics["selection_by_intermediate_count"] = {
            str(count): {
                "selected_scenes": int((counts == count).sum()),
                "successes": int(selected[counts == count].sum()),
                "success_rate": (
                    float(selected[counts == count].mean())
                    if np.any(counts == count)
                    else None
                ),
            }
            for count in sorted({len(links) for links in actions})
        }
        metrics["fixed_action_success_rates"] = {
            action_id(links): float(labels[:, i].mean())
            for i, links in enumerate(actions)
        }
        if actions == LINK_ACTIONS:
            pair_count = len(LINK_PAIRS)
            metrics["new_actions_rescue_scenes"] = int(
                (
                    (labels[:, :pair_count].max(axis=1) == 0)
                    & (labels[:, pair_count:].max(axis=1) == 1)
                ).sum()
            )
    # Keep legacy consumers of ten-pair reports working.
    if actions is None or actions == LINK_PAIRS:
        metrics["fixed_pair_successes"] = metrics["fixed_action_successes"]
        metrics["fixed_pair_success_rate"] = metrics["fixed_action_success_rate"]
    return metrics


def train_seed(
    output,
    state,
    obstacles,
    targets,
    split,
    mean,
    std,
    settings,
    seed,
    fixed_pair,
    device,
    *,
    actions=LINK_PAIRS,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    actions = validate_actions(actions)
    if targets.shape[1] != len(actions):
        raise ValueError("training label/action dimension mismatch")
    model = LinkSuccessNet(mean, std, actions=actions).to(device)
    optimizer = make_optimizer(model, settings)
    batch_rng = np.random.default_rng(seed)
    destination = output / f"seed-{seed}"
    destination.mkdir()
    metric = SELECTION_METRIC
    objective = objective_metadata(settings)
    best_rank = None
    best_loss, best_successes, best_epoch, best_state, stale = (
        float("inf"),
        -1,
        0,
        None,
        0,
    )
    history = []
    started = time.perf_counter()
    validation = torch.as_tensor(split["validation"], device=device)
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        loss_sum = bce_sum = selection_sum = 0.0
        permutation = batch_rng.permutation(split["train"])
        for start in range(0, len(permutation), settings["batch_size"]):
            index = torch.as_tensor(
                permutation[start : start + settings["batch_size"]], device=device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(state[index], obstacles[index])
            loss, bce, selection = loss_components(logits, targets[index])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite training loss at seed {seed}, epoch {epoch}"
                )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            bce_sum += float(bce.detach()) * len(index)
            selection_sum += float(selection.detach()) * len(index)
        model.eval()
        with torch.no_grad():
            logits = model(state[validation], obstacles[validation])
            validation_total, validation_bce, validation_selection = loss_components(
                logits, targets[validation]
            )
            validation_loss = float(validation_bce)
            choice = logits.sigmoid().argmax(dim=1)
            validation_successes = int(
                targets[validation].gather(1, choice[:, None]).sum()
            )
            selection_success = validation_successes / len(validation)
        if not all(
            (
                torch.isfinite(x)
                for x in (validation_total, validation_bce, validation_selection)
            )
        ):
            raise FloatingPointError("nonfinite validation loss")
        rank = checkpoint_rank(validation_loss, validation_successes)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_loss, best_successes, best_epoch, stale = (
                validation_loss,
                validation_successes,
                epoch,
                0,
            )
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_bce": bce_sum / len(permutation),
            "train_selection_loss": selection_sum / len(permutation),
            "train_total_loss": loss_sum / len(permutation),
            "validation_bce": validation_loss,
            "validation_selection_loss": float(validation_selection),
            "validation_total_loss": float(validation_total),
            "validation_selected_successes": validation_successes,
            "validation_selected_success_rate": selection_success,
        }
        history.append(row)
        write_json(
            output / "progress.json",
            {
                "status": "training",
                "seed": seed,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_bce": best_loss,
                "best_validation_selected_successes": best_successes,
                "selection_metric": metric,
                **objective,
                "updated_utc": utc_now(),
            },
        )
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"seed={seed} epoch={epoch} train_bce={row['train_bce']:.5f} val_bce={validation_loss:.5f} val_selection={selection_success:.3f}",
                flush=True,
            )
        if stale >= settings["patience"]:
            break
    write_csv(destination / "history.csv", history)
    checkpoint = {
        "schema_version": 17,
        "model": MODEL_SPEC,
        "state_dict": best_state,
        **catalog_metadata(actions),
        "output_dim": len(actions),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_bce": best_loss,
        "best_validation_selected_successes": best_successes,
        "best_validation_selected_success_rate": best_successes / len(validation),
        "selection_metric": metric,
        **objective,
        "input_encoding": "q_zscore_7,ik_zscore_7,goal_position_metres_3,rotation_first_two_columns_6",
        "obstacle_encoding": "world_xyz_radius_metres_raw_shared_mlp_cross_attention",
        "numerical_error_policy": "failure",
        "dataset_sha256": settings["dataset_sha256"],
    }
    torch.save(checkpoint, destination / "best.pt")
    result = {
        "seed": seed,
        "epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_bce": best_loss,
        "wall_seconds": time.perf_counter() - started,
        "best_validation_selected_successes": best_successes,
        "best_validation_selected_success_rate": best_successes / len(validation),
        "minimum_validation_bce": min((row["validation_bce"] for row in history)),
        "selection_metric": metric,
        **objective,
        "checkpoint": str((destination / "best.pt").relative_to(output)),
        "mlp_dropout": 0.1,
        "mlp_dropout_scope": "all_hidden",
    }
    write_json(destination / "training.json", result)
    print(
        f"seed={seed} finished epochs={len(history)} best_epoch={best_epoch} val_success={best_successes}/{len(validation)} val_bce={best_loss:.5f} selection={metric}",
        flush=True,
    )
    return result


def write_report(output, report):
    has_test = report["test_evaluation_performed"]
    rows = [
        "# Panda selected link-combination model",
        "",
        f"Robot MLP 23→64→64; obstacle MLP 4→32→32; one-head cross-attention (32D); context projection 32→64; concatenated prediction MLP 128→64→{report['output_dim']}.",
        f"{report['parameter_count']:,} parameters. ReLU and dropout 0.1 at all five hidden MLP sites; dropout disabled for evaluation.",
        f"AdamW with weight_decay={report['weight_decay']:g}; BCE + 3 × negative log successful softmax mass, averaged across all scenes. Numerical errors remain failures.",
        "Only 14 joint-angle inputs are standardized using training statistics; geometry stays in metres.",
        f"Checkpoint selection: {selection_description()}; exact ties retain earlier epoch/lower seed.",
        (
            "All choices are frozen before test evaluation."
            if has_test
            else "No test set was requested or evaluated. Validation is used for checkpoint/seed selection; its score is not an independent test estimate."
        ),
        f"Selected seed: {report['selected_seed']}; fixed action from training labels: {report['fixed_action_id']}.",
        "",
        (
            "| Seed | Best epoch | Validation success | Test success | Test BCE |"
            if has_test
            else "| Seed | Best epoch | Validation success | Validation BCE |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: |"
            if has_test
            else "| --- | ---: | ---: | ---: |"
        ),
    ]
    for run in report["runs"]:
        v = run["metrics"]["validation"]
        if has_test:
            t = run["metrics"]["test"]
            rows.append(
                f"| {run['seed']} | {run['best_epoch']} | {v['selected_success_rate']:.2%} | {t['selected_success_rate']:.2%} | {t['bce']:.4f} |"
            )
        else:
            rows.append(
                f"| {run['seed']} | {run['best_epoch']} | {v['selected_success_rate']:.2%} | {v['bce']:.4f} |"
            )
    rows += [
        "",
        "Seeds share the same evaluation scenes. Results and paired intervals are exploratory; no multiplicity correction. This report does not deploy a controller.",
        "",
        "![Learning curves](learning_curves.png)",
    ]
    (output / "report.md").write_text("\n".join(rows) + "\n")


def plot_history(output, runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for run in runs:
        with (output / f"seed-{run['seed']}" / "history.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        epochs = [int(row["epoch"]) for row in rows]
        axes[0].plot(
            epochs, [float(row["train_bce"]) for row in rows], label=str(run["seed"])
        )
        axes[1].plot(
            epochs,
            [float(row["validation_bce"]) for row in rows],
            label=str(run["seed"]),
        )
    for axis, title in zip(axes, ["Training BCE", "Validation BCE"]):
        axis.set(xlabel="Epoch", ylabel="BCE", title=title)
        axis.grid(alpha=0.25)
        axis.legend(title="Seed")
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)


def run(dataset_directory, config_path, output, device_name):
    started = time.perf_counter()
    settings = yaml.safe_load(Path(config_path).read_text())
    validate_settings(settings)
    metric = SELECTION_METRIC
    objective = objective_metadata(settings["training"])
    data = load_dataset(dataset_directory, settings["dataset_sha256"])
    labels = data["labels"]
    actions = data["actions"]
    specification = settings["split"]
    split = load_scene_split(data, specification, dataset_directory)
    has_test = "test" in split
    split_policy = (
        "frozen_scene_ids"
        if "file" in specification
        else "stratified_successful_action_count"
    )
    state = make_state(
        data["initial_q"],
        data["selected_ik"],
        data["target_position"],
        data["target_quaternion"],
    )
    mean, std = joint_statistics(state, split["train"])
    fixed_pair = int(labels[split["train"]].mean(axis=0).argmax())
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda") or (
        device.type == "cuda" and (not torch.cuda.is_available())
    ):
        raise RuntimeError("requested device unavailable; no implicit fallback")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        provenance = {
            "created_utc": utc_now(),
            "dataset_directory": str(Path(dataset_directory).resolve()),
            "dataset_sha256": data["dataset_sha256"],
            "dataset_sources": data["dataset_sources"],
            "split_policy": split_policy,
            "evaluation_splits": list(split),
            "test_evaluation_performed": has_test,
            "settings": settings,
            "model": MODEL_SPEC,
            **catalog_metadata(actions),
            "output_dim": len(actions),
            "source_sha256": {
                "train_link_selection.py": sha256(__file__),
                "link_selection.py": sha256(
                    Path(__file__).parents[1] / "learning/link_selection.py"
                ),
                "link_actions.py": sha256(
                    Path(__file__).parents[1] / "domain/link_actions.py"
                ),
            },
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
            },
            "normalization": {
                "joint_mean": mean.tolist(),
                "joint_std": std.tolist(),
                "joint_method": "train_only_per_column_zscore_ddof0",
                "goal_position": "raw_metres",
                "obstacles": "raw_metres",
                "rotation": "first_two_columns",
            },
            "numerical_error_labels_reclassified": int((~data["label_valid"]).sum()),
            "numerical_error_scenes": int((~data["label_valid"]).any(axis=1).sum()),
            "all_fail_scenes": int((labels.sum(axis=1) == 0).sum()),
            "all_success_scenes": int((labels.sum(axis=1) == len(actions)).sum()),
            "parameter_count": sum(
                parameter.numel()
                for parameter in LinkSuccessNet(actions=actions).parameters()
            ),
            "selection_policy": "argmax; ties use lowest index; no IK returns Direct",
            "selection_metric": metric,
            "checkpoint_selection": selection_description()
            + "; exact ties keep earlier epoch/lower seed"
            + (
                "; all seeds frozen before test evaluation"
                if has_test
                else "; no test set"
            ),
            "early_stopping_selection": selection_description(),
            **objective,
        }
        write_json(output / "manifest.json", provenance)
        shutil.copyfile(config_path, output / "training_config.yaml")
        split_record = {
            key: {
                "indices": indices.tolist(),
                "case_ids": data["case_id"][indices].tolist(),
                "successful_action_count_histogram": np.bincount(
                    labels[indices].sum(axis=1).astype(int), minlength=len(actions) + 1
                ).tolist(),
                "numerical_error_labels": int((~data["label_valid"][indices]).sum()),
            }
            for key, indices in split.items()
        }
        write_json(output / "split.json", split_record)
        np.savez_compressed(
            output / "effective_labels.npz",
            case_id=data["case_id"],
            labels=labels.astype(np.int8),
            original_label_valid=data["label_valid"],
            **catalog_arrays(actions),
        )
        tensors = [
            torch.as_tensor(array, dtype=torch.float32, device=device)
            for array in [state, data["obstacles"], labels]
        ]
        training = {**settings["training"], "dataset_sha256": data["dataset_sha256"]}
        runs = [
            train_seed(
                output,
                *tensors,
                split,
                mean,
                std,
                training,
                seed,
                fixed_pair,
                device,
                actions=actions,
            )
            for seed in training["seeds"]
        ]
        selected = select_run(runs)
        write_json(
            output / "selection.json",
            {
                "selected_seed": selected["seed"],
                "checkpoint": selected["checkpoint"],
                "criterion": metric,
                "tie_breakers": ["validation_bce", "seed"],
                "validation_selected_successes": selected[
                    "best_validation_selected_successes"
                ],
                "validation_bce": selected["best_validation_bce"],
                "test_evaluation_performed": has_test,
                **({"selected_before_test_evaluation": True} if has_test else {}),
                "selected_utc": utc_now(),
            },
        )
        prediction_rows = []
        for trained in runs:
            model = load_model(output / trained["checkpoint"], device)
            with torch.no_grad():
                logits = model(tensors[0], tensors[1])
                probabilities = logits.sigmoid().cpu().numpy()
                selection_probabilities = logits.softmax(dim=1).cpu().numpy()
            trained["metrics"] = {
                key: prediction_metrics(
                    labels[index],
                    probabilities[index],
                    fixed_pair,
                    specification["seed"],
                    actions=actions,
                )
                for key, index in split.items()
            }
            with torch.no_grad():
                for key, index in split.items():
                    total, bce, selection = loss_components(
                        logits[index], tensors[2][index]
                    )
                    trained["metrics"][key].update(
                        selection_loss=float(selection),
                        total_loss=float(total),
                        bce_from_logits=float(bce),
                    )
            np.savez_compressed(
                output / f"seed-{trained['seed']}" / "predictions.npz",
                case_id=data["case_id"],
                probabilities=probabilities,
                selection_probabilities=selection_probabilities,
                chosen_action_index=probabilities.argmax(axis=1),
                **catalog_arrays(actions),
                **(
                    {"chosen_pair_index": probabilities.argmax(axis=1)}
                    if actions == LINK_PAIRS
                    else {}
                ),
            )
            write_json(
                output / f"seed-{trained['seed']}" / "metrics.json", trained["metrics"]
            )
            for key, indices in split.items():
                for index in indices:
                    choice = int(probabilities[index].argmax())
                    prediction_rows.append(
                        {
                            "seed": trained["seed"],
                            "split": key,
                            "case_id": data["case_id"][index],
                            "selected_action_index": choice,
                            "selected_action_id": action_id(actions[choice]),
                            "selected_method": (
                                "sequential" if actions[choice] else "direct"
                            ),
                            "intermediate_count": len(actions[choice]),
                            "selected_success": int(labels[index, choice]),
                            "selected_source_numerical_error": bool(
                                not data["label_valid"][index, choice]
                            ),
                            "fixed_action_success": int(labels[index, fixed_pair]),
                            "oracle_success": int(labels[index].max()),
                            **{
                                f"p_{action_id(links)}": float(probabilities[index, k])
                                for k, links in enumerate(actions)
                            },
                        }
                    )
        write_csv(output / "predictions.csv", prediction_rows)
        baselines = {}
        train_rates = labels[split["train"]].mean(axis=0)
        for key, indices in split.items():
            baselines[key] = prediction_metrics(
                labels[indices],
                np.broadcast_to(train_rates, (len(indices), len(actions))),
                fixed_pair,
                specification["seed"],
                actions=actions,
            )
        test_summary = {}
        if has_test:
            test_rates = [r["metrics"]["test"]["selected_success_rate"] for r in runs]
            test_summary = {
                "mean_test_selected_success_rate": float(np.mean(test_rates)),
                "std_test_selected_success_rate_across_seeds": (
                    float(np.std(test_rates, ddof=1)) if len(runs) > 1 else 0.0
                ),
            }
        report = {
            **objective,
            "model": MODEL_SPEC,
            **catalog_metadata(actions),
            "output_dim": len(actions),
            "selection_metric": metric,
            "split_policy": split_policy,
            "evaluation_splits": list(split),
            "test_evaluation_performed": has_test,
            "parameter_count": provenance["parameter_count"],
            "selected_seed": selected["seed"],
            "fixed_action_index": fixed_pair,
            "fixed_action_id": action_id(actions[fixed_pair]),
            **(
                {
                    "fixed_pair_index": fixed_pair,
                    "fixed_pair": list(actions[fixed_pair]),
                }
                if actions == LINK_PAIRS
                else {}
            ),
            "baselines": baselines,
            "runs": runs,
            "mean_validation_selected_success_rate": float(
                np.mean(
                    [r["metrics"]["validation"]["selected_success_rate"] for r in runs]
                )
            ),
            **test_summary,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(output / "evaluation.json", report)
        write_report(output, report)
        plot_history(output, runs)
        write_json(
            output / "progress.json",
            {
                "status": "completed",
                "runs": len(runs),
                "selection_metric": metric,
                **objective,
                "selected_seed": selected["seed"],
                "test_evaluation_performed": has_test,
                "finished_utc": utc_now(),
            },
        )
        hashes = {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        write_json(
            output / "complete.json",
            {
                "runs": len(runs),
                "scenes": len(labels),
                "selection_metric": metric,
                **objective,
                "labels": int(labels.size),
                **catalog_metadata(actions),
                "numerical_error_policy": "failure",
                "test_evaluation_performed": has_test,
                "selected_seed": selected["seed"],
                "finished_utc": utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "data_sha256": hashes,
            },
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "selected_seed": selected["seed"],
                    "validation": selected["metrics"]["validation"],
                    "test_evaluation_performed": has_test,
                    **({"test": selected["metrics"]["test"]} if has_test else {}),
                    "output": str(output),
                }
            ),
            flush=True,
        )
        return report
    except BaseException as error:
        write_json(output / "failed.json", {"error": repr(error), "at_utc": utc_now()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.dataset, args.config, args.output, args.device)


if __name__ == "__main__":
    main()

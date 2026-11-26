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
from safe_ik.learning.ik_target_features import (
    compute_ik_target_features,
    make_fk_geometry,
)
from safe_ik.learning.link_selection import (
    LinkSuccessNet,
    joint_statistics,
    load_dataset,
    load_model,
    load_scene_split,
    make_state,
    model_spec,
    sha256,
    validate_bce_loss_weight,
    validate_mlp_dropout,
    validate_negative_bce_weight,
    validate_obstacle_context,
    validate_selection_loss_weight,
)
from safe_ik.learning.relative_geometry import (
    RelativeGeometrySuccessNet,
    relative_model_spec,
    validate_query_conditioning,
)
from safe_ik.learning.relative_geometry_features import (
    compute_relative_geometry_features,
    make_relative_geometry,
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


def weighted_binary_loss(logits, targets, beta, raw_bce):
    """Binary diagnostic before its objective coefficient, including beta."""
    if beta == 1.0:
        return raw_bce
    elements = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (elements * (targets + beta * (1 - targets))).mean()


def loss_components(
    logits,
    targets,
    *,
    selection_loss_weight=3.0,
    negative_bce_weight=1.0,
    bce_loss_weight=1.0,
):
    """Return optimization loss, raw BCE and the selection diagnostic.

    Beta scales only failure labels. Reduce over all scene/action entries,
    without dividing by the weight sum, to implement the stated cost exactly.
    Raw BCE remains comparable across beta values and breaks checkpoint ties.
    """
    weight = validate_selection_loss_weight(selection_loss_weight)
    bce_weight = validate_bce_loss_weight(bce_loss_weight)
    if bce_weight == 0 and weight == 0:
        raise ValueError(
            "bce_loss_weight and selection_loss_weight cannot both be zero"
        )
    beta = validate_negative_bce_weight(negative_bce_weight)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    weighted_bce = weighted_binary_loss(logits, targets, beta, bce)
    selection = successful_mass_loss(logits, targets)
    return bce_weight * weighted_bce + weight * selection, bce, selection


def objective_metadata(settings):
    weight = validate_selection_loss_weight(settings.get("selection_loss_weight", 3.0))
    beta = validate_negative_bce_weight(settings.get("negative_bce_weight", 1.0))
    bce_weight = validate_bce_loss_weight(settings.get("bce_loss_weight", 1.0))
    if bce_weight == 0 and weight == 0:
        raise ValueError(
            "bce_loss_weight and selection_loss_weight cannot both be zero"
        )
    binary_name = "BCE" if beta == 1.0 else f"negative_weighted_BCE(beta={beta:g})"
    binary_term = (
        binary_name if bce_weight == 1.0 else f"{bce_weight:g} * {binary_name}"
    )
    return {
        "optimization_loss": f"{binary_term} + {weight:g} * negative_log_successful_softmax_mass",
        "bce_loss_weight": bce_weight,
        "negative_bce_weight": beta,
        "bce_reduction": "mean_over_all_scene_action_entries; not_weight_normalized",
        "sigmoid_score_semantics": (
            "ordinal_selection_score"
            if bce_weight == 0
            else (
                "success_probability_estimate"
                if beta == 1.0
                else "cost_weighted_success_score"
            )
        ),
        "selection_loss_weight": weight,
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


def stopping_metadata(settings):
    enabled = settings.get("early_stopping", False)
    return {
        "early_stopping": enabled,
        "max_epochs": settings["max_epochs"],
        "early_stopping_patience": settings["patience"] if enabled else None,
    }


def validate_training_settings(training, *, internal=False):
    allowed = {
        "seeds",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "early_stopping",
        "selection_loss_weight",
        "mlp_dropout",
        "negative_bce_weight",
        "bce_loss_weight",
    }
    if internal:
        allowed.add("dataset_sha256")
    unknown = set(training) - allowed
    if unknown:
        raise ValueError(
            f"unsupported training settings for the selected model: {sorted(unknown)}"
        )
    selection_weight = validate_selection_loss_weight(
        training.get("selection_loss_weight", 3.0)
    )
    bce_weight = validate_bce_loss_weight(training.get("bce_loss_weight", 1.0))
    if bce_weight == 0 and selection_weight == 0:
        raise ValueError(
            "bce_loss_weight and selection_loss_weight cannot both be zero"
        )
    validate_mlp_dropout(training.get("mlp_dropout", 0.1))
    validate_negative_bce_weight(training.get("negative_bce_weight", 1.0))
    if type(training.get("early_stopping", False)) is not bool:
        raise ValueError("early_stopping must be a boolean")
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
    required = {"dataset_sha256", "numerical_error_policy", "split", "training"}
    if (
        not isinstance(settings, dict)
        or not required <= set(settings)
        or set(settings)
        - required
        - {"ik_target_features", "relative_geometry", "obstacle_context"}
    ):
        raise ValueError(
            "config requires dataset_sha256, numerical_error_policy, split and training; optional geometry or obstacle_context is supported"
        )
    context_mode = validate_obstacle_context(settings.get("obstacle_context", "full"))
    if context_mode != "full" and (
        "ik_target_features" in settings or "relative_geometry" in settings
    ):
        raise ValueError("zero obstacle context requires the original 23D model")
    if "ik_target_features" in settings and "relative_geometry" in settings:
        raise ValueError("select only one geometry model")
    if "relative_geometry" in settings:
        geometry = settings["relative_geometry"]
        if (
            not isinstance(geometry, dict)
            or set(geometry) - {"query_conditioning"}
            != {
                "model_path",
                "model_sha256",
                "capsule_config_path",
                "capsule_config_sha256",
            }
            or any(
                not isinstance(geometry.get(k), str) or not geometry[k]
                for k in ("model_path", "capsule_config_path")
            )
            or any(
                not isinstance(geometry.get(k), str)
                or len(geometry[k]) != 64
                or any(c not in "0123456789abcdef" for c in geometry[k])
                for k in ("model_sha256", "capsule_config_sha256")
            )
        ):
            raise ValueError(
                "relative_geometry requires pinned model and capsule configuration"
            )
        validate_query_conditioning(geometry.get("query_conditioning", "local"))
    if "ik_target_features" in settings:
        geometry = settings["ik_target_features"]
        if (
            not isinstance(geometry, dict)
            or set(geometry) != {"model_path", "model_sha256"}
            or not isinstance(geometry["model_path"], str)
            or not geometry["model_path"]
            or not isinstance(geometry["model_sha256"], str)
            or len(geometry["model_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in geometry["model_sha256"])
        ):
            raise ValueError("ik_target_features requires model_path and model_sha256")
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
    fk_geometry=None,
    relative_geometry=None,
    query_conditioning="local",
    obstacle_context="full",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    actions = validate_actions(actions)
    if targets.shape[1] != len(actions):
        raise ValueError("training label/action dimension mismatch")
    dropout = validate_mlp_dropout(settings.get("mlp_dropout", 0.1))
    obstacle_context = validate_obstacle_context(obstacle_context)
    if obstacle_context != "full" and (
        fk_geometry is not None or relative_geometry is not None
    ):
        raise ValueError("zero obstacle context requires the original 23D model")
    model_options = {"ik_target_features": True} if fk_geometry is not None else {}
    if obstacle_context != "full":
        model_options["obstacle_context"] = obstacle_context
    model = (
        RelativeGeometrySuccessNet(
            mean, std, actions=actions, mlp_dropout=dropout,
            geometry=relative_geometry, query_conditioning=query_conditioning,
        )
        if relative_geometry is not None
        else LinkSuccessNet(
            mean, std, actions=actions, mlp_dropout=dropout, **model_options
        )
    ).to(device)
    optimizer = make_optimizer(model, settings)
    batch_rng = np.random.default_rng(seed)
    destination = output / f"seed-{seed}"
    destination.mkdir()
    metric = SELECTION_METRIC
    objective = objective_metadata(settings)
    stopping = stopping_metadata(settings)
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
        loss_sum = bce_sum = selection_sum = weighted_bce_sum = 0.0
        permutation = batch_rng.permutation(split["train"])
        for start in range(0, len(permutation), settings["batch_size"]):
            index = torch.as_tensor(
                permutation[start : start + settings["batch_size"]], device=device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(state[index], obstacles[index])
            loss, bce, selection = loss_components(
                logits,
                targets[index],
                selection_loss_weight=objective["selection_loss_weight"],
                negative_bce_weight=objective["negative_bce_weight"],
                bce_loss_weight=objective["bce_loss_weight"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite training loss at seed {seed}, epoch {epoch}"
                )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            bce_sum += float(bce.detach()) * len(index)
            selection_sum += float(selection.detach()) * len(index)
            with torch.no_grad():
                weighted_bce_sum += float(
                    weighted_binary_loss(
                        logits, targets[index], objective["negative_bce_weight"], bce
                    )
                ) * len(index)
        model.eval()
        with torch.no_grad():
            logits = model(state[validation], obstacles[validation])
            validation_total, validation_bce, validation_selection = loss_components(
                logits,
                targets[validation],
                selection_loss_weight=objective["selection_loss_weight"],
                negative_bce_weight=objective["negative_bce_weight"],
                bce_loss_weight=objective["bce_loss_weight"],
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
            "train_weighted_bce": weighted_bce_sum / len(permutation),
            "validation_bce": validation_loss,
            "validation_selection_loss": float(validation_selection),
            "validation_total_loss": float(validation_total),
            "validation_weighted_bce": float(
                weighted_binary_loss(
                    logits,
                    targets[validation],
                    objective["negative_bce_weight"],
                    validation_bce,
                )
            ),
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
                **stopping,
                "updated_utc": utc_now(),
            },
        )
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"seed={seed} epoch={epoch} train_bce={row['train_bce']:.5f} val_bce={validation_loss:.5f} val_selection={selection_success:.3f}",
                flush=True,
            )
        if stopping["early_stopping"] and stale >= settings["patience"]:
            break
    write_csv(destination / "history.csv", history)
    checkpoint = {
        "schema_version": (
            (20 if query_conditioning == "local" else 21)
            if relative_geometry is not None
            else (
                19
                if fk_geometry is not None
                else (22 if obstacle_context == "zero" else 17)
            )
        ),
        "model": (
            relative_model_spec(
                dropout, relative_geometry, query_conditioning=query_conditioning
            )
            if relative_geometry is not None
            else model_spec(dropout, **model_options)
        ),
        **({"fk_geometry": fk_geometry} if fk_geometry is not None else {}),
        **(
            {"relative_geometry": relative_geometry}
            if relative_geometry is not None
            else {}
        ),
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
        **stopping,
        "input_encoding": "q_zscore_7,ik_zscore_7,goal_position_metres_3,rotation_first_two_columns_6"
        + (
            ",ik_target_link2_to_link6_world_pose_5x9"
            if fk_geometry is not None
            else ""
        ),
        "obstacle_encoding": (
            "link_current_target_relative_xyz_radius_capsule_clearances; ordered_actions; IK_independent_direct"
            if relative_geometry is not None
            else "world_xyz_radius_metres_raw_shared_mlp_cross_attention"
        ),
        "numerical_error_policy": "failure",
        "dataset_sha256": settings["dataset_sha256"],
        "checkpoint_kind": "best",
        "checkpoint_epoch": best_epoch,
        "checkpoint_validation_bce": best_loss,
        "checkpoint_validation_selected_successes": best_successes,
    }
    if relative_geometry is not None:
        checkpoint[
            "input_encoding"
        ] += ",current_target_link_poses,current_ee_pose,current_target_capsule_endpoints"
    torch.save(checkpoint, destination / "best.pt")
    last_checkpoint = {
        **checkpoint,
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "checkpoint_kind": "last",
        "checkpoint_epoch": len(history),
        "checkpoint_validation_bce": history[-1]["validation_bce"],
        "checkpoint_validation_selected_successes": history[-1][
            "validation_selected_successes"
        ],
    }
    torch.save(last_checkpoint, destination / "last.pt")
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
        **stopping,
        "stopped_early": len(history) < settings["max_epochs"],
        "checkpoint": str((destination / "best.pt").relative_to(output)),
        "last_checkpoint": str((destination / "last.pt").relative_to(output)),
        "last_epoch": len(history),
        "last_validation_bce": history[-1]["validation_bce"],
        "last_validation_selected_successes": history[-1][
            "validation_selected_successes"
        ],
        "mlp_dropout": dropout,
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
    relative = (
        report["model"].get("architecture") == "relative_geometry_cross_attention"
    )
    architecture_text = (
        "Shared 32D link-motion and relative-obstacle encoders; one-head 32D cross-attention per link; ordered action slots and a shared Sequential scorer; separate IK-independent Direct branch. Whole-body current capsule clearances are retained."
        if relative
        else f"Robot MLP {report['model'].get('state_input_dim', 23)}→64→64; obstacle MLP 4→32→32; one-head cross-attention (32D); context projection 32→64; concatenated prediction MLP 128→64→{report['output_dim']}."
    )
    if report["model"].get("obstacle_context") == "zero":
        architecture_text += (
            " Diagnostic ablation: projected obstacle context is multiplied by zero"
            " before concatenation in training and inference. All modules still run;"
            " registered parameter count is unchanged but effective capacity is lower."
        )
    rows = [
        "# Panda selected link-combination model",
        "",
        architecture_text,
        f"{report['parameter_count']:,} parameters. ReLU and dropout {report['model']['mlp_dropout']:g} at all hidden MLP sites; dropout disabled for evaluation.",
        f"AdamW with weight_decay={report['weight_decay']:g}; {report['optimization_loss'].replace(' * ', ' × ')}. Negative-label BCE weight beta={report['negative_bce_weight']:g}; positive-label weight 1. Numerical errors remain failures.",
        "Raw BCE is unweighted for checkpoint tie-breaking; weighted_bce is the binary diagnostic before multiplication by bce_loss_weight. Cost-weighted or selection-only sigmoid scores are not calibrated success probabilities.",
        "Only 14 joint-angle inputs are standardized using training statistics; geometry stays in metres.",
        f"Checkpoint selection: {selection_description()}; exact ties retain earlier epoch/lower seed.",
        (
            f"Early stopping enabled with patience {report['early_stopping_patience']}; maximum {report['max_epochs']} epochs."
            if report["early_stopping"]
            else f"Early stopping disabled: every seed trains for {report['max_epochs']} epochs."
        ),
        "Each seed saves both best.pt and last.pt. This report's selection metrics use best.pt; final-epoch validation metrics are separate fields in evaluation.json.",
        (
            "All choices are frozen before test evaluation."
            if has_test
            else (
                "Test evaluation was not performed. Reserved test scenes remain excluded from training, model inference and exported predictions."
                if report.get("reserved_test_scenes", 0)
                else "No test set was requested or evaluated."
            )
            + " Validation is used for checkpoint/seed selection; its score is not an independent test estimate."
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


def run(
    dataset_directory, config_path, output, device_name, *, skip_test_evaluation=False
):
    started = time.perf_counter()
    settings = yaml.safe_load(Path(config_path).read_text())
    validate_settings(settings)
    metric = SELECTION_METRIC
    objective = objective_metadata(settings["training"])
    stopping = stopping_metadata(settings["training"])
    use_fk = "ik_target_features" in settings
    use_relative = "relative_geometry" in settings
    obstacle_context = settings.get("obstacle_context", "full")
    query_conditioning = settings.get("relative_geometry", {}).get(
        "query_conditioning", "local"
    )
    relative_geometry = None
    relative_model_path = relative_config_path = None
    if use_relative:
        spec = settings["relative_geometry"]
        relative_model_path = (
            Path(config_path).resolve().parent / spec["model_path"]
        ).resolve()
        relative_config_path = (
            Path(config_path).resolve().parent / spec["capsule_config_path"]
        ).resolve()
        if (
            sha256(relative_model_path) != spec["model_sha256"]
            or sha256(relative_config_path) != spec["capsule_config_sha256"]
        ):
            raise ValueError("relative geometry model/configuration checksum mismatch")
        relative_geometry = make_relative_geometry(
            relative_model_path, relative_config_path
        )
    dropout = settings["training"].get("mlp_dropout", 0.1)
    selected_model_spec = (
        relative_model_spec(
            dropout, relative_geometry, query_conditioning=query_conditioning
        )
        if use_relative
        else model_spec(
            dropout, ik_target_features=use_fk, obstacle_context=obstacle_context
        )
    )
    data = load_dataset(dataset_directory, settings["dataset_sha256"])
    labels = data["labels"]
    actions = data["actions"]
    specification = settings["split"]
    split = load_scene_split(data, specification, dataset_directory)
    evaluation_split = {
        key: indices
        for key, indices in split.items()
        if key != "test" or not skip_test_evaluation
    }
    evaluation_indices = np.sort(np.concatenate(list(evaluation_split.values())))
    has_test = "test" in evaluation_split
    split_policy = (
        "frozen_scene_ids"
        if "file" in specification
        else "stratified_successful_action_count"
    )
    fk_path = None
    fk_values = None
    if use_fk:
        fk_path = (
            Path(config_path).resolve().parent
            / settings["ik_target_features"]["model_path"]
        ).resolve()
        if sha256(fk_path) != settings["ik_target_features"]["model_sha256"]:
            raise ValueError(
                "IK target FK model hash does not match the configured geometry"
            )
        fk_values = compute_ik_target_features(
            data["selected_ik"][evaluation_indices], fk_path
        )
    # Reserved test scenes receive no feature calculation or model forward pass.
    # Original indices are retained to preserve the frozen split's identity.
    evaluated_state = make_state(
        data["initial_q"][evaluation_indices],
        data["selected_ik"][evaluation_indices],
        data["target_position"][evaluation_indices],
        data["target_quaternion"][evaluation_indices],
        **({"ik_target_pose": fk_values} if use_fk else {}),
    )
    if use_relative:
        relative_values = compute_relative_geometry_features(
            data["initial_q"][evaluation_indices],
            data["selected_ik"][evaluation_indices],
            relative_model_path,
            relative_geometry,
        )
        evaluated_state = np.concatenate((evaluated_state, relative_values), axis=1)
    state = np.zeros((len(labels), evaluated_state.shape[1]), dtype=np.float32)
    state[evaluation_indices] = evaluated_state
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
        geometry_path = None
        relative_config_snapshot = None
        if use_relative:
            geometry_path = output / "relative_geometry" / "panda.xml"
            geometry_path.parent.mkdir()
            relative_config_snapshot = geometry_path.parent / "controller.yaml"
            shutil.copyfile(relative_model_path, geometry_path)
            shutil.copyfile(relative_config_path, relative_config_snapshot)
            spec = settings["relative_geometry"]
            if (
                sha256(geometry_path) != spec["model_sha256"]
                or sha256(relative_config_snapshot) != spec["capsule_config_sha256"]
            ):
                raise ValueError("relative geometry changed while preparing the run")
            relative_geometry = make_relative_geometry(
                geometry_path, relative_config_snapshot
            )
            np.savez_compressed(
                output / "relative_geometry_features.npz",
                case_id=data["case_id"][evaluation_indices],
                features=relative_values,
            )
        if use_fk:
            geometry_path = output / "fk_geometry" / "panda.xml"
            geometry_path.parent.mkdir()
            shutil.copyfile(fk_path, geometry_path)
            if sha256(geometry_path) != settings["ik_target_features"]["model_sha256"]:
                raise ValueError(
                    "IK target FK geometry changed while preparing the run"
                )
            np.savez_compressed(
                output / "ik_target_features.npz",
                case_id=data["case_id"][evaluation_indices],
                features=fk_values,
            )
        provenance = {
            "created_utc": utc_now(),
            "dataset_directory": str(Path(dataset_directory).resolve()),
            "dataset_sha256": data["dataset_sha256"],
            "dataset_sources": data["dataset_sources"],
            "split_policy": split_policy,
            "evaluation_splits": list(evaluation_split),
            "test_evaluation_performed": has_test,
            **({"reserved_test_scenes": len(split["test"])} if "test" in split else {}),
            "skip_test_evaluation": skip_test_evaluation,
            **({"fk_geometry": make_fk_geometry(geometry_path)} if use_fk else {}),
            **({"relative_geometry": relative_geometry} if use_relative else {}),
            "settings": settings,
            "model": selected_model_spec,
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
                **(
                    {
                        "ik_target_features.py": sha256(
                            Path(__file__).parents[1] / "learning/ik_target_features.py"
                        ),
                        "panda_kinematics.py": sha256(
                            Path(__file__).parents[1] / "adapters/panda_kinematics.py"
                        ),
                    }
                    if use_fk or use_relative
                    else {}
                ),
                **(
                    {
                        name: sha256(Path(__file__).parents[1] / "learning" / name)
                        for name in (
                            "relative_geometry.py",
                            "relative_geometry_features.py",
                        )
                    }
                    if use_relative
                    else {}
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
                **(
                    {"ik_target_link_poses": "raw_metres_and_rotation_columns"}
                    if use_fk or use_relative
                    else {}
                ),
            },
            "numerical_error_labels_reclassified": int((~data["label_valid"]).sum()),
            "numerical_error_scenes": int((~data["label_valid"]).any(axis=1).sum()),
            "all_fail_scenes": int((labels.sum(axis=1) == 0).sum()),
            "all_success_scenes": int((labels.sum(axis=1) == len(actions)).sum()),
            "parameter_count": sum(
                parameter.numel()
                for parameter in (
                    RelativeGeometrySuccessNet(
                        actions=actions, mlp_dropout=dropout,
                        geometry=relative_geometry, query_conditioning=query_conditioning,
                    )
                    if use_relative
                    else LinkSuccessNet(
                        actions=actions,
                        mlp_dropout=selected_model_spec["mlp_dropout"],
                        ik_target_features=use_fk,
                        obstacle_context=obstacle_context,
                    )
                ).parameters()
            ),
            "selection_policy": "argmax; ties use lowest index; no IK returns Direct",
            "selection_metric": metric,
            "checkpoint_selection": selection_description()
            + "; exact ties keep earlier epoch/lower seed"
            + (
                "; all seeds frozen before test evaluation"
                if has_test
                else "; test evaluation omitted"
            ),
            "early_stopping_selection": (
                selection_description() if stopping["early_stopping"] else None
            ),
            **objective,
            **stopping,
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
            case_id=data["case_id"][evaluation_indices],
            labels=labels[evaluation_indices].astype(np.int8),
            original_label_valid=data["label_valid"][evaluation_indices],
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
                query_conditioning=query_conditioning,
                obstacle_context=obstacle_context,
                fk_geometry=(
                    make_fk_geometry(
                        geometry_path, checkpoint_directory=output / f"seed-{seed}"
                    )
                    if use_fk
                    else None
                ),
                relative_geometry=(
                    make_relative_geometry(
                        geometry_path,
                        relative_config_snapshot,
                        checkpoint_directory=output / f"seed-{seed}",
                    )
                    if use_relative
                    else None
                ),
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
                evaluated_logits = model(
                    tensors[0][evaluation_indices], tensors[1][evaluation_indices]
                )
                logits = torch.full_like(tensors[2], torch.nan)
                logits[evaluation_indices] = evaluated_logits
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
                for key, index in evaluation_split.items()
            }
            with torch.no_grad():
                for key, index in evaluation_split.items():
                    total, bce, selection = loss_components(
                        logits[index],
                        tensors[2][index],
                        selection_loss_weight=objective["selection_loss_weight"],
                        negative_bce_weight=objective["negative_bce_weight"],
                        bce_loss_weight=objective["bce_loss_weight"],
                    )
                    trained["metrics"][key].update(
                        selection_loss=float(selection),
                        total_loss=float(total),
                        bce_from_logits=float(bce),
                        weighted_bce=float(
                            weighted_binary_loss(
                                logits[index],
                                tensors[2][index],
                                objective["negative_bce_weight"],
                                bce,
                            )
                        ),
                    )
            np.savez_compressed(
                output / f"seed-{trained['seed']}" / "predictions.npz",
                case_id=data["case_id"][evaluation_indices],
                probabilities=probabilities[evaluation_indices],
                selection_probabilities=selection_probabilities[evaluation_indices],
                chosen_action_index=probabilities[evaluation_indices].argmax(axis=1),
                **catalog_arrays(actions),
                **(
                    {
                        "chosen_pair_index": probabilities[evaluation_indices].argmax(
                            axis=1
                        )
                    }
                    if actions == LINK_PAIRS
                    else {}
                ),
            )
            write_json(
                output / f"seed-{trained['seed']}" / "metrics.json", trained["metrics"]
            )
            for key, indices in evaluation_split.items():
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
        for key, indices in evaluation_split.items():
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
            **stopping,
            "model": selected_model_spec,
            **catalog_metadata(actions),
            "output_dim": len(actions),
            "selection_metric": metric,
            "split_policy": split_policy,
            "evaluation_splits": list(evaluation_split),
            **({"reserved_test_scenes": len(split["test"])} if "test" in split else {}),
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
                **stopping,
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
                "evaluated_scenes": len(evaluation_indices),
                "selection_metric": metric,
                **objective,
                **stopping,
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
    parser.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help="Keep frozen test scenes reserved; do not infer or export their predictions",
    )
    args = parser.parse_args()
    run(
        args.dataset,
        args.config,
        args.output,
        args.device,
        skip_test_evaluation=args.skip_test_evaluation,
    )


if __name__ == "__main__":
    main()

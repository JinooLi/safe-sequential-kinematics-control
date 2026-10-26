"""Train the existing small backbone with softmax and discounted success time."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import yaml

from ..learning.link_selection import (
    LINK_PAIRS,
    MODEL_SPEC,
    joint_statistics,
    load_dataset,
    load_model,
    load_scene_split,
    make_state,
    sha256,
)
from ..learning.link_time_selection import (
    OBJECTIVE,
    SELECTION_METRIC,
    LinkTimeSelectionNet,
    discounted_rewards,
    load_completion_times,
    time_reward_loss,
    timing_metrics,
    validate_time_scale,
)
from .train_link_selection import (
    make_optimizer,
    utc_now,
    validate_settings,
    write_csv,
    write_json,
)


def validate_time_settings(config):
    if not isinstance(config, dict) or "objective" not in config:
        raise ValueError("softmax time-selection requires an explicit objective")
    validate_settings({k: v for k, v in config.items() if k != "objective"})
    objective = config["objective"]
    if (
        not isinstance(objective, dict)
        or set(objective) != {"name", "time_scale_seconds"}
        or objective["name"] != OBJECTIVE
    ):
        raise ValueError("unsupported time-selection objective")
    validate_time_scale(objective["time_scale_seconds"])


def objective_metadata(settings, scale):
    return dict(
        optimization_loss=OBJECTIVE,
        time_scale_seconds=scale,
        output_activation="softmax",
        output_semantics="link_pair_selection_probability",
        bce_loss_weight=0.0,
        optimizer="adamw",
        weight_decay=settings["weight_decay"],
        numerical_error_policy="failure",
        selection_metric=SELECTION_METRIC,
        reduction="mean_over_all_scenes; all_failure_rows_zero",
        reward="success * exp(-simulation_completion_time / time_scale_seconds)",
    )


def checkpoint_rank(reward, successes, loss):
    return (-reward, -successes, loss)


def select_run(runs):
    return min(
        runs,
        key=lambda r: (
            *checkpoint_rank(
                r["best_validation_time_reward"],
                r["best_validation_selected_successes"],
                r["best_validation_loss"],
            ),
            r["seed"],
        ),
    )


def train_seed(output, tensors, split, mean, std, settings, scale, seed):
    state, obstacles, labels, times, rewards = tensors
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if state.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = LinkTimeSelectionNet(mean, std).to(state.device)
    optimizer = make_optimizer(model, settings)
    batch_rng = np.random.default_rng(seed)
    dest = output / f"seed-{seed}"
    dest.mkdir()
    validation = torch.as_tensor(split["validation"], device=state.device)
    history, best_rank, best_state, best_record, stale = [], None, None, None, 0
    started = time.perf_counter()
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        loss_sum = 0.0
        permutation = batch_rng.permutation(split["train"])
        for start in range(0, len(permutation), settings["batch_size"]):
            index = torch.as_tensor(
                permutation[start : start + settings["batch_size"]], device=state.device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model.logits(state[index], obstacles[index])
            loss = time_reward_loss(logits, labels[index], times[index], scale)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite loss at seed {seed}, epoch {epoch}"
                )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
        model.eval()
        with torch.no_grad():
            logits = model.logits(state[validation], obstacles[validation])
            vloss = float(
                time_reward_loss(logits, labels[validation], times[validation], scale)
            )
            choices = logits.softmax(dim=1).argmax(dim=1)
            successes = int(labels[validation].gather(1, choices[:, None]).sum())
            reward = float(rewards[validation].gather(1, choices[:, None]).mean())
        if not np.isfinite([vloss, reward, loss_sum]).all():
            raise FloatingPointError("nonfinite training/validation metrics")
        row = dict(
            epoch=epoch,
            train_loss=loss_sum / len(permutation),
            validation_loss=vloss,
            validation_selected_successes=successes,
            validation_selected_success_rate=successes / len(validation),
            validation_selected_time_reward=reward,
        )
        rank = checkpoint_rank(reward, successes, vloss)
        if best_rank is None or rank < best_rank:
            best_rank, best_record, stale = rank, row.copy(), 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            stale += 1
        history.append(row)
        write_json(
            output / "progress.json",
            dict(
                status="training",
                seed=seed,
                epoch=epoch,
                best_epoch=best_record["epoch"],
                best_validation_time_reward=-best_rank[0],
                updated_utc=utc_now(),
            ),
        )
        if epoch == 1 or epoch % 25 == 0:
            print(
                f'seed={seed} epoch={epoch} train_loss={row["train_loss"]:.5f} val_reward={reward:.5f} val_success={successes}/{len(validation)}',
                flush=True,
            )
        if stale >= settings["patience"]:
            break
    metadata = objective_metadata(settings, scale)
    result = dict(
        seed=seed,
        epochs=len(history),
        best_epoch=best_record["epoch"],
        best_validation_time_reward=best_record["validation_selected_time_reward"],
        best_validation_selected_successes=best_record["validation_selected_successes"],
        best_validation_loss=best_record["validation_loss"],
        wall_seconds=time.perf_counter() - started,
        checkpoint=f"seed-{seed}/best.pt",
    )
    checkpoint = dict(
        schema_version=16,
        model=MODEL_SPEC,
        state_dict=best_state,
        link_pairs=[list(p) for p in LINK_PAIRS],
        dataset_sha256=settings["dataset_sha256"],
        input_encoding="q_zscore_7,ik_zscore_7,goal_position_metres_3,rotation_first_two_columns_6",
        obstacle_encoding="world_xyz_radius_metres_raw_shared_mlp_cross_attention",
        **metadata,
        **result,
    )
    torch.save(checkpoint, dest / "best.pt")
    write_csv(dest / "history.csv", history)
    write_json(dest / "training.json", {**metadata, **result})
    print(
        f'seed={seed} complete best_epoch={result["best_epoch"]} val_reward={result["best_validation_time_reward"]:.5f}',
        flush=True,
    )
    return result


def plot_history(output, runs):
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for run in runs:
        with (output / f'seed-{run["seed"]}' / "history.csv").open() as f:
            rows = list(csv.DictReader(f))
        epochs = [int(r["epoch"]) for r in rows]
        for ax, key in zip(
            axes, ["train_loss", "validation_loss", "validation_selected_time_reward"]
        ):
            ax.plot(epochs, [float(r[key]) for r in rows], label=str(run["seed"]))
    for ax, title in zip(
        axes, ["Training loss", "Validation loss", "Validation selected reward"]
    ):
        ax.set(xlabel="Epoch", title=title)
        ax.grid(alpha=0.25)
        ax.legend(title="Seed")
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)


def run(dataset_directory, config_path, output, device_name):
    started = time.perf_counter()
    config = yaml.safe_load(Path(config_path).read_text())
    validate_time_settings(config)
    scale = config["objective"]["time_scale_seconds"]
    data = load_dataset(dataset_directory, config["dataset_sha256"])
    split = load_scene_split(data, config["split"], dataset_directory)
    times, timing_sources = load_completion_times(data, dataset_directory)
    rewards = discounted_rewards(data["labels"], times, scale)
    state = make_state(
        data["initial_q"],
        data["selected_ik"],
        data["target_position"],
        data["target_quaternion"],
    )
    mean, std = joint_statistics(state, split["train"])
    device = torch.device(device_name)
    if (
        device.type not in ("cpu", "cuda")
        or device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("requested device unavailable; no implicit fallback")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = {**config["training"], "dataset_sha256": data["dataset_sha256"]}
    metadata = objective_metadata(settings, scale)
    try:
        manifest = dict(
            created_utc=utc_now(),
            dataset_directory=str(Path(dataset_directory).resolve()),
            dataset_sha256=data["dataset_sha256"],
            settings=config,
            model=MODEL_SPEC,
            **metadata,
            timing_sources=timing_sources,
            parameter_count=sum(p.numel() for p in LinkTimeSelectionNet().parameters()),
            normalization=dict(
                joint_mean=mean.tolist(),
                joint_std=std.tolist(),
                joint_method="train_only_zscore",
                geometry="raw_metres",
            ),
            source_sha256={
                str(p.relative_to(Path(__file__).parents[1])): sha256(p)
                for p in [
                    Path(__file__),
                    Path(__file__).with_name("train_link_selection.py"),
                    Path(__file__).parents[1] / "learning/link_time_selection.py",
                    Path(__file__).parents[1] / "learning/link_selection.py",
                ]
            },
            environment=dict(
                python=platform.python_version(),
                torch=str(torch.__version__),
                numpy=np.__version__,
                device=str(device),
                gpu=(
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
            ),
            checkpoint_selection="maximum validation selected reward; ties: successes, loss, earlier epoch/lower seed",
            numerical_error_labels_reclassified=int((~data["label_valid"]).sum()),
            train_all_failure_zero_gradient_scenes=int(
                (data["labels"][split["train"]].sum(axis=1) == 0).sum()
            ),
        )
        write_json(output / "manifest.json", manifest)
        shutil.copyfile(config_path, output / "training_config.yaml")
        write_json(
            output / "split.json",
            {
                k: dict(indices=v.tolist(), case_ids=data["case_id"][v].tolist())
                for k, v in split.items()
            },
        )
        np.savez_compressed(
            output / "time_targets.npz",
            case_id=data["case_id"],
            labels=data["labels"].astype(np.int8),
            completion_time=times,
            rewards=rewards,
            link_pairs=np.asarray(LINK_PAIRS),
        )
        tensors = [
            torch.as_tensor(x, dtype=torch.float32, device=device)
            for x in [state, data["obstacles"], data["labels"], times]
        ]
        tensors.append(torch.as_tensor(rewards, dtype=torch.float64, device=device))
        runs = [
            train_seed(output, tensors, split, mean, std, settings, scale, seed)
            for seed in settings["seeds"]
        ]
        selected = select_run(runs)
        write_json(
            output / "selection.json",
            dict(
                **selected,
                criterion=SELECTION_METRIC,
                selected_before_test_evaluation=True,
                selected_utc=utc_now(),
            ),
        )
        for trained in runs:
            model = load_model(output / trained["checkpoint"], device)
            with torch.no_grad():
                logits = model.logits(tensors[0], tensors[1])
                probabilities = logits.softmax(dim=1).cpu().numpy()
                raw_logits = logits.cpu().numpy()
            if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=3e-7):
                raise AssertionError("softmax row sums do not equal one")
            metrics = {
                key: timing_metrics(
                    data["labels"][idx], times[idx], probabilities[idx], scale
                )
                for key, idx in split.items()
            }
            for key, idx in split.items():
                metrics[key]["loss"] = float(
                    time_reward_loss(
                        logits[idx], tensors[2][idx], tensors[3][idx], scale
                    )
                )
                metrics[key]["softmax_logit_argmax_disagreements"] = int(
                    (
                        probabilities[idx].argmax(axis=1)
                        != raw_logits[idx].argmax(axis=1)
                    ).sum()
                )
            np.savez_compressed(
                output / f'seed-{trained["seed"]}' / "predictions.npz",
                case_id=data["case_id"],
                probabilities=probabilities,
                logits=raw_logits,
                chosen_pair_index=probabilities.argmax(axis=1),
            )
            trained["metrics"] = metrics
            write_json(output / f'seed-{trained["seed"]}' / "metrics.json", metrics)
        report = dict(
            **metadata,
            model=MODEL_SPEC,
            selected_seed=selected["seed"],
            runs=runs,
            mean_test_selected_success_rate=float(
                np.mean([r["metrics"]["test"]["selected_success_rate"] for r in runs])
            ),
            mean_test_selected_time_reward=float(
                np.mean([r["metrics"]["test"]["selected_time_reward"] for r in runs])
            ),
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(output / "evaluation.json", report)
        plot_history(output, runs)
        write_json(
            output / "progress.json",
            dict(
                status="completed",
                runs=len(runs),
                selected_seed=selected["seed"],
                finished_utc=utc_now(),
            ),
        )
        hashes = {
            str(p.relative_to(output)): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        }
        write_json(
            output / "complete.json",
            dict(
                runs=len(runs),
                scenes=len(data["labels"]),
                labels=int(data["labels"].size),
                **metadata,
                selected_seed=selected["seed"],
                elapsed_seconds=time.perf_counter() - started,
                finished_utc=utc_now(),
                data_sha256=hashes,
            ),
        )
        return report
    except BaseException as exc:
        write_json(output / "failed.json", dict(error=repr(exc), utc=utc_now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.dataset, args.config, args.output, args.device)


if __name__ == "__main__":
    main()

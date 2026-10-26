"""Outcome/timing integrity, reward gradients and softmax checkpoint semantics."""

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml

from safe_ik.application.train_link_time_selection import (
    checkpoint_rank,
    objective_metadata,
    run,
    validate_time_settings,
)
from safe_ik.learning.link_selection import (
    LINK_PAIRS,
    MODEL_SPEC,
    LinkSuccessNet,
    load_dataset,
    load_model,
    make_state,
    predict_selection,
    sha256,
)
from safe_ik.learning.link_time_selection import (
    LinkTimeSelectionNet,
    discounted_rewards,
    load_completion_times,
    time_reward_loss,
    timing_metrics,
)


def dataset(root):
    rng = np.random.default_rng(17)
    labels = rng.integers(0, 2, (20, 10), dtype=np.int8)
    labels[0] = 0
    labels[1] = 1
    labels[0, 0] = -1
    valid = labels >= 0
    ids = np.array([f"case-{i}" for i in range(20)])
    arrays = dict(
        case_id=ids,
        initial_q=rng.normal(size=(20, 7)),
        selected_ik=rng.normal(size=(20, 7)),
        target_position=rng.normal(size=(20, 3)),
        target_quaternion=np.tile([1.0, 0, 0, 0], (20, 1)),
        obstacles=np.tile([[[0.1, 0.2, 0.3, 0.1]]], (20, 2, 1)),
        success=labels,
        label_valid=valid,
        link_pairs=np.array(LINK_PAIRS),
    )
    np.savez_compressed(root / "dataset.npz", **arrays)
    rows = []
    for i, case in enumerate(ids):
        for j, pair in enumerate(LINK_PAIRS):
            ok = labels[i, j] == 1
            t = 10.0 + j if ok else 0.02
            rows.append(
                dict(
                    case_id=case,
                    pair=json.dumps(pair),
                    label_valid=bool(valid[i, j]),
                    success=bool(ok),
                    goal_reached=bool(ok),
                    reason="goal_reached" if ok else "stalled",
                    elapsed=t,
                    completion_time=t if ok else "",
                    interpolated_success=bool(ok),
                    candidate_switch_count=0,
                )
            )
    with (root / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows[::-1])
    (root / "complete.json").write_text(
        json.dumps(
            dict(
                contexts=20,
                valid_labels=int(valid.sum()),
                success_labels=int((labels == 1).sum()),
                data_sha256={
                    n: sha256(root / n) for n in ["dataset.npz", "results.csv"]
                },
            )
        )
    )


class LinkTimeSelectionTests(unittest.TestCase):
    def test_time_rewards_favor_fast_success_and_never_fast_failure(self):
        logits = torch.zeros((3, 10), dtype=torch.float64, requires_grad=True)
        labels = torch.zeros_like(logits)
        labels[0, :2] = 1
        labels[1] = 1
        times = torch.full_like(logits, float("nan"))
        times[0, :2] = torch.tensor([10.0, 100.0])
        times[1] = 30.0
        loss = time_reward_loss(logits, labels, times)
        rewards = np.zeros((3, 10))
        rewards[0, :2] = np.exp(-np.array([10.0, 100.0]) / 120)
        rewards[1] = np.exp(-30 / 120)
        expected = (-np.log(rewards[0].mean()) + 30 / 120) / 3
        self.assertAlmostEqual(float(loss.detach()), expected)
        loss.backward()
        self.assertLess(float(logits.grad[0, 0]), float(logits.grad[0, 1]))
        self.assertGreater(float(logits.grad[0, 2]), 0.0)
        torch.testing.assert_close(logits.grad[1:], torch.zeros_like(logits.grad[1:]))
        self.assertTrue(
            torch.autograd.gradcheck(
                time_reward_loss, (logits.detach().requires_grad_(), labels, times)
            )
        )
        shifted = logits.detach() + 5000
        torch.testing.assert_close(
            time_reward_loss(shifted, labels, times), loss.detach()
        )
        r = discounted_rewards(labels.numpy(), times.numpy(), 120.0)
        np.testing.assert_allclose(r, rewards)

    def test_extreme_logits_all_failure_and_invalid_success_time(self):
        logits = torch.tensor([[1000.0] + [-1000.0] * 9] * 2, requires_grad=True)
        labels = torch.zeros_like(logits)
        labels[0, 1] = 1
        times = torch.full_like(logits, float("nan"))
        times[0, 1] = 60
        loss = time_reward_loss(logits, labels, times)
        self.assertAlmostEqual(float(loss.detach()), 1000.25, places=4)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[1], torch.zeros(10))
        with self.assertRaisesRegex(ValueError, "successful completion"):
            discounted_rewards(labels.numpy(), np.full((2, 10), np.nan), 120)
        for value in [True, 0, -1, float("nan")]:
            with self.assertRaises(ValueError):
                discounted_rewards(labels.numpy(), times.numpy(), value)

    def test_same_backbone_normalized_output_and_checkpoint_fallback(self):
        torch.manual_seed(11)
        baseline = LinkSuccessNet().eval()
        torch.manual_seed(11)
        model = LinkTimeSelectionNet().eval()
        for k, v in baseline.state_dict().items():
            torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
        state, obstacles = torch.randn(4, 23), torch.randn(4, 3, 4)
        torch.testing.assert_close(
            model.logits(state, obstacles), baseline(state, obstacles), rtol=0, atol=0
        )
        probabilities = model(state, obstacles)
        torch.testing.assert_close(probabilities.sum(1), torch.ones(4))
        torch.testing.assert_close(
            probabilities.argmax(1), baseline(state, obstacles).argmax(1)
        )
        checkpoint = dict(
            schema_version=16,
            model=MODEL_SPEC,
            state_dict=model.state_dict(),
            link_pairs=[list(p) for p in LINK_PAIRS],
            **objective_metadata({"weight_decay": 0.0001}, 120.0),
        )
        with TemporaryDirectory() as temp:
            path = Path(temp) / "best.pt"
            torch.save(checkpoint, path)
            loaded = load_model(path)
            torch.testing.assert_close(
                loaded(state, obstacles), probabilities, rtol=0, atol=0
            )
            answer = predict_selection(
                loaded, [0] * 7, [0] * 7, [0] * 3, [1, 0, 0, 0], []
            )
            self.assertAlmostEqual(sum(answer["probabilities"]), 1.0, places=6)
            self.assertEqual(answer["probability_semantics"], "link_pair_selection")
            self.assertEqual(
                predict_selection(None, None, None, None, None, None)["method"],
                "direct",
            )
            for field, value in [
                ("output_activation", "sigmoid"),
                ("bce_loss_weight", 1.0),
                ("time_scale_seconds", 0),
            ]:
                torch.save({**checkpoint, field: value}, path)
                with self.assertRaises(ValueError):
                    load_model(path)

    def test_timing_csv_join_hashes_failure_mask_and_metrics(self):
        with TemporaryDirectory() as temp:
            p = Path(temp)
            dataset(p)
            data = load_dataset(p)
            times, sources = load_completion_times(data, p)
            self.assertTrue(np.isnan(times[data["labels"] == 0]).all())
            self.assertTrue((times[data["labels"] == 1] >= 10).all())
            self.assertEqual(len(sources), 1)
            pred = np.zeros((20, 10))
            pred[:, 0] = 1
            metrics = timing_metrics(data["labels"], times, pred)
            n = int(data["labels"][:, 0].sum())
            self.assertEqual(metrics["selected_successes"], n)
            self.assertAlmostEqual(
                metrics["failure_penalized_120s_cost"], (n * 10 + (20 - n) * 120) / 20
            )
            with (p / "results.csv").open("a") as f:
                f.write("\n")
            with self.assertRaisesRegex(ValueError, "CSV checksum"):
                load_completion_times(data, p)

    def test_training_roundtrip_and_validation_only_reward_selection(self):
        self.assertLess(checkpoint_rank(0.6, 4, 2), checkpoint_rank(0.5, 5, 1))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "data"
            source.mkdir()
            dataset(source)
            config = dict(
                dataset_sha256=sha256(source / "dataset.npz"),
                numerical_error_policy="failure",
                split=dict(seed=9, train=12, validation=4, test=4),
                objective=dict(
                    name="negative_log_discounted_success_mass",
                    time_scale_seconds=120.0,
                ),
                training=dict(
                    seeds=[11],
                    learning_rate=0.001,
                    weight_decay=0.0001,
                    batch_size=6,
                    max_epochs=3,
                    patience=2,
                ),
            )
            validate_time_settings(config)
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            out = root / "training"
            report = run(source, path, out, "cpu")
            complete = json.loads((out / "complete.json").read_text())
            for n, h in complete["data_sha256"].items():
                self.assertEqual(sha256(out / n), h)
            selection = json.loads((out / "selection.json").read_text())
            self.assertTrue(selection["selected_before_test_evaluation"])
            with (out / "seed-11/history.csv").open() as f:
                history = list(csv.DictReader(f))
            best = min(
                history,
                key=lambda r: (
                    *checkpoint_rank(
                        float(r["validation_selected_time_reward"]),
                        int(r["validation_selected_successes"]),
                        float(r["validation_loss"]),
                    ),
                    int(r["epoch"]),
                ),
            )
            self.assertEqual(report["runs"][0]["best_epoch"], int(best["epoch"]))
            data = load_dataset(source)
            state = torch.from_numpy(
                make_state(
                    data["initial_q"],
                    data["selected_ik"],
                    data["target_position"],
                    data["target_quaternion"],
                )
            )
            model = load_model(out / "seed-11/best.pt")
            with torch.no_grad():
                probs = model(
                    state, torch.tensor(data["obstacles"], dtype=torch.float32)
                ).numpy()
            with np.load(out / "seed-11/predictions.npz") as z:
                np.testing.assert_array_equal(probs, z["probabilities"])
            with self.assertRaises(FileExistsError):
                run(source, path, out, "cpu")

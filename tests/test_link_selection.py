"""Regression coverage for the sole supported Panda link selector."""

import copy
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml

from safe_ik.application.train_link_selection import (
    checkpoint_rank,
    loss_components,
    make_optimizer,
    objective_metadata,
    prediction_metrics,
    run,
    select_run,
    successful_mass_loss,
    train_seed,
    validate_settings,
)
from safe_ik.learning.link_selection import (
    LINK_PAIRS,
    MODEL_SPEC,
    LinkSuccessNet,
    joint_statistics,
    load_dataset,
    load_model,
    load_scene_split,
    make_state,
    predict_selection,
    sha256,
    split_scenes,
)


def example_dataset(root, size=20):
    rng = np.random.default_rng(123)
    labels = rng.integers(0, 2, size=(size, 10), dtype=np.int8)
    valid = np.ones_like(labels, dtype=bool)
    valid[0, 0] = False
    labels[0, 0] = -1
    path = root / "dataset.npz"
    obstacles = rng.normal(size=(size, 10, 4))
    obstacles[..., 3] = 0.1
    np.savez_compressed(
        path,
        case_id=np.array([f"scene-{i}" for i in range(size)]),
        initial_q=rng.normal(size=(size, 7)),
        selected_ik=rng.normal(size=(size, 7)),
        target_position=rng.normal(size=(size, 3)),
        target_quaternion=np.tile([1.0, 0, 0, 0], (size, 1)),
        obstacles=obstacles,
        link_pairs=np.array(LINK_PAIRS),
        success=labels,
        label_valid=valid,
    )
    (root / "complete.json").write_text(
        json.dumps(
            {
                "contexts": size,
                "valid_labels": int(valid.sum()),
                "success_labels": int((labels == 1).sum()),
                "data_sha256": {"dataset.npz": sha256(path)},
            }
        )
    )
    return path


class LinkSelectionTests(unittest.TestCase):

    def test_selection_loss_rewards_total_success_mass_and_masks_uniform_outcomes(self):
        probabilities = torch.tensor([[0.4, 0.2] + [0.05] * 8] * 3, dtype=torch.float64)
        logits = probabilities.log().requires_grad_()
        labels = torch.zeros_like(logits)
        labels[0, :2] = 1
        labels[2] = 1
        loss = successful_mass_loss(logits, labels)
        self.assertAlmostEqual(float(loss.detach()), -np.log(0.6) / 3)
        loss.backward()
        expected = torch.zeros_like(logits)
        expected[0] = (probabilities[0] - probabilities[0] * labels[0] / 0.6) / 3
        torch.testing.assert_close(logits.grad, expected)
        self.assertTrue(
            torch.autograd.gradcheck(
                successful_mass_loss, (logits.detach().requires_grad_(), labels)
            )
        )
        for shares in ([0.8, 0.1], [0.45, 0.45]):
            p = torch.tensor([list(shares) + [0.0125] * 8], dtype=torch.float64)
            self.assertAlmostEqual(
                float(successful_mass_loss(p.log(), labels[:1])), -np.log(0.9)
            )

    def test_selection_loss_is_finite_for_extreme_logits_and_all_failure_batches(self):
        logits = torch.tensor([[1000.0] + [-1000.0] * 9] * 3, requires_grad=True)
        labels = torch.zeros_like(logits)
        labels[0, 1] = 1
        labels[2] = 1
        loss = successful_mass_loss(logits, labels)
        self.assertAlmostEqual(float(loss.detach()), 2000 / 3, places=3)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[1:], torch.zeros_like(logits.grad[1:]))
        for value in (0, 1):
            isolated = logits.detach().clone().requires_grad_()
            loss = successful_mass_loss(isolated, torch.full_like(isolated, value))
            self.assertEqual(float(loss.detach()), 0.0)
            loss.backward()
            torch.testing.assert_close(isolated.grad, torch.zeros_like(isolated))

    def test_numerical_errors_are_failures_without_changing_source(self):
        with TemporaryDirectory() as temp:
            path = example_dataset(Path(temp))
            before = path.read_bytes()
            data = load_dataset(temp, sha256(path))
            self.assertEqual(data["labels"][0, 0], 0)
            self.assertEqual(data["success"][0, 0], -1)
            self.assertFalse(data["label_valid"][0, 0])
            self.assertTrue(np.isin(data["labels"], [0, 1]).all())
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_dataset(temp, "0" * 64)

    def test_scenes_split_once_and_deterministically(self):
        labels = np.array([[int(j < i % 11) for j in range(10)] for i in range(1000)])
        a, b = (split_scenes(labels), split_scenes(labels))
        self.assertEqual([len(a[key]) for key in a], [700, 150, 150])
        self.assertEqual(len(set(np.concatenate(list(a.values())))), 1000)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        with self.assertRaises(ValueError):
            split_scenes(labels, test_size=149)

    def test_statistics_use_only_training_joints(self):
        state = np.zeros((4, 23), dtype=np.float32)
        state[0, :14] = 2
        state[1, :14] = 4
        state[2:, :14] = 10000
        mean, std = joint_statistics(state, [0, 1])
        np.testing.assert_array_equal(mean, np.full(14, 3))
        np.testing.assert_array_equal(std, np.ones(14))
        _, constant_std = joint_statistics(state, [0])
        self.assertTrue(np.all(constant_std > 0))

    def test_frozen_split_follows_scene_ids_not_row_positions(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            example_dataset(root)
            data = load_dataset(root)
            partition = split_scenes(data["labels"], 12, 4, 4, 9)
            record = {
                k: {"case_ids": data["case_id"][v].tolist()}
                for k, v in partition.items()
            }
            path = root / "frozen.json"
            path.write_text(json.dumps(record))
            spec = {
                "train": 12,
                "validation": 4,
                "test": 4,
                "file": path.name,
                "sha256": sha256(path),
            }
            reordered = {**data, "case_id": data["case_id"][::-1]}
            loaded = load_scene_split(reordered, spec, root)
            for key, indices in loaded.items():
                self.assertEqual(
                    reordered["case_id"][indices].tolist(), record[key]["case_ids"]
                )
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_scene_split(data, {**spec, "sha256": "0" * 64}, root)
            record["train"]["case_ids"][0] = record["test"]["case_ids"][0]
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "exactly once"):
                load_scene_split(data, {**spec, "sha256": sha256(path)}, root)
            record["train"]["case_ids"][0] = "unknown-scene"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_scene_split(data, {**spec, "sha256": sha256(path)}, root)

    def test_positions_radii_and_rotation_are_not_standardized(self):
        mean = np.arange(14, dtype=np.float32)
        model = LinkSuccessNet(mean, np.full(14, 2))
        state = torch.zeros((2, 23))
        state[:, :14] = torch.as_tensor(mean + 6)
        state[:, 14:] = torch.arange(9) / 10
        obstacles = torch.tensor([[[0.1, -0.2, 0.3, 0.15]]] * 2)
        received = {}
        model.state_encoder[0].register_forward_pre_hook(
            lambda module, args: received.update(state=args[0].detach().clone())
        )
        model.obstacle_encoder[0].register_forward_pre_hook(
            lambda module, args: received.update(obstacles=args[0].detach().clone())
        )
        model(state, obstacles)
        torch.testing.assert_close(received["state"][:, :14], torch.full((2, 14), 3.0))
        torch.testing.assert_close(received["state"][:, 14:], state[:, 14:])
        torch.testing.assert_close(received["obstacles"], obstacles)

    def test_scalar_first_rotation_is_invariant_to_quaternion_sign(self):
        w = 2 ** (-0.5)
        quaternion = np.array([[w, 0, 0, w]])
        args = [np.zeros((1, 7)), np.ones((1, 7)), np.array([[0.2, 0.3, 0.4]])]
        a = make_state(*args, quaternion)
        b = make_state(*args, -quaternion)
        np.testing.assert_allclose(a, b, atol=1e-06)
        np.testing.assert_allclose(a[0, 17:], [0, 1, 0, -1, 0, 0], atol=1e-06)
        np.testing.assert_allclose(a[0, 14:17], args[2][0])

    def test_selection_metrics_use_paired_outcomes(self):
        labels = np.zeros((4, 10), dtype=np.float32)
        labels[:, 0] = [1, 0, 1, 0]
        labels[:, 1] = [1, 1, 0, 1]
        probabilities = np.full_like(labels, 0.1)
        probabilities[:, 1] = 0.9
        metrics = prediction_metrics(labels, probabilities, 0)
        self.assertEqual(metrics["selected_successes"], 3)
        self.assertEqual(metrics["paired_wins"], 2)
        self.assertEqual(metrics["paired_losses"], 1)
        self.assertEqual(metrics["oracle_successes"], 4)
        self.assertEqual(metrics["delta_vs_fixed"], 0.25)

    def test_fixed_model_has_five_dropout_sites_and_query_dependent_context(self):
        torch.manual_seed(11)
        model = LinkSuccessNet().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 24234)
        drops = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        self.assertEqual(len(drops), 5)
        self.assertTrue(all(m.p == 0.1 for m in drops))
        self.assertEqual(model.cross_attention.num_heads, 1)
        state, obstacles = torch.randn(6, 23), torch.randn(6, 4, 4)
        a = model(state, obstacles)
        torch.testing.assert_close(a, model(state, obstacles[:, [2, 0, 3, 1]]))
        features = model.state_encoder(state)
        context = model.obstacle_features(obstacles, state_features=features)
        other = model.obstacle_features(obstacles, state_features=features + 10)
        self.assertFalse(torch.equal(context, other))
        model.train()
        self.assertFalse(torch.equal(model(state, obstacles), model(state, obstacles)))
        model.eval()
        torch.testing.assert_close(a, model(state, obstacles), rtol=0, atol=0)

    def test_padding_empty_obstacles_and_gradients(self):
        torch.manual_seed(23)
        model = LinkSuccessNet().eval()
        state, obstacles = torch.randn(4, 23), torch.randn(4, 3, 4)
        mask = torch.ones(4, 3, dtype=torch.bool)
        padded = torch.cat((obstacles, torch.full((4, 2, 4), torch.nan)), dim=1)
        padded_mask = torch.cat((mask, torch.zeros(4, 2, dtype=torch.bool)), dim=1)
        torch.testing.assert_close(
            model(state, obstacles), model(state, padded, padded_mask)
        )
        empty = model(state, obstacles[:, :0])
        all_masked = model(state, padded, torch.zeros_like(padded_mask))
        torch.testing.assert_close(empty, all_masked)
        mixed_mask = padded_mask.clone()
        mixed_mask[0] = False
        loss = model(state, padded, mixed_mask).square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )
        self.assertGreater(model.cross_attention.in_proj_weight.grad.abs().sum(), 0)

    def test_fixed_loss_value_and_gradient(self):
        logits = torch.randn(4, 10, dtype=torch.float64, requires_grad=True)
        labels = torch.zeros_like(logits)
        labels[1] = 1
        labels[2, :3] = 1
        labels[3, 6:] = 1
        expected_selection = (
            sum(
                -torch.log(logits[i].softmax(0)[labels[i].bool()].sum()) for i in (2, 3)
            )
            / 4
        )
        expected = (
            torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            + 3 * expected_selection
        )
        total, bce, selection = loss_components(logits, labels)
        torch.testing.assert_close(total, expected)
        torch.testing.assert_close(selection, expected_selection)
        torch.testing.assert_close(
            torch.autograd.grad(total, logits, retain_graph=True)[0],
            torch.autograd.grad(expected, logits)[0],
        )
        with self.assertRaises(TypeError):
            loss_components(logits, labels, selection_loss_weight=1)

    def test_config_and_optimizer_do_not_accept_retired_options(self):
        path = Path(__file__).parents[1] / "configs/training/panda_link_selection.yaml"
        config = yaml.safe_load(path.read_text())
        validate_settings(config)
        for key, value in {
            "architecture": "pooling",
            "attention_heads": 4,
            "prediction_self_attention": True,
            "fk_features": True,
            "mlp_dropout": 0.1,
            "robot_hidden_dims": [128, 128],
        }.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_settings({**config, key: value})
        for key, value in {
            "optimizer": "adam",
            "parameter_l2_weight": 0,
            "ranking_loss_weight": 0,
            "time_loss_weight": 0,
            "selection_loss_weight": 1,
            "bce_loss_weight": 0,
            "selection_metric": "validation_bce",
        }.items():
            candidate = copy.deepcopy(config)
            candidate["training"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_settings(candidate)
        model = LinkSuccessNet()
        optimizer = make_optimizer(model, config["training"])
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0001)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)
        for value in (-1, float("nan"), float("inf"), True, None):
            candidate = copy.deepcopy(config)
            candidate["training"]["weight_decay"] = value
            with self.subTest(decay=value), self.assertRaises(ValueError):
                validate_settings(candidate)
        with self.assertRaises(TypeError):
            LinkSuccessNet(attention_heads=4)

    def test_adamw_decays_parameters_with_zero_task_gradient(self):
        config = yaml.safe_load(
            (
                Path(__file__).parents[1] / "configs/training/panda_link_selection.yaml"
            ).read_text()
        )
        model = torch.nn.Linear(2, 1).double()
        optimizer = make_optimizer(model, config["training"])
        before = [p.detach().clone() for p in model.parameters()]
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        factor = (
            1 - config["training"]["learning_rate"] * config["training"]["weight_decay"]
        )
        for parameter, initial in zip(model.parameters(), before):
            torch.testing.assert_close(parameter, initial * factor, rtol=1e-14, atol=0)
            self.assertFalse(torch.equal(parameter, initial))

    def test_current_and_selected_legacy_checkpoints_and_direct_fallback(self):
        model = LinkSuccessNet().eval()
        checkpoint = {
            "schema_version": 13,
            "model": MODEL_SPEC,
            "state_dict": model.state_dict(),
            "link_pairs": [list(pair) for pair in LINK_PAIRS],
            "selection_metric": "validation_selected_success_rate",
            **objective_metadata({"weight_decay": 0.0001}),
        }
        state, obstacles = torch.randn(3, 23), torch.randn(3, 2, 4)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            torch.save(checkpoint, path)
            loaded = load_model(path)
            torch.testing.assert_close(
                model(state, obstacles), loaded(state, obstacles), rtol=0, atol=0
            )
            for field, value in (
                ("optimizer", "adam"),
                ("weight_decay", -1),
                ("weight_decay", float("nan")),
            ):
                with self.subTest(field=field, value=value):
                    torch.save({**checkpoint, field: value}, path)
                    with self.assertRaises(ValueError):
                        load_model(path)
            previous = {**checkpoint, "schema_version": 12, "optimizer": "adam"}
            previous.pop("weight_decay")
            torch.save(previous, path)
            torch.testing.assert_close(
                model(state, obstacles),
                load_model(path)(state, obstacles),
                rtol=0,
                atol=0,
            )
            legacy = {
                **previous,
                "schema_version": 11,
                **{
                    k: v for k, v in MODEL_SPEC.items() if k != "prediction_hidden_dims"
                },
                "head_hidden_layers": 1,
                "obstacle_self_attention": False,
                "prediction_self_attention": False,
                "fk_features": None,
            }
            legacy.pop("model")
            torch.save(legacy, path)
            torch.testing.assert_close(
                model(state, obstacles),
                load_model(path)(state, obstacles),
                rtol=0,
                atol=0,
            )
            for field, value in (
                ("attention_heads", 4),
                ("schema_version", 3),
                ("mlp_dropout_scope", "robot_head_first"),
                ("prediction_self_attention", True),
                ("selection_loss_weight", 1),
                ("optimizer", "adamw"),
            ):
                with self.subTest(field=field):
                    torch.save({**legacy, field: value}, path)
                    with self.assertRaises(ValueError):
                        load_model(path)
        never_called = Mock(side_effect=AssertionError)
        fallback = predict_selection(never_called, None, None, None, None, None)
        self.assertEqual(fallback["method"], "direct")
        never_called.assert_not_called()
        result = predict_selection(model, [0] * 7, [0] * 7, [0] * 3, [1, 0, 0, 0], [])
        self.assertEqual(len(result["probabilities"]), 10)
        self.assertEqual(
            result["link_frames"],
            [f"link{i}" for i in LINK_PAIRS[result["pair_index"]]],
        )

    def test_success_selection_and_early_stopping_keep_the_same_rule(self):
        runs = [
            {
                "seed": 11,
                "best_validation_bce": 0.1,
                "best_validation_selected_successes": 1,
            },
            {
                "seed": 23,
                "best_validation_bce": 0.5,
                "best_validation_selected_successes": 2,
            },
            {
                "seed": 37,
                "best_validation_bce": 0.4,
                "best_validation_selected_successes": 2,
            },
            {
                "seed": 17,
                "best_validation_bce": 0.4,
                "best_validation_selected_successes": 2,
            },
        ]
        self.assertEqual(select_run(runs)["seed"], 17)
        low = torch.full((2, 10), -5.0)
        low[0, 0] = 5
        low[1, 0], low[1, 1] = -4, -3
        high = torch.zeros(2, 10)
        high[:, 0] = 0.1
        tie = torch.full((2, 10), -1.0)
        tie[:, 0] = 1
        schedule = [low, high, tie, tie, low]

        class ScheduledNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
                self.register_buffer("epoch_marker", torch.tensor(0))

            def forward(self, state, obstacles):
                if self.training:
                    return self.weight.expand(len(state), 10)
                index = int(self.epoch_marker)
                self.epoch_marker.add_(1)
                return schedule[index]

        with (
            TemporaryDirectory() as temp,
            patch(
                "safe_ik.application.train_link_selection.LinkSuccessNet",
                return_value=ScheduledNet(),
            ),
        ):
            labels = torch.zeros(4, 10)
            labels[:, 0] = 1
            result = train_seed(
                Path(temp),
                torch.zeros(4, 23),
                torch.zeros(4, 1, 4),
                labels,
                {"train": np.array([0, 1]), "validation": np.array([2, 3])},
                np.zeros(14),
                np.ones(14),
                {
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "batch_size": 2,
                    "max_epochs": 10,
                    "patience": 2,
                    "dataset_sha256": "test",
                },
                1,
                0,
                torch.device("cpu"),
            )
            self.assertEqual(result["best_epoch"], 3)
            self.assertEqual(result["epochs"], 5)

    def test_training_export_roundtrip_preserves_frozen_splits_and_source(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "data"
            source.mkdir()
            dataset = example_dataset(source)
            source_hash = sha256(dataset)
            ids = load_dataset(source)["case_id"].tolist()
            frozen = {
                "train": {"case_ids": ids[:12]},
                "validation": {"case_ids": ids[12:16]},
                "test": {"case_ids": ids[16:]},
            }
            split_path = source / "frozen.json"
            split_path.write_text(json.dumps(frozen))
            config = {
                "dataset_sha256": source_hash,
                "numerical_error_policy": "failure",
                "split": {
                    "seed": 9,
                    "train": 12,
                    "validation": 4,
                    "test": 4,
                    "file": split_path.name,
                    "sha256": sha256(split_path),
                },
                "training": {
                    "seeds": [1, 2],
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "batch_size": 6,
                    "max_epochs": 3,
                    "patience": 2,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            output = root / "training"
            report = run(source, config_path, output, "cpu")
            complete = json.loads((output / "complete.json").read_text())
            for name, expected in complete["data_sha256"].items():
                self.assertEqual(sha256(output / name), expected)
            self.assertEqual(sha256(dataset), source_hash)
            self.assertEqual(report["model"], MODEL_SPEC)
            self.assertEqual(report["optimizer"], "adamw")
            self.assertEqual(report["weight_decay"], 0.0001)
            saved_split = json.loads((output / "split.json").read_text())
            for key in frozen:
                self.assertEqual(saved_split[key]["case_ids"], frozen[key]["case_ids"])
            selection = json.loads((output / "selection.json").read_text())
            self.assertTrue(selection["selected_before_test_evaluation"])
            data = load_dataset(source)
            state = torch.from_numpy(
                make_state(
                    data["initial_q"],
                    data["selected_ik"],
                    data["target_position"],
                    data["target_quaternion"],
                )
            )
            obstacles = torch.as_tensor(data["obstacles"], dtype=torch.float32)
            for trained in report["runs"]:
                folder = output / f"seed-{trained['seed']}"
                model = load_model(folder / "best.pt")
                with torch.no_grad():
                    actual = model(state, obstacles).sigmoid().numpy()
                stored = np.load(folder / "predictions.npz")
                np.testing.assert_array_equal(actual, stored["probabilities"])
                with (folder / "history.csv").open() as f:
                    history = list(csv.DictReader(f))
                best = min(
                    history,
                    key=lambda row: (
                        -int(row["validation_selected_successes"]),
                        float(row["validation_bce"]),
                        int(row["epoch"]),
                    ),
                )
                self.assertEqual(trained["best_epoch"], int(best["epoch"]))
                for row in history:
                    for phase in ("train", "validation"):
                        self.assertAlmostEqual(
                            float(row[phase + "_total_loss"]),
                            float(row[phase + "_bce"])
                            + 3 * float(row[phase + "_selection_loss"]),
                            places=6,
                        )
            self.assertEqual(
                report["selected_seed"], select_run(report["runs"])["seed"]
            )
            with self.assertRaises(FileExistsError):
                run(source, config_path, output, "cpu")

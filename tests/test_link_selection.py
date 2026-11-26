"""Regression coverage for the sole supported Panda link selector."""

import copy
import csv
import json
import subprocess
import sys
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
from safe_ik.domain.link_actions import LINK_ACTIONS, catalog_arrays, catalog_metadata
from safe_ik.learning.link_selection import (
    LINK_PAIRS,
    MODEL_SPEC,
    LinkSuccessNet,
    joint_statistics,
    load_dataset,
    load_model,
    load_scene_split,
    make_state,
    model_spec,
    predict_selection,
    sha256,
    split_scenes,
)


def example_dataset(root, size=20, *, actions=LINK_PAIRS):
    rng = np.random.default_rng(123)
    labels = rng.integers(0, 2, size=(size, len(actions)), dtype=np.int8)
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
        **(
            {"link_pairs": np.array(LINK_PAIRS)}
            if actions == LINK_PAIRS
            else {**catalog_arrays(actions), "observed": np.ones_like(valid)}
        ),
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
                **(catalog_metadata(actions) if actions == LINK_ACTIONS else {}),
            }
        )
    )
    return path


class LinkSelectionTests(unittest.TestCase):

    def test_new_actions_require_observed_outcomes_not_fake_failure_labels(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = example_dataset(root, actions=LINK_ACTIONS)
            data = load_dataset(root)
            self.assertEqual(data["actions"], LINK_ACTIONS)
            self.assertEqual(data["labels"].shape, (20, 16))
            self.assertEqual(data["labels"][0, 0], 0)
            with np.load(path) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["observed"][0, -1] = False
            arrays["success"][0, -1] = -1
            arrays["label_valid"][0, -1] = False
            np.savez_compressed(path, **arrays)
            complete = json.loads((root / "complete.json").read_text())
            complete["data_sha256"][path.name] = sha256(path)
            (root / "complete.json").write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, "unobserved"):
                load_dataset(root)
            arrays.pop("observed")
            np.savez_compressed(path, **arrays)
            complete["data_sha256"][path.name] = sha256(path)
            (root / "complete.json").write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, "observed outcome mask"):
                load_dataset(root)

    def test_new_checkpoint_routes_single_link_and_direct_without_changing_legacy(self):
        model = LinkSuccessNet().eval()
        checkpoint = {
            "schema_version": 17,
            "model": MODEL_SPEC,
            "output_dim": len(LINK_ACTIONS),
            **catalog_metadata(),
            "selection_metric": "validation_selected_success_rate",
            **objective_metadata({"weight_decay": 0.0001}),
        }
        with TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            for index, method, frames in [
                (10, "sequential", ["link2"]),
                (15, "direct", []),
            ]:
                with torch.no_grad():
                    model.head[-1].weight.zero_()
                    model.head[-1].bias.fill_(-1)
                    model.head[-1].bias[index] = 1
                torch.save({**checkpoint, "state_dict": model.state_dict()}, path)
                loaded = load_model(path)
                result = predict_selection(
                    loaded, [0] * 7, [0] * 7, [0] * 3, [1, 0, 0, 0], []
                )
                self.assertEqual(result["action_index"], index)
                self.assertEqual(result["method"], method)
                self.assertEqual(result["link_frames"], frames)
                self.assertIsNone(result["pair_index"])
                self.assertEqual(len(result["probabilities"]), 16)
            state, obstacles = torch.randn(3, 23), torch.randn(3, 2, 4)
            for weight in (0, 1.0, 3.0):
                with self.subTest(weight=weight):
                    torch.save(
                        {
                            **checkpoint,
                            "state_dict": model.state_dict(),
                            **objective_metadata(
                                {
                                    "weight_decay": 0.0001,
                                    "selection_loss_weight": weight,
                                }
                            ),
                        },
                        path,
                    )
                    torch.testing.assert_close(
                        model(state, obstacles),
                        load_model(path)(state, obstacles),
                        rtol=0,
                        atol=0,
                    )
            for weight in (-1, float("nan"), float("inf"), True, None, "1"):
                with self.subTest(weight=weight):
                    torch.save(
                        {
                            **checkpoint,
                            "state_dict": model.state_dict(),
                            "selection_loss_weight": weight,
                        },
                        path,
                    )
                    with self.assertRaises(ValueError):
                        load_model(path)
            for beta in (None, 1.0, 2.0):
                with self.subTest(negative_bce_weight=beta):
                    saved = {**checkpoint, "state_dict": model.state_dict()}
                    if beta is None:
                        saved.pop("negative_bce_weight", None)
                    else:
                        saved["negative_bce_weight"] = beta
                    torch.save(saved, path)
                    loaded = load_model(path)
                    self.assertEqual(
                        loaded.negative_bce_weight, 1.0 if beta is None else beta
                    )
                    torch.testing.assert_close(
                        model(state, obstacles),
                        loaded(state, obstacles),
                        rtol=0,
                        atol=0,
                    )
            for beta in (0, -1, float("nan"), float("inf"), True, None, "2"):
                with self.subTest(invalid_negative_bce_weight=beta):
                    torch.save(
                        {
                            **checkpoint,
                            "state_dict": model.state_dict(),
                            "negative_bce_weight": beta,
                        },
                        path,
                    )
                    with self.assertRaisesRegex(ValueError, "negative_bce_weight"):
                        load_model(path)
            for coefficient in (None, 0, 0.5, 1.0):
                with self.subTest(bce_loss_weight=coefficient):
                    saved = {**checkpoint, "state_dict": model.state_dict()}
                    if coefficient is None:
                        saved.pop("bce_loss_weight", None)
                    else:
                        saved["bce_loss_weight"] = coefficient
                    torch.save(saved, path)
                    loaded = load_model(path)
                    self.assertEqual(
                        loaded.bce_loss_weight,
                        1 if coefficient is None else coefficient,
                    )
                    torch.testing.assert_close(
                        model(state, obstacles),
                        loaded(state, obstacles),
                        rtol=0,
                        atol=0,
                    )
            for coefficient in (-1, float("nan"), float("inf"), True, None, "1"):
                with self.subTest(invalid_bce_loss_weight=coefficient):
                    torch.save(
                        {
                            **checkpoint,
                            "state_dict": model.state_dict(),
                            "bce_loss_weight": coefficient,
                        },
                        path,
                    )
                    with self.assertRaisesRegex(ValueError, "bce_loss_weight"):
                        load_model(path)
            torch.save(
                {
                    **checkpoint,
                    "state_dict": model.state_dict(),
                    "bce_loss_weight": 0,
                    "selection_loss_weight": 0,
                },
                path,
            )
            with self.assertRaises(ValueError):
                load_model(path)
            torch.save(
                {
                    **checkpoint,
                    "model": model_spec(0.2),
                    "state_dict": model.state_dict(),
                },
                path,
            )
            loaded = load_model(path)
            drops = [m for m in loaded.modules() if isinstance(m, torch.nn.Dropout)]
            self.assertEqual(len(drops), 5)
            self.assertTrue(all(m.p == 0.2 for m in drops))
            torch.testing.assert_close(
                model(state, obstacles), loaded(state, obstacles), rtol=0, atol=0
            )
            for dropout in (-0.1, 1.0, float("nan"), float("inf"), True, None, "0.2"):
                with self.subTest(dropout=dropout):
                    torch.save(
                        {
                            **checkpoint,
                            "model": {**MODEL_SPEC, "mlp_dropout": dropout},
                            "state_dict": model.state_dict(),
                        },
                        path,
                    )
                    with self.assertRaises(ValueError):
                        load_model(path)
            torch.save(
                {**checkpoint, "state_dict": model.state_dict(), "output_dim": 10}, path
            )
            with self.assertRaisesRegex(ValueError, "dimension"):
                load_model(path)

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

    def test_no_test_split_is_stratified_and_frozen_by_scene_id(self):
        counts = np.arange(10000) % 17
        labels = (np.arange(16)[None, :] < counts[:, None]).astype(np.int8)
        split = split_scenes(labels, 9000, 1000, 0, 20261016)
        self.assertEqual(
            {k: len(v) for k, v in split.items()}, {"train": 9000, "validation": 1000}
        )
        np.testing.assert_array_equal(
            np.sort(np.concatenate(list(split.values()))), np.arange(10000)
        )
        expected = np.bincount(counts, minlength=17) / 10
        actual = np.bincount(counts[split["validation"]], minlength=17)
        self.assertTrue(np.all(np.abs(actual - expected) < 1))
        for key, indices in split_scenes(labels, 9000, 1000, 0, 20261016).items():
            np.testing.assert_array_equal(indices, split[key])
        with self.assertRaises(ValueError):
            split_scenes(labels, 9001, 1000, -1)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ids = np.array([f"scene-{i}" for i in range(10000)])
            record = {k: {"case_ids": ids[v].tolist()} for k, v in split.items()}
            path = root / "split.json"
            path.write_text(json.dumps(record))
            spec = dict(
                train=9000, validation=1000, test=0, file=path.name, sha256=sha256(path)
            )
            data = {"case_id": ids[::-1]}
            loaded = load_scene_split(data, spec, root)
            self.assertEqual(set(loaded), {"train", "validation"})
            for key, indices in loaded.items():
                self.assertEqual(
                    data["case_id"][indices].tolist(), record[key]["case_ids"]
                )
            record["test"] = {"case_ids": [ids[0]]}
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "size mismatch: test"):
                load_scene_split(data, {**spec, "sha256": sha256(path)}, root)

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
        labels = np.zeros((4, 16), dtype=np.float32)
        labels[:, 0] = [1, 0, 1, 0]
        labels[:, 10] = [1, 1, 0, 1]
        probabilities = np.full_like(labels, 0.1)
        probabilities[np.arange(4), [0, 10, 15, 10]] = 0.9
        metrics = prediction_metrics(labels, probabilities, 0, actions=LINK_ACTIONS)
        self.assertEqual(metrics["selected_successes"], 3)
        self.assertEqual(metrics["paired_wins"], 2)
        self.assertEqual(metrics["paired_losses"], 1)
        self.assertEqual(metrics["oracle_successes"], 4)
        self.assertEqual(metrics["delta_vs_fixed"], 0.25)
        self.assertEqual(metrics["new_actions_rescue_scenes"], 2)
        self.assertEqual(
            metrics["selection_by_intermediate_count"]["0"]["selected_scenes"], 1
        )
        self.assertEqual(
            metrics["selection_by_intermediate_count"]["1"]["successes"], 2
        )

    def test_fixed_model_has_five_dropout_sites_and_query_dependent_context(self):
        torch.manual_seed(11)
        model = LinkSuccessNet().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 24624)
        drops = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        self.assertEqual(len(drops), 5)
        self.assertTrue(all(m.p == 0.1 for m in drops))
        self.assertEqual(model.cross_attention.num_heads, 1)
        state, obstacles = torch.randn(6, 23), torch.randn(6, 4, 4)
        a = model(state, obstacles)
        torch.manual_seed(11)
        stronger_dropout = LinkSuccessNet(mlp_dropout=0.2).eval()
        self.assertEqual(set(model.state_dict()), set(stronger_dropout.state_dict()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(
                value, stronger_dropout.state_dict()[key], rtol=0, atol=0
            )
        torch.testing.assert_close(
            a, stronger_dropout(state, obstacles), rtol=0, atol=0
        )
        stronger_drops = [
            m for m in stronger_dropout.modules() if isinstance(m, torch.nn.Dropout)
        ]
        self.assertEqual(len(stronger_drops), 5)
        self.assertTrue(all(m.p == 0.2 for m in stronger_drops))
        self.assertEqual(stronger_dropout.cross_attention.dropout, 0)
        self.assertEqual(model_spec(), MODEL_SPEC)
        self.assertEqual(model_spec(0.2), {**MODEL_SPEC, "mlp_dropout": 0.2})
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

    def test_selection_weight_preserves_loss_value_and_gradient(self):
        logits = torch.randn(4, 16, dtype=torch.float64, requires_grad=True)
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
        expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels
        )
        for weight, arguments in (
            (3, {}),
            (1, {"selection_loss_weight": 1}),
            (0, {"selection_loss_weight": 0}),
        ):
            with self.subTest(weight=weight):
                expected = expected_bce + weight * expected_selection
                total, bce, selection = loss_components(logits, labels, **arguments)
                torch.testing.assert_close(total, expected)
                torch.testing.assert_close(bce, expected_bce)
                torch.testing.assert_close(selection, expected_selection)
                torch.testing.assert_close(
                    torch.autograd.grad(total, logits, retain_graph=True)[0],
                    torch.autograd.grad(expected, logits, retain_graph=True)[0],
                )

    def test_negative_bce_weight_changes_only_failure_loss_and_gradient(self):
        logits = torch.tensor(
            [[-2.0, 0.5, 1.5, -0.25], [0.75, -1.0, 2.0, -1.5]],
            dtype=torch.float64,
            requires_grad=True,
        )
        labels = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.float64)
        per_label = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        legacy = per_label.mean() + 3 * successful_mass_loss(logits, labels)
        default = loss_components(logits, labels)
        explicit = loss_components(
            logits, labels, bce_loss_weight=1, negative_bce_weight=1
        )
        for implicit_value, explicit_value in zip(default, explicit):
            torch.testing.assert_close(implicit_value, explicit_value, rtol=0, atol=0)
        torch.testing.assert_close(default[0], legacy, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(default[0], logits, retain_graph=True)[0],
            torch.autograd.grad(legacy, logits, retain_graph=True)[0],
            rtol=0,
            atol=0,
        )

        baseline, baseline_bce, baseline_selection = loss_components(
            logits, labels, selection_loss_weight=0, negative_bce_weight=1
        )
        weighted, raw_bce, selection = loss_components(
            logits, labels, selection_loss_weight=0, negative_bce_weight=2
        )
        torch.testing.assert_close(weighted, (per_label * (2 - labels)).mean())
        torch.testing.assert_close(raw_bce, baseline_bce, rtol=0, atol=0)
        torch.testing.assert_close(selection, baseline_selection, rtol=0, atol=0)
        baseline_gradient = torch.autograd.grad(baseline, logits, retain_graph=True)[0]
        weighted_gradient = torch.autograd.grad(weighted, logits, retain_graph=True)[0]
        torch.testing.assert_close(
            weighted_gradient[labels == 0], 2 * baseline_gradient[labels == 0]
        )
        torch.testing.assert_close(
            weighted_gradient[labels == 1], baseline_gradient[labels == 1]
        )
        total, _, _ = loss_components(
            logits, labels, selection_loss_weight=3, negative_bce_weight=2
        )
        torch.testing.assert_close(total, weighted + 3 * selection)

    def test_pure_selection_preserves_mass_loss_and_uniform_rows_have_no_gradient(self):
        logits = torch.tensor(
            [[1.0, -2.0, 0.5], [-0.5, 0.0, 2.0], [1.5, -0.5, 0.25]],
            dtype=torch.float64,
            requires_grad=True,
        )
        labels = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
            dtype=torch.float64,
        )
        expected = successful_mass_loss(logits, labels)
        total, raw_bce, selection = loss_components(
            logits, labels, bce_loss_weight=0, selection_loss_weight=1
        )
        torch.testing.assert_close(total, expected, rtol=0, atol=0)
        torch.testing.assert_close(selection, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            raw_bce,
            torch.nn.functional.binary_cross_entropy_with_logits(logits, labels),
        )
        gradient = torch.autograd.grad(total, logits, retain_graph=True)[0]
        torch.testing.assert_close(
            gradient,
            torch.autograd.grad(expected, logits, retain_graph=True)[0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(gradient[:2], torch.zeros_like(gradient[:2]))
        self.assertGreater(gradient[2, 0], 0)
        self.assertTrue((gradient[2, 1:] < 0).all())
        weighted, _, _ = loss_components(
            logits,
            labels,
            bce_loss_weight=0.5,
            selection_loss_weight=2,
            negative_bce_weight=2,
        )
        per_label = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        torch.testing.assert_close(
            weighted, 0.5 * (per_label * (2 - labels)).mean() + 2 * selection
        )

    def test_bce_coefficient_requires_finite_nonnegative_and_a_nonzero_objective(self):
        path = Path(__file__).parents[1] / "configs/training/panda_link_selection.yaml"
        config = yaml.safe_load(path.read_text())
        config["training"].pop("bce_loss_weight", None)
        self.assertEqual(objective_metadata(config["training"])["bce_loss_weight"], 1)
        logits, labels = torch.zeros(2, 3), torch.ones(2, 3)
        for coefficient in (0, 0.5, 1, 2.0):
            candidate = copy.deepcopy(config)
            candidate["training"]["bce_loss_weight"] = coefficient
            with self.subTest(coefficient=coefficient):
                validate_settings(candidate)
                self.assertEqual(
                    objective_metadata(candidate["training"])["bce_loss_weight"],
                    coefficient,
                )
        for coefficient in (-1, float("nan"), float("inf"), True, False, None, "1"):
            candidate = copy.deepcopy(config)
            candidate["training"]["bce_loss_weight"] = coefficient
            with self.subTest(coefficient=coefficient):
                with self.assertRaisesRegex(ValueError, "bce_loss_weight"):
                    validate_settings(candidate)
                with self.assertRaisesRegex(ValueError, "bce_loss_weight"):
                    loss_components(logits, labels, bce_loss_weight=coefficient)
        candidate = copy.deepcopy(config)
        candidate["training"].update(bce_loss_weight=0, selection_loss_weight=0)
        with self.assertRaises(ValueError):
            validate_settings(candidate)
        with self.assertRaises(ValueError):
            loss_components(logits, labels, bce_loss_weight=0, selection_loss_weight=0)

    def test_negative_bce_weight_is_stable_for_extreme_logits(self):
        logits = torch.tensor([[1000.0, -1000.0, -1000.0, 1000.0]], requires_grad=True)
        labels = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        total, raw_bce, selection = loss_components(
            logits, labels, selection_loss_weight=0, negative_bce_weight=2
        )
        self.assertEqual(float(total.detach()), 750.0)
        self.assertEqual(float(raw_bce.detach()), 500.0)
        self.assertTrue(torch.isfinite(selection))
        total.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[0.5, -0.25, 0.0, 0.0]]))

    def test_negative_bce_weight_config_requires_positive_finite_number(self):
        path = Path(__file__).parents[1] / "configs/training/panda_link_selection.yaml"
        config = yaml.safe_load(path.read_text())
        config["training"].pop("negative_bce_weight", None)
        validate_settings(config)
        self.assertEqual(
            objective_metadata(config["training"])["negative_bce_weight"], 1.0
        )
        for beta in (0.5, 1, 2.0):
            candidate = copy.deepcopy(config)
            candidate["training"]["negative_bce_weight"] = beta
            with self.subTest(beta=beta):
                validate_settings(candidate)
                self.assertEqual(
                    objective_metadata(candidate["training"])["negative_bce_weight"],
                    beta,
                )
        for beta in (
            0,
            -1,
            float("nan"),
            float("inf"),
            -float("inf"),
            True,
            False,
            None,
            "2",
        ):
            candidate = copy.deepcopy(config)
            candidate["training"]["negative_bce_weight"] = beta
            with (
                self.subTest(beta=beta),
                self.assertRaisesRegex(ValueError, "negative_bce_weight"),
            ):
                validate_settings(candidate)

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
        self.assertEqual(
            objective_metadata(config["training"])["selection_loss_weight"], 3
        )
        for weight in (0, 1, 3.0):
            candidate = copy.deepcopy(config)
            candidate["training"]["selection_loss_weight"] = weight
            with self.subTest(weight=weight):
                validate_settings(candidate)
                self.assertEqual(
                    objective_metadata(candidate["training"])["selection_loss_weight"],
                    weight,
                )
        for weight in (-1, float("nan"), float("inf"), True, None, "1"):
            candidate = copy.deepcopy(config)
            candidate["training"]["selection_loss_weight"] = weight
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                validate_settings(candidate)
        for dropout in (0, 0.1, 0.2):
            candidate = copy.deepcopy(config)
            candidate["training"]["mlp_dropout"] = dropout
            with self.subTest(dropout=dropout):
                validate_settings(candidate)
        for dropout in (-0.1, 1, float("nan"), float("inf"), True, None, "0.2"):
            candidate = copy.deepcopy(config)
            candidate["training"]["mlp_dropout"] = dropout
            with self.subTest(dropout=dropout):
                with self.assertRaises(ValueError):
                    validate_settings(candidate)
                with self.assertRaises(ValueError):
                    LinkSuccessNet(mlp_dropout=dropout)
        for enabled in (False, True):
            candidate = copy.deepcopy(config)
            candidate["training"]["early_stopping"] = enabled
            with self.subTest(early_stopping=enabled):
                validate_settings(candidate)
        for enabled in (0, 1, None, "false"):
            candidate = copy.deepcopy(config)
            candidate["training"]["early_stopping"] = enabled
            with self.subTest(early_stopping=enabled), self.assertRaises(ValueError):
                validate_settings(candidate)
        with self.assertRaises(TypeError):
            LinkSuccessNet(attention_heads=4)

    def test_current_and_selected_legacy_checkpoints_and_direct_fallback(self):
        model = LinkSuccessNet(actions=LINK_PAIRS).eval()
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
                ("model", {**MODEL_SPEC, "mlp_dropout": 0.2}),
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
                ("mlp_dropout", 0.2),
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
                    "early_stopping": True,
                    "dataset_sha256": "test",
                },
                1,
                0,
                torch.device("cpu"),
            )
            self.assertEqual(result["best_epoch"], 3)
            self.assertEqual(result["epochs"], 5)
            self.assertTrue(result["early_stopping"])
            self.assertTrue(result["stopped_early"])

    def test_disabled_early_stopping_finishes_epochs_and_preserves_best_and_last(self):
        class ScheduledNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
                self.register_buffer("epoch_marker", torch.tensor(0))

            def forward(self, state, obstacles):
                if self.training:
                    return self.weight.expand(len(state), 10)
                logits = torch.zeros(len(state), 10)
                logits[:, 0 if int(self.epoch_marker) == 0 else 1] = 5
                self.epoch_marker.add_(1)
                return logits

        for option in ({}, {"early_stopping": False}):
            with (
                self.subTest(option=option),
                TemporaryDirectory() as temp,
                patch(
                    "safe_ik.application.train_link_selection.LinkSuccessNet",
                    return_value=ScheduledNet(),
                ),
            ):
                labels = torch.zeros(4, 10)
                labels[:, 0] = 1
                output = Path(temp)
                result = train_seed(
                    output,
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
                        "max_epochs": 5,
                        "patience": 1,
                        "dataset_sha256": "test",
                        **option,
                    },
                    1,
                    0,
                    torch.device("cpu"),
                )
                self.assertEqual(result["epochs"], 5)
                self.assertEqual(result["last_epoch"], 5)
                self.assertEqual(result["best_epoch"], 1)
                self.assertFalse(result["early_stopping"])
                self.assertFalse(result["stopped_early"])
                best = torch.load(output / result["checkpoint"], weights_only=False)
                last = torch.load(
                    output / result["last_checkpoint"], weights_only=False
                )
                self.assertEqual(int(best["state_dict"]["epoch_marker"]), 1)
                self.assertEqual(int(last["state_dict"]["epoch_marker"]), 5)
                self.assertFalse(
                    torch.equal(
                        best["state_dict"]["weight"], last["state_dict"]["weight"]
                    )
                )
                self.assertEqual(result["best_validation_selected_successes"], 2)
                self.assertEqual(result["last_validation_selected_successes"], 0)
                self.assertGreater(
                    result["last_validation_bce"], result["best_validation_bce"]
                )

    def test_training_export_roundtrip_preserves_frozen_splits_and_source(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "data"
            source.mkdir()
            dataset = example_dataset(source, actions=LINK_ACTIONS)
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
            self.assertEqual(report["output_dim"], 16)
            self.assertEqual(report["parameter_count"], 24624)
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

    def test_objective_training_roundtrip_preserves_coefficients_and_bce_diagnostics(
        self,
    ):
        for beta, coefficient, semantics in (
            (2, 1, "cost_weighted_success_score"),
            (1, 0, "ordinal_selection_score"),
        ):
            with (
                self.subTest(beta=beta, bce_loss_weight=coefficient),
                TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                source = root / "data"
                source.mkdir()
                dataset = example_dataset(source, actions=LINK_ACTIONS)
                config = {
                    "dataset_sha256": sha256(dataset),
                    "numerical_error_policy": "failure",
                    "split": dict(seed=9, train=16, validation=4, test=0),
                    "training": dict(
                        seeds=[11],
                        learning_rate=0.001,
                        weight_decay=0.0001,
                        negative_bce_weight=beta,
                        bce_loss_weight=coefficient,
                        selection_loss_weight=1,
                        batch_size=8,
                        max_epochs=2,
                        patience=2,
                    ),
                }
                config_path = root / "config.yaml"
                config_path.write_text(yaml.safe_dump(config))
                output = root / "training"
                report = run(source, config_path, output, "cpu")
                trained = report["runs"][0]
                manifest = json.loads((output / "manifest.json").read_text())
                for metadata in (report, trained, manifest):
                    self.assertEqual(metadata["negative_bce_weight"], beta)
                    self.assertEqual(metadata["bce_loss_weight"], coefficient)
                    self.assertEqual(metadata["sigmoid_score_semantics"], semantics)
                for key in ("checkpoint", "last_checkpoint"):
                    checkpoint = torch.load(output / trained[key], weights_only=True)
                    self.assertEqual(checkpoint["negative_bce_weight"], beta)
                    self.assertEqual(checkpoint["bce_loss_weight"], coefficient)
                    self.assertEqual(
                        load_model(output / trained[key]).bce_loss_weight, coefficient
                    )
                model = load_model(output / trained["checkpoint"])
                self.assertEqual(model.negative_bce_weight, beta)
                self.assertEqual(model.bce_loss_weight, coefficient)
                prediction = predict_selection(
                    model, [0] * 7, [0] * 7, [0] * 3, [1, 0, 0, 0], []
                )
                self.assertEqual(prediction["negative_bce_weight"], beta)
                self.assertEqual(prediction["bce_loss_weight"], coefficient)
                self.assertEqual(prediction["sigmoid_score_semantics"], semantics)

                if coefficient == 0:
                    self.assertEqual(
                        prediction["reason"], "highest_ordinal_selection_score"
                    )
                    self.assertEqual(
                        prediction["probability_semantics"],
                        "per_action_ordinal_selection_score",
                    )
                data = load_dataset(source)
                split = load_scene_split(data, config["split"], source)
                state = torch.from_numpy(
                    make_state(
                        data["initial_q"],
                        data["selected_ik"],
                        data["target_position"],
                        data["target_quaternion"],
                    )
                )
                with torch.no_grad():
                    logits = model(
                        state, torch.as_tensor(data["obstacles"], dtype=torch.float32)
                    )
                    labels = torch.as_tensor(data["labels"], dtype=logits.dtype)
                    for phase, index in split.items():
                        per_label = (
                            torch.nn.functional.binary_cross_entropy_with_logits(
                                logits[index], labels[index], reduction="none"
                            )
                        )
                        raw_bce = float(per_label.mean())
                        weighted_bce = float(
                            (
                                per_label * (labels[index] + beta * (1 - labels[index]))
                            ).mean()
                        )
                        metrics = trained["metrics"][phase]
                        self.assertAlmostEqual(metrics["bce_from_logits"], raw_bce)
                        self.assertAlmostEqual(
                            metrics["weighted_bce"], weighted_bce, places=6
                        )
                        self.assertAlmostEqual(
                            metrics["total_loss"],
                            coefficient * weighted_bce + metrics["selection_loss"],
                            places=6,
                        )
                    self.assertAlmostEqual(
                        trained["best_validation_bce"],
                        trained["metrics"]["validation"]["bce_from_logits"],
                    )
                with (output / "seed-11/history.csv").open() as handle:
                    history = list(csv.DictReader(handle))
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
                        if beta == 1:
                            self.assertAlmostEqual(
                                float(row[phase + "_weighted_bce"]),
                                float(row[phase + "_bce"]),
                                places=6,
                            )
                        else:
                            self.assertGreater(
                                float(row[phase + "_weighted_bce"]),
                                float(row[phase + "_bce"]),
                            )
                        self.assertAlmostEqual(
                            float(row[phase + "_total_loss"]),
                            coefficient * float(row[phase + "_weighted_bce"])
                            + float(row[phase + "_selection_loss"]),
                            places=6,
                        )

    def test_cli_training_without_test_reports_only_train_and_validation(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "data"
            source.mkdir()
            path = example_dataset(source, actions=LINK_ACTIONS)
            config = {
                "dataset_sha256": sha256(path),
                "numerical_error_policy": "failure",
                "split": dict(seed=9, train=16, validation=4, test=0),
                "training": dict(
                    seeds=[11],
                    learning_rate=0.001,
                    weight_decay=0.0001,
                    selection_loss_weight=1,
                    mlp_dropout=0.2,
                    batch_size=8,
                    max_epochs=2,
                    patience=2,
                ),
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            output = root / "training"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "safe_ik.application.train_link_selection",
                    "--dataset",
                    str(source),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output),
                    "--device",
                    "cpu",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            report = json.loads((output / "evaluation.json").read_text())
            self.assertEqual(report["selection_loss_weight"], 1)
            self.assertEqual(report["model"], model_spec(0.2))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["model"], model_spec(0.2))
            self.assertEqual(manifest["settings"]["training"]["mlp_dropout"], 0.2)
            self.assertIn("BCE + 1 *", report["optimization_loss"])
            trained = report["runs"][0]
            self.assertEqual(trained["selection_loss_weight"], 1)
            self.assertEqual(trained["mlp_dropout"], 0.2)
            checkpoint_path = output / trained["checkpoint"]
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            self.assertEqual(checkpoint["selection_loss_weight"], 1)
            self.assertEqual(checkpoint["model"], model_spec(0.2))
            loaded = load_model(checkpoint_path)
            self.assertTrue(
                all(
                    m.p == 0.2
                    for m in loaded.modules()
                    if isinstance(m, torch.nn.Dropout)
                )
            )
            with (checkpoint_path.parent / "history.csv").open() as handle:
                history = list(csv.DictReader(handle))
            for row in history:
                for phase in ("train", "validation"):
                    self.assertAlmostEqual(
                        float(row[phase + "_total_loss"]),
                        float(row[phase + "_bce"])
                        + float(row[phase + "_selection_loss"]),
                        places=6,
                    )
            for metrics in trained["metrics"].values():
                self.assertAlmostEqual(
                    metrics["total_loss"],
                    metrics["bce_from_logits"] + metrics["selection_loss"],
                    places=6,
                )
            self.assertFalse(report["test_evaluation_performed"])
            self.assertEqual(report["evaluation_splits"], ["train", "validation"])
            self.assertEqual(set(report["runs"][0]["metrics"]), {"train", "validation"})
            self.assertEqual(set(report["baselines"]), {"train", "validation"})
            self.assertFalse(
                any(
                    "test" in key
                    for key in report
                    if key != "test_evaluation_performed"
                )
            )
            self.assertEqual(
                set(json.loads((output / "split.json").read_text())),
                {"train", "validation"},
            )
            selection = json.loads((output / "selection.json").read_text())
            self.assertNotIn("selected_before_test_evaluation", selection)
            self.assertFalse(selection["test_evaluation_performed"])
            self.assertFalse(
                json.loads((output / "complete.json").read_text())[
                    "test_evaluation_performed"
                ]
            )
            text = (output / "report.md").read_text()
            self.assertIn("BCE + 1 ×", text)
            self.assertIn("dropout 0.2", text)
            self.assertIn("No test set was requested or evaluated", text)
            self.assertNotIn("Test success", text)
            with (output / "predictions.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({r["split"] for r in rows}, {"train", "validation"})
            self.assertEqual(len(rows), 20)

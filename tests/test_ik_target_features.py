"""Selected-IK pose features, portable inference and held-out scene isolation."""

import copy
import csv
import json
import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from safe_ik.adapters.panda_kinematics import PandaKinematics
from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.application.train_link_selection import objective_metadata, run
from safe_ik.application.train_link_time_selection import validate_time_settings
from safe_ik.control.panda_controller import PandaTrajectoryController
from safe_ik.domain.ik_contracts import SafeIKCandidate
from safe_ik.domain.link_actions import LINK_ACTIONS, catalog_metadata
from safe_ik.learning.ik_target_features import (
    compute_ik_target_features,
    make_fk_geometry,
    validate_fk_geometry,
)
from safe_ik.learning.link_selection import (
    LinkSuccessNet,
    load_dataset,
    load_model,
    make_state,
    model_spec,
    predict_selection,
    sha256,
    split_scenes,
)
from tests.test_link_selection import example_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/robots/panda.xml"
FRAMES = tuple(f"link{i}" for i in range(2, 7))


class IKTargetFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ik = np.array(
            [
                [0, 0, 0, -np.pi / 2, 0, np.pi / 2, -np.pi / 4],
                [0.3, -0.4, 0.2, -1.8, 0.5, 1.4, -0.2],
                [-1.3, 0.8, -0.7, -0.6, -0.9, 2.5, 1.1],
            ]
        )

    def state(self):
        return make_state(
            self.ik / 2,
            self.ik,
            np.tile([0.4, -0.2, 0.6], (3, 1)),
            np.tile([1.0, 0, 0, 0], (3, 1)),
            ik_target_pose=compute_ik_target_features(self.ik, MODEL),
        )

    def test_features_match_mujoco_and_actual_sequential_targets(self):
        features = compute_ik_target_features(self.ik, MODEL)
        self.assertEqual(features.shape, (3, 45))
        self.assertEqual(features.dtype, np.float32)
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        data = mujoco.MjData(model)
        base = load_panda_config(ROOT / "tests/fixtures/panda.yaml")
        config = replace(
            base,
            trajectory=replace(base.trajectory, stage_frames=FRAMES, stage_task="pose"),
        )
        kin = PandaKinematics(MODEL, base.ee_frame, FRAMES)
        controller = PandaTrajectoryController(
            config,
            kin,
            Mock(),
            tuple(SafeIKCandidate(q, i, 1, 0, 0, 1) for i, q in enumerate(self.ik)),
            Mock(),
        )
        for row, q in enumerate(self.ik):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            for column, frame in enumerate(FRAMES):
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame)
                rotation = data.xmat[body].reshape(3, 3)
                expected = np.r_[data.xpos[body], rotation[:, 0], rotation[:, 1]]
                actual = features[row, column * 9 : (column + 1) * 9]
                with self.subTest(scene=row, frame=frame):
                    np.testing.assert_allclose(actual, expected, atol=6e-8, rtol=0)
                    np.testing.assert_allclose(
                        actual[:3],
                        controller._candidate_stage_targets[row][column],
                        atol=6e-8,
                        rtol=0,
                    )
                    target_rotation = Rotation.from_quat(
                        controller._candidate_stage_orientations[row][column],
                        scalar_first=True,
                    ).as_matrix()
                    np.testing.assert_allclose(
                        actual[3:],
                        np.r_[target_rotation[:, 0], target_rotation[:, 1]],
                        atol=6e-8,
                        rtol=0,
                    )

    def test_pose_inputs_retain_units_and_existing_backbone(self):
        state = torch.tensor(self.state())
        mean, std = np.arange(14), np.full(14, 2.0)
        model = LinkSuccessNet(mean, std, mlp_dropout=0.2, ik_target_features=True)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 27504)
        self.assertEqual(model.cross_attention.num_heads, 1)
        self.assertEqual(model.cross_attention.dropout, 0)
        drops = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        self.assertEqual([m.p for m in drops], [0.2] * 5)
        obstacles = torch.tensor([[[0.25, -0.4, 0.6, 0.07]]] * 3)
        received = {}
        model.state_encoder[0].register_forward_pre_hook(
            lambda module, args: received.update(state=args[0].detach().clone())
        )
        model.obstacle_encoder[0].register_forward_pre_hook(
            lambda module, args: received.update(obstacles=args[0].detach().clone())
        )
        logits = model(state, obstacles)
        self.assertEqual(logits.shape, (3, 16))
        torch.testing.assert_close(
            received["state"][:, :14],
            (state[:, :14] - torch.tensor(mean, dtype=torch.float32)) / 2,
        )
        torch.testing.assert_close(
            received["state"][:, 14:], state[:, 14:], rtol=0, atol=0
        )
        torch.testing.assert_close(received["obstacles"], obstacles, rtol=0, atol=0)
        logits.square().mean().backward()
        self.assertGreater(
            float(model.state_encoder[0].weight.grad[:, 23:].abs().sum()), 0
        )
        with self.assertRaises(ValueError):
            model(state[:, :23], obstacles)

    def test_checkpoint_portability_hash_guard_and_inference_parity(self):
        state = self.state()
        obstacles = np.tile([[[0.1, 0.2, 0.5, 0.08]]], (3, 1, 1)).astype(np.float32)
        model = LinkSuccessNet(mlp_dropout=0.2, ik_target_features=True).eval()
        with TemporaryDirectory() as temp:
            original = Path(temp) / "original"
            original.mkdir()
            shutil.copyfile(MODEL, original / "panda.xml")
            geometry = make_fk_geometry(
                original / "panda.xml", checkpoint_directory=original
            )
            checkpoint = {
                "schema_version": 19,
                "model": model_spec(0.2, ik_target_features=True),
                "fk_geometry": geometry,
                "output_dim": len(LINK_ACTIONS),
                **catalog_metadata(),
                "selection_metric": "validation_selected_success_rate",
                **objective_metadata(
                    {"weight_decay": 0.0001, "selection_loss_weight": 1}
                ),
                "state_dict": model.state_dict(),
            }
            torch.save(checkpoint, original / "model.pt")
            moved = Path(temp) / "moved"
            shutil.move(original, moved)
            loaded = load_model(moved / "model.pt")
            expected = model(torch.tensor(state), torch.tensor(obstacles))
            torch.testing.assert_close(
                loaded(torch.tensor(state), torch.tensor(obstacles)),
                expected,
                rtol=0,
                atol=0,
            )
            result = predict_selection(
                loaded,
                self.ik[0] / 2,
                self.ik[0],
                [0.4, -0.2, 0.6],
                [1, 0, 0, 0],
                obstacles[0],
            )
            np.testing.assert_allclose(
                result["probabilities"],
                expected[0].sigmoid().detach().numpy(),
                rtol=1e-6,
                atol=1e-7,
            )
            self.assertEqual(result["action_index"], int(expected[0].argmax()))
            changed = moved / "panda.xml"
            changed.write_text(changed.read_text() + "\n<!-- modified -->\n")
            with self.assertRaisesRegex(ValueError, "hash|checksum|SHA256|sha256"):
                load_model(moved / "model.pt")
            with self.assertRaisesRegex(ValueError, "hash|checksum|SHA256|sha256"):
                validate_fk_geometry(geometry, checkpoint_directory=moved)
            # An explicit identical geometry is a valid relocation; an altered one is not.
            reloaded = load_model(moved / "model.pt", fk_model_path=MODEL)
            torch.testing.assert_close(
                reloaded(torch.tensor(state), torch.tensor(obstacles)),
                expected,
                rtol=0,
                atol=0,
            )
            never_called = Mock(side_effect=AssertionError("network must be bypassed"))
            with patch(
                "safe_ik.learning.ik_target_features.compute_ik_target_features",
                side_effect=AssertionError("FK must be bypassed"),
            ):
                fallback = predict_selection(never_called, None, None, None, None, None)
            self.assertEqual(fallback["method"], "direct")
            self.assertEqual(fallback["reason"], "no_validated_ik")
            never_called.assert_not_called()

    def test_existing_baseline_checkpoint_remains_23_dimensional(self):
        model = LinkSuccessNet(mlp_dropout=0.2).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 24624)
        state = torch.tensor(self.state()[:, :23])
        obstacles = torch.zeros((3, 0, 4))
        with TemporaryDirectory() as temp:
            path = Path(temp) / "baseline.pt"
            torch.save(
                {
                    "schema_version": 17,
                    "model": model_spec(0.2),
                    "output_dim": 16,
                    **catalog_metadata(),
                    "selection_metric": "validation_selected_success_rate",
                    **objective_metadata(
                        {"weight_decay": 0.0001, "selection_loss_weight": 1}
                    ),
                    "state_dict": model.state_dict(),
                },
                path,
            )
            with patch(
                "safe_ik.learning.ik_target_features.compute_ik_target_features",
                side_effect=AssertionError("baseline does not need FK"),
            ):
                loaded = load_model(path)
                torch.testing.assert_close(
                    model(state, obstacles), loaded(state, obstacles), rtol=0, atol=0
                )
                result = predict_selection(
                    loaded,
                    self.ik[0] / 2,
                    self.ik[0],
                    [0.4, -0.2, 0.6],
                    [1, 0, 0, 0],
                    [],
                )
            self.assertEqual(len(result["probabilities"]), 16)

    def test_fk_rejects_invalid_inputs_and_misidentified_frames(self):
        for values in (np.zeros(7), np.zeros((2, 8)), np.full((1, 7), np.nan)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                compute_ik_target_features(values, MODEL)
        geometry = make_fk_geometry(MODEL)
        changed = copy.deepcopy(geometry)
        changed["link_frames"] = list(reversed(FRAMES))
        with self.assertRaises(ValueError):
            validate_fk_geometry(changed)
        args = (self.ik / 2, self.ik, np.zeros((3, 3)), np.tile([1, 0, 0, 0], (3, 1)))
        for extra in (np.zeros((3, 44)), np.zeros((2, 45)), np.full((3, 45), np.inf)):
            with self.subTest(shape=extra.shape), self.assertRaises(ValueError):
                make_state(*args, ik_target_pose=extra)

    def test_time_training_cannot_silently_ignore_success_only_features(self):
        config = yaml.safe_load(
            (ROOT / "configs/training/panda_link_time_selection.yaml").read_text()
        )
        validate_time_settings(config)
        config["ik_target_features"] = {
            "model_path": str(MODEL),
            "model_sha256": sha256(MODEL),
        }
        with self.assertRaisesRegex(ValueError, "only supported for success training"):
            validate_time_settings(config)

    def test_skip_test_keeps_frozen_partition_but_never_forwards_heldout_rows(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "data"
            source.mkdir()
            path = example_dataset(source, actions=LINK_ACTIONS)
            data = load_dataset(source)
            split = split_scenes(data["labels"], 12, 4, 4, 9)
            with np.load(path) as archive:
                arrays = {name: archive[name] for name in archive.files}
            # This value survives joint normalization and uniquely marks held-out rows.
            arrays["target_position"][split["test"], 0] = 12345.0
            np.savez_compressed(path, **arrays)
            complete = json.loads((source / "complete.json").read_text())
            complete["data_sha256"][path.name] = sha256(path)
            (source / "complete.json").write_text(json.dumps(complete))
            record = {
                key: {
                    "case_ids": data["case_id"][indices].tolist(),
                    "indices": indices.tolist(),
                }
                for key, indices in split.items()
            }
            frozen = source / "split.json"
            frozen.write_text(json.dumps(record))
            config = {
                "dataset_sha256": sha256(path),
                "numerical_error_policy": "failure",
                "split": {
                    "train": 12,
                    "validation": 4,
                    "test": 4,
                    "seed": 9,
                    "file": frozen.name,
                    "sha256": sha256(frozen),
                },
                "ik_target_features": {
                    "model_path": str(MODEL),
                    "model_sha256": sha256(MODEL),
                },
                "training": {
                    "seeds": [11],
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "batch_size": 4,
                    "max_epochs": 1,
                    "patience": 1,
                    "early_stopping": True,
                    "selection_loss_weight": 1,
                    "mlp_dropout": 0.2,
                    "negative_bce_weight": 1,
                    "bce_loss_weight": 1,
                },
            }
            config_path = root / "training.yaml"
            config_path.write_text(yaml.safe_dump(config))
            original_forward = LinkSuccessNet.forward
            forwards = []

            def guarded_forward(model, state, obstacles, mask=None):
                self.assertFalse(bool((state[:, 14] == 12345.0).any()))
                self.assertEqual(state.shape[1], 68)
                forwards.append(len(state))
                return original_forward(model, state, obstacles, mask)

            output = root / "training"
            with patch.object(LinkSuccessNet, "forward", guarded_forward):
                report = run(
                    source, config_path, output, "cpu", skip_test_evaluation=True
                )
            self.assertTrue(forwards)
            self.assertFalse(report["test_evaluation_performed"])
            self.assertEqual(report["evaluation_splits"], ["train", "validation"])
            self.assertEqual(report["parameter_count"], 27504)
            self.assertEqual(set(report["runs"][0]["metrics"]), {"train", "validation"})
            saved_split = json.loads((output / "split.json").read_text())
            self.assertEqual(
                saved_split["test"]["case_ids"], record["test"]["case_ids"]
            )
            self.assertEqual(saved_split["test"]["indices"], record["test"]["indices"])
            expected_ids = set(
                record["train"]["case_ids"] + record["validation"]["case_ids"]
            )
            with np.load(output / "seed-11/predictions.npz") as predictions:
                self.assertEqual(set(predictions["case_id"]), expected_ids)
                self.assertEqual(predictions["probabilities"].shape, (16, 16))
            with np.load(output / "ik_target_features.npz") as features:
                self.assertEqual(set(features["case_id"]), expected_ids)
            with np.load(output / "effective_labels.npz") as labels:
                self.assertEqual(set(labels["case_id"]), expected_ids)
            with (output / "predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["case_id"] for row in rows}, expected_ids)
            self.assertEqual({row["split"] for row in rows}, {"train", "validation"})
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse(manifest["test_evaluation_performed"])
            self.assertEqual(manifest["settings"]["split"]["sha256"], sha256(frozen))


if __name__ == "__main__":
    unittest.main()

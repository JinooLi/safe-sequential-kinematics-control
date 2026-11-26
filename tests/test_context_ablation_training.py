"""Train/export/reload the context ablation without evaluating reserved scenes."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import yaml

from safe_ik.application.train_link_selection import run, validate_settings
from safe_ik.domain.link_actions import LINK_ACTIONS
from safe_ik.learning.link_selection import (
    LinkSuccessNet,
    load_dataset,
    load_model,
    make_state,
    sha256,
)
from tests.test_link_selection import example_dataset


class ContextAblationTrainingTests(unittest.TestCase):
    def test_mode_survives_training_export_reload_without_test_forward(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "data"
            source.mkdir()
            dataset = example_dataset(source, actions=LINK_ACTIONS)
            data = load_dataset(source)
            partitions = {
                "train": np.arange(12),
                "validation": np.arange(12, 16),
                "test": np.arange(16, 20),
            }
            frozen = source / "split.json"
            frozen.write_text(
                json.dumps(
                    {
                        name: {
                            "indices": ids.tolist(),
                            "case_ids": data["case_id"][ids].tolist(),
                        }
                        for name, ids in partitions.items()
                    }
                )
            )
            config = {
                "dataset_sha256": sha256(dataset),
                "numerical_error_policy": "failure",
                "split": {
                    "train": 12,
                    "validation": 4,
                    "test": 4,
                    "seed": 9,
                    "file": frozen.name,
                    "sha256": sha256(frozen),
                },
                "training": {
                    "seeds": [11],
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "batch_size": 4,
                    "max_epochs": 1,
                    "patience": 1,
                    "early_stopping": True,
                    "mlp_dropout": 0.2,
                    "negative_bce_weight": 1.0,
                    "bce_loss_weight": 1.0,
                    "selection_loss_weight": 1.0,
                },
            }
            state = make_state(
                data["initial_q"][:16],
                data["selected_ik"][:16],
                data["target_position"][:16],
                data["target_quaternion"][:16],
            )
            allowed = torch.as_tensor(state[:, :7], dtype=torch.float32)
            original_forward = LinkSuccessNet.forward

            def guarded_forward(model, states, obstacles, mask=None):
                membership = (states[:, None, :7] == allowed[None]).all(-1)
                self.assertTrue(bool(membership.any(1).all()))
                return original_forward(model, states, obstacles, mask)

            for mode in ("full", "zero"):
                with self.subTest(obstacle_context=mode):
                    config["obstacle_context"] = mode
                    config_path = root / f"{mode}.yaml"
                    config_path.write_text(yaml.safe_dump(config))
                    output = root / mode
                    with patch.object(LinkSuccessNet, "forward", guarded_forward):
                        report = run(
                            source,
                            config_path,
                            output,
                            "cpu",
                            skip_test_evaluation=True,
                        )
                    self.assertFalse(report["test_evaluation_performed"])
                    self.assertEqual(report["parameter_count"], 24624)
                    checkpoint = output / "seed-11/best.pt"
                    saved = torch.load(checkpoint, weights_only=True)
                    self.assertEqual(
                        saved["schema_version"], 17 if mode == "full" else 22
                    )
                    self.assertEqual(
                        saved["model"].get("obstacle_context", "full"), mode
                    )
                    model = load_model(checkpoint)
                    self.assertEqual(model.obstacle_context, mode)
                    states = torch.as_tensor(state)
                    obstacles = torch.as_tensor(
                        data["obstacles"][:16], dtype=torch.float32
                    )
                    with torch.no_grad():
                        expected = model(states, obstacles).sigmoid().numpy()
                        if mode == "zero":
                            torch.testing.assert_close(
                                model(states, obstacles),
                                model(states, obstacles + 0.75),
                                rtol=0,
                                atol=0,
                            )
                    with np.load(output / "seed-11/predictions.npz") as predictions:
                        self.assertEqual(
                            predictions["case_id"].tolist(),
                            data["case_id"][:16].tolist(),
                        )
                        np.testing.assert_allclose(
                            predictions["probabilities"], expected, rtol=1e-6, atol=1e-7
                        )
            for bad in (None, True, "off"):
                config["obstacle_context"] = bad
                with self.assertRaises(ValueError):
                    validate_settings(config)
            config["obstacle_context"] = "zero"
            config["relative_geometry"] = {}
            with self.assertRaisesRegex(ValueError, "original 23D"):
                validate_settings(config)


if __name__ == "__main__":
    unittest.main()

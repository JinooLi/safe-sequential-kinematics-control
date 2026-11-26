"""Training integration: frozen holdout isolation and portable action scoring."""

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import yaml

from safe_ik.application.train_link_selection import run
from safe_ik.domain.link_actions import LINK_ACTIONS, catalog_arrays, catalog_metadata
from safe_ik.learning.link_selection import load_dataset, load_model, make_state, predict_selection, sha256
from safe_ik.learning.relative_geometry import RelativeGeometrySuccessNet
from safe_ik.learning.relative_geometry_features import compute_relative_geometry_features
from tests.test_link_selection import example_dataset


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/robots/panda.xml"
CONTROLLER = ROOT / "configs/panda.yaml"


class RelativeGeometryTrainingTests(unittest.TestCase):
    def test_training_preserves_frozen_test_and_reloads_selected_action_scores(self):
        for mode in ("local", "reordered_control", "global_query"):
            with self.subTest(query_conditioning=mode):
                self._assert_training_preserves_frozen_test(mode)

    def _assert_training_preserves_frozen_test(self, mode):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "data"
            source.mkdir()
            dataset_path = example_dataset(source, size=12, actions=LINK_ACTIONS)
            with np.load(dataset_path) as archive:
                arrays = {key: archive[key] for key in archive.files}
            # Held-out states remain valid serialized inputs, but must never
            # reach FK or model execution while test evaluation is disabled.
            arrays["initial_q"][10:] = 12345.0
            arrays["selected_ik"][10:] = -12345.0
            arrays["target_position"][10:, 0] = 12345.0
            np.savez_compressed(dataset_path, **arrays)
            complete_path = source / "complete.json"
            complete = json.loads(complete_path.read_text())
            complete["data_sha256"][dataset_path.name] = sha256(dataset_path)
            complete_path.write_text(json.dumps(complete))
            data = load_dataset(source)
            partitions = {
                "train": np.arange(8),
                "validation": np.arange(8, 10),
                "test": np.arange(10, 12),
            }
            frozen = source / "split.json"
            record = {
                name: {
                    "indices": indices.tolist(),
                    "case_ids": data["case_id"][indices].tolist(),
                }
                for name, indices in partitions.items()
            }
            frozen.write_text(json.dumps(record))
            config = {
                "dataset_sha256": sha256(dataset_path),
                "numerical_error_policy": "failure",
                "split": {
                    "train": 8,
                    "validation": 2,
                    "test": 2,
                    "seed": 9,
                    "file": frozen.name,
                    "sha256": sha256(frozen),
                },
                "relative_geometry": {
                    "model_path": str(MODEL),
                    "model_sha256": sha256(MODEL),
                    "capsule_config_path": str(CONTROLLER),
                    "capsule_config_sha256": sha256(CONTROLLER),
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
            if mode != "local":
                config["relative_geometry"]["query_conditioning"] = mode
            config_path = root / "training.yaml"
            config_path.write_text(yaml.safe_dump(config))
            feature_calls = []
            forward_sizes = []
            original_forward = RelativeGeometrySuccessNet.forward
            allowed_q = torch.as_tensor(data["initial_q"][:10], dtype=torch.float32)

            def guarded_features(q, ik, model_path, geometry):
                np.testing.assert_array_equal(q, data["initial_q"][:10])
                np.testing.assert_array_equal(ik, data["selected_ik"][:10])
                result = compute_relative_geometry_features(q, ik, model_path, geometry)
                feature_calls.append(len(result))
                return result

            def guarded_forward(model, state, obstacles, mask=None):
                # This also rejects the zero-filled reserved rows in the
                # trainer's full-size tensor, rather than only the sentinel.
                membership = (state[:, None, :7] == allowed_q[None]).all(dim=-1)
                self.assertTrue(bool(membership.any(dim=1).all()))
                self.assertEqual(state.shape[1], model.relative_geometry["state_dim"])
                forward_sizes.append(len(state))
                return original_forward(model, state, obstacles, mask)

            output = root / "training"
            with patch(
                "safe_ik.application.train_link_selection.compute_relative_geometry_features",
                side_effect=guarded_features,
            ), patch.object(RelativeGeometrySuccessNet, "forward", guarded_forward):
                report = run(source, config_path, output, "cpu", skip_test_evaluation=True)

            self.assertEqual(feature_calls, [10])
            self.assertTrue(forward_sizes)
            self.assertFalse(report["test_evaluation_performed"])
            self.assertEqual(report["evaluation_splits"], ["train", "validation"])
            self.assertEqual(set(report["runs"][0]["metrics"]), {"train", "validation"})
            self.assertEqual(report["runs"][0]["epochs"], 1)
            saved_split = json.loads((output / "split.json").read_text())
            for name in partitions:
                self.assertEqual(saved_split[name]["case_ids"], record[name]["case_ids"])
                self.assertEqual(saved_split[name]["indices"], record[name]["indices"])

            expected_ids = set(data["case_id"][:10])
            with np.load(output / "relative_geometry_features.npz") as artifact:
                self.assertEqual(set(artifact["case_id"]), expected_ids)
                self.assertEqual(artifact["case_id"].tolist(), data["case_id"][:10].tolist())
                features = artifact["features"].copy()
            with np.load(output / "effective_labels.npz") as artifact:
                self.assertEqual(set(artifact["case_id"]), expected_ids)
                self.assertEqual(artifact["labels"][0, 0], 0)
                self.assertFalse(artifact["original_label_valid"][0, 0])
                np.testing.assert_array_equal(artifact["labels"], data["labels"][:10])
            with np.load(output / "seed-11/predictions.npz") as artifact:
                self.assertEqual(set(artifact["case_id"]), expected_ids)
                probabilities = artifact["probabilities"].copy()
                chosen = artifact["chosen_action_index"].copy()
                self.assertEqual(probabilities.shape, (10, 16))
                for name, expected in catalog_arrays(LINK_ACTIONS).items():
                    np.testing.assert_array_equal(artifact[name], expected)
            with (output / "predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["case_id"] for row in rows}, expected_ids)
            self.assertEqual({row["split"] for row in rows}, {"train", "validation"})
            self.assertEqual(len(rows), 10)

            selection = json.loads((output / "selection.json").read_text())
            self.assertEqual(selection["criterion"], "validation_selected_success_rate")
            self.assertEqual(selection["selected_seed"], 11)
            self.assertFalse(selection["test_evaluation_performed"])
            checkpoint_path = output / selection["checkpoint"]
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["schema_version"], 20 if mode == "local" else 21)
            self.assertEqual(checkpoint["model"].get("query_conditioning", "local"), mode)
            self.assertEqual(checkpoint["numerical_error_policy"], "failure")
            self.assertEqual(checkpoint["selection_loss_weight"], 1)
            self.assertEqual(checkpoint["model"]["mlp_dropout"], 0.2)
            for key, expected in catalog_metadata(LINK_ACTIONS).items():
                self.assertEqual(checkpoint[key], expected)
            self.assertEqual(checkpoint["best_validation_selected_successes"], selection["validation_selected_successes"])
            loaded = load_model(checkpoint_path)
            self.assertEqual(loaded.actions, LINK_ACTIONS)
            self.assertEqual(sum(parameter.numel() for parameter in loaded.parameters()), report["parameter_count"])
            state = np.concatenate((make_state(
                data["initial_q"][:10], data["selected_ik"][:10],
                data["target_position"][:10], data["target_quaternion"][:10],
            ), features), axis=1)
            with torch.no_grad():
                replay = loaded(torch.as_tensor(state), torch.as_tensor(data["obstacles"][:10], dtype=torch.float32)).sigmoid().numpy()
            np.testing.assert_array_equal(replay.argmax(axis=1), chosen)
            np.testing.assert_allclose(replay, probabilities, atol=1e-7, rtol=1e-6)
            result = predict_selection(
                loaded, data["initial_q"][8], data["selected_ik"][8],
                data["target_position"][8], data["target_quaternion"][8], data["obstacles"][8],
            )
            self.assertEqual(result["action_index"], int(chosen[8]))
            np.testing.assert_allclose(result["probabilities"], probabilities[8], atol=1e-7, rtol=1e-6)

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse(manifest["test_evaluation_performed"])
            self.assertEqual(manifest["settings"]["split"]["sha256"], sha256(frozen))
            self.assertEqual(manifest["relative_geometry"]["model_sha256"], sha256(MODEL))
            self.assertEqual(manifest["relative_geometry"]["capsule_config_sha256"], sha256(CONTROLLER))
            self.assertEqual(sha256(dataset_path), config["dataset_sha256"])


if __name__ == "__main__":
    unittest.main()

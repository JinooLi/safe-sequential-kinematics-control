"""Only the approved saved-data weight may differ from the 1x experiment."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from safe_ik.adapters.rdf_checkpoint import file_sha256
from safe_ik.application.panda_rdf_hand_continuation import (
    REUSE_FILES,
    copy_reweighted_data,
    validate_protocol,
    verify_reuse,
)


class HandReweightTest(unittest.TestCase):
    def protocol(self):
        root = Path(__file__).resolve().parents[1]
        return json.loads(
            (root / "configs/panda_rdf_hand_weight4_20260930.json").read_text()
        )

    def fixture(self, source):
        p = self.protocol()
        original = copy.deepcopy(p)
        original.pop("reuse_files")
        original.update(purpose="targeted_append_from_weighted_800", new_weight=1)
        for name in REUSE_FILES:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"preserved evidence")
        (source / "protocol.json").write_text(json.dumps(original))
        (source / "hand/comparison.json").write_text('{"complete": true}')
        (source / "hand/data.report.json").write_text(
            '{"new_weight": 1, "new_count": 5000}'
        )
        data = dict(
            points=np.random.default_rng(7).uniform(-0.01, 0.01, (5000, 3)),
            truth=np.full(5000, 0.005),
            targets=np.full(5000, 0.004, dtype=np.float32),
            weights=np.ones(5000),
        )
        np.savez(source / "hand/training.npz", **data)
        np.savez(
            source / "hand/batch_indices.npz",
            indices=np.arange(5000),
            offsets=np.array([0, 1280, 2560, 3840, 5000]),
        )
        p["reuse_files"] = {name: file_sha256(source / name) for name in REUSE_FILES}
        return p, data, SimpleNamespace(center=np.zeros(3), radius=0.2)

    def test_fixed_4x_protocol_rejects_budget_changes(self):
        p = self.protocol()
        validate_protocol(p)
        for key, value in [
            ("new_weight", 8),
            ("new_weight", 1),
            ("parent_step", 804),
            ("passes", 4),
            ("seed", 20261001),
            ("new_count", 20000),
        ]:
            changed = copy.deepcopy(p)
            changed[key] = value
            with self.assertRaisesRegex(ValueError, "fixed protocol"):
                validate_protocol(changed)
        p["reuse_files"].pop("hand/training.npz")
        with self.assertRaisesRegex(ValueError, "reuse coverage"):
            validate_protocol(p)

    def test_copy_preserves_all_inputs_and_only_changes_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "source", Path(tmp) / "dest"
            p, data, model = self.fixture(source)
            dest.mkdir()
            with (
                patch(
                    "safe_ik.application.panda_rdf_hand_continuation.scan_targets",
                    side_effect=AssertionError("no relabel"),
                ),
                patch(
                    "safe_ik.application.panda_rdf_hand_continuation.draw_mining_points",
                    side_effect=AssertionError("no sample"),
                ),
            ):
                points, targets, weights, offsets, report = copy_reweighted_data(
                    source, dest, p, model, np.dtype("float32")
                )
            for name in [
                "mining.npz",
                "proposals.npz",
                "scan_cloud.npz",
                "batch_indices.npz",
            ]:
                self.assertEqual(
                    file_sha256(dest / name), p["reuse_files"]["hand/" + name]
                )
            with np.load(dest / "training.npz") as saved:
                for key in ["points", "targets", "truth"]:
                    np.testing.assert_array_equal(saved[key], data[key])
            np.testing.assert_array_equal(weights, np.full(5000, 4.0))
            self.assertEqual(report["effective_new_weight_sum"], 20000.0)
            self.assertEqual(report["generated_points_this_run"], 0)
            for name, digest in p["reuse_files"].items():
                self.assertEqual(file_sha256(source / name), digest)

    def test_saved_hash_and_nonunit_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            p, data, model = self.fixture(source)
            data["weights"][:] = 4
            np.savez(source / "hand/training.npz", **data)
            with self.assertRaisesRegex(ValueError, "saved 1x SHA"):
                verify_reuse(source, p, model, np.dtype("float32"))
            p["reuse_files"]["hand/training.npz"] = file_sha256(
                source / "hand/training.npz"
            )
            with self.assertRaisesRegex(ValueError, "saved unit weights"):
                verify_reuse(source, p, model, np.dtype("float32"))


if __name__ == "__main__":
    unittest.main()

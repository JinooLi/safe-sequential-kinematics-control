"""Checkpoint integrity and compatibility are checked without optional torch."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from safe_ik.adapters.rdf_checkpoint import (
    RidgeState,
    array_sha256,
    load_checkpoint,
    model_signature,
    save_checkpoint,
)
from safe_ik.domain.panda_rdf import BernsteinRDF


class RidgeCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "state"
        self.model = BernsteinRDF(np.array([0.1, -0.2, 0.3]), 1.4, np.zeros((2,) * 3))
        self.context = {"input_sha256": "abc", "batch_sizes": [3, 2]}
        self.prefix = lambda n: {"near": array_sha256(np.arange(n * 3))}
        self.state = RidgeState(
            np.arange(8, dtype=float),
            np.eye(8),
            {
                "schema_version": 1,
                "model": model_signature(self.model, 1e-4),
                "context": self.context,
                "batch_prefix_sha256": self.prefix(2),
                "completed_steps": 2,
                "consumed_samples": 10,
            },
        )

    def save(self, state=None):
        return save_checkpoint(self.root, state or self.state, self.model, 1e-4)

    def load(self, **kwargs):
        options = dict(
            path=self.root,
            model=self.model,
            regularization=1e-4,
            context=self.context,
            prefix_hashes=self.prefix,
        )
        return load_checkpoint(**(options | kwargs))

    def test_roundtrip_is_exact_readonly_and_non_pickle(self):
        meta = self.save()
        restored = self.load()
        np.testing.assert_array_equal(restored.weights, self.state.weights)
        np.testing.assert_array_equal(restored.covariance, self.state.covariance)
        self.assertFalse(restored.covariance.flags.writeable)
        self.assertEqual(restored.metadata, meta)
        self.assertEqual(
            set(p.name for p in self.root.iterdir()),
            {"metadata.json", "normalized_weights.npy", "inverse_information.npy"},
        )

    def test_snapshot_never_overwritten(self):
        self.save()
        with self.assertRaises(FileExistsError):
            self.save()

    def test_corrupt_state_or_incomplete_snapshot_rejected(self):
        self.save()
        with (self.root / "inverse_information.npy").open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\xff")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.load()
        with self.assertRaises(FileNotFoundError):
            self.load(path=self.root / "missing")

    def test_context_prefix_and_model_mismatch_rejected(self):
        self.save()
        for overrides in (
            {"context": {"different_pool": True}},
            {"prefix_hashes": lambda n: {"near": "different"}},
            {"regularization": 0.01},
            {"model": BernsteinRDF(np.zeros(3), 1.4, np.zeros((2,) * 3))},
            {"model": BernsteinRDF(self.model.center, 2.0, np.zeros((2,) * 3))},
            {"model": BernsteinRDF(self.model.center, 1.4, np.zeros((3,) * 3))},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.load(**overrides)

    def test_invalid_state_rejected_before_publication(self):
        for weights, covariance, meta in (
            (self.state.weights.astype(np.float32), np.eye(8), self.state.metadata),
            (np.full(8, np.nan), np.eye(8), self.state.metadata),
            (self.state.weights, np.full((8, 8), np.inf), self.state.metadata),
            (self.state.weights, np.triu(np.ones((8, 8))), self.state.metadata),
            (
                self.state.weights,
                np.eye(8),
                self.state.metadata | {"completed_steps": -1},
            ),
            (
                self.state.weights,
                np.eye(8),
                self.state.metadata | {"consumed_samples": 0},
            ),
        ):
            with self.assertRaises(ValueError):
                self.save(RidgeState(weights, covariance, meta))
            self.assertFalse(self.root.exists())

    def test_unknown_schema_rejected(self):
        self.save()
        file = self.root / "metadata.json"
        meta = json.loads(file.read_text())
        meta["schema_version"] = 2
        file.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            self.load()


if __name__ == "__main__":
    unittest.main()

"""Fixed-input ablations must preserve the reference batches and all other settings."""

import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from safe_ik.application.panda_rdf_training import TrainingOptions
from safe_ik.application.refit_panda_rdf import prepare_fixed_batches


class FixedBatchTest(unittest.TestCase):
    def setUp(self):
        self.reference = TrainingOptions(
            basis_count=3, updates=2, near_batch=3, random_batch=2, pool_size=20
        )
        self.near = np.array([[0, 1, 2], [3, 4, 5]])
        self.random = np.array([[0, 1], [2, 3]])

    def plan(self, **changes):
        return prepare_fixed_batches(
            self.near,
            self.random,
            self.reference,
            replace(self.reference, **changes),
            20,
            20,
        )

    def test_original_batches_unchanged_and_not_aliased(self):
        a, b, plan = self.plan(basis_count=4)
        np.testing.assert_array_equal(a, self.near)
        np.testing.assert_array_equal(b, self.random)
        self.assertFalse(np.shares_memory(a, self.near))
        self.assertEqual(plan["appended_updates"], 0)
        self.assertIsNone(plan["extra_batch_seed"])
        json.dumps(plan, allow_nan=False)

    def test_append_preserves_prefix_is_deterministic_and_nested(self):
        a, b, plan = self.plan(updates=4, basis_count=4)
        long_a, long_b, _ = self.plan(updates=8, basis_count=4)
        np.testing.assert_array_equal(a[:2], self.near)
        np.testing.assert_array_equal(b[:2], self.random)
        np.testing.assert_array_equal(a, long_a[:4])
        np.testing.assert_array_equal(b, long_b[:4])
        again_a, again_b, again_plan = self.plan(updates=4, basis_count=4)
        np.testing.assert_array_equal(a, again_a)
        np.testing.assert_array_equal(b, again_b)
        self.assertEqual(plan, again_plan)
        self.assertEqual(plan["consumed_samples"], 20)
        self.assertEqual(plan["extra_batch_spawn_key"], [4, 6])
        for ids in (a, b):
            self.assertTrue(np.all((ids >= 0) & (ids < 20)))
            self.assertTrue(all(len(np.unique(row)) == len(row) for row in ids))

    def test_shortened_prefix_or_changed_training_settings_rejected(self):
        for changes in (
            {"updates": 1},
            {"regularization": 0.1},
            {"near_batch": 4},
            {"pool_size": 30},
            {"normal_count": 10},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan(**changes)

    def test_malformed_indices_or_wrong_pool_size_rejected(self):
        for near, count in (
            (self.near.astype(float), 20),
            (self.near[:-1], 20),
            (np.array([[0, 0, 1], [2, 3, 4]]), 20),
            (np.array([[0, 1, 20], [2, 3, 4]]), 20),
            (self.near, 21),
        ):
            with self.subTest(count=count), self.assertRaises(ValueError):
                prepare_fixed_batches(
                    near, self.random, self.reference, self.reference, count, 20
                )

    def test_different_append_seed_keeps_original_prefix(self):
        options = replace(self.reference, updates=4)
        a, b, _ = self.plan(updates=4)
        other_a, other_b, _ = prepare_fixed_batches(
            self.near, self.random, self.reference, options, 20, 20, 1
        )
        np.testing.assert_array_equal(other_a[:2], a[:2])
        np.testing.assert_array_equal(other_b[:2], b[:2])
        self.assertFalse(np.array_equal(other_a[2:], a[2:]))
        with self.assertRaises(ValueError):
            prepare_fixed_batches(
                self.near, self.random, self.reference, options, 20, 20, -1
            )

    def test_800_config_only_changes_update_count(self):
        config = Path(__file__).resolve().parents[1] / "configs"
        old = yaml.safe_load((config / "panda_rdf_training_bp24_400.yaml").read_text())
        new = yaml.safe_load((config / "panda_rdf_training_bp24_800.yaml").read_text())
        self.assertEqual(new, old | {"updates": 800})


if __name__ == "__main__":
    unittest.main()

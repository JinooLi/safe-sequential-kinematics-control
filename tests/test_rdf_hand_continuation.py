"""Bounded independent hard-region sampling without validation-data fitting."""

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from safe_ik.application.panda_rdf_failure_study import rng_for
from safe_ik.application.panda_rdf_hand_continuation import (
    draw_local_proposals,
    select_anchors,
    select_new_points,
    validate_protocol,
)


class HandSamplingTest(unittest.TestCase):
    def test_anchor_mask_includes_negative_predictions_only_reference_exterior(self):
        points = np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [2, 0, 0], [0, 0, 0]], dtype=float
        )
        truth = np.array([0.002, -0.001, 0.061, 0.002, 0.002])
        prediction = np.array([-0.007, 0.02, 0.1, -0.007, 0.001])
        np.testing.assert_array_equal(
            select_anchors(points, truth, prediction, np.zeros(3), 1.0),
            [True, False, False, False, False],
        )

    def test_local_proposal_reproducible_within_radius_and_other_seed_distinct(self):
        anchors = np.array([[0.1, 0, 0], [0, 0.1, 0]])
        a, ids = draw_local_proposals(anchors, rng_for(20260930, 8, 202), 100, 0.01)
        b, ids2 = draw_local_proposals(anchors, rng_for(20260930, 8, 202), 100, 0.01)
        c, _ = draw_local_proposals(anchors, rng_for(20260930, 8, 203), 100, 0.01)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ids, ids2)
        self.assertTrue(np.all(np.linalg.norm(a - anchors[ids], axis=1) <= 0.01))
        self.assertFalse(np.array_equal(a, c))
        with self.assertRaisesRegex(ValueError, "anchors"):
            draw_local_proposals(np.empty((0, 3)), rng_for(0, 0, 0), 10, 0.01)

    def test_selection_excludes_old_duplicates_interior_and_out_of_domain(self):
        points = np.array(
            [
                [0, 0, 0],
                [0.1, 0, 0],
                [0.1, 0, 0],
                [0.2, 0, 0],
                [2, 0, 0],
                [0.3, 0, 0],
                [0.4, 0, 0],
            ]
        )
        truth = np.array([0.01, 0.01, 0.01, -0.01, 0.01, 0.061, 0.02])
        chosen, mask, dist = select_new_points(
            points, truth, np.zeros(3), 1.0, np.zeros((1, 3)), 2
        )
        np.testing.assert_array_equal(chosen, [1, 6])
        self.assertTrue(np.all(dist[chosen] > 1e-12))
        self.assertEqual(mask.sum(), 2)
        with self.assertRaisesRegex(ValueError, "fixed candidate budget"):
            select_new_points(points, truth, np.zeros(3), 1.0, np.zeros((1, 3)), 3)

    def test_protocol_does_not_silently_increase_steps_or_weight(self):
        root = Path(__file__).resolve().parents[1]
        p = json.loads(
            (root / "configs/panda_rdf_hand_targeted_20260930.json").read_text()
        )
        validate_protocol(p)
        for key, value in [
            ("new_count", 10000),
            ("new_weight", 4.0),
            ("passes", 2),
            ("seed", 20260931),
            ("error_threshold", 0.007),
            ("label_method", "mesh"),
            ("local_radius", 0.02),
        ]:
            altered = copy.deepcopy(p)
            altered[key] = value
            with self.assertRaisesRegex(ValueError, "fixed protocol"):
                validate_protocol(altered)


if __name__ == "__main__":
    unittest.main()

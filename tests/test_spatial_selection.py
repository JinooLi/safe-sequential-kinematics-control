"""Spatial votes must preserve strict-majority and frozen-scene semantics."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from safe_ik.domain.link_actions import LINK_ACTIONS, LINK_PAIRS, action_id
from safe_ik.learning.spatial_selection import (
    SpatialSelectionConfig, SpatialSequenceSelector, load_spatial_selection_config,
)


def prediction(index):
    scores = [0.1] * len(LINK_ACTIONS)
    scores[index] = 0.9
    return dict(action_index=index, probabilities=scores)


class SpatialSelectionTests(unittest.TestCase):
    def selector(self, *, config=None, position=17, **overrides):
        context = dict(
            model=SimpleNamespace(actions=LINK_ACTIONS),
            selected_ik=np.arange(7) / 10,
            target_position=[0.1, 0.2, 0.3],
            target_quaternion=[1.0, 0.0, 0.0, 0.0],
            obstacles=[[0.7, 0.8, 0.9, 0.2]],
        )
        context.update(overrides)
        return SpatialSequenceSelector(config or SpatialSelectionConfig(), position, **context)

    def choose(self, votes, current=(2, 3), **kwargs):
        selector = self.selector(**kwargs)
        with patch("safe_ik.learning.spatial_selection.predict_selection", side_effect=[prediction(i) for i in votes]):
            return selector.select(np.zeros(7), current)

    def test_half_does_not_switch_but_26_of_50_does(self):
        tied = self.choose([1] * 25 + [2] * 25)
        self.assertEqual(tied["action_index"], 0)
        self.assertIsNone(tied["spatial_vote"]["majority_action_index"])
        self.assertEqual(tied["reason"], "spatial_no_majority")
        majority = self.choose([1] * 26 + [2] * 24)
        self.assertEqual(majority["action_index"], 1)
        self.assertEqual(majority["spatial_vote"]["majority_action_index"], 1)
        self.assertEqual(majority["spatial_vote"]["vote_fractions"][1], 0.52)

    def test_plurality_and_current_majority_preserve_complete_current_route(self):
        plurality = self.choose([1] * 24 + [2] * 20 + [3] * 6, current=("link4", "link6"))
        self.assertEqual(plurality["link_frames"], ["link4", "link6"])
        self.assertEqual(plurality["spatial_vote"]["winner_action_index"], 1)
        kept = self.choose([8] * 26 + [2] * 24, current=("link4", "link6"))
        self.assertEqual(kept["reason"], "spatial_current_majority")
        self.assertEqual(kept["action_index"], 8)

    def test_direct_and_single_action_metadata_and_odd_sample_count(self):
        for selected in (10, 15):
            result = self.choose([selected] * 3 + [0] * 2, config=SpatialSelectionConfig(sample_count=5))
            links = LINK_ACTIONS[selected]
            self.assertEqual(result["action_id"], action_id(links))
            self.assertEqual(result["link_frames"], [f"link{i}" for i in links])
            self.assertIsNone(result["pair_index"])
            self.assertEqual(result["method"], "sequential" if links else "direct")
            self.assertNotIn("probabilities", result)
        retained = self.choose([0] * 25 + [1] * 25, current=())
        self.assertEqual(retained["action_id"], "direct")

    def test_sampling_is_exact_untruncated_gaussian_and_context_is_frozen(self):
        ik = np.arange(7) / 10
        obstacles = np.asarray([[0.7, 0.8, 0.9, 0.2]])
        selector = self.selector(selected_ik=ik, obstacles=obstacles)
        ik[:] = -99
        obstacles[:] = -99
        q = np.full(7, 2.8973)  # Some hypothetical samples cross a joint limit.
        original_q = q.copy()
        expected_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261126, 17])))
        with patch("safe_ik.learning.spatial_selection.predict_selection", return_value=prediction(1)) as infer:
            for poll in range(2):
                expected = q + expected_rng.normal(0.0, 0.1 / 3, size=(50, 7))
                selected = selector.select(q, (2, 3))
                vote = selected["spatial_vote"]
                np.testing.assert_array_equal(vote["samples"], expected)
                self.assertEqual(vote["poll_index"], poll)
                self.assertEqual(vote["sigma_rad"], 0.1 / 3)
                self.assertTrue(np.any(np.asarray(vote["samples"]) > q))
                self.assertTrue(np.any(np.linalg.norm(expected - q, axis=1) > 0.1))
                for call in infer.call_args_list[poll * 50:(poll + 1) * 50]:
                    np.testing.assert_array_equal(call.args[2], np.arange(7) / 10)
                    np.testing.assert_array_equal(call.args[3], [0.1, 0.2, 0.3])
                    np.testing.assert_array_equal(call.args[4], [1, 0, 0, 0])
                    np.testing.assert_array_equal(call.args[5], [[0.7, 0.8, 0.9, 0.2]])
            self.assertEqual(infer.call_count, 100)
        np.testing.assert_array_equal(q, original_q)

    def test_scene_random_stream_is_independent_of_worker_order_or_subset(self):
        def run(order):
            outputs = {}
            for position in order:
                selector = self.selector(position=position, config=SpatialSelectionConfig(sample_count=3))
                with patch("safe_ik.learning.spatial_selection.predict_selection", return_value=prediction(0)):
                    outputs[position] = [selector.select(np.zeros(7), ())['spatial_vote']['samples'] for _ in range(2)]
            return outputs
        forward = run([1, 9, 17])
        self.assertEqual(forward, run([17, 1, 9]))
        self.assertEqual(forward[9], run([9])[9])
        self.assertNotEqual(forward[1], forward[9])

    def test_timing_separates_queries_from_whole_vote(self):
        selector = self.selector(config=SpatialSelectionConfig(sample_count=2))
        with patch("safe_ik.learning.spatial_selection.predict_selection", return_value=prediction(0)), patch(
            "safe_ik.learning.spatial_selection.perf_counter", side_effect=[0, 1, 2, 3, 5, 7]
        ):
            result = selector.select(np.zeros(7), ())
        self.assertEqual(result["inference_wall_seconds"], 3)
        self.assertEqual(result["spatial_selection_wall_seconds"], 7)

    def test_invalid_scores_or_inconsistent_argmax_fail_without_returning_route(self):
        invalid = [
            dict(probabilities=[0.5] * 15, action_index=0),
            dict(probabilities=[float("nan")] * 16, action_index=0),
            dict(probabilities=[-0.1] * 16, action_index=0),
            dict(probabilities=[1.1] * 16, action_index=0),
            dict(probabilities=[0.5] * 16, action_index=1),
            dict(probabilities=None, action_index=None),
        ]
        for result in invalid:
            with self.subTest(result=result), patch("safe_ik.learning.spatial_selection.predict_selection", return_value=result):
                with self.assertRaises(ValueError):
                    self.selector().select(np.zeros(7), ())

    def test_invalid_model_current_route_and_scene_inputs_rejected(self):
        for overrides in (
            dict(model=SimpleNamespace(actions=LINK_PAIRS)), dict(selected_ik=None),
            dict(selected_ik=[0] * 6), dict(obstacles=[[0, 0, 0, -1]]),
            dict(target_position=[0, float("nan"), 0]),
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.selector(**overrides)
        for q, current in (([0] * 6, ()), ([float("nan")] * 7, ()), ([0] * 7, (2, 3, 4)), ([0] * 7, (True, 3))):
            with self.subTest(q=q, current=current), self.assertRaises(ValueError):
                self.selector().select(q, current)
        with self.assertRaises(ValueError):
            self.selector(position=-1)


class SpatialConfigurationTests(unittest.TestCase):
    def test_invalid_count_seed_radius_or_period_rejected(self):
        for field, values in (
            ("sample_count", [0, -1, True, 50.0]),
            ("temporal_confirmations", [0, -1, True, 12.0, "12"]),
            ("sampling_seed", [-1, False, 1.0]),
            ("three_sigma_radius_rad", [0, -0.1, True, float("nan"), float("inf")]),
            ("period_seconds", [0, -0.3, False, float("nan"), "0.3"]),
        ):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    SpatialSelectionConfig(**{field: value})

    def test_yaml_preserves_explicit_parameters_and_rejects_misspelled_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spatial.yaml"
            path.write_text("spatial_reselection:\n  sample_count: 30\n  three_sigma_radius_rad: 0.06\n  period_seconds: 0.4\n  sampling_seed: 5\n")
            self.assertEqual(load_spatial_selection_config(path), SpatialSelectionConfig(30, .06, .4, 5))
            path.write_text(path.read_text() + "  temporal_confirmations: 12\n")
            self.assertEqual(load_spatial_selection_config(path).temporal_confirmations, 12)
            for contents in ("", "[]", "spatial_reselection: []", "spatial_reselection:\n  sampl_count: 50", "spatial_reselection:\n  sample_count: true"):
                path.write_text(contents)
                with self.subTest(contents=contents), self.assertRaises(ValueError):
                    load_spatial_selection_config(path)


if __name__ == "__main__":
    unittest.main()

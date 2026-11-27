"""Check spatial policy evidence and runner wiring, not controller internals."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from safe_ik.application import panda_reselect_reached as replay
from safe_ik.domain.link_actions import LINK_ACTIONS
from safe_ik.learning.spatial_selection import (
    SpatialSelectionConfig,
    SpatialSequenceSelector,
)


def prediction(index):
    scores = [0.1] * len(LINK_ACTIONS)
    scores[index] = 0.9
    return dict(action_index=index, probabilities=scores)


class SpatialEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.config = SpatialSelectionConfig(period_seconds=0.1)
        self.position = 17
        self.trace = dict(
            q=np.arange(13 * 7, dtype=float).reshape(13, 7) / 100,
            time=np.arange(13) * 0.02,
            command_time=np.arange(12) * 0.02,
            stage_frames=np.asarray([f"link{i}" for i in range(2, 7)]),
            stage_index=np.zeros(12, dtype=int),
        )
        self.initial = ["link2", "link3"]
        selector = SpatialSequenceSelector(
            self.config, self.position, SimpleNamespace(actions=LINK_ACTIONS),
            np.zeros(7), [0.1, 0.2, 0.3], [1, 0, 0, 0], [],
        )
        self.calls = []
        for index, votes in ((5, [1] * 25 + [2] * 25), (10, [1] * 26 + [2] * 24)):
            with patch(
                "safe_ik.learning.spatial_selection.predict_selection",
                side_effect=[prediction(action) for action in votes],
            ):
                proposal = selector.select(self.trace["q"][index], self.initial)
            self.calls.append(dict(
                proposal, time=float(self.trace["command_time"][index]),
                q=self.trace["q"][index].copy(),
            ))
        self.events = []
        for index, proposal, accepted in ((5, self.initial, False), (10, ["link2", "link4"], True)):
            self.events.append(dict(
                control_call_index=index,
                simulated_time=index * 0.02,
                q=self.trace["q"][index].tolist(),
                current_sequence_before=self.initial.copy(),
                current_sequence_after=list(proposal),
                proposed_sequence=list(proposal),
                accepted=accepted,
                confirmation_streak=int(accepted),
                streak_before=0, streak_after=0,
                candidate_before=None, candidate_after=None,
                previous_stage="link2_pose", entry_stage="link2_pose",
                next_stage="link2_pose",
                max_switches=None,
                switch_count_before=0, switch_count_after=int(accepted),
                blocked_by_switch_limit=False,
                lock_on_ee=False, ee_locked_after=False,
            ))
        self.transitions = [dict(
            time=0.2, previous="link2_pose", next="link2_pose", reason="sequence_switch",
        )]

    def validate(self, *, calls=None, events=None, config=None, position=None):
        return replay.timestamp_periodic_events(
            self.events if events is None else events,
            self.calls if calls is None else calls,
            self.trace, self.transitions, self.initial,
            period=0.1, confirmations=1, integration_period=0.02,
            spatial_config=self.config if config is None else config,
            spatial_position=self.position if position is None else position,
        )

    def test_tied_votes_retain_route_and_next_majority_switches_without_temporal_streak(self):
        records = self.validate()
        self.assertEqual([row["accepted"] for row in records], [False, True])
        self.assertEqual(records[0]["proposed_sequence"], self.initial)
        self.assertEqual(records[0]["spatial_vote"]["winner_action_index"], 1)
        self.assertEqual(records[0]["action_index"], 0)
        self.assertEqual(records[1]["proposed_sequence"], ["link2", "link4"])
        self.assertEqual(records[1]["confirmation_streak"], 1)
        self.assertEqual([row["time"] for row in records], [0.1, 0.2])
        self.assertNotIn("probabilities", records[0])

    def test_corrupted_sample_vote_or_proposal_cannot_be_certified(self):
        changes = (
            lambda call: call["spatial_vote"]["samples"][0].__setitem__(0, 123.0),
            lambda call: call["spatial_vote"]["vote_counts"].__setitem__(1, 26),
            lambda call: call["spatial_vote"].__setitem__("majority_action_index", 1),
            lambda call: call["spatial_vote"].__setitem__("sample_count", 49),
            lambda call: call["spatial_vote"].__setitem__("poll_index", 1),
            lambda call: call.__setitem__("action_index", 1),
        )
        for change in changes:
            calls = copy.deepcopy(self.calls)
            change(calls[0])
            with self.subTest(change=change), self.assertRaises((ValueError, AssertionError)):
                self.validate(calls=calls)
        with self.assertRaises((ValueError, AssertionError)):
            self.validate(position=self.position + 1)
        with self.assertRaises((ValueError, AssertionError)):
            self.validate(config=SpatialSelectionConfig(period_seconds=0.1, three_sigma_radius_rad=0.2))

    def test_spatial_poll_counts_and_neural_evaluations_keep_requested_denominator(self):
        base = dict(
            execution_status="completed", policy="spatial", success=False,
            goal_reached=False, reason="stalled", numerical_failure=False,
            completion_time=None, stored_static_success=False, static_replay_matches=None,
            initial_inference_count=1,
        )
        rows = [
            dict(base, spatial_poll_count=2, spatial_neural_evaluations=100,
                 total_inference_count=101, periodic_poll_count=2, periodic_switch_count=1,
                 spatial_majority_polls=1, spatial_no_majority_polls=1),
            dict(base, success=True, goal_reached=True, completion_time=4.0, reason="goal_reached",
                 spatial_poll_count=1, spatial_neural_evaluations=50,
                 total_inference_count=51, periodic_poll_count=1, periodic_switch_count=0,
                 spatial_majority_polls=1, spatial_no_majority_polls=0),
        ]
        result = replay.aggregate(rows, requested=3)
        self.assertEqual(result["spatial_poll_count"], 3)
        self.assertEqual(result["spatial_neural_evaluations"], 150)
        self.assertEqual(result["total_inference_count"], 152)
        self.assertEqual(result["spatial_majority_polls"], 2)
        self.assertEqual(result["spatial_no_majority_polls"], 1)
        self.assertEqual(result["periodic_switch_count"], 1)
        self.assertEqual(result["missing_cases"], 1)
        self.assertEqual(result["success_rate_requested"], 1 / 3)


class SpatialRunnerConfigurationTests(unittest.TestCase):
    def args(self, **overrides):
        result = dict(mode="spatial", spatial_config=None,
                      sequence_period=None, sequence_confirmations=None)
        result.update(overrides)
        return SimpleNamespace(**result)

    def test_yaml_controls_spatial_sampling_and_defaults_to_one_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.yaml"
            path.write_text(
                "spatial_reselection:\n"
                "  sample_count: 31\n"
                "  three_sigma_radius_rad: 0.06\n"
                "  period_seconds: 0.4\n"
                "  sampling_seed: 19\n"
            )
            config, period, confirmations = replay.resolve_sequence_settings(self.args(spatial_config=path))
            self.assertEqual(config.sample_count, 31)
            self.assertEqual(config.sigma_rad, 0.02)
            self.assertEqual(config.sampling_seed, 19)
            self.assertEqual((period, confirmations), (0.4, 1))
            path.write_text(path.read_text() + "  temporal_confirmations: 12\n")
            config, period, confirmations = replay.resolve_sequence_settings(self.args(spatial_config=path))
            self.assertEqual((period, confirmations), (0.4, 12))
            self.assertEqual(config.temporal_confirmations, 12)
            for overrides in (dict(sequence_period=0.2), dict(sequence_confirmations=5), dict(mode="static")):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    replay.resolve_sequence_settings(self.args(spatial_config=path, **overrides))
        with self.assertRaises(ValueError):
            replay.resolve_sequence_settings(self.args())

    def test_periodic_cli_defaults_and_explicit_settings_are_preserved(self):
        self.assertEqual(
            replay.resolve_sequence_settings(self.args(mode="periodic")),
            (None, 0.1, 5),
        )
        self.assertEqual(
            replay.resolve_sequence_settings(self.args(mode="periodic", sequence_period=0.4, sequence_confirmations=9)),
            (None, 0.4, 9),
        )
        self.assertEqual(replay.resolve_sequence_settings(self.args(mode="static")), (None, None, None))


class HybridSpatialEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.config = SpatialSelectionConfig(
            three_sigma_radius_rad=0.3, temporal_confirmations=12,
        )
        self.initial = ["link2", "link3"]
        # Eleven matching majorities must not survive a changed winner or a
        # tied vote. Only the final twelve matching other-route majorities act.
        winners = [1] * 11 + [2, None] + [1] * 12 + [1]
        streaks = list(range(1, 12)) + [1, 0] + list(range(1, 13)) + [0]
        commands = len(winners) * 15 + 1
        self.trace = dict(
            q=np.arange((commands + 1) * 7, dtype=float).reshape(commands + 1, 7) / 10000,
            time=np.arange(commands + 1) * 0.02,
            command_time=np.arange(commands) * 0.02,
            stage_frames=np.asarray(replay.STAGE_FRAMES),
            stage_index=np.zeros(commands, dtype=int),
        )
        selector = SpatialSequenceSelector(
            self.config, 17, SimpleNamespace(actions=LINK_ACTIONS),
            np.zeros(7), [0.1, 0.2, 0.3], [1, 0, 0, 0], [],
        )
        self.calls, self.events = [], []
        for poll, (winner, streak) in enumerate(zip(winners, streaks)):
            index = (poll + 1) * 15
            current = self.initial if poll <= 24 else ["link2", "link4"]
            votes = [1] * 25 + [2] * 25 if winner is None else [winner] * 26 + [3] * 24
            with patch("safe_ik.learning.spatial_selection.predict_selection", side_effect=[prediction(i) for i in votes]):
                proposal = selector.select(self.trace["q"][index], current)
            self.calls.append(dict(proposal, q=self.trace["q"][index].copy(), time=float(self.trace["command_time"][index])))
            accepted = poll == 24
            proposed = proposal["link_frames"]
            self.events.append(dict(
                control_call_index=index, simulated_time=index * 0.02,
                q=self.trace["q"][index].tolist(), current_sequence_before=current,
                current_sequence_after=proposed if accepted else current,
                proposed_sequence=proposed, accepted=accepted,
                confirmation_streak=streak,
                streak_after=0 if accepted else streak,
                candidate_after=None if accepted or streak == 0 else proposed,
                previous_stage="link2_pose", entry_stage="link2_pose", next_stage="link2_pose",
                max_switches=3, switch_count_before=int(poll > 24),
                switch_count_after=int(poll >= 24), blocked_by_switch_limit=False,
                switch_guard_evaluated=False, blocked_by_switch_guard=False,
            ))
        self.transitions = [dict(
            time=float(self.trace["command_time"][375]), previous="link2_pose",
            next="link2_pose", reason="sequence_switch",
        )]

    def validate(self, events=None, confirmations=12):
        return replay.timestamp_periodic_events(
            self.events if events is None else events, self.calls, self.trace,
            self.transitions, self.initial, period=0.3, confirmations=confirmations,
            integration_period=0.02, max_switches=3,
            spatial_config=self.config, spatial_position=17,
        )

    def test_every_poll_votes_and_only_twelve_same_consecutive_majorities_switch(self):
        records = self.validate()
        self.assertEqual([i for i, event in enumerate(records) if event["accepted"]], [24])
        self.assertEqual(records[11]["confirmation_streak"], 1)
        self.assertEqual(records[12]["confirmation_streak"], 0)
        self.assertEqual(records[23]["confirmation_streak"], 11)
        self.assertEqual(records[24]["confirmation_streak"], 12)
        self.assertEqual(sum(len(event["spatial_vote"]["sample_actions"]) for event in records), 26 * 50)

    def test_fabricated_streak_or_wrong_hybrid_configuration_is_rejected(self):
        events = copy.deepcopy(self.events)
        events[11]["confirmation_streak"] = 12
        with self.assertRaisesRegex(ValueError, "confirmation evidence"):
            self.validate(events)
        events = copy.deepcopy(self.events)
        events[12]["candidate_after"] = ["link2", "link5"]
        with self.assertRaisesRegex(ValueError, "confirmation evidence"):
            self.validate(events)
        with self.assertRaisesRegex(ValueError, "matching temporal settings"):
            self.validate(confirmations=1)


if __name__ == "__main__":
    unittest.main()

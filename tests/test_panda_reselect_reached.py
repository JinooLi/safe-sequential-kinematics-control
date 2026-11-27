"""Protect frozen scene identity and denominators in adaptive policy replay."""

import copy
import unittest
from unittest.mock import Mock, patch

import numpy as np

from safe_ik.application.panda_reselect_reached import (
    aggregate,
    build_tasks,
    make_probability_gain_switch_guard,
    periodic_schedule_steps,
    select_partition_rows,
    select_test_rows,
    timestamp_collision_stall_events,
    timestamp_periodic_events,
    validate_sequence_lock_on_ee,
    validate_sequence_max_switches,
    validate_sequence_min_probability_gain,
    ThresholdWeightedSelector,
)
from safe_ik.domain.link_actions import LINK_ACTIONS, action_id


def prediction_with_scores(scores):
    index = int(np.argmax(scores))
    links = LINK_ACTIONS[index]
    return dict(
        action_index=index, action_id=action_id(links),
        pair_index=index if len(links) == 2 else None,
        method="sequential" if links else "direct",
        link_frames=[f"link{i}" for i in links],
        probabilities=list(scores), reason="highest_predicted_success",
    )


class ThresholdWeightedSelectionTests(unittest.TestCase):
    def test_initial_argmax_does_not_draw_or_mutate_original_prediction(self):
        scores = [0.8, 0.9] + [0.1] * 14
        original = prediction_with_scores(scores)
        saved = copy.deepcopy(original)
        selector = ThresholdWeightedSelector(107, 23)
        selected = selector.select(original)
        self.assertEqual(selected["action_index"], 1)
        self.assertEqual(selected["sampling"]["rng_draw_count"], 0)
        self.assertFalse(selected["sampling"]["sampling_applied"])
        self.assertEqual(original, saved)
        independent_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([107, 23])))
        weights = np.array([0.8 / 1.7, 0.9 / 1.7] + [0.] * 14)
        expected = int(independent_rng.choice(16, p=weights))
        self.assertEqual(selector.select(original)["action_index"], expected)

    def test_strict_threshold_and_probability_weights_are_applied_at_every_reach(self):
        scores = [0.5, 0.5001, 0.75, 0.99] + [0.1] * 12
        prediction = prediction_with_scores(scores)
        selector = ThresholdWeightedSelector(1, 2)
        selector.select(prediction)
        selector.rng = Mock()
        selector.rng.choice.side_effect = [1, 2]
        first, second = selector.select(prediction), selector.select(prediction)
        self.assertEqual([first["action_index"], second["action_index"]], [1, 2])
        self.assertEqual(first["sampling"]["eligible_action_indices"], [1, 2, 3])
        weights = np.asarray(first["sampling"]["sampling_weights"])
        np.testing.assert_allclose(weights[[1, 2, 3]], np.asarray(scores)[[1, 2, 3]] / sum(scores[1:4]))
        self.assertEqual(weights[0], 0)
        self.assertEqual(second["sampling"]["rng_draw_count"], 2)
        self.assertEqual(selector.rng.choice.call_count, 2)
        self.assertEqual(first["sampling"]["original_argmax_action_index"], 3)
        self.assertEqual(first["sampling"]["original_argmax_score"], 0.99)

    def test_no_eligible_action_falls_back_without_consuming_rng(self):
        selector = ThresholdWeightedSelector(99, 9)
        prediction = prediction_with_scores([0.2, 0.5] + [0.1] * 14)
        selector.select(prediction)
        selector.rng = Mock()
        selector.rng.choice.side_effect = AssertionError("unexpected random draw")
        selected = selector.select(prediction)
        self.assertEqual(selected["action_index"], 1)
        self.assertEqual(selected["sampling"]["eligible_action_indices"], [])
        self.assertTrue(selected["sampling"]["fallback_to_argmax"])
        self.assertFalse(selected["sampling"]["sampling_applied"])
        self.assertEqual(selected["sampling"]["rng_draw_count"], 0)
        self.assertEqual(selected["sampling"]["sampling_weights"][1], 1)

    def test_scene_rng_is_independent_of_schedule_and_subset(self):
        prediction = prediction_with_scores([0.6] * 16)
        def run(order):
            output = {}
            for position in order:
                selector = ThresholdWeightedSelector(234, position)
                output[position] = [selector.select(prediction)["action_index"] for _ in range(25)]
            return output
        ordered = run([1, 3, 9])
        self.assertEqual(ordered, run([9, 1, 3]))
        self.assertEqual(ordered[3], run([3])[3])
        self.assertNotEqual(ordered[1], ordered[9])

    def test_selected_aliases_describe_pair_single_and_direct_actions_consistently(self):
        for action in (4, 11, 15):
            with self.subTest(action=action):
                scores = [0.1] * 16
                scores[action] = 0.9
                selector = ThresholdWeightedSelector(0, 0)
                prediction = prediction_with_scores(scores)
                selector.select(prediction)
                selected = selector.select(prediction)
                self.assertEqual(selected["action_id"], action_id(LINK_ACTIONS[action]))
                self.assertEqual(selected["action_index"], action)
                self.assertEqual(selected["pair_index"], action if len(LINK_ACTIONS[action]) == 2 else None)
                self.assertEqual(selected["method"], "sequential" if LINK_ACTIONS[action] else "direct")
                self.assertEqual(selected["link_frames"], [f"link{i}" for i in LINK_ACTIONS[action]])

    def test_original_actions_keep_distinct_mass_before_completed_link_filtering(self):
        # After link3, these three actions all resolve to link4. The sampling
        # experiment nevertheless retains three separate original choices.
        aliases = [LINK_ACTIONS.index(links) for links in ((2, 4), (3, 4), (4,))]
        scores = [0.1] * 16
        for index in aliases:
            scores[index] = 0.6
        selector = ThresholdWeightedSelector(9, 4)
        prediction = prediction_with_scores(scores)
        selector.select(prediction)
        selected = selector.select(prediction)
        self.assertEqual(selected["sampling"]["eligible_action_indices"], aliases)
        np.testing.assert_allclose(np.asarray(selected["sampling"]["sampling_weights"])[aliases], [1 / 3] * 3)

    def test_invalid_sampling_inputs_are_rejected(self):
        for seed, position, threshold in ((-1, 0, .5), (True, 0, .5), (0, -1, .5), (0, 0, float("nan")), (0, 0, 1.1)):
            with self.subTest(seed=seed, position=position, threshold=threshold), self.assertRaises(ValueError):
                ThresholdWeightedSelector(seed, position, threshold)
        selector = ThresholdWeightedSelector(0, 0)
        for scores in ([0.5] * 15, [float("nan")] * 16, [1.01] * 16):
            prediction = dict(action_index=0, probabilities=scores)
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                selector.select(prediction)


class ReplayInputTests(unittest.TestCase):
    def setUp(self):
        self.data = dict(
            case_id=np.asarray(["scene0", "scene1", "scene2"]),
            initial_q=np.arange(21, dtype=np.float64).reshape(3, 7) / 17,
            selected_ik=np.arange(21, dtype=np.float64).reshape(3, 7) / 19,
            target_position=np.arange(9, dtype=np.float64).reshape(3, 3) / 7,
            target_quaternion=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
            obstacles=np.tile([[[0.1, 0.2, 0.3, 0.1]]], (3, 1, 1)),
            labels=np.zeros((3, 16), dtype=np.float32),
            label_valid=np.ones((3, 16), dtype=bool),
        )
        self.split = dict(
            train=dict(indices=[0], case_ids=["scene0"]),
            validation=dict(indices=[], case_ids=[]),
            test=dict(indices=[2, 1], case_ids=["scene2", "scene1"]),
        )

    def test_test_positions_preserve_existing_order_and_identity(self):
        self.assertEqual(select_test_rows(self.data, self.split), [(0, 2), (1, 1)])
        self.assertEqual(
            select_test_rows(self.data, self.split, indices=[1, 0]), [(1, 1), (0, 2)]
        )
        self.assertEqual(
            select_test_rows(self.data, self.split, offset=1, limit=1), [(1, 1)]
        )
        broken = copy.deepcopy(self.split)
        broken["test"]["case_ids"].reverse()
        with self.assertRaisesRegex(ValueError, "do not match"):
            select_test_rows(self.data, broken)
        broken = copy.deepcopy(self.split)
        broken["train"]["case_ids"] = ["scene1"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_test_rows(self.data, broken)

    def test_duplicate_or_out_of_range_requested_positions_are_rejected(self):
        for indices in ([], [0, 0], [-1], [2], [True]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                select_test_rows(self.data, self.split, indices=indices)
        with self.assertRaises(ValueError):
            select_test_rows(self.data, self.split, limit=3)
        with self.assertRaises(ValueError):
            select_test_rows(self.data, self.split, indices=[0], limit=1)

    def test_validation_partition_preserves_identity_and_cannot_overlap_test(self):
        split = copy.deepcopy(self.split)
        split["validation"], split["test"] = split["test"], split["validation"]
        rows = select_partition_rows(self.data, split, partition="validation", indices=[1, 0])
        self.assertEqual(rows, [(1, 1), (0, 2)])
        task = build_tasks(self.data, rows, partition="validation")[0]
        self.assertEqual(task["partition"], "validation")
        self.assertEqual(task["partition_position"], 1)
        self.assertEqual(task["test_position"], 1)  # Documented legacy alias.
        split["test"] = dict(indices=[1], case_ids=["scene1"])
        with self.assertRaisesRegex(ValueError, "overlap test"):
            select_partition_rows(self.data, split, partition="validation")
        with self.assertRaisesRegex(ValueError, "test or validation"):
            select_partition_rows(self.data, split, partition="train")

    def test_original_metadata_and_float64_values_survive_replay(self):
        task = build_tasks(self.data, [(0, 2)])[0]
        original = dict(task["case"], selected_ik=dict(task["selected_ik"]))
        original["ik_seed"] = 123
        original["selected_ik"].update(seed_index=11, iterations=331)
        context = {"scene2": original}
        replay = build_tasks(self.data, [(0, 2)], context)[0]
        self.assertEqual(replay["case"]["ik_seed"], 123)
        self.assertEqual(replay["selected_ik"]["seed_index"], 11)
        self.assertEqual(replay["selected_ik"]["iterations"], 331)
        self.assertTrue(replay["original_candidate_metadata"])
        self.assertEqual(replay["case"]["initial_q"].dtype, np.float64)
        np.testing.assert_array_equal(replay["selected_ik"]["q"], self.data["selected_ik"][2])
        for key in ("initial_q", "target_position", "target_quaternion"):
            np.testing.assert_array_equal(replay["case"][key], self.data[key][2])
        # Same ID from a different scene/IK run must not be accepted.
        changed = copy.deepcopy(context)
        changed["scene2"]["selected_ik"]["q"][0] += 1e-12
        with self.assertRaises(AssertionError):
            build_tasks(self.data, [(0, 2)], changed)
        changed = copy.deepcopy(context)
        changed["scene2"]["obstacles"][0]["radius"] += 1e-12
        with self.assertRaises(AssertionError):
            build_tasks(self.data, [(0, 2)], changed)

    def test_no_ik_search_is_implied_when_original_metadata_is_missing(self):
        task = build_tasks(self.data, [(1, 1)])[0]
        self.assertFalse(task["original_candidate_metadata"])
        np.testing.assert_array_equal(task["selected_ik"]["q"], self.data["selected_ik"][1])
        # Invalid numerical labels are failures, not a scene exclusion.
        self.data["label_valid"][1, 0] = False
        task = build_tasks(self.data, [(1, 1)])[0]
        self.assertFalse(task["stored_label_valid"][0])
        self.assertFalse(task["stored_success"][0])


class ReplayStatisticsTests(unittest.TestCase):
    def test_runner_errors_and_missing_scenes_do_not_inflate_success(self):
        base = dict(
            execution_status="completed", success=True, goal_reached=True,
            reason="goal_reached", numerical_failure=False,
            completion_time=7.0, stored_static_success=False,
            static_replay_matches=None,
        )
        rows = [
            base,
            dict(base, success=False, goal_reached=False, reason="qp_solver_failure",
                 numerical_failure=True, completion_time=None, stored_static_success=True),
            dict(base, execution_status="error", reason="runner_error", success=False),
        ]
        result = aggregate(rows, requested=4)
        self.assertEqual(result["success_rate_requested"], 0.25)
        self.assertEqual(result["runner_errors"], 1)
        self.assertEqual(result["missing_cases"], 1)

        self.assertEqual(result["numerical_failures"], 1)
        self.assertEqual(result["paired_cases"], 2)
        self.assertEqual((result["recovered"], result["lost"]), (1, 1))
        self.assertEqual(result["mean_success_completion_time"], 7.0)

    def test_periodic_polls_switches_and_scene_counts_are_distinct(self):
        base = dict(
            execution_status="completed", policy="periodic", success=False,
            goal_reached=False, reason="timeout", numerical_failure=False,
            completion_time=None, stored_static_success=False, static_replay_matches=None,
            initial_inference_count=1,
        )
        rows = [
            dict(base, periodic_poll_count=1199, periodic_switch_count=3, total_inference_count=1200,
                 periodic_blocked_switch_proposals=17, periodic_switch_limit_exhausted=True,
                 periodic_switch_guard_evaluations=5, periodic_blocked_by_switch_guard=2),
            dict(base, periodic_poll_count=20, periodic_switch_count=0, total_inference_count=21,
                 periodic_blocked_switch_proposals=0, periodic_switch_limit_exhausted=False,
                 periodic_switch_guard_evaluations=4, periodic_blocked_by_switch_guard=4),
        ]
        result = aggregate(rows, requested=3)
        self.assertEqual(result["periodic_poll_count"], 1219)
        self.assertEqual(result["periodic_switch_count"], 3)
        self.assertEqual(result["periodic_switched_scenes"], 1)
        self.assertEqual(result["initial_inference_count"], 2)
        self.assertEqual(result["total_inference_count"], 1221)
        self.assertEqual(result["periodic_blocked_switch_proposals"], 17)
        self.assertEqual(result["periodic_scenes_with_blocked_switch_proposals"], 1)
        self.assertEqual(result["periodic_scenes_at_switch_limit"], 1)
        self.assertEqual(result["periodic_switch_guard_evaluations"], 9)
        self.assertEqual(result["periodic_blocked_by_switch_guard"], 6)
        self.assertEqual(result["periodic_scenes_blocked_by_switch_guard"], 2)
        self.assertEqual(result["missing_cases"], 1)

    def test_collision_gate_checks_and_skips_have_distinct_counts(self):
        base = dict(
            execution_status="completed", policy="collision_stall_skip", success=False,
            goal_reached=False, reason="stalled", numerical_failure=False,
            completion_time=None, stored_static_success=False, static_replay_matches=None,
        )
        rows = [
            dict(base, collision_stall_gate_checks=2, collision_stall_skipped_stages=2,
                 collision_stall_diagnostic_failures=0),
            dict(base, collision_stall_gate_checks=1, collision_stall_skipped_stages=0,
                 collision_stall_diagnostic_failures=1),
            dict(base, collision_stall_gate_checks=0, collision_stall_skipped_stages=0,
                 collision_stall_diagnostic_failures=0),
        ]
        result = aggregate(rows, requested=4)
        self.assertEqual(result["collision_stall_gate_checks"], 3)
        self.assertEqual(result["collision_stall_skipped_stages"], 2)
        self.assertEqual(result["collision_stall_affected_scenes"], 1)
        self.assertEqual(result["collision_stall_diagnostic_failures"], 1)
        self.assertEqual(result["missing_cases"], 1)


class PeriodicEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.trace = dict(
            q=np.arange(17 * 7, dtype=float).reshape(17, 7) / 100,
            time=np.arange(17) * .02, command_time=np.arange(16) * .02,
            stage_frames=np.asarray([f"link{i}" for i in range(2, 7)]),
            stage_index=np.full(16, 5),
        )
        action = LINK_ACTIONS.index((2,))
        scores = [.1] * 16
        scores[action] = .9
        self.calls = [
            dict(prediction_with_scores(scores), time=float(self.trace["command_time"][i]),
                 q=self.trace["q"][i].copy(), inference_wall_seconds=.001)
            for i in (5, 10, 15)
        ]
        self.events = []
        for index, before, after, accepted in ((5, [], [], False), (10, [], ["link2"], True), (15, ["link2"], ["link2"], False)):
            self.events.append(dict(
                control_call_index=index, simulated_time=index * .02,
                q=self.trace["q"][index].tolist(),
                current_sequence_before=before, current_sequence_after=after,
                proposed_sequence=["link2"], accepted=accepted,
                confirmation_streak=2 if accepted else (1 if index == 5 else 0),
                streak_after=0 if index != 5 else 1,
                candidate_after=None if index != 5 else ["link2"],
                previous_stage="end_effector_pose",
                # The restarted link is already satisfied and passes instantly.
                entry_stage="link2_pose" if accepted else "end_effector_pose",
                next_stage="end_effector_pose",
                max_switches=None, switch_count_before=0 if index < 15 else 1,
                switch_count_after=0 if index == 5 else 1,
                blocked_by_switch_limit=False,
            ))
        self.transitions = [
            dict(time=.2, previous="end_effector_pose", next="link2_pose", reason="sequence_switch"),
            dict(time=.2, previous="link2_pose", next="end_effector_pose", reason="target_reached"),
        ]

    def validate(self, events=None, calls=None, transitions=None, max_switches=None):
        return timestamp_periodic_events(
            self.events if events is None else events,
            self.calls if calls is None else calls, self.trace,
            self.transitions if transitions is None else transitions, [],
            period=.1, confirmations=2, integration_period=.02, max_switches=max_switches,
        )

    def test_direct_can_switch_to_an_already_reached_link_without_losing_poll_identity(self):
        events = self.validate()
        self.assertEqual([event["time"] for event in events], [.1, .2, .3])
        self.assertEqual(events[1]["current_sequence_before"], [])
        self.assertEqual(events[1]["entry_stage"], "link2_pose")
        self.assertEqual(events[1]["next_stage"], "end_effector_pose")
        self.assertEqual(events[1]["probabilities"], self.calls[1]["probabilities"])
        self.assertEqual(events[1]["inference_wall_seconds"], .001)

    def test_missing_poll_wrong_state_or_wrong_action_invalidates_evidence(self):
        with self.assertRaisesRegex(ValueError, "every scheduled"):
            self.validate(events=self.events[:-1], calls=self.calls[:-1])
        calls = copy.deepcopy(self.calls)
        calls[0]["q"][0] += 1e-8
        with self.assertRaises(AssertionError):
            self.validate(calls=calls)
        calls = copy.deepcopy(self.calls)
        calls[0]["action_index"] = 15
        with self.assertRaisesRegex(ValueError, "model argmax"):
            self.validate(calls=calls)
        events = copy.deepcopy(self.events)
        events[1]["confirmation_streak"] = 1
        with self.assertRaisesRegex(ValueError, "confirmation evidence"):
            self.validate(events=events)
        with self.assertRaisesRegex(ValueError, "sequence switches"):
            self.validate(transitions=[])

    def test_poll_clock_is_integer_steps_and_terminal_state_does_not_poll(self):
        self.assertEqual([periodic_schedule_steps(period, 5, .02) for period in (.1, .2, .4)], [5, 10, 20])
        for period, confirmations in ((.11, 5), (0, 5), (.1, 0), (.1, True), (float("nan"), 5)):
            with self.subTest(period=period, confirmations=confirmations), self.assertRaises(ValueError):
                periodic_schedule_steps(period, confirmations, .02)
        # A final state at 0.1 has no controller call, so it cannot poll.
        trace = dict(self.trace, q=self.trace["q"][:6], time=self.trace["time"][:6],
                     command_time=self.trace["command_time"][:5], stage_index=self.trace["stage_index"][:5])
        self.assertEqual(timestamp_periodic_events(
            [], [], trace, [], [], period=.1, confirmations=5, integration_period=.02,
        ), [])

    def test_switch_cap_blocks_confirmed_proposals_but_preserves_every_poll(self):
        self.trace = dict(
            q=np.arange(32 * 7, dtype=float).reshape(32, 7) / 100,
            time=np.arange(32) * .02, command_time=np.arange(31) * .02,
            stage_frames=self.trace["stage_frames"], stage_index=np.full(31, 5),
        )
        self.events = copy.deepcopy(self.events[:2])
        self.calls = copy.deepcopy(self.calls[:2])
        for event in self.events:
            event["max_switches"] = 1
        # The initial Direct -> link2 acceptance exhausts the single-change cap.
        # Direct is still proposed at t=.3,.4,.5; two confirmed proposals are
        # blocked and the evidence remains visible, rather than ending polling.
        for index, streak in ((15, 1), (20, 2), (25, 3), (30, 0)):
            current = index == 30
            scores = [.1] * 16
            action = LINK_ACTIONS.index((2,)) if current else LINK_ACTIONS.index(())
            scores[action] = .9
            call = dict(prediction_with_scores(scores), time=float(self.trace["command_time"][index]),
                        q=self.trace["q"][index].copy(), inference_wall_seconds=.001)
            self.calls.append(call)
            self.events.append(dict(
                control_call_index=index, simulated_time=index * .02,
                q=self.trace["q"][index].tolist(), current_sequence_before=["link2"],
                current_sequence_after=["link2"], proposed_sequence=call["link_frames"],
                accepted=False, confirmation_streak=streak, streak_after=streak,
                candidate_after=None if current else [],
                previous_stage="end_effector_pose", entry_stage="end_effector_pose", next_stage="end_effector_pose",
                max_switches=1, switch_count_before=1, switch_count_after=1,
                blocked_by_switch_limit=streak >= 2,
            ))
        records = self.validate(max_switches=1)
        self.assertEqual(len(records), 6)
        self.assertEqual(sum(event["accepted"] for event in records), 1)
        self.assertEqual(sum(event["blocked_by_switch_limit"] for event in records), 2)
        self.assertEqual(records[-2]["confirmation_streak"], 3)
        self.assertEqual(records[-1]["confirmation_streak"], 0)
        tampered = copy.deepcopy(self.events)
        tampered[3]["accepted"] = True
        with self.assertRaisesRegex(ValueError, "switch-limit"):
            self.validate(events=tampered, max_switches=1)
        with self.assertRaisesRegex(ValueError, "every scheduled"):
            self.validate(events=self.events[:2], calls=self.calls[:2], max_switches=1)

    def test_optional_cap_validation_and_zero_cap_are_unambiguous(self):
        self.assertIsNone(validate_sequence_max_switches(None, "static"))
        self.assertEqual(validate_sequence_max_switches(np.int64(3)), 3)
        self.assertEqual(validate_sequence_max_switches(0), 0)
        for value, mode in ((-1, "periodic"), (True, "periodic"), (np.bool_(False), "periodic"),
                            (1., "periodic"), (3, "static"), (0, "adaptive")):
            with self.subTest(value=value, mode=mode), self.assertRaises(ValueError):
                validate_sequence_max_switches(value, mode)
        events = copy.deepcopy(self.events)
        for i, event in enumerate(events):
            event.update(
                max_switches=0, accepted=False, current_sequence_before=[], current_sequence_after=[],
                switch_count_before=0, switch_count_after=0,
                confirmation_streak=i + 1, streak_after=i + 1, candidate_after=["link2"],
                blocked_by_switch_limit=i >= 1,
            )
        records = self.validate(events=events, transitions=[], max_switches=0)
        self.assertEqual(len(records), 3)
        self.assertFalse(any(event["accepted"] for event in records))
        self.assertEqual(sum(event["blocked_by_switch_limit"] for event in records), 2)


class ProbabilityGainEvidenceTests(unittest.TestCase):
    def make_evidence(self, threshold=0.05, cap=3):
        # The first eleven polls have a large gain but cannot switch yet. The
        # twelfth fails the guard; the next poll can pass without twelve more.
        commands = 13 * 15 + 1
        trace = dict(
            q=np.arange((commands + 1) * 7, dtype=float).reshape(commands + 1, 7) / 10000,
            time=np.arange(commands + 1) * .02, command_time=np.arange(commands) * .02,
            stage_frames=np.asarray([f"link{i}" for i in range(2, 7)]),
            stage_index=np.ones(commands, dtype=int),
        )
        initial, proposed = ["link2", "link3"], ["link2", "link4"]
        calls, events = [], []
        for poll in range(1, 14):
            index = poll * 15
            scores = [.1] * 16
            scores[0] = .5
            scores[1] = .5 + (threshold / 2 if poll == 12 else threshold * 2)
            # The reached link2 is not removed from the current full route.
            # Its remaining suffix link3 has a different, much lower score.
            scores[LINK_ACTIONS.index((3,))] = .05
            calls.append(dict(prediction_with_scores(scores), q=trace["q"][index].copy(), time=float(trace["command_time"][index])))
            confirmed = poll >= 12
            evaluated = confirmed and cap > 0
            accepted = poll == 13 and cap > 0
            events.append(dict(
                control_call_index=index, simulated_time=index * .02,
                q=trace["q"][index].tolist(), current_sequence_before=initial,
                current_sequence_after=proposed if accepted else initial,
                proposed_sequence=proposed, accepted=accepted,
                confirmation_streak=poll, streak_before=poll - 1,
                candidate_before=proposed if poll > 1 else None,
                streak_after=0 if accepted else poll,
                candidate_after=None if accepted else proposed,
                previous_stage="link3_pose", entry_stage="link2_pose" if accepted else "link3_pose",
                next_stage="link2_pose" if accepted else "link3_pose",
                max_switches=cap, switch_count_before=0, switch_count_after=int(accepted),
                blocked_by_switch_limit=confirmed and cap == 0,
                switch_guard_evaluated=evaluated,
                blocked_by_switch_guard=evaluated and poll == 12,
            ))
        transitions = [dict(time=float(trace["command_time"][195]), previous="link3_pose", next="link2_pose", reason="sequence_switch")] if cap else []
        return events, calls, trace, transitions, initial

    def validate(self, evidence, threshold=.05, cap=3):
        return timestamp_periodic_events(
            *evidence, period=.3, confirmations=12, integration_period=.02,
            max_switches=cap, min_probability_gain=threshold,
        )

    def test_margin_checked_only_when_temporally_ready_and_rejection_keeps_streak(self):
        for threshold in (.05, .10):
            with self.subTest(threshold=threshold):
                records = self.validate(self.make_evidence(threshold), threshold)
                self.assertFalse(any(event["switch_guard_evaluated"] for event in records[:11]))
                self.assertTrue(records[11]["blocked_by_switch_guard"])
                self.assertAlmostEqual(records[11]["switch_probability_gain"], threshold / 2)
                self.assertEqual(records[11]["streak_after"], 12)
                self.assertTrue(records[12]["accepted"])
                self.assertEqual(records[12]["confirmation_streak"], 13)
                self.assertEqual(records[12]["streak_after"], 0)

    def test_cap_has_priority_over_score_guard(self):
        records = self.validate(self.make_evidence(cap=0), cap=0)
        self.assertFalse(any(event["switch_guard_evaluated"] for event in records))
        self.assertFalse(any(event["blocked_by_switch_guard"] for event in records))
        self.assertTrue(all(event["switch_probability_gain"] is None for event in records))
        self.assertEqual(sum(event["blocked_by_switch_limit"] for event in records), 2)

    def test_replay_rejects_fabricated_streak_guard_or_pending_state(self):
        for poll, key, value in (
            (0, "confirmation_streak", 12),
            (0, "switch_guard_evaluated", True),
            (11, "blocked_by_switch_guard", False),
            (11, "streak_after", 0),
            (12, "candidate_before", None),
        ):
            evidence = self.make_evidence()
            evidence[0][poll][key] = value
            with self.subTest(poll=poll, key=key), self.assertRaisesRegex(ValueError, "evidence"):
                self.validate(evidence)

    def test_guard_reuses_latest_exact_q_and_full_route_scores_without_inference(self):
        evidence = self.make_evidence()
        calls = evidence[1][:12]
        q = calls[-1]["q"].copy()
        guard = make_probability_gain_switch_guard(calls, .05)
        with patch("safe_ik.application.panda_reselect_reached.predict_selection", side_effect=AssertionError("extra inference")):
            self.assertFalse(guard(q, ["link2", "link3"], ["link2", "link4"]))
            calls.append(evidence[1][12])
            self.assertTrue(guard(calls[-1]["q"], ["link2", "link3"], ["link2", "link4"]))
        with self.assertRaises(AssertionError):
            guard(q, ["link2", "link3"], ["link2", "link4"])
        calls[-1]["probabilities"][0:2] = [.5, .75]
        self.assertTrue(make_probability_gain_switch_guard(calls, .25)(calls[-1]["q"], ["link2", "link3"], ["link2", "link4"]))
        self.assertIsNone(make_probability_gain_switch_guard(calls, None))

    def test_probability_gain_validation_rejects_nonperiodic_or_nonfinite_values(self):
        self.assertIsNone(validate_sequence_min_probability_gain(None, "spatial"))
        self.assertEqual(validate_sequence_min_probability_gain(0), 0.)
        self.assertEqual(validate_sequence_min_probability_gain(1), 1.)
        for value, mode in (
            (.05, "spatial"), (.1, "static"), (-.01, "periodic"), (1.01, "periodic"),
            (float("nan"), "periodic"), (float("inf"), "periodic"),
            (True, "periodic"), ("0.05", "periodic"),
        ):
            with self.subTest(value=value, mode=mode), self.assertRaises(ValueError):
                validate_sequence_min_probability_gain(value, mode)


class PeriodicEELockEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.trace = dict(
            q=np.zeros((17, 7)), time=np.arange(17) * .02,
            command_time=np.arange(16) * .02,
            stage_frames=np.asarray([f"link{i}" for i in range(2, 7)]),
            stage_index=np.full(16, 5),
        )

    def validate(self, initial, lock, events=(), calls=(), transitions=(), trace=None, enabled=True):
        return timestamp_periodic_events(
            events, calls, self.trace if trace is None else trace, transitions, initial,
            period=.1, confirmations=1, integration_period=.02,
            lock_on_ee=enabled, ee_lock_event=lock,
        )

    def test_initial_direct_has_no_polls_and_requires_zero_time_lock(self):
        lock = dict(control_call_index=0, simulated_time=0., reason="initial_direct", initial_direct=True)
        self.assertEqual(self.validate([], lock), [])
        with self.assertRaisesRegex(ValueError, "first EE entry"):
            self.validate([], None)
        with self.assertRaisesRegex(ValueError, "first EE entry"):
            self.validate([], dict(lock, control_call_index=5, simulated_time=.1))
        with self.assertRaisesRegex(ValueError, "disabled"):
            self.validate([], lock, enabled=False)
        terminal = dict(self.trace, q=self.trace["q"][:1], time=self.trace["time"][:1],
                        command_time=np.empty(0), stage_index=np.empty(0, dtype=int))
        self.assertEqual(self.validate([], lock, trace=terminal), [])

    def test_normal_ee_reach_suppresses_same_tick_poll_and_verifies_absorption(self):
        self.trace["stage_index"][:5] = 0
        lock = dict(control_call_index=5, simulated_time=.1, reason="reached_ee", initial_direct=False)
        transitions = [dict(time=.1, previous="link2_pose", next="end_effector_pose", reason="target_reached")]
        self.assertEqual(self.validate(["link2"], lock, transitions=transitions), [])
        with self.assertRaisesRegex(ValueError, "first EE entry"):
            self.validate(["link2"], None, transitions=transitions)
        with self.assertRaisesRegex(ValueError, "first EE entry"):
            self.validate(["link2"], dict(lock, reason="accepted_sequence_ee"), transitions=transitions)
        with self.assertRaisesRegex(ValueError, "no matching EE entry"):
            self.validate(["link2"], lock)
        self.trace["stage_index"][4] = 5
        with self.assertRaisesRegex(ValueError, "before the recorded"):
            self.validate(["link2"], lock, transitions=transitions)
        self.trace["stage_index"][4] = 0
        self.trace["stage_index"][12] = 1
        with self.assertRaisesRegex(ValueError, "left EE"):
            self.validate(["link2"], lock, transitions=transitions)

    def test_between_poll_reach_keeps_earlier_poll_and_terminal_link_needs_no_lock(self):
        self.trace["stage_index"][:7] = 0
        lock = dict(control_call_index=7, simulated_time=.14, reason="reached_ee", initial_direct=False)
        scores = [.1] * len(LINK_ACTIONS)
        scores[LINK_ACTIONS.index((2,))] = .9
        call = dict(prediction_with_scores(scores), time=.1, q=self.trace["q"][5].copy())
        event = dict(
            control_call_index=5, simulated_time=.1, q=self.trace["q"][5].tolist(),
            current_sequence_before=["link2"], current_sequence_after=["link2"],
            proposed_sequence=["link2"], confirmation_streak=0, streak_after=0,
            candidate_after=None, accepted=False, max_switches=None,
            switch_count_before=0, switch_count_after=0, blocked_by_switch_limit=False,
            previous_stage="link2_pose", entry_stage="link2_pose", next_stage="link2_pose",
            lock_on_ee=True, ee_locked_after=False,
        )
        transitions = [dict(time=.14, previous="link2_pose", next="end_effector_pose", reason="target_reached")]
        self.assertEqual(len(self.validate(["link2"], lock, [event], [call], transitions)), 1)
        with self.assertRaisesRegex(ValueError, "every scheduled"):
            self.validate(["link2"], lock, transitions=transitions)
        # A goal/timeout state can terminate before another controller call,
        # leaving the prior active link with no synthetic transition or lock.
        terminal = dict(self.trace, q=self.trace["q"][:6], time=self.trace["time"][:6],
                        command_time=self.trace["command_time"][:5], stage_index=np.zeros(5, dtype=int))
        self.assertEqual(self.validate(["link2"], None, trace=terminal), [])

    def test_accepted_route_keeps_triggering_poll_and_rejects_missing_or_later_polls(self):
        for proposal in ([], ["link3"]):
            with self.subTest(proposal=proposal):
                self.trace["stage_index"][:5] = 0
                lock = dict(control_call_index=5, simulated_time=.1,
                            reason="accepted_sequence_ee", initial_direct=False)
                scores = [.1] * len(LINK_ACTIONS)
                scores[LINK_ACTIONS.index(tuple(int(frame[4:]) for frame in proposal))] = .9
                call = dict(prediction_with_scores(scores), time=.1, q=self.trace["q"][5].copy())
                entry = "link3_pose" if proposal else "end_effector_pose"
                event = dict(
                    control_call_index=5, simulated_time=.1, q=self.trace["q"][5].tolist(),
                    current_sequence_before=["link2"], current_sequence_after=proposal,
                    proposed_sequence=proposal, confirmation_streak=1, streak_after=0,
                    candidate_after=None, accepted=True, max_switches=None,
                    switch_count_before=0, switch_count_after=1, blocked_by_switch_limit=False,
                    previous_stage="link2_pose", entry_stage=entry, next_stage="end_effector_pose",
                    lock_on_ee=True, ee_locked_after=True,
                )
                transitions = [dict(time=.1, previous="link2_pose", next=entry, reason="sequence_switch")]
                if proposal:
                    transitions.append(dict(time=.1, previous=entry, next="end_effector_pose", reason="target_reached"))
                records = self.validate(["link2"], lock, [event], [call], transitions)
                self.assertEqual(len(records), 1)
                self.assertTrue(records[0]["ee_locked_after"])
                with self.assertRaisesRegex(ValueError, "every scheduled"):
                    self.validate(["link2"], lock, transitions=transitions)
                with self.assertRaisesRegex(ValueError, "every scheduled"):
                    self.validate(["link2"], lock, [event, dict(event, control_call_index=10)], [call, call], transitions)
                with self.assertRaisesRegex(ValueError, "lock state"):
                    self.validate(["link2"], lock, [dict(event, ee_locked_after=False)], [call], transitions)

    def test_ee_lock_option_validation_and_summary_classification(self):
        self.assertFalse(validate_sequence_lock_on_ee(False, "static"))
        self.assertTrue(validate_sequence_lock_on_ee(np.bool_(True)))
        for value, mode in ((True, "static"), (1, "periodic"), ("true", "periodic"), (None, "periodic")):
            with self.subTest(value=value, mode=mode), self.assertRaises(ValueError):
                validate_sequence_lock_on_ee(value, mode)
        base = dict(execution_status="completed", policy="periodic", success=False,
                    goal_reached=False, reason="timeout", numerical_failure=False,
                    completion_time=None, stored_static_success=None, static_replay_matches=None)
        result = aggregate([
            dict(base, initial_direct=True, periodic_ee_locked=True, periodic_ee_lock_reason="initial_direct"),
            dict(base, initial_direct=False, periodic_ee_locked=True, periodic_ee_lock_reason="reached_ee"),
            dict(base, initial_direct=False, periodic_ee_locked=False, periodic_ee_lock_reason=None),
        ], requested=4)
        self.assertEqual(result["initial_direct_scenes"], 1)
        self.assertEqual(result["periodic_ee_locked_scenes"], 2)
        self.assertEqual(result["periodic_ee_lock_reasons"], dict(initial_direct=1, reached_ee=1))
        self.assertEqual(result["missing_cases"], 1)



class CollisionStallEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.trace = dict(
            q=np.zeros((4, 7)), time=np.array([0., .02, .04, .06]),
            command_time=np.array([0., .02, .04]),
        )
        self.rejected = dict(
            control_call_index=0, q=np.zeros(7).tolist(),
            previous_stage="link2_pose", next_stage="link2_pose", skipped=False,
            diagnostic_valid=False, active_collision_labels=[],
        )
        self.accepted = dict(
            self.rejected, control_call_index=2, skipped=True, diagnostic_valid=True,
            next_stage="link5_pose", active_collision_labels=["obstacle1_link2"],
        )
        self.transition = dict(
            time=.04, previous="link2_pose", next="link5_pose",
            reason="collision_constrained_stall",
        )

    def test_rejected_checks_and_repeated_q_use_exact_command_index(self):
        transitions = [
            self.transition,
            dict(time=.04, previous="link5_pose", next="end_effector_pose", reason="target_reached"),
        ]
        events = timestamp_collision_stall_events([self.rejected, self.accepted], self.trace, transitions)
        self.assertEqual([e["time"] for e in events], [0., .04])
        self.assertFalse(events[0]["diagnostic_valid"])
        self.assertEqual(events[1]["next_stage"], "link5_pose")
        self.assertNotIn("time", self.rejected)

    def test_mismatched_state_index_or_transition_cannot_certify_a_skip(self):
        bad_q = dict(self.accepted, q=np.ones(7).tolist())
        with self.assertRaises(AssertionError):
            timestamp_collision_stall_events([bad_q], self.trace, [self.transition])
        for index in (-1, 3, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                timestamp_collision_stall_events([dict(self.accepted, control_call_index=index)], self.trace, [self.transition])
        with self.assertRaises(ValueError):
            timestamp_collision_stall_events([self.accepted], self.trace, [])
        with self.assertRaises(ValueError):
            timestamp_collision_stall_events([dict(self.accepted, active_collision_labels=[])], self.trace, [self.transition])
        with self.assertRaises(ValueError):
            timestamp_collision_stall_events([self.accepted, self.rejected], self.trace, [self.transition])


if __name__ == "__main__":
    unittest.main()

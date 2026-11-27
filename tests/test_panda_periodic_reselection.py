"""Periodic, confirmed full-route switching with a fixed IK and unchanged QP."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.control.panda_controller import PandaTrajectoryController
from safe_ik.domain.ik_contracts import SafeIKCandidate
from safe_ik.domain.safety_contracts import BarrierEvaluation
from safe_ik.domain.types import QPSolution, QPStatus


class Kinematics:
    lower_limits = np.full(7, -10.0)
    upper_limits = np.full(7, 10.0)

    def pose_and_jacobian(self, q):
        return q[:3].copy(), np.array([1.0, 0, 0, 0]), np.eye(6, 7)

    def frame_point_and_jacobian(self, frame, q):
        joint = int(frame[4:]) - 2
        jacobian = np.zeros((3, 7))
        jacobian[0, joint] = 1.0
        return np.array([q[joint], 0, 0]), jacobian

    def frame_pose_and_jacobian(self, frame, q):
        position, jacobian = self.frame_point_and_jacobian(frame, q)
        return np.eye(3), position, np.vstack([jacobian, np.zeros((3, 7))])


class Safety:
    def evaluate(self, q):
        return BarrierEvaluation(("obstacle",), np.ones(1), np.zeros((1, 7)))


class Solver:
    def __init__(self, status=QPStatus.SOLVED):
        self.problems = []
        self.status = status

    def solve(self, problem):
        self.problems.append(problem)
        return QPSolution(self.status, np.zeros(8), 0.0)


class PeriodicReselectionTests(unittest.TestCase):
    def setUp(self):
        base = load_panda_config(Path(__file__).parent / "fixtures/panda.yaml")
        self.config = replace(
            base,
            initial_q=np.zeros(7),
            target=replace(
                base.target,
                position=np.full(3, 5.0),
                quaternion=np.array([1.0, 0, 0, 0]),
            ),
            trajectory=replace(
                base.trajectory,
                mode="sequential",
                stage_frames=tuple(f"link{i}" for i in range(2, 7)),
                stage_task="position",
                integration_period=0.02,
                stop_window=0.5,
                stop_progress_window=0.5,
            ),
        )
        self.candidate = SafeIKCandidate(np.ones(7), 0, 1, 0, 0, 1)

    def controller(self, selector, initial=("link2", "link4"), **kwargs):
        return PandaTrajectoryController(
            self.config,
            Kinematics(),
            Safety(),
            (self.candidate,),
            Solver(),
            periodic_stage_selector=selector,
            initial_stage_sequence=initial,
            **kwargs,
        )

    @staticmethod
    def step(controller, q=None, moving=True):
        q = np.zeros(7) if q is None else q
        controller.get_velocity(q)
        events = controller.consume_periodic_stage_selection_events()
        controller.record_applied_velocity(np.ones(7) if moving else np.zeros(7))
        return events

    def test_all_requested_periods_poll_on_simulated_grid_and_switch_after_five(self):
        for period in (0.1, 0.2, 0.4):
            with self.subTest(period=period):
                queried = []
                c = self.controller(
                    lambda q: queried.append(q.copy()) or ("link3", "link5"),
                    periodic_reselection_period=period,
                    periodic_reselection_confirmations=5,
                )
                ticks = round(period / 0.02)
                events = []
                for tick in range(5 * ticks + 1):
                    q = np.zeros(7)
                    q[6] = tick / 1000
                    events.extend(self.step(c, q))
                self.assertEqual(len(queried), 5)
                self.assertEqual(
                    [e["control_call_index"] for e in events],
                    [ticks * n for n in range(1, 6)],
                )
                np.testing.assert_allclose(
                    [e["simulated_time"] for e in events],
                    [period * n for n in range(1, 6)],
                )
                self.assertEqual([e["accepted"] for e in events], [False] * 4 + [True])
                self.assertEqual(c.active_sequence, ("link3", "link5"))
                self.assertEqual(c.active_stage_name, "link3_position")
                self.assertEqual(c.stage_index, 1)
                self.assertEqual(c.periodic_reselection_count, 1)
                self.assertEqual(c.candidate_switch_count, 0)
                self.assertEqual(c.stage_transition_count, 0)
                np.testing.assert_array_equal(c.target_q, self.candidate.q)
                np.testing.assert_array_equal(queried, [e["q"] for e in events])
                self.assertEqual(c.consume_periodic_stage_selection_events(), ())

    def test_current_action_resets_streak_and_different_candidate_starts_at_one(self):
        a, b, d = ("link2",), ("link3",), ("link4",)
        proposals = iter([b, b, a, b, d, b, b, b])
        c = self.controller(
            lambda q: next(proposals),
            initial=a,
            periodic_reselection_period=0.02,
            periodic_reselection_confirmations=3,
        )
        events = []
        for _ in range(9):
            events.extend(self.step(c))
        self.assertEqual(
            [e["confirmation_streak"] for e in events], [1, 2, 0, 1, 1, 1, 2, 3]
        )
        self.assertEqual([e["accepted"] for e in events], [False] * 7 + [True])
        self.assertIsNone(events[2]["candidate_after"])
        self.assertIsNone(events[-1]["candidate_after"])
        self.assertEqual(events[-1]["streak_after"], 0)

    def test_reach_advances_only_current_route_and_same_full_action_never_restarts(
        self,
    ):
        calls = []
        c = self.controller(
            lambda q: calls.append(q.copy()) or ("link2", "link5"),
            initial=("link2", "link5"),
            periodic_reselection_period=0.04,
            periodic_reselection_confirmations=1,
        )
        self.step(c)
        q = np.zeros(7)
        q[0] = 1.0
        self.step(c, q)  # reach between polls
        self.assertEqual(calls, [])
        self.assertEqual(c.stage_index, 3)
        self.assertEqual(c.active_sequence, ("link2", "link5"))
        q[0] = 0.0  # the old target need not remain reached
        (event,) = self.step(c, q)
        self.assertFalse(event["accepted"])
        self.assertEqual(c.active_stage_name, "link5_position")
        self.assertEqual(c.periodic_reselection_count, 0)
        self.assertEqual(
            c.consume_stage_transition_events(),
            (("link2_position", "link5_position", "target_reached"),),
        )

    def test_accepted_route_restarts_and_immediately_skips_only_currently_reached_links(
        self,
    ):
        c = self.controller(
            lambda q: ("link2", "link4"),
            initial=("link5",),
            periodic_reselection_period=0.02,
            periodic_reselection_confirmations=1,
        )
        self.step(c)
        q = np.zeros(7)
        q[0] = 1.0
        (event,) = self.step(c, q)
        self.assertEqual(event["entry_stage"], "link2_position")
        self.assertEqual(event["next_stage"], "link4_position")
        self.assertEqual(c.stage_index, 2)
        self.assertEqual(
            c.consume_stage_transition_events(),
            (
                ("link5_position", "link2_position", "sequence_switch"),
                ("link2_position", "link4_position", "target_reached"),
            ),
        )
        self.assertEqual(c.last_completed_link, 0)  # no monotone filter is used
        self.assertEqual(c.stage_names[-1], "end_effector_pose")
        self.assertEqual(len(c.stage_names), 6)

    def test_initial_direct_and_later_ee_can_both_return_to_intermediate_tasks(self):
        proposals = iter([("link6",), (), ("link2",)])
        c = self.controller(
            lambda q: next(proposals),
            initial=(),
            periodic_reselection_period=0.02,
            periodic_reselection_confirmations=1,
        )
        self.assertEqual(c.active_stage_name, "end_effector_pose")
        self.step(c)
        for expected in ("link6_position", "end_effector_pose", "link2_position"):
            (event,) = self.step(c)
            self.assertTrue(event["accepted"])
            self.assertEqual(c.active_stage_name, expected)
        self.assertEqual(c.periodic_reselection_count, 3)
        self.assertFalse(c.periodic_ee_locked)
        self.assertIsNone(c.periodic_ee_lock_event)

    def test_ee_lock_initial_direct_never_polls_or_changes_route(self):
        calls = []
        c = self.controller(
            lambda q: calls.append(q.copy()) or ("link6",),
            initial=(), periodic_reselection_period=0.02,
            periodic_reselection_confirmations=1,
            periodic_reselection_lock_on_ee=True,
        )
        self.assertTrue(c.periodic_ee_locked)
        self.assertEqual(c.periodic_ee_lock_event, dict(
            control_call_index=0, simulated_time=0.,
            reason="initial_direct", initial_direct=True,
        ))
        for _ in range(10):
            self.assertEqual(self.step(c), ())
        self.assertEqual(calls, [])
        self.assertEqual(c.active_sequence, ())
        self.assertEqual(c.active_stage_name, "end_effector_pose")
        self.assertEqual(c.periodic_reselection_count, 0)
        self.assertEqual(c.consume_stage_transition_events(), ())

    def test_ee_lock_normal_reach_precedes_same_tick_confirmed_switch(self):
        calls = []
        c = self.controller(
            lambda q: calls.append(q.copy()) or ("link3",),
            initial=("link2",), periodic_reselection_period=0.02,
            periodic_reselection_confirmations=2,
            periodic_reselection_lock_on_ee=True,
        )
        self.step(c)
        (pending,) = self.step(c)
        self.assertEqual(pending["confirmation_streak"], 1)
        q = np.zeros(7)
        q[0] = 1.
        self.assertEqual(self.step(c, q), ())
        self.assertEqual(c.periodic_ee_lock_event, dict(
            control_call_index=2, simulated_time=.04,
            reason="reached_ee", initial_direct=False,
        ))
        lock_event = c.periodic_ee_lock_event.copy()
        for _ in range(4):
            self.assertEqual(self.step(c), ())  # Old link no longer reached.
        self.assertEqual(len(calls), 1)
        self.assertEqual(c.active_sequence, ("link2",))
        self.assertEqual(c.active_stage_name, "end_effector_pose")
        self.assertEqual(c.periodic_reselection_count, 0)
        self.assertEqual(c.periodic_ee_lock_event, lock_event)
        self.assertIsNone(c._periodic_proposal)
        self.assertEqual(c._periodic_streak, 0)
        self.assertEqual(c.consume_stage_transition_events(), (
            ("link2_position", "end_effector_pose", "target_reached"),
        ))

    def test_ee_lock_accepting_direct_records_last_real_poll_then_stops(self):
        calls = []
        c = self.controller(
            lambda q: calls.append(q.copy()) or (),
            initial=("link2",), periodic_reselection_period=0.02,
            periodic_reselection_confirmations=2,
            periodic_reselection_lock_on_ee=True,
        )
        events = []
        for _ in range(8):
            events.extend(self.step(c))
        self.assertEqual(len(calls), 2)
        self.assertEqual([e["accepted"] for e in events], [False, True])
        self.assertEqual([e["ee_locked_after"] for e in events], [False, True])
        self.assertTrue(all(e["lock_on_ee"] for e in events))
        self.assertEqual(c.periodic_ee_lock_event, dict(
            control_call_index=2, simulated_time=.04,
            reason="accepted_sequence_ee", initial_direct=False,
        ))
        self.assertEqual(c.periodic_reselection_count, 1)
        self.assertEqual(c.active_sequence, ())
        self.assertEqual(c.active_stage_name, "end_effector_pose")

    def test_ee_lock_new_route_already_reached_locks_after_same_tick_advance(self):
        c = self.controller(
            lambda q: ("link2",), initial=("link5",),
            periodic_reselection_period=0.02, periodic_reselection_confirmations=1,
            periodic_reselection_lock_on_ee=True,
        )
        self.step(c)
        q = np.zeros(7)
        q[0] = 1.
        (event,) = self.step(c, q)
        self.assertTrue(event["accepted"])
        self.assertTrue(event["ee_locked_after"])
        self.assertEqual(event["entry_stage"], "link2_position")
        self.assertEqual(event["next_stage"], "end_effector_pose")
        self.assertEqual(c.periodic_ee_lock_event["reason"], "accepted_sequence_ee")
        self.assertEqual(c.periodic_ee_lock_event["simulated_time"], .02)
        self.assertEqual(self.step(c), ())
        self.assertEqual(c.consume_stage_transition_events(), (
            ("link5_position", "link2_position", "sequence_switch"),
            ("link2_position", "end_effector_pose", "target_reached"),
        ))

    def test_ee_lock_does_not_impose_switch_cap_before_ee(self):
        routes = [("link2",), ("link3",)] * 4
        proposals = iter(routes)
        c = self.controller(
            lambda q: next(proposals), initial=("link5",),
            periodic_reselection_period=0.02, periodic_reselection_confirmations=1,
            periodic_reselection_lock_on_ee=True,
        )
        self.step(c)
        for route in routes:
            (event,) = self.step(c)
            self.assertTrue(event["accepted"])
            self.assertFalse(event["ee_locked_after"])
            self.assertEqual(c.active_sequence, route)
        self.assertEqual(c.periodic_reselection_count, 8)
        self.assertIsNone(c.periodic_ee_lock_event)

    def test_ee_lock_requires_boolean_and_periodic_mode(self):
        for value in (0, 1, "false", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                self.controller(lambda q: (), periodic_reselection_lock_on_ee=value)
        with self.assertRaisesRegex(ValueError, "requires periodic reselection"):
            PandaTrajectoryController(
                self.config, Kinematics(), Safety(), (self.candidate,), Solver(),
                periodic_reselection_lock_on_ee=True,
            )

    def test_same_first_link_changed_suffix_is_a_real_switch(self):
        c = self.controller(
            lambda q: ("link2", "link6"),
            periodic_reselection_period=0.02,
            periodic_reselection_confirmations=1,
        )
        self.step(c)
        (event,) = self.step(c)
        self.assertTrue(event["accepted"])
        self.assertEqual(event["previous_stage"], event["next_stage"])
        self.assertEqual(c.active_sequence, ("link2", "link6"))
        self.assertEqual(
            c.consume_stage_transition_events(),
            (("link2_position", "link2_position", "sequence_switch"),),
        )
        self.assertEqual(len(c._progress_best_errors), 1)

    def test_total_switch_cap_counts_full_routes_and_keeps_reaching_through_ee(self):
        last_route = ("link3", "link5")
        proposals = iter([("link2", "link6"), (), last_route] + [("link6",)] * 3)
        queried = []

        def selector(q):
            queried.append(q.copy())
            return next(proposals)

        c = self.controller(
            selector,
            periodic_reselection_period=0.02,
            periodic_reselection_confirmations=1,
            periodic_reselection_max_switches=3,
        )
        self.assertEqual(c.periodic_reselection_count, 0)
        self.assertEqual(self.step(c), ())
        self.assertEqual(queried, [])  # initial selection does not consume the cap
        events = []
        for _ in range(4):
            events.extend(self.step(c))
        self.assertEqual([e["accepted"] for e in events], [True, True, True, False])
        self.assertEqual([e["switch_count_before"] for e in events], [0, 1, 2, 3])
        self.assertEqual([e["switch_count_after"] for e in events], [1, 2, 3, 3])
        self.assertEqual([e["max_switches"] for e in events], [3] * 4)
        self.assertEqual(
            [e["blocked_by_switch_limit"] for e in events], [False] * 3 + [True]
        )
        self.assertEqual(events[0]["previous_stage"], events[0]["next_stage"])
        self.assertEqual(events[1]["next_stage"], "end_effector_pose")
        self.assertEqual(events[2]["previous_stage"], "end_effector_pose")
        self.assertEqual(c.active_sequence, last_route)
        self.assertEqual(c.periodic_reselection_count, 3)
        q = np.zeros(7)
        q[1] = 1.0
        (event,) = self.step(c, q)
        self.assertTrue(event["blocked_by_switch_limit"])
        self.assertEqual(c.active_stage_name, "link5_position")
        q[3] = 1.0
        (event,) = self.step(c, q)
        self.assertTrue(event["blocked_by_switch_limit"])
        self.assertEqual(c.active_stage_name, "end_effector_pose")
        self.assertEqual(c.active_sequence, last_route)
        self.assertEqual(c.periodic_reselection_count, 3)
        self.assertEqual(c.stage_transition_count, 2)
        transitions = c.consume_stage_transition_events()
        self.assertEqual(sum(t[2] == "sequence_switch" for t in transitions), 3)
        self.assertEqual(sum(t[2] == "target_reached" for t in transitions), 2)
        self.assertEqual(self.step(c, np.full(7, 5.0)), ())
        self.assertTrue(c.last_status.success)
        self.assertEqual(len(queried), 6)

    def test_exhausted_cap_keeps_pending_streak_and_ordinary_stall_termination(self):
        proposals = iter([("link3",)] * 2 + [()] * 2 + [("link4",)] * 2)
        c = self.controller(
            lambda q: next(proposals, ("link6",)),
            periodic_reselection_period=0.02,
            periodic_reselection_confirmations=2,
            periodic_reselection_max_switches=3,
        )
        events = []
        for tick in range(100):
            events.extend(self.step(c, moving=False))
            if c.last_status.candidates_exhausted:
                break
        self.assertEqual(tick, 31)  # usual 25-step stall window after switch at tick 6
        self.assertEqual(len(events), 31)
        self.assertEqual(c.periodic_reselection_count, 3)
        self.assertEqual(c.active_sequence, ("link4",))
        self.assertEqual(c.last_status.stage_name, "link4_position")
        blocked = events[6:]
        self.assertEqual(
            [e["confirmation_streak"] for e in blocked], list(range(1, 26))
        )
        self.assertEqual([e["streak_after"] for e in blocked], list(range(1, 26)))
        self.assertEqual(
            [e["blocked_by_switch_limit"] for e in blocked], [False] + [True] * 24
        )
        self.assertTrue(all(e["candidate_after"] == ["link6"] for e in blocked))
        self.assertTrue(all(e["switch_count_after"] == 3 for e in blocked))
        self.assertTrue(all(not e["accepted"] for e in blocked))
        self.assertEqual(len(c._progress_best_errors), c._progress_best_errors.maxlen)
        self.assertEqual(len(c.applied_velocity_norms), c.applied_velocity_norms.maxlen)

    def test_zero_cap_preserves_pending_resets_and_fixed_route_qps(self):
        route, b, d = ("link2", "link4"), ("link3",), ("link5",)
        proposals = iter([b, b, b, d, d, route, b, b])
        capped = self.controller(
            lambda q: next(proposals),
            initial=route,
            periodic_reselection_period=0.02,
            periodic_reselection_confirmations=2,
            periodic_reselection_max_switches=0,
        )
        fixed = PandaTrajectoryController(
            replace(
                self.config,
                trajectory=replace(self.config.trajectory, stage_frames=route),
            ),
            Kinematics(),
            Safety(),
            (self.candidate,),
            Solver(),
        )
        events = []
        for tick in range(9):
            q = np.zeros(7)
            if tick >= 4:
                q[0] = 1.0
            if tick >= 7:
                q[2] = 1.0
            events.extend(self.step(capped, q))
            self.step(fixed, q)
            self.assertEqual(capped.active_stage_name, fixed.active_stage_name)
            self.assertEqual(
                list(capped._progress_best_errors), list(fixed._progress_best_errors)
            )
            for key in ("P", "q", "A", "lower", "upper"):
                np.testing.assert_array_equal(
                    getattr(capped.solver.problems[-1], key),
                    getattr(fixed.solver.problems[-1], key),
                )
        self.assertEqual(
            [e["confirmation_streak"] for e in events], [1, 2, 3, 1, 2, 0, 1, 2]
        )
        self.assertEqual(
            [e["blocked_by_switch_limit"] for e in events],
            [False, True, True, False, True, False, False, True],
        )
        self.assertTrue(all(not e["accepted"] for e in events))
        self.assertTrue(
            all(
                e["switch_count_before"] == e["switch_count_after"] == 0 for e in events
            )
        )
        self.assertEqual(capped.periodic_reselection_count, 0)
        self.assertEqual(capped.active_sequence, route)
        self.assertEqual(capped.stage_transition_count, 2)
        self.assertEqual(capped.active_stage_name, "end_effector_pose")

    def test_explicit_none_preserves_unlimited_switches_events_and_qps(self):
        proposals = [("link3",)] * 2 + [()] * 2 + [("link2",)] * 2
        proposals += [("link2", "link6")] * 2 + [("link5",)] * 2
        default_proposals, explicit_proposals = iter(proposals), iter(proposals)
        default = self.controller(
            lambda q: next(default_proposals),
            periodic_reselection_period=0.02,
            periodic_reselection_confirmations=2,
        )
        explicit = self.controller(
            lambda q: next(explicit_proposals),
            periodic_reselection_period=0.02,
            periodic_reselection_confirmations=2,
            periodic_reselection_max_switches=None,
            periodic_reselection_lock_on_ee=False,
            periodic_stage_switch_guard=None,
        )
        for _ in range(len(proposals) + 1):
            left, right = self.step(default), self.step(explicit)
            self.assertEqual(left, right)
            self.assertTrue(all(e["max_switches"] is None for e in left))
            self.assertTrue(all(not e["blocked_by_switch_limit"] for e in left))
            self.assertTrue(all(not e["switch_guard_evaluated"] for e in left))
            self.assertTrue(all(not e["blocked_by_switch_guard"] for e in left))
            self.assertEqual(default.active_sequence, explicit.active_sequence)
            self.assertEqual(
                default.consume_stage_transition_events(),
                explicit.consume_stage_transition_events(),
            )
            for key in ("P", "q", "A", "lower", "upper"):
                np.testing.assert_array_equal(
                    getattr(default.solver.problems[-1], key),
                    getattr(explicit.solver.problems[-1], key),
                )
        self.assertEqual(default.periodic_reselection_count, 5)
        self.assertEqual(explicit.periodic_reselection_count, 5)

    def test_switch_guard_waits_for_confirmation_and_reconsiders_next_poll(self):
        route, proposed = ("link2", "link4"), ("link3", "link5")
        queried, guarded = [], []
        decisions = iter((False, True))

        def selector(q):
            queried.append(q.copy())
            return proposed

        def guard(q, current, chosen):
            guarded.append((q.copy(), current, chosen))
            q[:] = -10.0  # The guard cannot mutate the controller's input.
            return next(decisions)

        c = self.controller(
            selector,
            initial=route,
            periodic_reselection_period=0.02,
            periodic_reselection_confirmations=3,
            periodic_stage_switch_guard=guard,
        )
        unchanged = self.controller(
            lambda q: route,
            initial=route,
            periodic_reselection_period=0.02,
        )
        events = []
        for tick in range(4):
            q = np.zeros(7)
            if tick:
                q[0] = 1.0  # Complete link2; the guard still receives the full route.
            events.extend(self.step(c, q))
            self.step(unchanged, q)
            self.assertEqual(len(guarded), int(tick == 3))
            self.assertEqual(c.active_sequence, route)
            self.assertEqual(c.active_stage_name, unchanged.active_stage_name)
            self.assertEqual(
                list(c._progress_best_errors), list(unchanged._progress_best_errors)
            )
            self.assertEqual(
                list(c.applied_velocity_norms), list(unchanged.applied_velocity_norms)
            )
            for key in ("P", "q", "A", "lower", "upper"):
                np.testing.assert_array_equal(
                    getattr(c.solver.problems[-1], key),
                    getattr(unchanged.solver.problems[-1], key),
                )
        self.assertEqual(len(queried), 3)
        self.assertEqual(
            [e["switch_guard_evaluated"] for e in events], [False, False, True]
        )
        self.assertEqual(
            [e["blocked_by_switch_guard"] for e in events], [False, False, True]
        )
        self.assertFalse(events[-1]["accepted"])
        self.assertEqual(events[-1]["candidate_after"], list(proposed))
        self.assertEqual(events[-1]["streak_after"], 3)
        self.assertEqual(c.periodic_reselection_count, 0)
        self.assertEqual(c.consume_stage_transition_events(), (
            ("link2_position", "link4_position", "target_reached"),
        ))

        q[6] = 0.25
        (accepted,) = self.step(c, q)
        self.assertEqual(len(queried), 4)  # One selector call per scheduled poll.
        self.assertEqual(len(guarded), 2)
        for index, (guard_q, current, chosen) in enumerate(guarded):
            np.testing.assert_array_equal(guard_q, queried[index + 2])
            self.assertEqual(current, route)
            self.assertEqual(chosen, proposed)
        self.assertEqual(accepted["confirmation_streak"], 4)
        self.assertTrue(accepted["switch_guard_evaluated"])
        self.assertFalse(accepted["blocked_by_switch_guard"])
        self.assertTrue(accepted["accepted"])
        self.assertIsNone(accepted["candidate_after"])
        self.assertEqual(accepted["streak_after"], 0)
        self.assertEqual(c.active_sequence, proposed)
        self.assertEqual(c.periodic_reselection_count, 1)
        self.assertEqual(len(c._progress_best_errors), 1)
        self.assertEqual(len(c.applied_velocity_norms), 1)

    def test_guard_rejection_keeps_normal_current_and_different_proposal_resets(self):
        a, b, d = ("link2",), ("link3",), ("link4",)
        proposals = iter((b, b, a, b, d, d))
        guarded = []
        c = self.controller(
            lambda q: next(proposals),
            initial=a,
            periodic_reselection_period=0.02,
            periodic_reselection_confirmations=2,
            periodic_stage_switch_guard=(
                lambda q, current, proposed: guarded.append(proposed) or False
            ),
        )
        events = []
        for _ in range(7):
            events.extend(self.step(c))
        self.assertEqual(
            [e["confirmation_streak"] for e in events], [1, 2, 0, 1, 1, 2]
        )
        self.assertEqual(
            [e["switch_guard_evaluated"] for e in events],
            [False, True, False, False, False, True],
        )
        self.assertIsNone(events[2]["candidate_after"])
        self.assertEqual(events[4]["candidate_after"], list(d))
        self.assertEqual(guarded, [b, d])
        self.assertTrue(all(not e["accepted"] for e in events))
        self.assertEqual(c.active_sequence, a)
        self.assertEqual(c.consume_stage_transition_events(), ())

    def test_switch_cap_takes_precedence_over_guard(self):
        for cap in (0, 1):
            with self.subTest(cap=cap):
                proposals = iter([("link3",)] * 2 + [("link5",)] * 3)
                guarded = []
                c = self.controller(
                    lambda q: next(proposals),
                    periodic_reselection_period=0.02,
                    periodic_reselection_confirmations=2,
                    periodic_reselection_max_switches=cap,
                    periodic_stage_switch_guard=(
                        lambda q, current, proposed: guarded.append(proposed) or True
                    ),
                )
                events = []
                for _ in range(6):
                    events.extend(self.step(c))
                self.assertEqual(len(guarded), cap)
                self.assertEqual(c.periodic_reselection_count, cap)
                self.assertTrue(all(not e["blocked_by_switch_guard"] for e in events))
                for event in events:
                    if event["blocked_by_switch_limit"]:
                        self.assertFalse(event["switch_guard_evaluated"])
                        self.assertFalse(event["accepted"])
                self.assertTrue(events[-1]["blocked_by_switch_limit"])
                self.assertEqual(events[-1]["candidate_after"], ["link5"])
                self.assertEqual(events[-1]["streak_after"], 3)

    def test_switch_guard_requires_callable_and_periodic_reselection(self):
        for guard in (False, True, 0, "guard"):
            with self.subTest(guard=guard), self.assertRaisesRegex(ValueError, "callable"):
                self.controller(lambda q: (), periodic_stage_switch_guard=guard)
        with self.assertRaisesRegex(ValueError, "requires periodic reselection"):
            PandaTrajectoryController(
                self.config, Kinematics(), Safety(), (self.candidate,), Solver(),
                periodic_stage_switch_guard=lambda q, current, proposed: True,
            )

    def test_switch_guard_requires_boolean_result(self):
        invalid = (None, 0, 1, "true", [], np.array(True), np.array([True]))
        for result in invalid:
            with self.subTest(result=result):
                c = self.controller(
                    lambda q: ("link5",),
                    periodic_reselection_period=0.02,
                    periodic_reselection_confirmations=1,
                    periodic_stage_switch_guard=lambda q, current, proposed: result,
                )
                self.step(c)
                with self.assertRaisesRegex(ValueError, "must return a boolean"):
                    self.step(c)
                self.assertEqual(c.active_sequence, ("link2", "link4"))
                self.assertEqual(c.periodic_reselection_count, 0)
                self.assertEqual(c.consume_stage_transition_events(), ())
        for result in (True, False, np.bool_(True), np.bool_(False)):
            with self.subTest(result=result):
                c = self.controller(
                    lambda q: ("link5",),
                    periodic_reselection_period=0.02,
                    periodic_reselection_confirmations=1,
                    periodic_stage_switch_guard=lambda q, current, proposed: result,
                )
                self.step(c)
                (event,) = self.step(c)
                self.assertIs(event["accepted"], bool(result))
                self.assertIs(event["blocked_by_switch_guard"], not bool(result))
                self.assertIs(event["switch_guard_evaluated"], True)

    def test_switch_cap_requires_nonnegative_integer_and_periodic_policy(self):
        for limit in (-1, 1.2, 3.0, True, False, np.bool_(True), "3", np.nan):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "nonnegative integer"),
            ):
                self.controller(lambda q: (), periodic_reselection_max_switches=limit)
        for limit in (0, 3, np.int64(3)):
            with self.subTest(limit=limit):
                c = self.controller(
                    lambda q: (), periodic_reselection_max_switches=limit
                )
                self.assertEqual(c.periodic_reselection_max_switches, int(limit))
                self.assertIs(type(c.periodic_reselection_max_switches), int)
                with self.assertRaisesRegex(
                    ValueError, "requires periodic reselection"
                ):
                    PandaTrajectoryController(
                        self.config,
                        Kinematics(),
                        Safety(),
                        (self.candidate,),
                        Solver(),
                        periodic_reselection_max_switches=limit,
                    )

    def test_unconfirmed_polls_do_not_reset_stall_history_or_prevent_terminal_stall(
        self,
    ):
        c = self.controller(
            lambda q: ("link5",),
            periodic_reselection_period=0.02,
            periodic_reselection_confirmations=100,
        )
        events = []
        for tick in range(100):
            events.extend(self.step(c, moving=False))
            if c.last_status.candidates_exhausted:
                break
        self.assertEqual(tick, 25)
        self.assertEqual(len(events), 25)
        self.assertEqual(c.periodic_reselection_count, 0)
        self.assertEqual(c.last_status.stage_name, "link2_position")

    def test_qp_failure_and_completed_ee_do_not_start_new_sequences(self):
        calls = []
        for succeeded in (True, False):
            with self.subTest(succeeded=succeeded):
                c = self.controller(
                    lambda q: calls.append(q.copy()) or ("link5",),
                    initial=(),
                    periodic_reselection_period=0.02,
                )
                q = np.full(7, 5.0) if succeeded else np.zeros(7)
                if not succeeded:
                    c.solver = Solver(QPStatus.FAILED)
                for _ in range(3):
                    self.step(c, q)
                self.assertEqual(calls, [])
                self.assertEqual(c.active_stage_name, "end_effector_pose")
                self.assertEqual(c.periodic_reselection_count, 0)
                self.assertTrue(
                    c.last_status.success if succeeded else c.qp_infeasible_stopped
                )

    def test_non_switching_periodic_qp_matches_fixed_path_for_position_and_pose(self):
        for task in ("position", "pose"):
            with self.subTest(task=task):
                self.config = replace(
                    self.config,
                    trajectory=replace(self.config.trajectory, stage_task=task),
                )
                route = ("link3", "link6")
                periodic = self.controller(
                    lambda q: route, initial=route, periodic_reselection_period=0.02
                )
                fixed = PandaTrajectoryController(
                    replace(
                        self.config,
                        trajectory=replace(self.config.trajectory, stage_frames=route),
                    ),
                    Kinematics(),
                    Safety(),
                    (self.candidate,),
                    Solver(),
                )
                for step in range(10):
                    q = np.zeros(7)
                    if step > 2:
                        q[1] = 1.0
                    if step > 5:
                        q[4] = 1.0
                    self.step(periodic, q)
                    self.step(fixed, q)
                    self.assertEqual(
                        periodic.active_stage_name, fixed.active_stage_name
                    )
                    for key in ("P", "q", "A", "lower", "upper"):
                        np.testing.assert_array_equal(
                            getattr(periodic.solver.problems[-1], key),
                            getattr(fixed.solver.problems[-1], key),
                        )
                self.assertEqual(periodic.periodic_reselection_count, 0)

    def test_rejects_ambiguous_periods_invalid_actions_and_combined_policies(self):
        for period in (0, -0.1, 0.03, np.nan, np.inf):
            with self.subTest(period=period), self.assertRaises(ValueError):
                self.controller(lambda q: (), periodic_reselection_period=period)
        for n in (0, -1, 1.2, True):
            with self.subTest(confirmations=n), self.assertRaises(ValueError):
                self.controller(lambda q: (), periodic_reselection_confirmations=n)
        for sequence in (("link7",), ("link3", "link2"), ("link2", "link2")):
            with self.subTest(sequence=sequence):
                with self.assertRaises(ValueError):
                    self.controller(lambda q: (), initial=sequence)
                c = self.controller(
                    lambda q: sequence, periodic_reselection_period=0.02
                )
                self.step(c)
                with self.assertRaises(ValueError):
                    self.step(c)
        with self.assertRaisesRegex(ValueError, "no other reselection"):
            self.controller(lambda q: (), stage_selector=lambda q: ())
        with self.assertRaisesRegex(ValueError, "no other reselection"):
            self.controller(
                lambda q: (),
                advance_on_collision_stall=True,
                collision_stall_solver=Solver(),
            )


if __name__ == "__main__":
    unittest.main()

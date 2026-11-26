"""Stage selection is monotone and occurs only at intermediate target reaches."""

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
        position, position_jacobian = self.frame_point_and_jacobian(frame, q)
        cosine, sine = np.cos(q[6]), np.sin(q[6])
        rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
        jacobian = np.vstack([position_jacobian, np.zeros((3, 7))])
        jacobian[5, 6] = 1.0
        return rotation, position, jacobian


class Safety:
    def evaluate(self, q):
        return BarrierEvaluation(("obstacle",), np.ones(1), np.zeros((1, 7)))


class Solver:
    def solve(self, problem):
        return QPSolution(QPStatus.SOLVED, np.zeros(8), 0.0)


class ReachedReselectionTest(unittest.TestCase):
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
                stop_progress_window=0.5,
            ),
        )
        self.target = np.array([1.0, 1, 1, 1, 1, 0, 0])
        self.candidate = SafeIKCandidate(self.target, 0, 1, 0, 0, 1)

    def controller(self, selector=None, config=None, candidates=None):
        return PandaTrajectoryController(
            config or self.config,
            Kinematics(),
            Safety(),
            (self.candidate,) if candidates is None else candidates,
            Solver(),
            stage_selector=selector,
        )

    def test_reinfer_from_reached_state_and_execute_only_first_remaining_link(self):
        actions = iter([("link2", "link4"), ("link3", "link5"), ("link2", "link5"), ()])
        query_states = []

        def select(q):
            query_states.append(q.copy())
            return next(actions)

        controller = self.controller(select)
        q = np.zeros(7)
        controller.get_velocity(q)
        self.assertEqual(controller.active_stage_name, "link2_position")
        for joint, expected in (
            (0, "link3_position"),
            (1, "link5_position"),
            (3, "end_effector_pose"),
        ):
            q[joint] = 1.0
            controller.get_velocity(q)
            self.assertEqual(controller.active_stage_name, expected)
            np.testing.assert_array_equal(query_states[-1], q)
        self.assertEqual(controller.last_completed_link, 5)
        self.assertEqual(controller.stage_transition_count, 3)
        self.assertEqual(controller.candidate_switch_count, 0)
        np.testing.assert_array_equal(controller.target_q, self.target)
        decisions = controller.consume_stage_selection_events()
        self.assertEqual([d["last_completed_link"] for d in decisions], [0, 2, 3, 5])
        self.assertEqual(decisions[2]["discarded_frames"], ["link2"])
        self.assertEqual(
            [d["next_frame"] for d in decisions], ["link2", "link3", "link5", None]
        )
        self.assertEqual(controller.consume_stage_selection_events(), ())
        self.assertEqual(
            controller.stage_names,
            tuple(f"link{i}_position" for i in range(2, 7)) + ("end_effector_pose",),
        )

    def test_empty_action_is_direct_and_final_stage_is_absorbing(self):
        calls = []
        controller = self.controller(lambda q: calls.append(q.copy()) or ())
        for _ in range(3):
            controller.get_velocity(np.zeros(7))
        self.assertEqual(len(calls), 1)
        self.assertEqual(controller.stage_index, 5)
        self.assertEqual(controller.last_completed_link, 0)
        self.assertEqual(controller.stage_transition_count, 0)

    def test_only_previously_completed_or_earlier_links_falls_back_to_ee(self):
        actions = iter([("link5",), ("link2", "link5")])
        controller = self.controller(lambda q: next(actions))
        controller.get_velocity(np.zeros(7))
        self.assertEqual(controller.active_stage_name, "link5_position")
        q = np.zeros(7)
        q[3] = 1.0
        controller.get_velocity(q)
        self.assertEqual(controller.active_stage_name, "end_effector_pose")
        self.assertEqual(
            controller.consume_stage_transition_events(),
            (("link5_position", "end_effector_pose", "target_reached"),),
        )
        self.assertEqual(
            controller.consume_stage_selection_events()[-1]["discarded_frames"],
            ["link2", "link5"],
        )

    def test_already_reached_targets_progress_in_same_tick_without_infinite_loop(self):
        actions = iter(
            [("link3", "link2"), ("link4", "link3"), ("link5",), ("link6",), ("link6",)]
        )
        controller = self.controller(lambda q: next(actions))
        controller.get_velocity(self.target)
        events = controller.consume_stage_selection_events()
        self.assertEqual(
            [event["next_frame"] for event in events],
            ["link2", "link3", "link5", "link6", None],
        )
        self.assertEqual(controller.last_completed_link, 6)
        self.assertEqual(controller.stage_index, 5)
        self.assertEqual(controller.stage_transition_count, 4)

    def test_stall_does_not_reselect_but_reach_resets_both_stall_histories(self):
        actions = iter([("link2",), ("link4",)])
        queries = []

        def select(q):
            queries.append(q.copy())
            return next(actions)

        controller = self.controller(select)
        q = np.zeros(7)
        for _ in range(100):
            controller.get_velocity(q)
            controller.record_applied_velocity(np.zeros(7))
        self.assertTrue(controller.last_status.candidates_exhausted)
        self.assertEqual(len(queries), 1)
        q[0] = 1.0
        controller.get_velocity(q)
        self.assertEqual(len(queries), 2)
        self.assertFalse(controller.last_status.stalled)
        self.assertEqual(len(controller.applied_velocity_norms), 0)
        self.assertEqual(len(controller._progress_best_errors), 1)
        self.assertEqual(controller.candidate_switch_count, 0)

    def test_pose_stage_requires_orientation_as_well_as_position(self):
        config = replace(
            self.config, trajectory=replace(self.config.trajectory, stage_task="pose")
        )
        actions = iter([("link2",), ()])
        controller = self.controller(lambda q: next(actions), config=config)
        q = np.zeros(7)
        q[0], q[6] = 1.0, 0.5
        controller.get_velocity(q)
        self.assertEqual(controller.stage_index, 0)
        q[6] = 0.0
        controller.get_velocity(q)
        self.assertEqual(controller.stage_index, 5)

    def test_selector_cannot_modify_the_control_state(self):
        def select(q):
            q[:] = 99.0
            return ("link4",)

        controller = self.controller(select)
        q = np.zeros(7)
        controller.get_velocity(q)
        np.testing.assert_array_equal(q, np.zeros(7))
        self.assertEqual(controller.consume_stage_selection_events()[0]["q"], [0.0] * 7)

    def test_invalid_policy_inputs_are_rejected_without_changing_ik(self):
        with self.assertRaisesRegex(ValueError, "one fixed IK"):
            self.controller(lambda q: (), candidates=(self.candidate, self.candidate))
        direct = replace(
            self.config, trajectory=replace(self.config.trajectory, mode="direct")
        )
        with self.assertRaisesRegex(ValueError, "sequential mode"):
            self.controller(lambda q: (), config=direct)
        unsorted = replace(
            self.config,
            trajectory=replace(self.config.trajectory, stage_frames=("link5", "link2")),
        )
        with self.assertRaisesRegex(ValueError, "ascending"):
            self.controller(lambda q: (), config=unsorted)
        controller = self.controller(lambda q: ("link7",))
        with self.assertRaisesRegex(ValueError, "unknown link frames"):
            controller.get_velocity(np.zeros(7))

    def test_default_fixed_sequence_still_visits_every_stage(self):
        controller = self.controller()
        q = np.zeros(7)
        controller.get_velocity(q)
        for joint in range(5):
            q[joint] = 1.0
            controller.get_velocity(q)
            self.assertEqual(controller.stage_index, joint + 1)
        self.assertEqual(controller.stage_transition_count, 5)
        self.assertEqual(controller.consume_stage_selection_events(), ())


if __name__ == "__main__":
    unittest.main()

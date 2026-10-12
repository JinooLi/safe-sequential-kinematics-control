"""Protect paired scoring against stale barriers, stage gates and biased cohorts."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from safe_ik.adapters.mujoco_plant import MujocoPlant
from safe_ik.adapters.panda_kinematics import PandaKinematics
from safe_ik.application.panda_comparison import (
    SceneSafety,
    atomic_json,
    digest,
    validate_completed_run,
    verify_resume,
)
from safe_ik.application.panda_comparison_rollout import comparison_rollout
from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.application.panda_runtime import PandaRuntime
from safe_ik.domain.ik_contracts import SafeIKCandidate
from safe_ik.domain.safety_contracts import BarrierEvaluation
from safe_ik.domain.types import QPSolution, QPStatus
from safe_ik.experiments.panda_comparison import paired_analysis

ROOT = Path(__file__).resolve().parents[1]


class Kinematics:
    lower_limits = np.full(7, -10.0)
    upper_limits = np.full(7, 10.0)

    def pose_and_jacobian(self, q):
        return q[:3].copy(), np.array([1.0, 0.0, 0.0, 0.0]), np.eye(6, 7)

    def frame_point_and_jacobian(self, name, q, local_offset=None):
        return q[3:6].copy(), np.eye(7)[3:6]


class Safety:
    def __init__(self, function=lambda q: 1.0, label="self_test"):
        self.function, self.label = function, label

    def evaluate(self, q):
        return BarrierEvaluation(
            (self.label,), np.array([self.function(q)]), np.zeros((1, 7))
        )


class Plant:
    def __init__(self):
        self.q = np.zeros(7)
        self.time = 0.0

    def advance(self, velocity, period):
        self.q += velocity * period
        self.time += period


class Solver:
    def __init__(self, status=QPStatus.SOLVED):
        self.status = status

    def solve(self, problem):
        return QPSolution(
            self.status,
            (
                np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                if self.status == QPStatus.SOLVED
                else None
            ),
            0.001,
            "test",
        )


class PandaComparisonTest(unittest.TestCase):
    def setUp(self):
        config = load_panda_config(ROOT / "configs/panda.yaml")
        self.config = replace(
            config,
            initial_q=np.zeros(7),
            target=replace(
                config.target,
                position=np.array([0.02, 0.0, 0.0]),
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                position_tolerance=0.001,
            ),
            trajectory=replace(
                config.trajectory,
                mode="direct",
                duration=0.02,
                integration_period=0.02,
                stage_task="position",
                stage_frames=("link3", "link5"),
            ),
        )

    def run_case(
        self, safety=None, config=None, candidate=None, status=QPStatus.SOLVED
    ):
        runtime = PandaRuntime(
            Kinematics(), safety or Safety(), Solver(status), Plant()
        )
        candidates = () if candidate is None else (candidate,)
        return comparison_rollout(config or self.config, runtime, candidates)

    def test_goal_on_final_step_is_counted_and_recorded(self):
        result, trace = self.run_case()
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 1)
        self.assertEqual(len(trace["q"]), 2)
        np.testing.assert_array_equal(trace["q"][-1], result["final_q"])
        self.assertAlmostEqual(result["completion_time"], 0.02)

    def test_joint_stop_records_actual_motion_without_hiding_requested_command(self):
        config = load_panda_config(ROOT / "tests/fixtures/panda.yaml")
        model_kin = PandaKinematics(config.model_path, config.ee_frame)
        initial = config.initial_q.copy()
        initial[0] = model_kin.upper_limits[0] - 1e-4
        config = replace(
            config,
            initial_q=initial,
            target=replace(
                config.target,
                position=initial[:3] + [0.02, 0, 0],
                quaternion=np.array([1.0, 0, 0, 0]),
                position_tolerance=0.001,
            ),
            trajectory=replace(
                config.trajectory,
                mode="direct",
                duration=0.04,
                integration_period=0.02,
                success_barrier_tolerance=0.001,
            ),
        )
        kin = Kinematics()
        kin.lower_limits = model_kin.lower_limits
        kin.upper_limits = model_kin.upper_limits
        runtime = PandaRuntime(kin, Safety(), Solver(), MujocoPlant(config))
        result, trace = comparison_rollout(config, runtime, subdivisions=1)
        self.assertEqual(result["reason"], "timeout")
        self.assertEqual(result["minimum_joint_margin"], 0.0)
        self.assertEqual(result["joint_limit_clipping_steps"], 2)
        self.assertEqual(result["joint_limit_clipping_counts"], [2, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(trace["command"][:, 0], [1.0, 1.0])
        np.testing.assert_allclose(trace["applied_velocity"][:, 0], [0.005, 0.0])
        np.testing.assert_allclose(
            np.diff(trace["q"], axis=0), trace["applied_velocity"] * 0.02
        )
        self.assertLess(result["applied_control_effort"], result["control_effort"])

    def test_newly_unsafe_final_state_is_not_success(self):
        result, trace = self.run_case(Safety(lambda q: 0.01 - q[0]))
        self.assertTrue(result["goal_reached"])
        self.assertFalse(result["success"])
        self.assertLess(trace["barrier_values"][-1, 0], 0)

    def test_comparison_keeps_strict_safety_score_separate_from_configured_tolerance(
        self,
    ):
        result, _ = self.run_case(Safety(lambda q: 1.0 if q[0] == 0 else -4.43e-11))
        self.assertTrue(result["goal_reached"])
        self.assertTrue(result["success"])
        self.assertFalse(result["success_strict"])
        self.assertFalse(result["safety_violation"])
        self.assertTrue(result["safety_violation_strict"])
        self.assertTrue(result["success_barrier_tolerance_1e6"])

    def test_configured_tolerance_scores_whole_path_for_both_methods_and_barrier_types(
        self,
    ):
        for mode in ("direct", "sequential"):
            config = replace(
                self.config,
                target=replace(self.config.target, position=np.array([0.04, 0, 0])),
                trajectory=replace(
                    self.config.trajectory,
                    mode=mode,
                    duration=0.04,
                    success_barrier_tolerance=5e-4,
                ),
            )
            candidate = SafeIKCandidate(
                np.array([0.04, 0, 0, 0, 0, 0, 0]), 0, 1, 0, 0, 1
            )
            for label in ("self_test", "obstacle_test"):
                for minimum, success in (
                    (-0.00049, True),
                    (-0.0005, True),
                    (-0.00051, False),
                ):
                    with self.subTest(mode=mode, label=label, minimum=minimum):
                        result, trace = self.run_case(
                            Safety(
                                lambda q: minimum if 0 < q[0] < 0.04 else 1.0, label
                            ),
                            config,
                            candidate,
                        )
                        self.assertTrue(result["goal_reached"])
                        self.assertEqual(result["success"], success)
                        self.assertEqual(result["interpolated_success"], success)
                        self.assertEqual(result["safety_violation"], not success)
                        self.assertFalse(result["success_strict"])
                        self.assertFalse(result["interpolated_success_strict"])
                        self.assertFalse(result["success_barrier_tolerance_1e6"])
                        self.assertEqual(result["success_barrier_tolerance"], 5e-4)
                        self.assertEqual(result["minimum_barrier"], minimum)
                        self.assertEqual(trace["barrier_values"][1, 0], minimum)
                        self.assertGreater(trace["barrier_values"][-1, 0], 0)

    def test_zero_tolerance_requires_nonnegative_barriers(self):
        config = replace(
            self.config,
            trajectory=replace(self.config.trajectory, success_barrier_tolerance=0),
        )
        result, _ = self.run_case(
            Safety(lambda q: 1.0 if q[0] == 0 else -1e-12), config
        )
        self.assertTrue(result["goal_reached"])
        self.assertFalse(result["success"])
        self.assertFalse(result["interpolated_success"])
        self.assertTrue(result["safety_violation"])

    def test_barrier_tolerance_does_not_relax_goal_or_joint_limits(self):
        config = replace(
            self.config,
            trajectory=replace(self.config.trajectory, success_barrier_tolerance=5e-4),
        )
        unreachable = replace(
            config, target=replace(config.target, position=np.ones(3))
        )
        safety = Safety(lambda q: 1.0 if q[0] == 0 else -0.0004)
        result, _ = self.run_case(safety, unreachable)
        self.assertFalse(result["goal_reached"])
        self.assertFalse(result["success"])
        kin = Kinematics()
        kin.upper_limits = np.full(7, 0.0199)
        runtime = PandaRuntime(kin, safety, Solver(), Plant())
        result, _ = comparison_rollout(config, runtime)
        self.assertTrue(result["goal_reached"])
        self.assertLess(result["minimum_joint_margin"], 0)
        self.assertFalse(result["success"])
        self.assertFalse(result["interpolated_success"])
        self.assertTrue(result["safety_violation"])

    def test_interpolation_applies_same_tolerance_between_recorded_states(self):
        config = replace(
            self.config,
            trajectory=replace(self.config.trajectory, success_barrier_tolerance=5e-4),
        )
        for minimum, success in ((-0.0005, True), (-0.00051, False)):
            with self.subTest(minimum=minimum):
                result, _ = self.run_case(
                    Safety(lambda q: minimum if 0 < q[0] < 0.02 else 1.0), config
                )
                self.assertTrue(result["success"])
                self.assertTrue(result["success_strict"])
                self.assertEqual(result["interpolated_success"], success)
                self.assertFalse(result["interpolated_success_strict"])
                self.assertEqual(result["interpolated_minimum_barrier"], minimum)

    def test_single_candidate_continues_slow_progress_until_goal(self):
        class SlowSolver(Solver):
            def solve(self, problem):
                result = super().solve(problem)
                return replace(result, x=result.x * 1e-4)

        config = replace(
            self.config,
            target=replace(
                self.config.target,
                position=np.array([0.0105, 0, 0]),
                position_tolerance=0.01,
            ),
            trajectory=replace(self.config.trajectory, mode="sequential", duration=6.0),
        )
        candidate = SafeIKCandidate(np.array([0.0105, 0, 0, 0, 0, 0, 0]), 0, 1, 0, 0, 1)
        runtime = PandaRuntime(Kinematics(), Safety(), SlowSolver(), Plant())
        result, _ = comparison_rollout(config, runtime, (candidate,), subdivisions=1)
        self.assertEqual(result["reason"], "goal_reached")
        self.assertTrue(result["success"])
        self.assertGreater(result["elapsed"], config.trajectory.stop_progress_window)

    def test_stalled_candidate_switches_and_records_new_targets(self):
        class StageStallsSolver(Solver):
            def solve(self, problem):
                result = super().solve(problem)
                # No motion for the first intermediate target. Once the second
                # candidate reaches its stages, allow motion to the final pose.
                return replace(result, x=np.zeros(8)) if problem.A[0, 3] else result

        config = replace(
            self.config,
            trajectory=replace(
                self.config.trajectory,
                mode="sequential",
                duration=0.2,
                stop_window=0.04,
                stop_progress_window=0.04,
            ),
        )
        first = SafeIKCandidate(np.array([0.02, 0, 0, 0.5, 0, 0, 0]), 4, 1, 0, 0, 1)
        second = replace(first, q=np.array([0.02, 0, 0, 0, 0, 0, 1]), seed_index=8)
        for candidates in ((first,), (first, second)):
            with self.subTest(candidate_count=len(candidates)):
                runtime = PandaRuntime(
                    Kinematics(), Safety(), StageStallsSolver(), Plant()
                )
                result, trace = comparison_rollout(
                    config, runtime, candidates, subdivisions=1
                )
                self.assertEqual(result["candidate_count"], len(candidates))
                if len(candidates) == 1:
                    self.assertEqual(result["reason"], "stalled")
                    self.assertEqual(result["candidate_switch_count"], 0)
                    np.testing.assert_array_equal(trace["candidate_seed_index"], 4)
                else:
                    self.assertTrue(result["success"])
                    self.assertEqual(result["candidate_switch_count"], 1)
                    self.assertEqual(result["final_candidate_seed_index"], 8)
                    (event,) = result["candidate_transitions"]
                    self.assertAlmostEqual(event["time"], 0.04)
                    self.assertEqual(event["previous_seed_index"], 4)
                    self.assertEqual(event["next_seed_index"], 8)
                    np.testing.assert_array_equal(
                        trace["candidate_index"], [0, 0, 1, 1]
                    )
                    np.testing.assert_array_equal(
                        trace["candidate_seed_index"], [4, 4, 8, 8]
                    )
                    np.testing.assert_allclose(trace["stage_errors"][:2], 0.5)
                    np.testing.assert_allclose(trace["stage_errors"][2:], 0.0)
                    np.testing.assert_array_equal(
                        trace["candidate_q"], [first.q, second.q]
                    )
                    self.assertAlmostEqual(result["elapsed"], 0.06)

    def test_candidate_switches_at_same_stage_are_logged_and_share_time_budget(self):
        class StoppedSolver(Solver):
            def solve(self, problem):
                return replace(super().solve(problem), x=np.zeros(8))

        first = SafeIKCandidate(np.array([0.02, 0, 0, 0.5, 0, 0, 0]), 4, 1, 0, 0, 1)
        candidates = (first, replace(first, q=first.q * 2, seed_index=8))
        for duration, reason, elapsed in (
            (0.06, "timeout", 0.06),
            (0.2, "stalled", 0.08),
        ):
            config = replace(
                self.config,
                trajectory=replace(
                    self.config.trajectory,
                    mode="sequential",
                    duration=duration,
                    stop_window=0.04,
                    stop_progress_window=0.04,
                ),
            )
            runtime = PandaRuntime(Kinematics(), Safety(), StoppedSolver(), Plant())
            result, trace = comparison_rollout(
                config, runtime, candidates, subdivisions=1
            )
            with self.subTest(duration=duration):
                self.assertEqual(result["reason"], reason)
                self.assertAlmostEqual(result["elapsed"], elapsed)
                self.assertEqual(result["candidate_switch_count"], 1)
                self.assertEqual(len(result["candidate_transitions"]), 1)
                self.assertEqual(result["stage_transitions"], [])
                np.testing.assert_allclose(trace["stage_errors"][-1], 1.0)

    def test_direct_ignores_ik_candidates(self):
        candidate = SafeIKCandidate(np.ones(7), 4, 1, 0, 0, 1)
        runtime = PandaRuntime(Kinematics(), Safety(), Solver(), Plant())
        result, trace = comparison_rollout(
            self.config, runtime, (candidate,), subdivisions=1
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["candidate_switch_count"], 0)
        np.testing.assert_array_equal(trace["candidate_index"], [-1, -1])

    def test_sequential_requires_prepared_candidates(self):
        config = replace(
            self.config, trajectory=replace(self.config.trajectory, mode="sequential")
        )
        runtime = PandaRuntime(Kinematics(), Safety(), Solver(), Plant())
        with self.assertRaisesRegex(ValueError, "prepared candidates"):
            comparison_rollout(config, runtime)

    def test_interpolation_detects_crossing_between_safe_samples(self):
        config = replace(
            self.config,
            trajectory=replace(self.config.trajectory, success_barrier_tolerance=0.0),
        )
        result, _ = self.run_case(
            Safety(lambda q: (q[0] - 0.01) ** 2 - 0.00003), config=config
        )
        self.assertTrue(result["success"])
        self.assertFalse(result["interpolated_success"])
        self.assertEqual(result["interpolation_checks"], 4)

    def test_saved_validation_uses_configured_subdivision_count(self):
        config = replace(
            self.config,
            trajectory=replace(self.config.trajectory, success_barrier_tolerance=5e-4),
        )
        runtime = PandaRuntime(
            Kinematics(),
            Safety(lambda q: 1.0 if q[0] == 0 else -0.0005),
            Solver(),
            Plant(),
        )
        result, trace = comparison_rollout(config, runtime, subdivisions=3)
        self.assertTrue(result["success"])
        self.assertEqual(result["interpolation_checks"], 2)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for method in ("direct", "sequential"):
                directory = root / "case" / method
                directory.mkdir(parents=True)
                np.savez_compressed(directory / "trajectory.npz", **trace)
                atomic_json(
                    directory / "summary.json",
                    {
                        **result,
                        "case_id": "case",
                        "method": method,
                        "trace_sha256": digest(directory / "trajectory.npz"),
                    },
                )
            cases = [{"case_id": "case", "initial_q": trace["q"][0].tolist()}]
            validate_completed_run(root, cases, subdivisions=3, barrier_tolerance=5e-4)
            with self.assertRaisesRegex(ValueError, "tolerance mismatch"):
                validate_completed_run(root, cases, subdivisions=3, barrier_tolerance=0)
            with self.assertRaisesRegex(ValueError, "interpolation"):
                validate_completed_run(root, cases, subdivisions=5)

    def test_goal_does_not_require_intermediate_stage_completion(self):
        config = replace(
            self.config, trajectory=replace(self.config.trajectory, mode="sequential")
        )
        candidate = SafeIKCandidate(
            np.array([0.02, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), 0, 1, 0.0, 0.0, 1.0
        )
        result, trace = self.run_case(config=config, candidate=candidate)
        self.assertTrue(result["success"])
        self.assertEqual(trace["stage_index"][0], 0)
        self.assertGreater(
            trace["stage_errors"][-1, 0], config.trajectory.stage_position_tolerance
        )

    def test_failed_qp_is_recorded_without_integrating_dummy_command(self):
        result, trace = self.run_case(status=QPStatus.INFEASIBLE)
        self.assertEqual(result["reason"], "qp_infeasible")
        self.assertFalse(result["success"])
        self.assertEqual(len(trace["q"]), 1)
        self.assertEqual(len(trace["command"]), 1)
        self.assertFalse(trace["command_applied"][0])

    def test_scene_obstacles_do_not_leak_through_shared_self_cache(self):
        shared = Safety()
        first = SceneSafety(shared, Safety(lambda q: 0.2, "obstacle"))
        second = SceneSafety(shared, Safety(lambda q: -0.1, "obstacle"))
        q = np.zeros(7)
        self.assertEqual(first.evaluate(q).minimum, 0.2)
        self.assertEqual(second.evaluate(q).minimum, -0.1)

    def test_ik_failures_stay_in_all_scene_denominator(self):
        rows = []
        for case, ik, direct, sequential in (
            ("a", True, False, True),
            ("b", False, True, False),
        ):
            for method, success in (("direct", direct), ("sequential", sequential)):
                rows.append(
                    dict(
                        case_id=case,
                        method=method,
                        ik_available=ik,
                        success=success,
                        goal_reached=success,
                        reason="goal_reached" if success else "timeout",
                        safety_violation=False,
                        completion_time=1.0 if success else None,
                    )
                )
        result = paired_analysis(rows, 2, 60.0)
        self.assertEqual(result["all_scenes"]["case_count"], 2)
        self.assertEqual(result["ik_conditional"]["case_count"], 1)
        self.assertEqual(result["all_scenes"]["success_rate_difference"], 0.0)
        self.assertEqual(result["ik_conditional"]["success_rate_difference"], 1.0)

    def test_resume_rejects_changed_settings_and_tampered_cases(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cases.jsonl").write_text("{}\n")
            atomic_json(
                root / "manifest.json",
                {
                    "identity": {"seed": 1},
                    "data_sha256": {"cases.jsonl": digest(root / "cases.jsonl")},
                },
            )
            verify_resume(root, {"seed": 1})
            with self.assertRaisesRegex(ValueError, "settings changed"):
                verify_resume(root, {"seed": 2})
            (root / "cases.jsonl").write_text(json.dumps({"changed": True}))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_resume(root, {"seed": 1})


if __name__ == "__main__":
    unittest.main()

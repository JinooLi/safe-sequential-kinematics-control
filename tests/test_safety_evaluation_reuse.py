"""Explicit per-iteration reuse preserves the original IK and soft-min math."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import test_panda_rdf_integration as integration

from safe_ik.application.panda_setup import build_panda_components
from safe_ik.domain.panda_config import SAFE_IK_STEP_SIZE
from safe_ik.domain.panda_rdf_safety import PandaRDFSafety
from safe_ik.domain.safe_pose_ik import IKSeed, SafePoseIKSolver, damped_pseudoinverse
from safe_ik.domain.spatial_safety import BarrierEvaluation, PandaCapsuleSafety


def legacy_solve_seed(solver, seed, require_safe):
    """Pre-reuse call order, retained as an independent update-loop oracle."""
    ik = solver.config.safe_ik
    lower, upper = solver.kinematics.lower_limits, solver.kinematics.upper_limits
    q = np.asarray(seed.q, dtype=float).copy()
    consecutive = 0
    for iteration in range(1, ik.iteration_number + 1):
        position_error, orientation_error, _ = solver._pose_errors(q)
        barriers = solver.safety.evaluate(q)
        candidate = solver._candidate(
            q,
            seed.index,
            iteration,
            require_safe,
            position_error=position_error,
            orientation_error=orientation_error,
            barriers=barriers,
        )
        if candidate is not None:
            consecutive += 1
            if consecutive >= ik.convergence_hold_steps:
                return candidate
        else:
            consecutive = 0
        position_error, orientation_error, jacobian = solver._pose_errors(q)
        task_error = np.concatenate((position_error, 2.0 * orientation_error))
        primary = damped_pseudoinverse(jacobian, ik.damping) @ (
            ik.pose_gain * task_error
        )
        projector = np.eye(solver.config.dof) - np.linalg.pinv(jacobian) @ jacobian
        if require_safe:
            _, gradient, _ = solver.safety.softmin(q, ik.softmin_temperature)
            secondary = projector @ (ik.safety_gradient_gain * gradient)
        else:
            secondary = np.zeros(solver.config.dof)
        q = np.clip(q + SAFE_IK_STEP_SIZE * (primary + secondary), lower, upper)
    return None


class BarrierAggregationTest(unittest.TestCase):
    def test_matches_original_formula_without_mutating_evaluation(self):
        rng = np.random.default_rng(20261001)
        values, gradients = rng.normal(size=12), rng.normal(size=(12, 7))
        evaluation = BarrierEvaluation(tuple(map(str, range(12))), values, gradients)
        for temperature in (0.001, 0.5, 1.0, 10.0):
            minimum = float(np.min(values))
            exponentials = np.exp(-(values - minimum) / temperature)
            weights = exponentials / np.sum(exponentials)
            expected_value = minimum - temperature * np.log(np.sum(exponentials))
            before = (values.copy(), gradients.copy())
            value, gradient = evaluation.softmin(temperature)
            self.assertEqual(value, expected_value)
            np.testing.assert_array_equal(gradient, weights @ gradients)
            np.testing.assert_array_equal(values, before[0])
            np.testing.assert_array_equal(gradients, before[1])

    def test_gradient_matches_finite_difference_and_extreme_values_stay_finite(self):
        values = np.array([-0.01, 0.02, 0.2])
        gradients = np.arange(21, dtype=float).reshape(3, 7) / 100
        evaluation = BarrierEvaluation(("a", "b", "c"), values, gradients)
        _, analytic = evaluation.softmin(0.02)
        numerical = []
        for joint in range(7):
            delta = gradients[:, joint] * 1e-7
            plus = replace(evaluation, values=values + delta).softmin(0.02)[0]
            minus = replace(evaluation, values=values - delta).softmin(0.02)[0]
            numerical.append((plus - minus) / 2e-7)
        np.testing.assert_allclose(analytic, numerical, rtol=0, atol=1e-9)
        extreme = replace(evaluation, values=np.array([-1e6, 0, 1e6]))
        value, gradient = extreme.softmin(1e-6)
        self.assertTrue(np.isfinite(value))
        np.testing.assert_array_equal(gradient, gradients[0])

    def test_empty_singleton_and_nonpositive_temperature(self):
        empty = BarrierEvaluation((), np.empty(0), np.empty((0, 7)))
        value, gradient = empty.softmin(1.0)
        self.assertEqual(value, float("inf"))
        np.testing.assert_array_equal(gradient, np.zeros(7))
        single = BarrierEvaluation(("a",), np.array([-0.02]), np.ones((1, 7)))
        value, gradient = single.softmin(0.2)
        self.assertEqual(value, -0.02)
        np.testing.assert_array_equal(gradient, np.ones(7))
        for evaluation in (empty, single):
            for temperature in (0, -1):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    evaluation.softmin(temperature)


class SafetyReuseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        integration.PandaRDFIntegrationTest.setUpClass()
        cls.fixture = integration.PandaRDFIntegrationTest()
        cls.config = cls.fixture.config

    def components(self, backend):
        kinematics, capsule = build_panda_components(self.config)
        if backend == "capsule":
            return kinematics, capsule
        bundle, options = self.fixture.bundle()
        obstacles = PandaCapsuleSafety(
            kinematics,
            self.config.safety.capsules,
            self.config.obstacles,
            self.config.safety.margin,
            include_self=False,
        )
        return kinematics, PandaRDFSafety(
            kinematics, bundle.models, bundle.queries, bundle.pairs, options, obstacles
        )

    def test_wrapper_and_reuse_match_for_both_backends_without_geometry_calls(self):
        for backend in ("capsule", "rdf"):
            with self.subTest(backend=backend):
                kin, safety = self.components(backend)
                evaluation = safety.evaluate(self.config.initial_q)
                totals = dict(getattr(safety, "total_timings", {}))
                timings = dict(getattr(safety, "last_timings", {}))
                with patch.object(
                    safety, "evaluate", side_effect=AssertionError("re-evaluation")
                ):
                    value, gradient = evaluation.softmin(0.5)
                self.assertEqual(getattr(safety, "total_timings", {}), totals)
                self.assertEqual(getattr(safety, "last_timings", {}), timings)
                old_value, old_gradient, old_evaluation = safety.softmin(
                    self.config.initial_q, 0.5
                )
                self.assertEqual(value, old_value)
                np.testing.assert_array_equal(gradient, old_gradient)
                self.assertEqual(evaluation.labels, old_evaluation.labels)
                np.testing.assert_array_equal(evaluation.values, old_evaluation.values)

    def test_one_evaluation_and_pose_query_per_iteration_with_identical_legacy_path(
        self,
    ):
        for backend in ("capsule", "rdf"):
            for require_safe in (True, False):
                for easy_target in (False, True):
                    with self.subTest(
                        backend=backend, require_safe=require_safe, easy=easy_target
                    ):
                        config = replace(
                            self.config,
                            safe_ik=replace(
                                self.config.safe_ik,
                                iteration_number=8,
                                convergence_hold_duration=0.03,
                                pose_gain=0.2,
                            ),
                        )
                        if easy_target:
                            config = replace(
                                config,
                                target=replace(
                                    config.target,
                                    position_tolerance=10.0,
                                    orientation_tolerance=10.0,
                                ),
                            )
                        results, traces, counts = [], [], []
                        for legacy in (True, False):
                            kin, safety = self.components(backend)
                            solver = SafePoseIKSolver(config, kin, safety)
                            with (
                                patch.object(
                                    solver, "_candidate", wraps=solver._candidate
                                ) as candidate,
                                patch.object(
                                    kin,
                                    "pose_and_jacobian",
                                    wraps=kin.pose_and_jacobian,
                                ) as pose,
                                patch.object(
                                    safety, "evaluate", wraps=safety.evaluate
                                ) as evaluate,
                                patch.object(
                                    safety, "softmin", wraps=safety.softmin
                                ) as softmin,
                            ):
                                seed = IKSeed(0, config.initial_q)
                                result = (
                                    legacy_solve_seed(solver, seed, require_safe)
                                    if legacy
                                    else solver.solve_seed(
                                        seed, require_safe=require_safe
                                    )
                                )
                            results.append(result)
                            traces.append(candidate.call_args_list)
                            counts.append((pose.call_count, evaluate.call_count))
                            if not legacy:
                                self.assertEqual(pose.call_count, candidate.call_count)
                                self.assertEqual(
                                    evaluate.call_count, candidate.call_count
                                )
                                softmin.assert_not_called()
                        self.assertEqual(len(traces[0]), len(traces[1]))
                        for before, after in zip(*traces):
                            np.testing.assert_array_equal(before.args[0], after.args[0])
                            a, b = before.kwargs["barriers"], after.kwargs["barriers"]
                            self.assertEqual(a.labels, b.labels)
                            np.testing.assert_array_equal(a.values, b.values)
                            np.testing.assert_array_equal(a.gradients, b.gradients)
                        before, after = results
                        if before is None:
                            self.assertIsNone(after)
                        else:
                            np.testing.assert_array_equal(before.q, after.q)
                            self.assertEqual(before.iterations, after.iterations)
                            self.assertEqual(
                                before.minimum_barrier, after.minimum_barrier
                            )
                            self.assertEqual(
                                before.position_error, after.position_error
                            )
                            self.assertEqual(
                                before.orientation_error, after.orientation_error
                            )
                        self.assertGreater(counts[0][0], counts[1][0])

    def test_rdf_snapshot_survives_changed_q_and_separate_workers(self):
        _, safety = self.components("rdf")
        _, other = self.components("rdf")
        q = self.config.initial_q.copy()
        before = safety.evaluate(q)
        values, gradients = before.values.copy(), before.gradients.copy()
        expected = before.softmin(0.5)
        q[0] += 0.1
        after = safety.evaluate(q)
        other.evaluate(q + 0.01)
        self.assertIsNot(before, after)
        np.testing.assert_array_equal(before.values, values)
        np.testing.assert_array_equal(before.gradients, gradients)
        self.assertEqual(before.softmin(0.5)[0], expected[0])
        np.testing.assert_array_equal(before.softmin(0.5)[1], expected[1])

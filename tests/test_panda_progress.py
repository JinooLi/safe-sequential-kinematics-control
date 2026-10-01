"""Progress counts completed work without changing numerical IK/control results."""

import io
import unittest
from concurrent.futures import Future
from dataclasses import replace
from queue import Queue
from unittest.mock import Mock

import numpy as np
import test_panda_runtime as runtime_fixture

from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.application.panda_progress import PandaProgress
from safe_ik.application.panda_rollout import run_panda
from safe_ik.application.panda_setup import build_panda_components
from safe_ik.application.parallel_ik import _ProcessIKSearch, panda_ik_search
from safe_ik.domain.safe_pose_ik import (
    IKProgress,
    IKSeed,
    SafeIKCandidate,
    SafePoseIKSolver,
)


class PandaProgressTest(unittest.TestCase):
    def setUp(self):
        self.base = load_panda_config(
            runtime_fixture.ROOT / "tests/fixtures/panda.yaml"
        )

    def test_serial_events_match_actual_iterations_on_convergence_and_exhaustion(self):
        for hold, expected_steps, found in ((0.02, 2, True), (1.0, 5, False)):
            config = replace(
                self.base,
                target=replace(
                    self.base.target,
                    position=self.base.initial_q[:3],
                    quaternion=np.array([1.0, 0, 0, 0]),
                ),
                safe_ik=replace(
                    self.base.safe_ik,
                    iteration_number=5,
                    convergence_hold_duration=hold,
                ),
            )
            solver = SafePoseIKSolver(
                config,
                runtime_fixture.LinearKinematics(),
                runtime_fixture.ConstantSafety(),
            )
            seed, events = IKSeed(0, config.initial_q), []
            expected = solver.solve_seed(seed)
            actual = solver.solve_seed(seed, on_progress=events.append)
            self.assertEqual(
                [e.iteration for e in events], list(range(1, expected_steps + 1))
            )
            self.assertEqual(sum(e.finished for e in events), 1)
            self.assertEqual(events[-1].candidate_found, found)
            if found:
                np.testing.assert_array_equal(actual.q, expected.q)
                self.assertEqual(actual.iterations, expected.iterations)
            else:
                self.assertIsNone(actual)
                self.assertIsNone(expected)

    def test_exception_does_not_report_seed_complete(self):
        safety = Mock()
        safety.evaluate.side_effect = RuntimeError("test calculation error")
        solver = SafePoseIKSolver(self.base, runtime_fixture.LinearKinematics(), safety)
        events = []
        with self.assertRaisesRegex(RuntimeError, "calculation error"):
            solver.solve_seed(IKSeed(0, self.base.initial_q), on_progress=events.append)
        self.assertEqual(events, [])

    def test_real_spawn_progress_preserves_results_and_pool_can_run_diagnostic(self):
        config = replace(
            self.base,
            target=replace(
                self.base.target, position_tolerance=100, orientation_tolerance=100
            ),
            safe_ik=replace(
                self.base.safe_ik,
                workers=2,
                max_restarts=4,
                iteration_number=4,
                convergence_hold_duration=0.02,
                safety_gradient_gain=1e-6,
            ),
        )
        kin, safety = build_panda_components(config)
        position, quaternion, _ = kin.pose_and_jacobian(config.initial_q)
        config = replace(
            config,
            target=replace(config.target, position=position, quaternion=quaternion),
        )
        serial = SafePoseIKSolver(config, kin, safety)
        with panda_ik_search(config, kin, safety) as parallel:
            for require_safe in (True, False):
                expected = serial.solve(require_safe=require_safe)
                self.assertIsNotNone(expected.selected)
                events = []
                actual = parallel.solve(
                    require_safe=require_safe, on_progress=events.append
                )
                self.assertEqual(
                    {e.seed_index for e in events if e.finished}, {0, 1, 2, 3}
                )
                self.assertEqual(
                    actual.selected.seed_index, expected.selected.seed_index
                )
                self.assertEqual(
                    [c.seed_index for c in actual.candidates],
                    [c.seed_index for c in expected.candidates],
                )
                for left, right in zip(actual.candidates, expected.candidates):
                    np.testing.assert_allclose(left.q, right.q, atol=1e-12, rtol=0)
                    self.assertEqual(left.iterations, right.iterations)
            parallel.solve(require_safe=False)  # optional reporting remains optional

    def test_fast_later_seed_is_reported_before_slow_first_seed(self):
        futures = [Future(), Future()]
        executor = Mock()
        executor.submit.side_effect = futures
        local = SafePoseIKSolver(
            self.base,
            runtime_fixture.LinearKinematics(),
            runtime_fixture.ConstantSafety(),
        )
        search = _ProcessIKSearch(local, executor, Queue())
        events = []
        search.on_progress = events.append
        seeds = [IKSeed(i, self.base.initial_q) for i in range(2)]
        candidates = [
            SafeIKCandidate(self.base.initial_q.copy(), i, 3, 0, 0, 1) for i in range(2)
        ]
        metrics = {
            "timings": {},
            "minimum_self_barrier": None,
            "minimum_self_pair": None,
        }
        futures[1].set_result((candidates[1], metrics))
        results = search._map_seeds(seeds, True)
        self.assertIs(next(results), candidates[1])
        self.assertEqual(events[-1].seed_index, 1)
        futures[0].set_result((candidates[0], metrics))
        self.assertIs(next(results), candidates[0])
        self.assertEqual(list(results), [])

    def test_old_phase_telemetry_is_ignored(self):
        queue, events = Queue(), []
        search = _ProcessIKSearch(Mock(), Mock(), queue)
        search.search_id, search.on_progress = 2, events.append
        queue.put((1, IKProgress(0, 999, True)))
        queue.put((2, IKProgress(0, 3)))
        search._drain_progress()
        self.assertEqual(events, [IKProgress(0, 3)])

    def test_tqdm_deduplicates_terminal_events_and_keeps_true_iteration_count(self):
        stream = io.StringIO()
        with PandaProgress(file=stream) as progress:
            progress.start_ik(2, 100, True)
            progress.ik_update(IKProgress(0, 20, True, True))
            progress.ik_update(IKProgress(0, 20, True, True))
            progress.ik_update(IKProgress(0, 1))  # delayed telemetry cannot regress
            self.assertEqual(progress.bar.n, 1)
            self.assertEqual(progress.iterations, {0: 20})
            progress.ik_update(IKProgress(1, 100, True))
            self.assertEqual(progress.bar.n, 2)
            self.assertEqual(sum(progress.iterations.values()), 120)
            progress.finish("candidates found")
        self.assertIsNone(progress.bar)
        self.assertIn("iter=120/200 max", stream.getvalue())
        self.assertIn("candidates found", stream.getvalue())

    def test_tqdm_iteration_refresh_is_not_blocked_by_unchanged_seed_count(self):
        with PandaProgress(file=io.StringIO()) as progress:
            progress.start_ik(2, 100, True)
            progress.bar.refresh = Mock()
            progress.last_refresh = 0
            progress.ik_update(IKProgress(0, 1))
            progress.bar.refresh.assert_called()

    def test_progress_closes_on_error_without_filling_budget(self):
        stream = io.StringIO()
        with self.assertRaises(RuntimeError):
            with PandaProgress(file=stream) as progress:
                progress.start("Trajectory", 100, "step")
                progress.update(2)
                raise RuntimeError("test")
        self.assertIsNone(progress.bar)
        self.assertIn("[error]", stream.getvalue())
        self.assertNotIn("100/100", stream.getvalue())

    def test_trajectory_progress_preserves_trace_and_marks_failure(self):
        fixture = runtime_fixture.PandaRuntimeTest()
        fixture.setUp()
        for status in (
            runtime_fixture.QPStatus.SOLVED,
            runtime_fixture.QPStatus.ITERATION_LIMIT,
        ):
            expected = run_panda(fixture.config, fixture.runtime(status))
            stream = io.StringIO()
            with PandaProgress(file=stream) as progress:
                result = run_panda(
                    fixture.config, fixture.runtime(status), progress=progress
                )
            self.assertEqual(result.steps, expected.steps)
            np.testing.assert_array_equal(result.trace.q, expected.trace.q)
            np.testing.assert_array_equal(result.final_q, expected.final_q)
            self.assertIn("Trajectory", stream.getvalue())
            if status == runtime_fixture.QPStatus.ITERATION_LIMIT:
                self.assertIn("[failed]", stream.getvalue())
                self.assertNotIn("2/2", stream.getvalue())

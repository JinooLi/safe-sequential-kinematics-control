"""Seed scheduling preserves serial IK semantics across real spawned processes."""

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import yaml

from safe_ik.application.panda_configuration import load_panda_config
from safe_ik.application.panda_rollout import run_panda
from safe_ik.application.panda_setup import build_panda_components, build_panda_runtime
from safe_ik.application.parallel_ik import panda_ik_search
from safe_ik.domain.safe_pose_ik import IKSeed, SafeIKCandidate, SafePoseIKSolver

ROOT = Path(__file__).resolve().parents[1]


class ParallelIKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load_panda_config(ROOT / "tests/fixtures/panda.yaml")
        cls.kinematics, cls.safety = build_panda_components(cls.base)

    def short_config(self, **kwargs):
        return replace(
            self.base,
            safe_ik=replace(
                self.base.safe_ik,
                iteration_number=4,
                convergence_hold_duration=0.02,
                max_restarts=4,
                workers=2,
                **kwargs,
            ),
        )

    def assert_same_results(self, left, right):
        self.assertEqual(left.failure_reason, right.failure_reason)
        self.assertEqual(
            [x.seed_index for x in left.candidates],
            [x.seed_index for x in right.candidates],
        )
        self.assertEqual(
            None if left.selected is None else left.selected.seed_index,
            None if right.selected is None else right.selected.seed_index,
        )
        for a, b in zip(left.candidates, right.candidates):
            self.assertEqual(a.iterations, b.iterations)
            np.testing.assert_allclose(a.q, b.q, rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                [a.position_error, a.orientation_error, a.minimum_barrier],
                [b.position_error, b.orientation_error, b.minimum_barrier],
                rtol=0,
                atol=1e-12,
            )

    def test_real_processes_match_serial_safe_and_pose_only_search(self):
        config = self.short_config()
        position, quaternion, _ = self.kinematics.pose_and_jacobian(config.initial_q)
        config = replace(
            config,
            target=replace(
                config.target,
                position=position,
                quaternion=quaternion,
                position_tolerance=100.0,
                orientation_tolerance=100.0,
            ),
            safe_ik=replace(config.safe_ik, safety_gradient_gain=1e-6),
        )
        serial = SafePoseIKSolver(config, self.kinematics, self.safety)
        with panda_ik_search(config, self.kinematics, self.safety) as parallel:
            for require_safe in (True, False):
                with self.subTest(require_safe=require_safe):
                    expected = serial.solve(require_safe=require_safe)
                    actual = parallel.solve(require_safe=require_safe)
                    self.assertGreater(len(expected.candidates), 0)
                    self.assert_same_results(expected, actual)
            # A bad task is a worker error, not a fabricated IK failure result.
            with self.assertRaisesRegex(ValueError, "Panda q must have shape"):
                list(parallel._map_seeds([IKSeed(0, np.zeros(2))], True))
            self.assert_same_results(
                serial.solve(require_safe=False), parallel.solve(require_safe=False)
            )

    def test_seed_generation_and_ties_are_independent_of_completion_order(self):
        config = self.short_config()
        initial = config.initial_q + 0.01
        captured = []

        def reverse_map(seeds, require_safe):
            captured.extend(seeds)
            self.assertTrue(require_safe)
            return [
                SafeIKCandidate(initial.copy(), seed.index, 1, 0.0, 0.0, 0.1)
                for seed in reversed(seeds)
            ]

        result = SafePoseIKSolver(config, self.kinematics, self.safety).solve(
            initial, seed_map=reverse_map
        )
        np.testing.assert_array_equal(captured[0].q, initial)
        rng = np.random.default_rng(config.safe_ik.random_seed)
        for seed in captured[1:]:
            np.testing.assert_array_equal(
                seed.q,
                rng.uniform(self.kinematics.lower_limits, self.kinematics.upper_limits),
            )
        self.assertEqual([c.seed_index for c in result.candidates], [0, 1, 2, 3])
        self.assertEqual(result.selected.seed_index, 0)

    def test_one_worker_or_one_seed_avoids_process_pool(self):
        for workers, seeds in [(1, 4), (8, 1)]:
            config = replace(
                self.base,
                safe_ik=replace(self.base.safe_ik, workers=workers, max_restarts=seeds),
            )
            with patch("safe_ik.application.parallel_ik.ProcessPoolExecutor") as pool:
                with panda_ik_search(config, self.kinematics, self.safety) as search:
                    self.assertIsInstance(search, SafePoseIKSolver)
            pool.assert_not_called()

    def test_worker_count_is_capped_at_seed_count(self):
        config = replace(
            self.base, safe_ik=replace(self.base.safe_ik, workers=20, max_restarts=3)
        )
        with patch("safe_ik.application.parallel_ik.ProcessPoolExecutor") as pool:
            with panda_ik_search(config, self.kinematics, self.safety):
                pass
        self.assertEqual(pool.call_args.kwargs["max_workers"], 3)
        self.assertEqual(
            pool.call_args.kwargs["mp_context"].get_start_method(), "spawn"
        )
        pool.return_value.__exit__.assert_called_once()

    def test_direct_mode_never_enters_search_context(self):
        position, quaternion, _ = self.kinematics.pose_and_jacobian(self.base.initial_q)
        config = replace(
            self.base,
            target=replace(self.base.target, position=position, quaternion=quaternion),
            trajectory=replace(self.base.trajectory, mode="direct"),
            safe_ik=replace(self.base.safe_ik, workers=4),
        )
        runtime = build_panda_runtime(config)
        with patch("safe_ik.application.parallel_ik.ProcessPoolExecutor") as pool:
            result = run_panda(config, runtime)
        self.assertTrue(result.ik_skipped)
        pool.assert_not_called()

    def test_worker_setting_rejects_invalid_values_and_defaults_to_serial(self):
        raw = yaml.safe_load((ROOT / "tests/fixtures/panda.yaml").read_text())
        raw["robot"]["model_path"] = str(self.base.model_path)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for value in (0, -1, 1.5, True, "2"):
                raw["safe_ik"]["workers"] = value
                path.write_text(yaml.safe_dump(raw))
                with (
                    self.subTest(value=value),
                    self.assertRaisesRegex(ValueError, "safe_ik.workers"),
                ):
                    load_panda_config(path)
            raw["safe_ik"].pop("workers")
            path.write_text(yaml.safe_dump(raw))
            self.assertEqual(load_panda_config(path).safe_ik.workers, 1)

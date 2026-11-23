"""Protect context/IK identity and supervised-label alignment in parallel export."""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from safe_ik.application import panda_link_dataset as dataset
from safe_ik.application.artifacts import json_value
from safe_ik.application.panda_experiment_configuration import load_panda_experiment
from safe_ik.application.panda_link_dataset import (
    LINK_ACTIONS,
    LINK_PAIRS,
    export_dataset,
    validate_trace,
)

ROOT = Path(__file__).resolve().parents[1]


class LinkDatasetTests(unittest.TestCase):
    def test_replenishment_continues_after_three_rounds_and_keeps_no_ik_records(self):
        # Production failure: 47/50, then 49/50, then still 49/50.
        contexts = []
        rounds = dataset.replenishment_plan(contexts, 50, 1000, 100)
        for expected, available in [
            ((1000, 50), 47),
            ((1001, 3), 2),
            ((1002, 1), 0),
            ((1003, 1), 1),
        ]:
            self.assertEqual(next(rounds), expected)
            contexts.extend({"ik_available": i < available} for i in range(expected[1]))
        self.assertEqual(list(rounds), [])
        self.assertEqual(sum(c["ik_available"] for c in contexts), 50)
        self.assertEqual(sum(not c["ik_available"] for c in contexts), 5)

    def test_replenishment_stops_at_reserved_seed_boundary(self):
        contexts = []
        rounds = dataset.replenishment_plan(contexts, 1, 1000, 4)
        for seed in range(1000, 1004):
            self.assertEqual(next(rounds), (seed, 1))
            contexts.append({"ik_available": False})
        self.assertEqual(list(rounds), [])
        for limit in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                list(dataset.replenishment_plan([], 1, 1000, limit))

    def test_gpu_sharing_and_pair_tasks_support_more_workers_than_contexts(self):
        visible = ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]
        self.assertEqual(dataset.gpu_assignments(visible, 4), visible * 4)
        self.assertEqual(dataset.gpu_assignments(visible, 1), visible)
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                dataset.gpu_assignments(visible, invalid)
        for invalid in ([], [""], ["0", " 0 "]):
            with self.assertRaises(ValueError):
                dataset.gpu_assignments(invalid, 4)
        contexts = [
            dict(case_id=str(i), ik_available=True, selected_ik={"q": [i] * 7})
            for i in range(10)
        ]
        tasks = dataset.pair_tasks(
            contexts + [dict(case_id="excluded", ik_available=False)]
        )
        self.assertEqual(len(tasks), 160)
        self.assertEqual(len({(c["case_id"], p) for c, p in tasks}), 160)
        for c, _ in tasks:
            self.assertIs(c["selected_ik"], contexts[int(c["case_id"])]["selected_ik"])

    def test_coordinator_collects_all_pairs_with_concurrent_workers(self):
        contexts = [dict(case_id=str(i), ik_available=True) for i in range(10)]

        def finish(task):
            context, pair = task
            return dict(case_id=context["case_id"], pair=list(pair), task_wall_time=2)

        with TemporaryDirectory() as temp:
            output = Path(temp)
            for context in contexts:
                (output / context["case_id"]).mkdir()
            with patch.object(dataset, "evaluate_pair", side_effect=finish):
                with ThreadPoolExecutor(max_workers=16) as pool:
                    rows = dataset.collect_pairs(pool, contexts, output, perf_counter())
            self.assertEqual(len(rows), 160)
            progress = json.loads((output / "progress.json").read_text())
            self.assertEqual(progress["completed_rollouts"], 160)
            self.assertEqual(progress["completed_contexts"], 10)
            for context in contexts:
                directory = output / context["case_id"]
                self.assertEqual(
                    json.loads((directory / "progress.json").read_text()),
                    {"completed_actions": 16},
                )
                self.assertEqual(
                    json.loads((directory / "context.json").read_text())[
                        "summed_action_wall_time"
                    ],
                    32,
                )

    def test_independent_pairs_use_shared_ik_without_search_or_overwrite(self):
        experiment = load_panda_experiment(
            ROOT / "configs/experiments/panda_link_dataset.yaml"
        )
        initial = experiment.config.initial_q.copy()
        selected = dict(
            q=initial.tolist(),
            seed_index=2,
            iterations=5,
            position_error=0.0,
            orientation_error=0.0,
            minimum_barrier=0.1,
        )
        context = dict(
            case_id="test",
            ik_available=True,
            selected_ik=selected,
            initial_q=initial.tolist(),
            ik_seed=4,
            obstacles=[],
            target_position=[0.1, 0.2, 0.3],
            target_quaternion=[1, 0, 0, 0],
        )
        received = []

        def rollout(config, runtime, candidates, *, subdivisions):
            received.append((config, candidates, subdivisions))
            result = dict(
                candidate_count=len(candidates),
                candidate_switch_count=0,
                final_q=initial,
                state_count=2,
                elapsed=0.02,
                reason="stalled",
                success=False,
                goal_reached=False,
                interpolation_checks=subdivisions - 1,
            )
            trace = dict(
                q=np.array([initial, initial]),
                time=np.array([0, 0.02]),
                candidate_q=np.array([c.q for c in candidates]),
                stage_frames=np.array(config.trajectory.stage_frames),
                candidate_index=np.full(2, 0 if candidates else -1, int),
                barrier_values=np.array([[0.01], [0.01]]),
                joint_margin=np.array([0.1, 0.1]),
                goal_reached=np.array([False, False]),
            )
            return result, trace

        with TemporaryDirectory() as temp:
            output = Path(temp)
            directory = output / "test"
            directory.mkdir()
            dataset.atomic_json(directory / "case.json", context)
            dataset.atomic_json(directory / "ik.json", {"selected": selected})
            with (
                patch.object(
                    dataset,
                    "_worker",
                    (experiment, object(), object(), output, "GPU-a"),
                ),
                patch.object(dataset, "prepare_ik") as search,
                patch.object(dataset, "scene_safety"),
                patch.object(dataset, "MujocoPlant"),
                patch.object(dataset, "OSQPSolver"),
                patch.object(dataset, "comparison_rollout", side_effect=rollout),
            ):
                for pair in LINK_ACTIONS:
                    row = dataset.evaluate_pair((context, pair))
                    self.assertEqual(row["candidate_switch_count"], 0)
                    self.assertEqual(row["gpu"], "GPU-a")
                search.assert_not_called()
                with self.assertRaises(FileExistsError):
                    dataset.evaluate_pair((context, LINK_PAIRS[0]))
                dataset.atomic_json(
                    directory / "ik.json", {"selected": dict(selected, seed_index=3)}
                )
                with self.assertRaisesRegex(ValueError, "prepared IK differs"):
                    dataset.evaluate_pair((context, LINK_PAIRS[0]))
            self.assertEqual(len(received), 16)
            for action, (config, candidates, subdivisions) in zip(
                LINK_ACTIONS, received
            ):
                self.assertEqual(
                    config.trajectory.mode, "sequential" if action else "direct"
                )
                self.assertEqual(
                    config.trajectory.stage_frames, tuple(f"link{i}" for i in action)
                )
                self.assertEqual(len(candidates), int(bool(action)))
                if candidates:
                    np.testing.assert_array_equal(candidates[0].q, initial)
                np.testing.assert_array_equal(config.initial_q, initial)
                self.assertEqual(config.trajectory.duration, 120)
                self.assertEqual(subdivisions, 1)

    def test_pilot_conditions(self):
        pilot = load_panda_experiment(
            ROOT / "configs/experiments/panda_link_dataset.yaml"
        )
        previous = load_panda_experiment(
            ROOT / "configs/experiments/panda_comparison.yaml"
        )
        self.assertEqual(pilot.config.trajectory.duration, 120)
        self.assertFalse(pilot.use_all_ik_candidates)
        self.assertTrue(pilot.sampling.randomize_initial_q)
        self.assertEqual(pilot.config.safe_ik.workers, 2)
        self.assertEqual(
            json_value(pilot.config.safety), json_value(previous.config.safety)
        )
        for key in (
            "clf_slack_weight",
            "clf_rate",
            "success_barrier_tolerance",
            "stage_task",
        ):
            self.assertEqual(
                getattr(pilot.config.trajectory, key),
                getattr(previous.config.trajectory, key),
            )
        self.assertEqual(
            LINK_PAIRS,
            (
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
                (3, 4),
                (3, 5),
                (3, 6),
                (4, 5),
                (4, 6),
                (5, 6),
            ),
        )

    def test_reused_ik_is_revalidated_without_search_and_preserves_inputs(self):
        experiment = load_panda_experiment(
            ROOT / "configs/experiments/panda_link_dataset.yaml"
        )
        experiment = replace(
            experiment,
            main=dict(seeds=[1], cases_per_seed=1),
            sampling=replace(experiment.sampling, obstacle_count=1),
        )
        initial = experiment.config.initial_q.tolist()
        selected = dict(
            q=initial,
            seed_index=2,
            iterations=5,
            position_error=0.0,
            orientation_error=0.0,
            minimum_barrier=0.1,
        )
        original = dict(
            case_id="saved-case",
            ik_available=True,
            selected_ik=selected,
            initial_q=initial,
            ik_seed=4,
            ik_wall_time=31.5,
            obstacles=[dict(name="sphere", position=[0.5, 0.5, 0.5], radius=0.1)],
            target_position=[0.1, 0.2, 0.3],
            target_quaternion=[1, 0, 0, 0],
        )
        safe = dict(goal_reached=True, joint_margin=0.2)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            source.write_text(json.dumps(original) + "\n")
            before = source.read_bytes()
            with (
                patch.object(dataset, "prepare_ik") as search,
                patch.object(dataset, "scene_safety") as safety,
                patch.object(dataset, "measure_state") as measure,
            ):
                safety.return_value.evaluate.return_value = SimpleNamespace(minimum=0.1)
                for label, samples, error in (
                    ("safe", [safe, safe], None),
                    (
                        "unsafe-start",
                        [dict(safe, joint_margin=0.05)],
                        "start is not safe",
                    ),
                    (
                        "invalid-ik",
                        [safe, dict(safe, goal_reached=False)],
                        "IK failed current validation",
                    ),
                ):
                    output = root / label
                    output.mkdir()
                    measure.side_effect = samples
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            dataset.reuse_contexts(
                                source, experiment, output, object(), object()
                            )
                        continue
                    contexts, audit = dataset.reuse_contexts(
                        source, experiment, output, object(), object()
                    )
                    self.assertEqual(contexts[0]["selected_ik"], selected)
                    self.assertIsNone(contexts[0]["ik_wall_time"])
                    self.assertEqual(contexts[0]["original_ik_wall_time"], 31.5)
                    self.assertFalse(audit["search_performed"])
                    self.assertEqual(audit["source_sha256"], dataset.digest(source))
                    self.assertEqual(
                        (output / "reused_contexts.jsonl").read_bytes(), before
                    )
                    ik = json.loads((output / "saved-case/ik.json").read_text())
                    self.assertEqual(ik["selected"], selected)
                    self.assertFalse(ik["search_performed"])
                search.assert_not_called()
            self.assertEqual(source.read_bytes(), before)

    def test_export_aligns_shuffled_results_and_masks_numerical_errors(self):
        context = dict(
            case_id="b",
            ik_available=True,
            selected_ik={"q": [1] * 7},
            initial_q=[0] * 7,
            obstacles=[dict(position=[1, 2, 3], radius=0.1)],
            target_position=[0.2, 0.3, 0.4],
            target_quaternion=[1, 0, 0, 0],
        )
        no_ik = dict(context, case_id="a", ik_available=False, selected_ik=None)
        rows = [
            dict(case_id="b", pair=list(p), label_valid=i != 3, success=i % 2 == 0)
            for i, p in enumerate(LINK_ACTIONS)
        ]
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            counts = export_dataset(directory, [context, no_ik], rows[::-1])
            with np.load(directory / "dataset.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["case_id"], ["b"])
                np.testing.assert_array_equal(
                    data["success"], [[1, 0, 1, -1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]
                )
                self.assertFalse(data["label_valid"][0, 3])
                self.assertTrue(data["observed"].all())
                self.assertEqual(data["obstacles"].shape, (1, 1, 4))
            self.assertEqual(counts["no_ik_contexts"], 1)
            self.assertEqual(counts["valid_labels"], 15)
            self.assertEqual(
                len((directory / "contexts.jsonl").read_text().splitlines()), 2
            )
            self.assertEqual(
                len(json.loads((directory / "results.json").read_text())), 16
            )
            with self.assertRaisesRegex(ValueError, "missing, extra or duplicate"):
                export_dataset(directory, [context], rows[:-1])
            with self.assertRaisesRegex(ValueError, "missing, extra or duplicate"):
                export_dataset(directory, [context], rows + rows[:1])
            with self.assertRaisesRegex(ValueError, "unobserved"):
                export_dataset(
                    directory, [context], [dict(rows[0], observed=False), *rows[1:]]
                )
            # An explicit legacy export keeps ten labels; it must never fabricate
            # failure labels for the six actions that were not run historically.
            legacy = directory / "legacy"
            legacy.mkdir()
            legacy_counts = export_dataset(legacy, [context], rows[:10], LINK_PAIRS)
            self.assertEqual(legacy_counts["action_schema_version"], 1)
            with np.load(legacy / "dataset.npz", allow_pickle=False) as data:
                self.assertEqual(data["success"].shape, (1, 10))

    def test_trace_validation_rejects_changed_ik_start_or_success(self):
        config = load_panda_experiment(
            ROOT / "configs/experiments/panda_link_dataset.yaml"
        ).config
        initial = config.initial_q.copy()
        selected = {"q": initial.copy()}
        case = {"initial_q": initial.copy()}
        trace = dict(
            q=np.array([initial, initial]),
            time=np.array([0, 0.02]),
            candidate_q=np.array([initial]),
            stage_frames=np.array(["link2", "link3"]),
            candidate_index=np.zeros(2, int),
            barrier_values=np.array([[0.01], [0.01]]),
            joint_margin=np.array([0.1, 0.1]),
            goal_reached=np.array([False, False]),
        )
        result = dict(
            candidate_count=1,
            candidate_switch_count=0,
            final_q=initial,
            state_count=2,
            elapsed=0.02,
            reason="stalled",
            success=False,
            goal_reached=False,
            interpolation_checks=4,
        )
        validate_trace(config, case, selected, (2, 3), result, trace, 5)
        with self.assertRaisesRegex(ValueError, "success label"):
            validate_trace(
                config, case, selected, (2, 3), dict(result, success=True), trace, 5
            )
        with self.assertRaises(AssertionError):
            validate_trace(config, case, {"q": initial + 0.1}, (2, 3), result, trace, 5)
        with self.assertRaises(AssertionError):
            validate_trace(
                config, {"initial_q": initial + 0.1}, selected, (2, 3), result, trace, 5
            )
        with self.assertRaisesRegex(ValueError, "fixed-IK"):
            validate_trace(
                config,
                case,
                selected,
                (2, 3),
                dict(result, candidate_switch_count=1),
                trace,
                5,
            )


if __name__ == "__main__":
    unittest.main()

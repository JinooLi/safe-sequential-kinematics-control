"""Protect context/IK identity and supervised-label alignment in parallel export."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from safe_ik.application.artifacts import json_value
from safe_ik.application.panda_experiment_configuration import load_panda_experiment
from safe_ik.application.panda_link_dataset import (
    LINK_PAIRS,
    export_dataset,
    validate_trace,
)

ROOT = Path(__file__).resolve().parents[1]


class LinkDatasetTests(unittest.TestCase):
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
            for i, p in enumerate(LINK_PAIRS)
        ]
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            counts = export_dataset(directory, [context, no_ik], rows[::-1])
            with np.load(directory / "dataset.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["case_id"], ["b"])
                np.testing.assert_array_equal(
                    data["success"], [[1, 0, 1, -1, 1, 0, 1, 0, 1, 0]]
                )
                self.assertFalse(data["label_valid"][0, 3])
                self.assertEqual(data["obstacles"].shape, (1, 1, 4))
            self.assertEqual(counts["no_ik_contexts"], 1)
            self.assertEqual(counts["valid_labels"], 9)
            self.assertEqual(
                len((directory / "contexts.jsonl").read_text().splitlines()), 2
            )
            self.assertEqual(
                len(json.loads((directory / "results.json").read_text())), 10
            )
            with self.assertRaisesRegex(ValueError, "missing, extra or duplicate"):
                export_dataset(directory, [context], rows[:-1])
            with self.assertRaisesRegex(ValueError, "missing, extra or duplicate"):
                export_dataset(directory, [context], rows + rows[:1])

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

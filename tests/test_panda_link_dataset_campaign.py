"""Batch boundaries, scientific settings and restart/integrity behavior."""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from safe_ik.application import panda_link_dataset_campaign as campaign
from safe_ik.application.artifacts import json_value
from safe_ik.application.panda_experiment_configuration import load_panda_experiment
from safe_ik.application.panda_link_dataset import LINK_PAIRS, export_dataset

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs/experiments/panda_link_dataset.yaml"


def request(total=3, size=2):
    return dict(
        total_contexts=total,
        batch_size=size,
        seed=2026101500,
        gpus=4,
        cpus=15,
        workers_per_gpu=4,
    )


class FinishedProcess:
    pid = 12345

    def __init__(self, returncode):
        self.returncode = returncode

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def wait(self, timeout=None):
        return self.returncode


def fake_batch(command, **kwargs):
    """Produce real small export files without simulating on a test GPU."""
    directory = Path(command[command.index("--output") + 1])
    experiment = load_panda_experiment(command[command.index("--experiment") + 1])
    directory.mkdir()
    contexts, rows = [], []
    for i in range(experiment.main["cases_per_seed"]):
        case_id = f"seed{experiment.main['seeds'][0]}_{i:04d}"
        contexts.append(
            dict(
                case_id=case_id,
                ik_available=True,
                selected_ik={"q": [1] * 7},
                initial_q=[0] * 7,
                obstacles=[dict(position=[1, 2, 3], radius=0.1)],
                target_position=[0.1, 0.2, 0.3],
                target_quaternion=[1, 0, 0, 0],
            )
        )
        rows.extend(
            dict(case_id=case_id, pair=list(pair), label_valid=True, success=j % 2 == 0)
            for j, pair in enumerate(LINK_PAIRS)
        )
    counts = export_dataset(directory, contexts, rows)
    campaign.atomic_json(
        directory / "manifest.json",
        dict(
            identity=campaign.identity(experiment),
            data_sha256={p.name: campaign.digest(p) for p in directory.iterdir()},
        ),
    )
    campaign.atomic_json(
        directory / "complete.json",
        dict(**counts, manifest_sha256=campaign.digest(directory / "manifest.json")),
    )
    return FinishedProcess(0)


class DatasetCampaignTests(unittest.TestCase):
    def setUp(self):
        # Synthetic child attempts must not inherit the live test allocation ID.
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("SLURM_JOB_ID", None)

    def test_resume_rejects_an_active_previous_allocation(self):
        with patch.object(
            campaign.subprocess, "check_output", return_value="RUNNING\n"
        ):
            with self.assertRaisesRegex(RuntimeError, "still active"):
                campaign.require_previous_job_stopped({"slurm_job_id": "123"})
        with patch.object(campaign.subprocess, "check_output", return_value=""):
            campaign.require_previous_job_stopped({"slurm_job_id": "123"})

    def test_resume_uses_accounting_when_old_job_left_squeue(self):
        import subprocess

        missing = subprocess.CalledProcessError(1, ["squeue"])
        with patch.object(
            campaign.subprocess, "check_output", side_effect=[missing, "COMPLETED|\n"]
        ):
            campaign.require_previous_job_stopped({"slurm_job_id": "123"})
        with patch.object(
            campaign.subprocess, "check_output", side_effect=[missing, "RUNNING|\n"]
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot verify"):
                campaign.require_previous_job_stopped({"slurm_job_id": "123"})

    def test_batch_count_seed_ranges_and_frozen_controller(self):
        plan = campaign.batch_plan(1000, 50, 2026101500)
        self.assertEqual(len(plan), 20)
        self.assertEqual(sum(b["contexts"] for b in plan), 1000)
        ranges = [s for b in plan for s in range(b["seed"], b["seed"] + 3)]
        self.assertEqual(len(set(ranges)), 60)
        self.assertNotIn(20261014, ranges)
        self.assertEqual(
            [b["contexts"] for b in campaign.batch_plan(101, 50, 0)], [50, 50, 1]
        )
        for total, size in [(0, 50), (10, 0), (True, 2)]:
            with self.assertRaises(ValueError):
                campaign.batch_plan(total, size, 0)
        experiment = load_panda_experiment(EXPERIMENT)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = campaign.prepare_campaign(experiment, root, request())
            for batch in metadata["batches"]:
                settings = load_panda_experiment(root / batch["config"])
                self.assertEqual(
                    json_value(settings.config), json_value(experiment.config)
                )
                self.assertEqual(settings.sampling, experiment.sampling)
                self.assertEqual(settings.subdivisions, 1)
                self.assertFalse(settings.use_all_ik_candidates)
            campaign.verify_inputs(metadata, experiment, root)
            (root / metadata["batches"][0]["config"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "campaign input changed"):
                campaign.verify_inputs(metadata, experiment, root)

    def test_complete_export_and_completed_resume_do_not_rerun(self):
        with TemporaryDirectory() as temp:
            output = Path(temp) / "campaign"
            with patch.object(
                campaign.subprocess, "Popen", side_effect=fake_batch
            ) as run:
                campaign.execute_campaign(EXPERIMENT, output, request())
                self.assertEqual(run.call_count, 2)
                before = (output / "complete.json").read_bytes()
                campaign.execute_campaign(EXPERIMENT, output, request(), resume=True)
                self.assertEqual(run.call_count, 2)
                self.assertEqual((output / "complete.json").read_bytes(), before)
                with self.assertRaises(FileExistsError):
                    campaign.execute_campaign(EXPERIMENT, output, request())
            with np.load(output / "dataset.npz") as data:
                self.assertEqual(data["success"].shape, (3, 10))
                self.assertEqual(len(set(data["case_id"])), 3)
            complete = json.loads((output / "complete.json").read_text())
            for name, expected in complete["data_sha256"].items():
                self.assertEqual(campaign.digest(output / name), expected)

    def test_expanded_replenishment_seeds_are_disjoint_and_forwarded(self):
        spec = dict(request(9000, 50), groups=3, ik_replenishment_rounds=100)
        plans = [campaign.campaign_plan(dict(spec, group_index=i)) for i in range(3)]
        seeds = [
            s for plan in plans for b in plan for s in range(b["seed"], b["seed"] + 100)
        ]
        self.assertEqual(len(seeds), 18000)
        self.assertEqual(len(set(seeds)), len(seeds))
        # Historical request files retain their original three-seed stride.
        self.assertEqual(
            campaign.campaign_plan(request())[1]["seed"], request()["seed"] + 3
        )
        with TemporaryDirectory() as temp:
            with patch.object(
                campaign.subprocess, "Popen", side_effect=fake_batch
            ) as run:
                campaign.execute_campaign(
                    EXPERIMENT,
                    Path(temp) / "campaign",
                    dict(request(), ik_replenishment_rounds=100),
                )
            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(
                    command[command.index("--ik-replenishment-rounds") + 1], "100"
                )

    def test_failed_batch_stops_and_resume_preserves_attempt_and_completed_batches(
        self,
    ):
        with TemporaryDirectory() as temp:
            output = Path(temp) / "campaign"

            def failure(command, **kwargs):
                directory = Path(command[command.index("--output") + 1])
                if "batch-001" in directory.name:
                    directory.mkdir()
                    (directory / "partial.txt").write_text("preserve me")
                    return FinishedProcess(1)
                return fake_batch(command, **kwargs)

            with patch.object(campaign.subprocess, "Popen", side_effect=failure) as run:
                with self.assertRaisesRegex(RuntimeError, "no further batches"):
                    campaign.execute_campaign(EXPERIMENT, output, request(5, 2))
                self.assertEqual(run.call_count, 2)
            self.assertFalse((output / "complete.json").exists())
            first = (output / "batch-000-attempt-01/complete.json").read_bytes()
            with patch.object(
                campaign.subprocess, "Popen", side_effect=fake_batch
            ) as run:
                campaign.execute_campaign(
                    EXPERIMENT, output, request(5, 2), resume=True
                )
                self.assertEqual(run.call_count, 2)
            self.assertEqual(
                (output / "batch-000-attempt-01/complete.json").read_bytes(), first
            )
            self.assertEqual(
                (output / "batch-001-attempt-01/partial.txt").read_text(), "preserve me"
            )
            self.assertTrue((output / "batch-001-attempt-02/complete.json").exists())
            self.assertEqual(
                json.loads((output / "complete.json").read_text())["contexts"], 5
            )

    def test_corrupt_completed_data_prevents_resume(self):
        with TemporaryDirectory() as temp:
            output = Path(temp) / "campaign"
            with patch.object(campaign.subprocess, "Popen", side_effect=fake_batch):
                campaign.execute_campaign(EXPERIMENT, output, request())
            (output / "batch-000-attempt-01/results.csv").write_text("corrupt")
            with patch.object(campaign.subprocess, "Popen") as run:
                with self.assertRaisesRegex(ValueError, "batch data hash mismatch"):
                    campaign.execute_campaign(
                        EXPERIMENT, output, request(), resume=True
                    )
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

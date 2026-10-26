"""Scientific parity, iteration-only checks, and bounded/disjoint scheduling."""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import test_rdf_torch_runtime as tensor_fixtures
from test_panda_link_dataset_campaign import EXPERIMENT, fake_batch, request

from safe_ik.adapters.panda_kinematics import PandaKinematics
from safe_ik.application import panda_link_dataset_campaign as campaign
from safe_ik.application import panda_link_dataset_pipeline as pipeline
from safe_ik.application.panda_components import build_panda_components
from safe_ik.application.panda_experiment_configuration import load_panda_experiment
from safe_ik.application.panda_link_dataset import verified_preparation
from safe_ik.application.panda_link_dataset_merge import merge_groups


@unittest.skipUnless(tensor_fixtures.HAS_TORCH, "optional torch")
class ValueBatchTests(unittest.TestCase):
    device = "cpu"

    def fixture(self, **kwargs):
        fixture = tensor_fixtures.TensorParityTest()
        fixture.device = self.device
        return fixture.fixture(**kwargs)

    def test_value_batches_match_full_gradients_and_do_not_request_derivatives(self):
        for gate in (False, True):
            for k in (1, 2, 20, None):
                plain, accelerated, engine = self.fixture(k=k, gate=gate)
                states = [[0.1, 0.13], [0.48, -0.05], [0.95, 0.12], [2.0, 0.3]]
                expected = [plain.evaluate(q) for q in states]
                with patch.object(engine, "_field", wraps=engine._field) as query:
                    actual = accelerated.evaluate_values_batch(states, batch_size=3)
                    self.assertTrue(
                        all(
                            not c.kwargs.get("gradient", False)
                            for c in query.call_args_list
                        )
                    )
                for a, e in zip(actual, expected):
                    self.assertEqual(a.labels, e.labels)
                    np.testing.assert_allclose(a.values, e.values, atol=2e-12, rtol=0)
                cpu = plain.evaluate_values_batch(states, batch_size=2)
                for a, e in zip(cpu, expected):
                    np.testing.assert_allclose(a.values, e.values, atol=2e-12, rtol=0)
                self.assertEqual(accelerated.evaluate_values_batch([]), [])
                # Value checks must not corrupt the gradients needed by the QP.
                for q, e in zip(states, expected):
                    np.testing.assert_allclose(
                        accelerated.evaluate(q).gradients,
                        e.gradients,
                        atol=2e-11,
                        rtol=0,
                    )

    def test_value_backend_rejects_nonfinite_fields(self):
        _, accelerated, engine = self.fixture()
        import torch

        with torch.no_grad():
            engine.weights[:] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            accelerated.evaluate_values_batch([[0.1, 0]])


@unittest.skipUnless(
    tensor_fixtures.HAS_TORCH and tensor_fixtures.torch.cuda.is_available(),
    "CUDA unavailable",
)
class CudaValueBatchTests(ValueBatchTests):
    device = "cuda"

    def test_real_panda_scene_values_match_full_evaluation(self):
        cfg = load_panda_experiment(EXPERIMENT).config
        kin, safety = build_panda_components(cfg)
        states = [
            cfg.initial_q + np.eye(7)[0] * x for x in (-0.04, -0.02, 0, 0.02, 0.04)
        ]
        expected = [safety.evaluate(q) for q in states]
        with patch(
            "safe_ik.adapters.panda_kinematics.pin.computeJointJacobians",
            side_effect=AssertionError("Jacobian requested by value check"),
        ):
            actual = safety.evaluate_values_batch(states)
        for a, e in zip(actual, expected):
            self.assertEqual(a.labels, e.labels)
            np.testing.assert_allclose(a.values, e.values, atol=1e-10, rtol=0)


class PoseOnlyTests(unittest.TestCase):
    def test_pose_cache_can_upgrade_to_jacobians_and_capsule_values_match(self):
        experiment = load_panda_experiment(EXPERIMENT)
        cfg = experiment.config
        kin, safety = build_panda_components(cfg, self_collision_model="capsule")
        reference = PandaKinematics(cfg.model_path, cfg.ee_frame)
        q = cfg.initial_q.copy()
        q[0] += 0.01
        with patch(
            "safe_ik.adapters.panda_kinematics.pin.computeJointJacobians",
            side_effect=AssertionError("value-only FK computed a Jacobian"),
        ):
            values = safety.evaluate_values_batch([q])[0]
        actual = safety.evaluate(q)
        np.testing.assert_allclose(values.values, actual.values, atol=1e-14, rtol=0)
        self.assertEqual(values.labels, actual.labels)
        for state in (q, q + 0.01, q):
            kin.frame_pose(cfg.ee_frame, state)
            actual = kin.pose_and_jacobian(state)
            expected = reference.pose_and_jacobian(state)
            for a, e in zip(actual, expected):
                np.testing.assert_allclose(a, e, atol=1e-14, rtol=0)


class PreparedArtifactTests(unittest.TestCase):
    def test_split_preparation_handoff_hashes_and_one_shot_rollout(self):
        from concurrent.futures import ThreadPoolExecutor
        from contextlib import contextmanager

        from safe_ik.application import panda_link_dataset as dataset

        experiment = load_panda_experiment(EXPERIMENT)
        case = dict(
            case_id="fixture",
            initial_q=experiment.config.initial_q.tolist(),
            obstacles=[],
            target_position=[0.1, 0.2, 0.3],
            target_quaternion=[1, 0, 0, 0],
        )

        @contextmanager
        def pool(*args):
            with ThreadPoolExecutor(max_workers=1) as executor:
                yield executor

        def prepare(case):
            directory = output / case["case_id"]
            directory.mkdir()
            context = dict(
                case,
                ik_available=True,
                selected_ik={"q": case["initial_q"]},
                ik_wall_time=0.01,
            )
            dataset.atomic_json(directory / "case.json", case)
            dataset.atomic_json(
                directory / "ik.json", {"selected": context["selected_ik"]}
            )
            return context

        def collect(pool, contexts, out, start):
            rows = []
            for pair in dataset.LINK_PAIRS:
                directory = out / "fixture"
                run = directory / f"link{pair[0]}_link{pair[1]}"
                run.mkdir()
                dataset.save_npz(
                    run / "trajectory.npz", q=np.array([case["initial_q"]])
                )
                rows.append(
                    dict(
                        case_id="fixture",
                        pair=list(pair),
                        success=False,
                        label_valid=True,
                        case_sha256=dataset.digest(directory / "case.json"),
                        ik_sha256=dataset.digest(directory / "ik.json"),
                        trace_sha256=dataset.digest(run / "trajectory.npz"),
                        rollout_wall_time=0.1,
                        validation_wall_time=0.0,
                        task_wall_time=0.1,
                    )
                )
            return rows

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            import yaml

            experiment.snapshot(root)
            config = experiment.resolved(root)
            config["main"]["cases_per_seed"] = 1
            path = root / "fixture.yaml"
            path.write_text(yaml.safe_dump(config))
            settings = load_panda_experiment(path)
            output = root / "batch"
            with (
                patch.dict(
                    os.environ,
                    {"CUDA_VISIBLE_DEVICES": "0", "SLURM_CPUS_PER_TASK": "4"},
                ),
                patch.object(dataset, "worker_pool", side_effect=pool),
                patch.object(
                    dataset, "build_panda_components", return_value=(object(), object())
                ),
                patch.object(dataset, "prepare_context", side_effect=prepare),
                patch.object(dataset, "generate_cases", return_value=([case], {})),
                patch.object(
                    dataset.subprocess, "check_output", return_value="fixture"
                ),
                patch.object(dataset, "collect_pairs", side_effect=collect),
            ):
                dataset.run(path, output, 1, 4, workers_per_gpu=1, phase="prepare")
                self.assertFalse((output / "complete.json").exists())
                verified_preparation(settings, output)
                ik = output / "fixture/ik.json"
                original = ik.read_bytes()
                ik.write_text("changed")
                with self.assertRaisesRegex(ValueError, "prepared data changed"):
                    dataset.run(path, output, 1, 4, workers_per_gpu=1, phase="rollout")
                ik.write_bytes(original)
                dataset.run(path, output, 1, 4, workers_per_gpu=1, phase="rollout")
                self.assertEqual(
                    json.loads((output / "complete.json").read_text())["rollouts"], 10
                )
                with self.assertRaises((ValueError, FileExistsError)):
                    dataset.run(path, output, 1, 4, workers_per_gpu=1, phase="rollout")


class PipelineTests(unittest.TestCase):
    def test_cpu_budget_and_disjoint_global_seeds(self):
        req = dict(request(1000, 50), cpus=25, prefetch_ik=True, ik_context_workers=2)
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "25"}):
            self.assertEqual(pipeline.validate_resources(req, 2), 25)
            with self.assertRaisesRegex(ValueError, "at least 25"):
                pipeline.validate_resources(dict(req, cpus=15), 2)
        plans = [
            campaign.campaign_plan(dict(req, groups=3, group_index=i)) for i in range(3)
        ]
        flattened = sorted(sum(plans, []), key=lambda b: b["index"])
        self.assertEqual(flattened, campaign.batch_plan(1000, 50, req["seed"]))
        self.assertEqual(
            [sum(b["contexts"] for b in p) for p in plans], [350, 350, 300]
        )

    def run_pipeline(self, output, fail_next=False):
        events, completions, children = [], {}, []

        class Child:
            def __init__(self, code, directory, phase):
                self.pid = 1000000 + len(children)
                self.code, self.directory, self.phase = code, directory, phase
                children.append(self)

            def poll(self):
                if self.phase == "rollout" and not self.code:
                    campaign.atomic_json(
                        self.directory / "complete.json", completions[self.directory]
                    )
                return self.code

            def wait(self, timeout=None):
                return self.code

        def spawn(command, **kwargs):
            self.assertEqual(
                command[command.index("--ik-replenishment-rounds") + 1], "100"
            )
            directory = Path(command[command.index("--output") + 1])
            phase = command[command.index("--phase") + 1]
            events.append((directory.name, phase))
            if phase == "prepare":
                fake_batch(command)
                completions[directory] = json.loads(
                    (directory / "complete.json").read_text()
                )
                (directory / "complete.json").unlink()
                settings = load_panda_experiment(
                    command[command.index("--experiment") + 1]
                )
                campaign.atomic_json(
                    directory / "preparation.json",
                    dict(
                        identity=campaign.identity(settings),
                        manifest_sha256=campaign.digest(directory / "manifest.json"),
                        data_sha256={
                            "contexts.jsonl": campaign.digest(
                                directory / "contexts.jsonl"
                            )
                        },
                    ),
                )
            return Child(
                int(fail_next and "batch-001" in directory.name and phase == "prepare"),
                directory,
                phase,
            )

        req = dict(
            request(5, 2),
            cpus=25,
            prefetch_ik=True,
            ik_context_workers=2,
            ik_replenishment_rounds=100,
        )
        with (
            patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "25"}),
            patch.object(pipeline.subprocess, "Popen", side_effect=spawn),
            patch.object(pipeline.os, "killpg") as kill,
        ):
            if fail_next:
                with self.assertRaisesRegex(RuntimeError, "no further batches"):
                    campaign.execute_campaign(EXPERIMENT, output, req)
                self.assertTrue(kill.called)
                self.assertFalse((output / "complete.json").exists())
                self.assertFalse(any("batch-002" in name for name, _ in events))
            else:
                campaign.execute_campaign(EXPERIMENT, output, req)
                self.assertEqual(
                    json.loads((output / "complete.json").read_text())["contexts"], 5
                )
                self.assertEqual(
                    events[:4],
                    [
                        ("batch-000-attempt-01", "prepare"),
                        ("batch-000-attempt-01", "rollout"),
                        ("batch-001-attempt-01", "prepare"),
                        ("batch-001-attempt-01", "rollout"),
                    ],
                )
                kill.assert_not_called()
        return events

    def test_prefetch_and_failure_stop(self):
        with TemporaryDirectory() as tmp:
            self.run_pipeline(Path(tmp) / "ok")
            self.run_pipeline(Path(tmp) / "failed", fail_next=True)

    def test_merge_rejects_missing_duplicate_and_corrupt_groups(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = [root / f"group-{i}" for i in range(2)]
            for i, directory in enumerate(dirs):
                with patch.object(campaign.subprocess, "Popen", side_effect=fake_batch):
                    campaign.execute_campaign(
                        EXPERIMENT,
                        directory,
                        dict(request(5, 2), groups=2, group_index=i),
                    )
            counts = merge_groups(dirs, root / "merged")
            self.assertEqual(counts["contexts"], 5)
            with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                merge_groups(dirs[:1], root / "missing")
            with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                merge_groups([dirs[0], dirs[0]], root / "duplicate")
            (dirs[0] / "results.csv").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                merge_groups(dirs, root / "corrupt")


if __name__ == "__main__":
    unittest.main()

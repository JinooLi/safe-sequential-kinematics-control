"""Bounded, restartable batches of the existing fixed-IK dataset experiment."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from .artifacts import json_value
from .panda_comparison import atomic_json, digest, fingerprint
from .panda_experiment_configuration import load_panda_experiment
from .panda_link_dataset import LINK_PAIRS, export_dataset, utc_now


def batch_plan(total_contexts, batch_size, seed):
    if any(type(n) is not int or n < 1 for n in (total_contexts, batch_size)):
        raise ValueError("context count and batch size must be positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("base seed must be a nonnegative integer")
    # Each batch may use three consecutive seeds for no-IK replenishment.
    return [
        dict(
            index=i,
            seed=seed + 3 * i,
            contexts=min(batch_size, total_contexts - offset),
        )
        for i, offset in enumerate(range(0, total_contexts, batch_size))
    ]


def identity(experiment):
    return fingerprint(
        experiment.config,
        experiment.config_path,
        experiment.main["seeds"],
        experiment.main["cases_per_seed"],
        experiment=experiment,
    )


def verify_inputs(campaign, experiment, output):
    if identity(experiment) != campaign["identity"]:
        raise ValueError("campaign source/config/model identity changed")
    for name, expected in campaign["input_sha256"].items():
        if digest(output / name) != expected:
            raise ValueError(f"campaign input changed: {name}")


def verified_batch(directory, expected_contexts, expected_identity=None):
    complete = json.loads((directory / "complete.json").read_text())
    if digest(directory / "manifest.json") != complete["manifest_sha256"]:
        raise ValueError(f"batch manifest hash mismatch: {directory}")
    manifest = json.loads((directory / "manifest.json").read_text())
    if expected_identity is not None and manifest["identity"] != expected_identity:
        raise ValueError(f"batch source/config/model identity mismatch: {directory}")
    for name, expected in manifest["data_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"batch data hash mismatch: {directory / name}")
    contexts = [
        json.loads(line)
        for line in (directory / "contexts.jsonl").read_text().splitlines()
    ]
    rows = json.loads((directory / "results.json").read_text())
    eligible = {c["case_id"] for c in contexts if c["ik_available"]}
    expected_keys = {(case_id, pair) for case_id in eligible for pair in LINK_PAIRS}
    keys = {(r["case_id"], tuple(r["pair"])) for r in rows}
    if (
        complete["contexts"] != expected_contexts
        or len(eligible) != expected_contexts
        or complete["rollouts"] != expected_contexts * len(LINK_PAIRS)
        or len(rows) != len(keys)
        or keys != expected_keys
    ):
        raise ValueError(f"batch context/pair count mismatch: {directory}")
    return contexts, rows


def prepare_campaign(experiment, output, request):
    inputs = output / "inputs"
    inputs.mkdir()
    experiment.snapshot(inputs)
    batches = batch_plan(
        request["total_contexts"], request["batch_size"], request["seed"]
    )
    for batch in batches:
        config = experiment.resolved(output)
        config["main"] = dict(seeds=[batch["seed"]], cases_per_seed=batch["contexts"])
        path = inputs / f"batch-{batch['index']:03d}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        # Validate paths/settings now, before reserving many hours of computation.
        settings = load_panda_experiment(path)
        if json_value(settings.config) != json_value(experiment.config):
            raise ValueError("batch configuration changed controller conditions")
        batch.update(
            config=str(path.relative_to(output)), status="pending", attempts=[]
        )
    return dict(
        schema_version=1,
        request=request,
        identity=identity(experiment),
        created_utc=utc_now(),
        status="running",
        batches=batches,
        input_sha256={str(p.relative_to(output)): digest(p) for p in inputs.iterdir()},
    )


def publish_progress(output, campaign):
    completed = [b for b in campaign["batches"] if b["status"] == "completed"]
    active = {}
    if campaign.get("active_output"):
        path = output / campaign["active_output"] / "progress.json"
        if path.exists():
            active = json.loads(path.read_text())
    atomic_json(output / "campaign.json", campaign)
    atomic_json(
        output / "progress.json",
        dict(
            status=campaign["status"],
            total_contexts=campaign["request"]["total_contexts"],
            total_rollouts=campaign["request"]["total_contexts"] * len(LINK_PAIRS),
            completed_batches=len(completed),
            total_batches=len(campaign["batches"]),
            completed_contexts=sum(b["contexts"] for b in completed)
            + active.get("completed_contexts", 0),
            completed_rollouts=sum(b["contexts"] for b in completed) * len(LINK_PAIRS)
            + active.get("completed_rollouts", 0),
            active_batch_progress=active,
            active_output=campaign.get("active_output"),
            updated_utc=utc_now(),
        ),
    )


def require_previous_job_stopped(attempt):
    job_id = attempt.get("slurm_job_id")
    if job_id:
        states = subprocess.check_output(
            ["squeue", "--noheader", "--jobs", str(job_id), "--format=%T"], text=True
        ).strip()
        if states:
            raise RuntimeError(f"previous Slurm job {job_id} is still active: {states}")


def execute_campaign(experiment_path, output, request, *, resume=False):
    experiment = load_panda_experiment(experiment_path)
    batch_plan(request["total_contexts"], request["batch_size"], request["seed"])
    if resume:
        if not (output / "campaign.json").is_file():
            raise ValueError("resume requires an existing campaign.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = (
            json.loads((output / "campaign.json").read_text())
            if resume
            else prepare_campaign(experiment, output, request)
        )
        if campaign["request"] != request:
            raise ValueError("resume must preserve campaign size, seed and resources")
        verify_inputs(campaign, experiment, output)
        if (output / "complete.json").exists():
            complete = json.loads((output / "complete.json").read_text())
            for name, expected in complete["data_sha256"].items():
                if digest(output / name) != expected:
                    raise ValueError(f"completed campaign hash mismatch: {name}")
            for batch in campaign["batches"]:
                directory = output / batch["attempts"][-1]["output"]
                if digest(directory / "complete.json") != batch["complete_sha256"]:
                    raise ValueError("completed batch marker changed")
                verified_batch(
                    directory,
                    batch["contexts"],
                    identity(load_panda_experiment(output / batch["config"])),
                )
            print(
                "Campaign already complete and verified; no work submitted.", flush=True
            )
            return
        campaign["status"] = "running"
        campaign["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
        publish_progress(output, campaign)
        contexts, rows = [], []
        try:
            for batch in campaign["batches"]:
                verify_inputs(campaign, experiment, output)
                expected_identity = identity(
                    load_panda_experiment(output / batch["config"])
                )
                attempt = batch["attempts"][-1] if batch["attempts"] else None
                directory = output / attempt["output"] if attempt else None
                # A complete marker may precede a killed coordinator's status write.
                if directory is not None and (directory / "complete.json").is_file():
                    if (
                        batch.get("complete_sha256")
                        and digest(directory / "complete.json")
                        != batch["complete_sha256"]
                    ):
                        raise ValueError("completed batch marker changed")
                    batch_contexts, batch_rows = verified_batch(
                        directory, batch["contexts"], expected_identity
                    )
                else:
                    if batch["status"] == "completed":
                        raise ValueError("completed batch lost its completion artifact")
                    if attempt:
                        require_previous_job_stopped(attempt)
                        attempt["status"] = "incomplete_preserved"
                    directory = (
                        output
                        / f"batch-{batch['index']:03d}-attempt-{len(batch['attempts']) + 1:02d}"
                    )
                    if directory.exists():
                        raise FileExistsError(directory)
                    command = [
                        sys.executable,
                        "-m",
                        "safe_ik.application.panda_link_dataset",
                        "--experiment",
                        str(output / batch["config"]),
                        "--output",
                        str(directory),
                        "--gpus",
                        str(request["gpus"]),
                        "--cpus",
                        str(request["cpus"]),
                        "--workers-per-gpu",
                        str(request["workers_per_gpu"]),
                    ]
                    attempt = dict(
                        output=str(directory.relative_to(output)),
                        command=command,
                        status="running",
                        started_utc=utc_now(),
                        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    )
                    batch["attempts"].append(attempt)
                    batch["status"] = "running"
                    campaign["active_output"] = attempt["output"]
                    publish_progress(output, campaign)
                    print(
                        f"batch {batch['index'] + 1}/{len(campaign['batches'])}: {directory}",
                        flush=True,
                    )
                    with subprocess.Popen(command, stdin=subprocess.DEVNULL) as process:
                        attempt["pid"] = process.pid
                        publish_progress(output, campaign)
                        while True:
                            try:
                                code = process.wait(timeout=30)
                                break
                            except subprocess.TimeoutExpired:
                                publish_progress(output, campaign)
                    attempt["returncode"] = code
                    if code:
                        attempt["status"] = "failed"
                        raise RuntimeError(
                            f"batch {batch['index']} exited with {code}; no further batches started"
                        )
                    batch_contexts, batch_rows = verified_batch(
                        directory, batch["contexts"], expected_identity
                    )
                # Never replace valid failures based on whether the controller succeeded.
                contexts.extend(batch_contexts)
                rows.extend(batch_rows)
                attempt["status"] = batch["status"] = "completed"
                attempt["verified_utc"] = utc_now()
                batch["complete_sha256"] = digest(directory / "complete.json")
                campaign["active_output"] = None
                publish_progress(output, campaign)
            if len({c["case_id"] for c in contexts}) != len(contexts):
                raise ValueError("duplicate scene identity across batches")
            verify_inputs(campaign, experiment, output)
            counts = export_dataset(output, contexts, rows)
            if counts["contexts"] != request["total_contexts"]:
                raise ValueError("campaign context count mismatch")
            campaign.update(status="completed", finished_utc=utc_now(), counts=counts)
            publish_progress(output, campaign)
            atomic_json(
                output / "complete.json",
                dict(
                    **counts,
                    finished_utc=utc_now(),
                    data_sha256={
                        name: digest(output / name)
                        for name in (
                            "dataset.npz",
                            "contexts.jsonl",
                            "results.json",
                            "results.csv",
                            "campaign.json",
                        )
                    },
                ),
            )
            print(json.dumps(counts), flush=True)
        except BaseException as exc:
            campaign.update(status="failed", error=repr(exc), error_utc=utc_now())
            publish_progress(output, campaign)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-contexts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026101500)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--cpus", type=int, default=15)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify/skip completed batches and retry incomplete batches in new directories",
    )
    args = parser.parse_args()
    request = {
        name: getattr(args, name)
        for name in (
            "total_contexts",
            "batch_size",
            "seed",
            "gpus",
            "cpus",
            "workers_per_gpu",
        )
    }
    execute_campaign(
        args.experiment, args.output.resolve(), request, resume=args.resume
    )


if __name__ == "__main__":
    main()

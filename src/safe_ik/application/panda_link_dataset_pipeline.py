"""Bounded one-batch IK lookahead with separately budgeted CPU workers."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager

from .panda_comparison import digest
from .panda_experiment_configuration import load_panda_experiment
from .panda_link_dataset import utc_now, verified_preparation


def validate_resources(request, ik_workers):
    gpus = request["gpus"]
    prep = request.get("ik_context_workers", 2)
    workers = request["workers_per_gpu"]
    if any(type(n) is not int or n < 1 for n in (gpus, prep, workers)) or prep > gpus:
        raise ValueError("invalid pipeline worker layout")
    # Two coordinators (campaign and child) plus the next batch's sampler,
    # prep workers and nested CPU IK searches, overlapping rollout workers.
    required = 3 + gpus * workers + prep * (1 + ik_workers)
    if request["cpus"] < required:
        raise ValueError(
            f"IK prefetch needs at least {required} CPUs; requested {request['cpus']}"
        )
    if int(os.environ.get("SLURM_CPUS_PER_TASK", request["cpus"])) < request["cpus"]:
        raise ValueError("pipeline CPU request exceeds allocation")
    return required


@contextmanager
def child_processes():
    """Stop children and their worker/IK descendants on failure or cancellation."""
    children = []

    def terminated(signum, frame):
        raise InterruptedError(f"campaign received signal {signum}")

    previous = signal.signal(signal.SIGTERM, terminated)
    try:
        yield children
    finally:
        signal.signal(signal.SIGTERM, previous)
        for process in children:
            # Even a failed coordinator may have surviving pool workers.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in children:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def execute_pipeline(
    experiment,
    output,
    campaign,
    *,
    identity,
    publish_progress,
    require_previous_job_stopped,
    verified_batch,
    verify_inputs,
):
    """Run lookahead using campaign-owned validation and progress callbacks."""
    request = campaign["request"]
    contexts, rows, pending = [], [], []

    def settings(batch):
        return load_panda_experiment(output / batch["config"])

    def accept(batch, directory):
        batch_contexts, batch_rows = verified_batch(
            directory, batch["contexts"], identity(settings(batch))
        )
        contexts.extend(batch_contexts)
        rows.extend(batch_rows)
        batch["status"] = batch["attempts"][-1]["status"] = "completed"
        batch["attempts"][-1]["verified_utc"] = utc_now()
        batch["complete_sha256"] = digest(directory / "complete.json")

    # Audit all existing attempts before starting any new work.
    for batch in campaign["batches"]:
        verify_inputs(campaign, experiment, output)
        attempt = batch["attempts"][-1] if batch["attempts"] else None
        directory = output / attempt["output"] if attempt else None
        if directory and (directory / "complete.json").exists():
            if (
                batch.get("complete_sha256")
                and digest(directory / "complete.json") != batch["complete_sha256"]
            ):
                raise ValueError("completed batch marker changed")
            accept(batch, directory)
        else:
            if batch["status"] == "completed":
                raise ValueError("completed batch lost its completion artifact")
            if attempt:
                require_previous_job_stopped(attempt)
                attempt["status"] = "incomplete_preserved"
            pending.append(batch)

    campaign["active_output"] = campaign["preparing_output"] = None
    with child_processes() as children:

        def launch(batch, phase):
            verify_inputs(campaign, experiment, output)
            if phase == "prepare":
                directory = (
                    output
                    / f"batch-{batch['index']:03d}-attempt-{len(batch['attempts'])+1:02d}"
                )
                if directory.exists():
                    raise FileExistsError(directory)
                attempt = dict(
                    output=str(directory.relative_to(output)),
                    status="preparing",
                    started_utc=utc_now(),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                )
                batch["attempts"].append(attempt)
            else:
                attempt = batch["attempts"][-1]
                directory = output / attempt["output"]
                verified_preparation(settings(batch), directory)
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
                "--ik-context-workers",
                str(request.get("ik_context_workers", 2)),
                "--ik-replenishment-rounds",
                str(request.get("ik_replenishment_rounds", 3)),
                "--phase",
                phase,
            ]
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, start_new_session=True
            )
            children.append(process)
            attempt[f"{phase}_command"] = command
            attempt[f"{phase}_pid"] = process.pid
            attempt[f"{phase}_started_utc"] = utc_now()
            batch["status"] = attempt["status"] = (
                "preparing" if phase == "prepare" else "running"
            )
            campaign["preparing_output" if phase == "prepare" else "active_output"] = (
                attempt["output"]
            )
            publish_progress(output, campaign)
            return process

        def check(process, batch, phase):
            code = process.poll()
            if code is not None:
                attempt = batch["attempts"][-1]
                attempt[f"{phase}_returncode"] = code
                if code:
                    batch["status"] = attempt["status"] = "failed"
                    raise RuntimeError(
                        f"batch {batch['index']} {phase} exited with {code}; no further batches started"
                    )
                if process in children:
                    children.remove(process)
            return code is not None

        def wait(process, batch, phase, other=None):
            last_update = 0.0
            while True:
                # A failed lookahead aborts the pipeline; never quietly replace it.
                if other:
                    check(other[0], other[1], "prepare")
                if check(process, batch, phase):
                    return
                if time.monotonic() - last_update >= 5:
                    publish_progress(output, campaign)
                    last_update = time.monotonic()
                time.sleep(0.25)

        preparation = launch(pending[0], "prepare") if pending else None
        for index, batch in enumerate(pending):
            wait(preparation, batch, "prepare")
            campaign["preparing_output"] = None
            rollout = launch(batch, "rollout")
            next_batch = pending[index + 1] if index + 1 < len(pending) else None
            preparation = launch(next_batch, "prepare") if next_batch else None
            wait(
                rollout,
                batch,
                "rollout",
                (preparation, next_batch) if next_batch else None,
            )
            accept(batch, output / batch["attempts"][-1]["output"])
            campaign["active_output"] = None
            publish_progress(output, campaign)
        campaign["preparing_output"] = None
    return contexts, rows

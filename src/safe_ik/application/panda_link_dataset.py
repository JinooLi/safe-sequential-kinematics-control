"""Fixed-IK Sequential labels with multiple independent workers per GPU.

No controller or IK algorithm is changed here. Each context is evaluated from
the same initial state against the same single validated IK for all ten pairs.
Outputs are exclusive to this invocation; existing directories are refused.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import socket
import subprocess
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from ..adapters.mujoco_plant import MujocoPlant
from ..adapters.osqp_solver import OSQPSolver
from ..domain.ik_contracts import SafeIKCandidate
from ..experiments.panda_comparison import case_config, generate_cases
from .artifacts import json_value
from .panda_comparison import atomic_json, digest, fingerprint, prepare_ik, scene_safety
from .panda_comparison_rollout import comparison_rollout
from .panda_components import build_panda_components
from .panda_experiment_configuration import load_panda_experiment
from .panda_runtime import PandaRuntime
from .paths import project_root

LINK_PAIRS = tuple(combinations(range(2, 7), 2))
_worker = None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        "".join(json.dumps(json_value(r), allow_nan=False) + "\n" for r in rows)
    )
    temporary.replace(path)


def gpu_assignments(visible, workers_per_gpu):
    """Round-robin assignments also support Slurm GPU UUIDs/MIG identifiers."""
    if type(workers_per_gpu) is not int or workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be a positive integer")
    devices = [item.strip() for item in visible]
    if (
        not devices
        or any(not item for item in devices)
        or len(set(devices)) != len(devices)
    ):
        raise ValueError("allocated GPUs must be nonempty and distinct")
    return devices * workers_per_gpu


def initialize_worker(gpu_queue, experiment_path, output, phase):
    global _worker
    gpu = gpu_queue.get(timeout=30)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    threadpool_limits(limits=1)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each worker must see exactly one allocated GPU")
    start = perf_counter()
    experiment = load_panda_experiment(experiment_path)
    kin, safety = build_panda_components(replace(experiment.config, obstacles=()))
    _worker = experiment, kin, safety, Path(output), gpu
    atomic_json(
        Path(output) / f"worker_{os.getpid()}.json",
        dict(
            gpu=gpu,
            gpu_name=torch.cuda.get_device_name(0),
            phase=phase,
            pid=os.getpid(),
            ready_utc=utc_now(),
            setup_wall_time=perf_counter() - start,
        ),
    )


@contextmanager
def worker_pool(visible, workers_per_gpu, experiment_path, output, phase):
    spawn = mp.get_context("spawn")
    assignments = gpu_assignments(visible, workers_per_gpu)
    queue = spawn.Queue()
    for gpu in assignments:
        queue.put(gpu)
    try:
        with ProcessPoolExecutor(
            max_workers=len(assignments),
            mp_context=spawn,
            initializer=initialize_worker,
            initargs=(queue, str(experiment_path), str(output), phase),
        ) as pool:
            yield pool
    finally:
        queue.close()
        queue.join_thread()


def validate_trace(config, case, selected, pair, result, trace, subdivisions):
    """Check actual saved labels/state identity, not just process completion."""
    q = np.asarray(trace["q"])
    time = np.asarray(trace["time"])
    if not np.isfinite(q).all() or not np.isfinite(trace["barrier_values"]).all():
        raise ValueError("nonfinite trajectory")
    np.testing.assert_array_equal(q[0], case["initial_q"])
    np.testing.assert_array_equal(trace["candidate_q"], [selected["q"]])
    np.testing.assert_array_equal(trace["stage_frames"], [f"link{i}" for i in pair])
    np.testing.assert_array_equal(trace["candidate_index"], np.zeros(len(q), int))
    np.testing.assert_array_equal(q[-1], result["final_q"])
    if result["candidate_count"] != 1 or result["candidate_switch_count"] != 0:
        raise ValueError("a fixed-IK label must never switch candidates")
    if len(q) != result["state_count"] or len(time) != len(q):
        raise ValueError("trajectory length mismatch")
    if not np.isclose(time[0], 0) or not np.isclose(time[-1], result["elapsed"]):
        raise ValueError("trajectory time mismatch")
    if not np.allclose(np.diff(time), config.trajectory.integration_period):
        raise ValueError("trajectory control period mismatch")
    if time[-1] > config.trajectory.duration + 1e-7:
        raise ValueError("simulation exceeded its time budget")
    if result["reason"] == "timeout" and not np.isclose(
        time[-1], config.trajectory.duration
    ):
        raise ValueError("early timeout")
    minimum = float(np.min(trace["barrier_values"]))
    joint_minimum = float(np.min(trace["joint_margin"]))
    reached = bool(np.any(trace["goal_reached"]))
    expected = (
        reached
        and minimum >= -config.trajectory.success_barrier_tolerance
        and joint_minimum >= 0
    )
    if result["success"] != expected or result["goal_reached"] != reached:
        raise ValueError("success label disagrees with trajectory")
    if result["interpolation_checks"] != (len(q) - 1) * (subdivisions - 1):
        raise ValueError("incomplete interpolation validation")


def prepare_context(case):
    """Prepare one shared candidate before any pair for this context is queued."""
    experiment, kin, self_safety, output, gpu = _worker
    start = perf_counter()
    started_utc = utc_now()
    directory = output / case["case_id"]
    directory.mkdir()  # Refuse accidental duplicate execution/overwrite.
    atomic_json(directory / "case.json", case)
    cfg = case_config(experiment.config, case)
    print(f"{case['case_id']}: IK start on GPU {gpu}", flush=True)
    ik = prepare_ik(cfg, kin, scene_safety(cfg, kin, self_safety))
    atomic_json(directory / "ik.json", ik)
    context = dict(
        **case,
        selected_ik=ik["selected"],
        ik_available=ik["selected"] is not None,
        ik_reason=ik["reason"],
        ik_wall_time=ik["wall_time"],
        preparation_wall_time=perf_counter() - start,
        preparation_started_utc=started_utc,
        prepared_utc=utc_now(),
        preparation_gpu=gpu,
    )
    atomic_json(directory / "prepared_context.json", context)
    if not context["ik_available"]:
        atomic_json(directory / "context.json", context)
    return context


def pair_tasks(contexts):
    """Schedule pairs independently, including when there are fewer scenes than workers."""
    usable = sorted(
        (c for c in contexts if c["ik_available"]), key=lambda c: c["case_id"]
    )
    return [(context, pair) for pair in LINK_PAIRS for context in usable]


def evaluate_pair(task):
    context, pair = task
    experiment, kin, self_safety, output, gpu = _worker
    if pair not in LINK_PAIRS or not context["ik_available"]:
        raise ValueError("rollout needs a supported pair and a prepared IK")
    run_start, started_utc = perf_counter(), utc_now()
    directory = output / context["case_id"]
    cfg = case_config(experiment.config, context)
    selected = context["selected_ik"]
    # Re-read immutable preparation artifacts: every pair must use the same IK.
    if (
        json_value(selected)
        != json.loads((directory / "ik.json").read_text())["selected"]
    ):
        raise ValueError("prepared IK differs from the shared context artifact")
    candidate = SafeIKCandidate(**{**selected, "q": np.asarray(selected["q"])})
    method_cfg = replace(
        cfg,
        trajectory=replace(
            cfg.trajectory,
            mode="sequential",
            stage_frames=tuple(f"link{i}" for i in pair),
        ),
    )
    run_dir = directory / f"link{pair[0]}_link{pair[1]}"
    run_dir.mkdir()  # Each pair has an exclusive output directory.
    runtime = PandaRuntime(
        kin,
        scene_safety(method_cfg, kin, self_safety),
        OSQPSolver(
            max_iterations=cfg.qp.max_iterations,
            absolute_tolerance=cfg.qp.absolute_tolerance,
            relative_tolerance=cfg.qp.relative_tolerance,
        ),
        MujocoPlant(method_cfg),
    )
    result, trace = comparison_rollout(
        method_cfg, runtime, (candidate,), subdivisions=experiment.subdivisions
    )
    validate_trace(
        method_cfg, context, selected, pair, result, trace, experiment.subdivisions
    )
    save_npz(run_dir / "trajectory.npz", **trace)
    result.update(
        case_id=context["case_id"],
        pair=list(pair),
        method="sequential",
        label_valid=result["reason"] != "qp_solver_failure",
        candidate_policy="one_nearest_no_switch",
        case_sha256=digest(directory / "case.json"),
        ik_sha256=digest(directory / "ik.json"),
        trace_sha256=digest(run_dir / "trajectory.npz"),
        task_wall_time=perf_counter() - run_start,
        started_utc=started_utc,
        finished_utc=utc_now(),
        gpu=gpu,
        worker_pid=os.getpid(),
    )
    atomic_json(run_dir / "summary.json", result)
    print(
        f"{context['case_id']} {pair}: {result['reason']} "
        f"success={result['success']} wall={result['task_wall_time']:.1f}s GPU={gpu}",
        flush=True,
    )
    return result


def collect_pairs(pool, contexts, output, start):
    """Only the coordinator writes shared progress/context metadata."""
    by_id = {c["case_id"]: c for c in contexts}
    if len(by_id) != len(contexts):
        raise ValueError("duplicate context identity")
    tasks = pair_tasks(contexts)
    futures = [pool.submit(evaluate_pair, task) for task in tasks]
    rows, counts = [], Counter()
    for future in as_completed(futures):
        row = future.result()
        rows.append(row)
        case_id = row["case_id"]
        counts[case_id] += 1
        atomic_json(
            output / case_id / "progress.json", dict(completed_pairs=counts[case_id])
        )
        if counts[case_id] == len(LINK_PAIRS):
            context = by_id[case_id]
            context["summed_pair_wall_time"] = sum(
                r["task_wall_time"] for r in rows if r["case_id"] == case_id
            )
            context["finished_utc"] = utc_now()
            atomic_json(output / case_id / "context.json", context)
        atomic_json(
            output / "progress.json",
            dict(
                phase="rollout",
                total_rollouts=len(tasks),
                completed_rollouts=len(rows),
                completed_contexts=sum(n == len(LINK_PAIRS) for n in counts.values()),
                prepared_contexts=sum(c["ik_available"] for c in contexts),
                elapsed_wall_time=perf_counter() - start,
                updated_utc=utc_now(),
            ),
        )
    return rows


def export_dataset(output, contexts, rows):
    """Exclude no-IK contexts; mask numerical failures instead of labeling them 0."""
    contexts = sorted(contexts, key=lambda c: c["case_id"])
    write_jsonl(output / "contexts.jsonl", contexts)
    usable = [c for c in contexts if c["ik_available"]]
    by_key = {(r["case_id"], tuple(r["pair"])): r for r in rows}
    expected_keys = {(c["case_id"], p) for c in usable for p in LINK_PAIRS}
    if len(by_key) != len(rows) or set(by_key) != expected_keys:
        raise ValueError("missing, extra or duplicate context/pair results")
    ordered = [by_key[c["case_id"], p] for c in usable for p in LINK_PAIRS]
    labels = np.full((len(usable), len(LINK_PAIRS)), -1, dtype=np.int8)
    valid = np.zeros_like(labels, dtype=bool)
    for i, c in enumerate(usable):
        for j, p in enumerate(LINK_PAIRS):
            row = by_key[c["case_id"], p]
            valid[i, j] = row["label_valid"]
            if valid[i, j]:
                labels[i, j] = int(row["success"])
    save_npz(
        output / "dataset.npz",
        case_id=np.asarray([c["case_id"] for c in usable]),
        initial_q=np.asarray([c["initial_q"] for c in usable]),
        selected_ik=np.asarray([c["selected_ik"]["q"] for c in usable]),
        obstacles=np.asarray(
            [[[*o["position"], o["radius"]] for o in c["obstacles"]] for c in usable]
        ),
        target_position=np.asarray([c["target_position"] for c in usable]),
        target_quaternion=np.asarray([c["target_quaternion"] for c in usable]),
        link_pairs=np.asarray(LINK_PAIRS, dtype=int),
        success=labels,
        label_valid=valid,
    )
    atomic_json(output / "results.json", ordered)
    columns = (
        "case_id",
        "pair",
        "label_valid",
        "success",
        "goal_reached",
        "reason",
        "elapsed",
        "completion_time",
        "minimum_barrier",
        "minimum_joint_margin",
        "interpolated_success",
        "candidate_switch_count",
        "rollout_wall_time",
        "validation_wall_time",
        "task_wall_time",
        "gpu",
        "worker_pid",
        "started_utc",
        "finished_utc",
    )
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    return dict(
        contexts=len(usable),
        attempted_contexts=len(contexts),
        no_ik_contexts=len(contexts) - len(usable),
        rollouts=len(rows),
        valid_labels=int(valid.sum()),
        success_labels=int((labels == 1).sum()),
        fully_labeled_contexts=int(valid.all(axis=1).sum()),
    )


def run(experiment_path, output, gpu_count, cpu_count, workers_per_gpu=4):
    start = perf_counter()
    started = utc_now()
    experiment = load_panda_experiment(experiment_path)
    if experiment.use_all_ik_candidates or not experiment.sampling.randomize_initial_q:
        raise ValueError("dataset requires random initial q and a single fixed IK")
    if len(experiment.main["seeds"]) != 1:
        raise ValueError(
            "dataset uses one base seed, incremented for replenishment rounds"
        )
    if experiment.config.safety.rdf.runtime_device != "cuda":
        raise ValueError("parallel GPU dataset runner requires rdf_device: cuda")
    # IK subprocesses run only in the preparation phase, never in rollout workers.
    required_cpus = 1 + gpu_count * (1 + experiment.config.safe_ik.workers)
    if gpu_count < 1 or cpu_count < required_cpus:
        raise ValueError(f"need at least {required_cpus} CPUs for this worker layout")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count)) < cpu_count:
        raise ValueError("requested CPU count exceeds Slurm allocation")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != gpu_count or any(not item.strip() for item in visible):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain exactly the allocated GPUs")
    rollout_workers = len(gpu_assignments(visible, workers_per_gpu))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    threadpool_limits(limits=1)
    output.mkdir(parents=True, exist_ok=False)
    root = project_root()
    identity = fingerprint(
        experiment.config,
        experiment.config_path,
        experiment.main["seeds"],
        experiment.main["cases_per_seed"],
        experiment=experiment,
    )
    manifest = dict(
        identity=identity,
        link_pairs=LINK_PAIRS,
        started_utc=started,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        hostname=socket.gethostname(),
        python=platform.python_version(),
        packages={
            name: version(name) for name in ("numpy", "mujoco", "pin", "osqp", "torch")
        },
        gpus=gpu_count,
        cpus=cpu_count,
        workers_per_gpu=workers_per_gpu,
        rollout_workers=rollout_workers,
        ik_context_workers=gpu_count,
        ik_workers_per_context=experiment.config.safe_ik.workers,
        scheduling_unit="context_link_pair",
        rollout_workers_share_allocated_cpus=rollout_workers > cpu_count,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True),
        data_sha256=experiment.snapshot(output),
        sampling_rounds=[],
    )
    with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as archive:
        for relative in identity["source_sha256"]:
            archive.add(root / relative, arcname=relative)
    atomic_json(output / "manifest.json", manifest)
    setup_start = perf_counter()
    kin, self_safety = build_panda_components(replace(experiment.config, obstacles=()))
    setup_wall = perf_counter() - setup_start
    contexts, rows, cases = [], [], []
    sampling_wall = ik_wall = 0.0
    target = experiment.main["cases_per_seed"]
    base_seed = experiment.main["seeds"][0]
    try:
        with worker_pool(visible, 1, experiment.path, output, "preparation") as pool:
            for round_index in range(3):
                deficit = target - sum(c["ik_available"] for c in contexts)
                if not deficit:
                    break
                phase_start = perf_counter()
                batch, sampling = generate_cases(
                    experiment.config,
                    kin,
                    self_safety,
                    [base_seed + round_index],
                    deficit,
                    experiment.sampling,
                )
                sampling_wall += perf_counter() - phase_start
                cases.extend(batch)
                write_jsonl(output / "cases.jsonl", cases)
                manifest["sampling_rounds"].append(
                    dict(
                        seed=base_seed + round_index, count=deficit, statistics=sampling
                    )
                )
                manifest["data_sha256"]["cases.jsonl"] = digest(output / "cases.jsonl")
                atomic_json(output / "manifest.json", manifest)
                phase_start = perf_counter()
                futures = [pool.submit(prepare_context, c) for c in batch]
                for future in as_completed(futures):
                    context = future.result()
                    contexts.append(context)
                    atomic_json(
                        output / "progress.json",
                        dict(
                            phase="preparation",
                            target_contexts=target,
                            prepared_contexts=sum(c["ik_available"] for c in contexts),
                            completed_contexts=0,
                            attempted_contexts=len(contexts),
                            completed_rollouts=len(rows),
                            elapsed_wall_time=perf_counter() - start,
                            updated_utc=utc_now(),
                        ),
                    )
                ik_wall += perf_counter() - phase_start
        write_jsonl(
            output / "contexts.jsonl", sorted(contexts, key=lambda c: c["case_id"])
        )
        if sum(c["ik_available"] for c in contexts) != target:
            raise RuntimeError(
                "IK search replenishment budget exhausted before rollouts"
            )
        phase_start = perf_counter()
        atomic_json(
            output / "progress.json",
            dict(
                phase="rollout",
                total_rollouts=target * len(LINK_PAIRS),
                completed_rollouts=0,
                completed_contexts=0,
                prepared_contexts=target,
                updated_utc=utc_now(),
                elapsed_wall_time=perf_counter() - start,
            ),
        )
        with worker_pool(
            visible, workers_per_gpu, experiment.path, output, "rollout"
        ) as pool:
            rows = collect_pairs(pool, contexts, output, start)
        rollout_wall = perf_counter() - phase_start
        counts = export_dataset(output, contexts, rows)
        if (
            fingerprint(
                experiment.config,
                experiment.config_path,
                experiment.main["seeds"],
                target,
                experiment=experiment,
            )
            != identity
        ):
            raise RuntimeError("source/config/resources changed during generation")
        # Verify every persisted trace and its shared case/IK provenance.
        for row in rows:
            directory = output / row["case_id"]
            run_dir = directory / f"link{row['pair'][0]}_link{row['pair'][1]}"
            for path, expected in (
                (directory / "case.json", row["case_sha256"]),
                (directory / "ik.json", row["ik_sha256"]),
                (run_dir / "trajectory.npz", row["trace_sha256"]),
            ):
                if digest(path) != expected:
                    raise ValueError(f"data hash mismatch: {path}")
        timing = dict(
            started_utc=started,
            finished_utc=utc_now(),
            total_wall_seconds=perf_counter() - start,
            sampler_setup_seconds=setup_wall,
            sampling_seconds=sampling_wall,
            parallel_processing_seconds=ik_wall + rollout_wall,
            parallel_ik_seconds=ik_wall,
            parallel_rollout_seconds=rollout_wall,
            summed_ik_seconds=sum(c["ik_wall_time"] for c in contexts),
            summed_rollout_seconds=sum(r["rollout_wall_time"] for r in rows),
            summed_validation_seconds=sum(r["validation_wall_time"] for r in rows),
            summed_task_seconds=sum(r["task_wall_time"] for r in rows),
        )
        atomic_json(output / "timing.json", timing)
        manifest["data_sha256"].update(
            {
                str(p.relative_to(output)): digest(p)
                for p in output.rglob("*")
                if p.is_file() and p.name not in ("manifest.json", "complete.json")
            }
        )
        atomic_json(output / "manifest.json", manifest)
        atomic_json(
            output / "complete.json",
            dict(
                **counts,
                finished_utc=utc_now(),
                manifest_sha256=digest(output / "manifest.json"),
            ),
        )
        print(json.dumps(dict(**counts, **timing)), flush=True)
    except BaseException as exc:
        atomic_json(
            output / "failed.json",
            dict(
                error=repr(exc), utc=utc_now(), elapsed_wall_time=perf_counter() - start
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--cpus", type=int, default=15)
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=4,
        help="independent rollout processes per allocated GPU (default: 4)",
    )
    args = parser.parse_args()
    run(
        args.experiment,
        args.output.resolve(),
        args.gpus,
        args.cpus,
        args.workers_per_gpu,
    )


if __name__ == "__main__":
    main()

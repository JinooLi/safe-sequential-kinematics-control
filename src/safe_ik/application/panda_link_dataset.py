"""Fixed-IK action labels with multiple independent workers per GPU.

No controller or IK algorithm is changed here. Each context is evaluated from
the same initial state against the same single validated IK for Sequential.
The action catalog also contains one-stage Sequential and IK-free Direct.
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
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from ..adapters.mujoco_plant import MujocoPlant
from ..adapters.osqp_solver import OSQPSolver
from ..domain.ik_contracts import SafeIKCandidate
from ..domain.link_actions import (
    LINK_ACTIONS,
    LINK_PAIRS,
    action_id,
    actions_from_metadata,
    catalog_arrays,
    catalog_metadata,
)
from ..experiments.panda_comparison import case_config, generate_cases, measure_state
from .artifacts import json_value
from .panda_comparison import atomic_json, digest, fingerprint, prepare_ik, scene_safety
from .panda_comparison_rollout import comparison_rollout
from .panda_components import build_panda_components
from .panda_experiment_configuration import load_panda_experiment
from .panda_runtime import PandaRuntime
from .paths import project_root

DEFAULT_IK_REPLENISHMENT_ROUNDS = 100
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


def replenishment_plan(contexts, target, base_seed, round_limit):
    """Recheck the remaining IK deficit after each round; reserve distinct seeds."""
    if type(round_limit) is not int or round_limit < 1:
        raise ValueError("IK replenishment rounds must be a positive integer")
    for round_index in range(round_limit):
        deficit = target - sum(c["ik_available"] for c in contexts)
        if deficit <= 0:
            return
        yield base_seed + round_index, deficit


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
    if pair:
        np.testing.assert_array_equal(trace["candidate_q"], [selected["q"]])
    elif np.asarray(trace["candidate_q"]).size:
        raise ValueError("Direct must not use an IK candidate")
    np.testing.assert_array_equal(trace["stage_frames"], [f"link{i}" for i in pair])
    np.testing.assert_array_equal(
        trace["candidate_index"], np.full(len(q), 0 if pair else -1, int)
    )
    np.testing.assert_array_equal(q[-1], result["final_q"])
    if (
        result["candidate_count"] != int(bool(pair))
        or result["candidate_switch_count"] != 0
    ):
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
    ik = prepare_ik(cfg, kin, scene_safety(cfg, kin, self_safety), values_only=True)
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


def reuse_contexts(source, experiment, output, kin, self_safety):
    """Validate saved scenes/IK with current models without searching or relabeling."""
    source = Path(source).resolve()
    payload = source.read_bytes()
    # Keep the exact imported input separate from new validation/output metadata.
    snapshot = output / "reused_contexts.jsonl"
    with snapshot.open("xb") as stream:
        stream.write(payload)
    source_hash = digest(snapshot)
    if digest(source) != source_hash:
        raise ValueError("reused context source changed while reading")
    imported = [
        json.loads(line) for line in payload.decode().splitlines() if line.strip()
    ]
    if len(imported) != experiment.main["cases_per_seed"]:
        raise ValueError("reused context count must match cases_per_seed")
    ids = [row["case_id"] for row in imported]
    if len(set(ids)) != len(ids) or any(
        not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name
        for name in ids
    ):
        raise ValueError("invalid or duplicate reused context identity")
    contexts = []
    for original in imported:
        if not original["ik_available"] or original.get("selected_ik") is None:
            raise ValueError("reused training context requires one saved IK")
        for name, size in (
            ("initial_q", 7),
            ("target_position", 3),
            ("target_quaternion", 4),
        ):
            value = np.asarray(original[name], dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"invalid reused {name}")
        if not np.isclose(np.linalg.norm(original["target_quaternion"]), 1.0):
            raise ValueError("reused goal quaternion must be normalized")
        if len(original["obstacles"]) != experiment.sampling.obstacle_count:
            raise ValueError("reused obstacle count differs from experiment")
        for obstacle in original["obstacles"]:
            position = np.asarray(obstacle["position"], dtype=float)
            radius = obstacle["radius"]
            if (
                position.shape != (3,)
                or not np.isfinite(position).all()
                or not np.isfinite(radius)
                or radius <= 0
            ):
                raise ValueError("invalid reused obstacle")
        selected = original["selected_ik"]
        candidate = SafeIKCandidate(
            **{**selected, "q": np.asarray(selected["q"], dtype=float)}
        )
        if candidate.q.shape != (7,) or not np.isfinite(candidate.q).all():
            raise ValueError("invalid reused IK configuration")
        started, started_utc = perf_counter(), utc_now()
        cfg = case_config(experiment.config, original)
        safety = scene_safety(cfg, kin, self_safety)
        initial_barrier = safety.evaluate(cfg.initial_q)
        initial = measure_state(cfg, kin, cfg.initial_q, initial_barrier)
        if (
            initial["joint_margin"] < experiment.sampling.joint_limit_margin
            or initial_barrier.minimum < experiment.sampling.minimum_barrier
        ):
            raise ValueError(f"reused start is not safe: {original['case_id']}")
        barrier = safety.evaluate(candidate.q)
        candidate_check = measure_state(cfg, kin, candidate.q, barrier)
        if (
            not candidate_check["goal_reached"]
            or candidate_check["joint_margin"] < 0
            or barrier.minimum < 0
        ):
            raise ValueError(
                f"reused IK failed current validation: {original['case_id']}"
            )
        elapsed = perf_counter() - started
        provenance = dict(source_path=str(source), source_sha256=source_hash)
        context = dict(
            **original,
        )
        context.update(
            ik_wall_time=None,
            ik_search_performed=False,
            ik_reused=True,
            original_ik_wall_time=original.get("ik_wall_time"),
            reused_source=provenance,
            preparation_wall_time=elapsed,
            preparation_started_utc=started_utc,
            prepared_utc=utc_now(),
            preparation_gpu=None,
            reuse_validation=dict(
                initial=initial,
                initial_minimum_barrier=float(initial_barrier.minimum),
                candidate=candidate_check,
                candidate_minimum_barrier=float(barrier.minimum),
                wall_time=elapsed,
            ),
        )
        directory = output / context["case_id"]
        directory.mkdir()
        case = {
            key: original[key]
            for key in (
                "case_id",
                "initial_q",
                "ik_seed",
                "obstacles",
                "target_position",
                "target_quaternion",
            )
        }
        atomic_json(directory / "case.json", case)
        atomic_json(
            directory / "ik.json",
            dict(
                selected=selected,
                accepted_candidates=[selected],
                checks=[
                    dict(
                        seed_index=candidate.seed_index,
                        passed=True,
                        minimum_barrier=float(barrier.minimum),
                    )
                ],
                reason=None,
                wall_time=None,
                search_performed=False,
                reused=True,
                source=provenance,
                validation=context["reuse_validation"],
            ),
        )
        atomic_json(directory / "prepared_context.json", context)
        contexts.append(context)
    return contexts, dict(
        source_path=str(source),
        source_sha256=source_hash,
        contexts=len(contexts),
        search_performed=False,
        validation_wall_time=sum(c["preparation_wall_time"] for c in contexts),
    )


def pair_tasks(contexts, actions=LINK_ACTIONS):
    """Schedule pairs independently, including when there are fewer scenes than workers."""
    usable = sorted(
        (c for c in contexts if c["ik_available"]), key=lambda c: c["case_id"]
    )
    return [(context, pair) for pair in actions for context in usable]


def evaluate_pair(task):
    context, pair = task
    experiment, kin, self_safety, output, gpu = _worker
    if pair not in LINK_ACTIONS or not context["ik_available"]:
        raise ValueError("rollout needs a supported action and a prepared context")
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
    candidates = (
        (SafeIKCandidate(**{**selected, "q": np.asarray(selected["q"])}),)
        if pair
        else ()
    )
    method_cfg = replace(
        cfg,
        trajectory=replace(
            cfg.trajectory,
            mode="sequential" if pair else "direct",
            stage_frames=tuple(f"link{i}" for i in pair),
        ),
    )
    run_dir = directory / action_id(pair)
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
        method_cfg, runtime, candidates, subdivisions=experiment.subdivisions
    )
    validate_trace(
        method_cfg, context, selected, pair, result, trace, experiment.subdivisions
    )
    save_npz(run_dir / "trajectory.npz", **trace)
    result.update(
        case_id=context["case_id"],
        pair=list(pair),
        action_id=action_id(pair),
        method=method_cfg.trajectory.mode,
        label_valid=result["reason"] != "qp_solver_failure",
        candidate_policy="one_nearest_no_switch" if pair else "no_ik",
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


def collect_pairs(pool, contexts, output, start, actions=LINK_ACTIONS):
    """Only the coordinator writes shared progress/context metadata."""
    by_id = {c["case_id"]: c for c in contexts}
    if len(by_id) != len(contexts):
        raise ValueError("duplicate context identity")
    tasks = pair_tasks(contexts, actions)
    futures = [pool.submit(evaluate_pair, task) for task in tasks]
    rows, counts = [], Counter()
    for future in as_completed(futures):
        row = future.result()
        rows.append(row)
        case_id = row["case_id"]
        counts[case_id] += 1
        atomic_json(
            output / case_id / "progress.json", dict(completed_actions=counts[case_id])
        )
        if counts[case_id] == len(actions):
            context = by_id[case_id]
            context["summed_action_wall_time"] = sum(
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
                completed_contexts=sum(n == len(actions) for n in counts.values()),
                prepared_contexts=sum(c["ik_available"] for c in contexts),
                elapsed_wall_time=perf_counter() - start,
                updated_utc=utc_now(),
            ),
        )
    return rows


def export_dataset(output, contexts, rows, actions=LINK_ACTIONS):
    """Exclude no-IK contexts; mask numerical failures instead of labeling them 0."""
    metadata = catalog_metadata(actions)
    contexts = sorted(contexts, key=lambda c: c["case_id"])
    if len({c["case_id"] for c in contexts}) != len(contexts):
        raise ValueError("duplicate context identity")
    usable = [c for c in contexts if c["ik_available"]]
    by_key = {(r["case_id"], tuple(r["pair"])): r for r in rows}
    expected_keys = {(c["case_id"], p) for c in usable for p in actions}
    if len(by_key) != len(rows) or set(by_key) != expected_keys:
        raise ValueError("missing, extra or duplicate context/pair results")
    ordered = [dict(by_key[c["case_id"], p]) for c in usable for p in actions]
    for row in ordered:
        links = tuple(row["pair"])
        if (
            row.get("action_id", action_id(links)) != action_id(links)
            or row.get("method", "sequential" if links else "direct")
            != ("sequential" if links else "direct")
            or row.get("observed", True) is not True
        ):
            raise ValueError("inconsistent or unobserved action outcome")
        row.update(
            action_id=action_id(links), method="sequential" if links else "direct"
        )
    write_jsonl(output / "contexts.jsonl", contexts)
    labels = np.full((len(usable), len(actions)), -1, dtype=np.int8)
    valid = np.zeros_like(labels, dtype=bool)
    for i, c in enumerate(usable):
        for j, p in enumerate(actions):
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
        **catalog_arrays(actions),
        success=labels,
        label_valid=valid,
        observed=np.ones_like(valid),
    )
    atomic_json(output / "results.json", ordered)
    columns = (
        "case_id",
        "pair",
        "action_id",
        "method",
        "label_valid",
        "success",
        "goal_reached",
        "reason",
        "elapsed",
        "completion_time",
        "minimum_barrier",
        "minimum_joint_margin",
        "interpolated_success",
        "safety_validation_mode",
        "interpolation_subdivisions",
        "interpolation_checks",
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
        **metadata,
        contexts=len(usable),
        attempted_contexts=len(contexts),
        no_ik_contexts=len(contexts) - len(usable),
        rollouts=len(rows),
        valid_labels=int(valid.sum()),
        success_labels=int((labels == 1).sum()),
        fully_labeled_contexts=int(valid.all(axis=1).sum()),
    )


def run(
    experiment_path,
    output,
    gpu_count,
    cpu_count,
    workers_per_gpu=4,
    *,
    phase="all",
    ik_context_workers=None,
    ik_replenishment_rounds=DEFAULT_IK_REPLENISHMENT_ROUNDS,
    contexts_file=None,
):
    start = perf_counter()
    started = utc_now()
    if type(ik_replenishment_rounds) is not int or ik_replenishment_rounds < 1:
        raise ValueError("IK replenishment rounds must be a positive integer")
    experiment = load_panda_experiment(experiment_path)
    if experiment.use_all_ik_candidates or not experiment.sampling.randomize_initial_q:
        raise ValueError("dataset requires random initial q and a single fixed IK")
    if len(experiment.main["seeds"]) != 1:
        raise ValueError(
            "dataset uses one base seed, incremented for replenishment rounds"
        )
    if experiment.config.safety.rdf.runtime_device != "cuda":
        raise ValueError("parallel GPU dataset runner requires rdf_device: cuda")
    if phase not in ("all", "prepare", "rollout"):
        raise ValueError("unknown dataset phase")
    if phase == "rollout" and contexts_file is not None:
        raise ValueError("--contexts-file is only used during preparation")
    ik_context_workers = gpu_count if ik_context_workers is None else ik_context_workers
    if not 1 <= ik_context_workers <= gpu_count:
        raise ValueError("IK context workers must be between one and GPU count")
    required_cpus = 1 + ik_context_workers * (1 + experiment.config.safe_ik.workers)
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
    if phase == "rollout":
        return run_prepared_rollouts(experiment, output, visible, workers_per_gpu)
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
        **catalog_metadata(),
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
        ik_context_workers=ik_context_workers if contexts_file is None else 0,
        ik_preparation_mode="search" if contexts_file is None else "reuse_and_validate",
        ik_replenishment_rounds=ik_replenishment_rounds,
        candidate_validation="values_only_batched_float64",
        ik_workers_per_context=experiment.config.safe_ik.workers,
        scheduling_unit="context_action",
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
        if contexts_file is not None:
            contexts, reuse = reuse_contexts(
                contexts_file, experiment, output, kin, self_safety
            )
            cases = [
                json.loads((output / c["case_id"] / "case.json").read_text())
                for c in contexts
            ]
            write_jsonl(output / "cases.jsonl", cases)
            manifest["reused_contexts"] = reuse
            manifest["candidate_validation"] = "reused_current_pose_joint_and_rdf"
            manifest["data_sha256"].update(
                {
                    name: digest(output / name)
                    for name in ("cases.jsonl", "reused_contexts.jsonl")
                }
            )
            atomic_json(output / "manifest.json", manifest)
        else:
            with worker_pool(
                visible[:ik_context_workers], 1, experiment.path, output, "preparation"
            ) as pool:
                for sampling_seed, deficit in replenishment_plan(
                    contexts, target, base_seed, ik_replenishment_rounds
                ):
                    phase_start = perf_counter()
                    batch, sampling = generate_cases(
                        experiment.config,
                        kin,
                        self_safety,
                        [sampling_seed],
                        deficit,
                        experiment.sampling,
                    )
                    sampling_wall += perf_counter() - phase_start
                    cases.extend(batch)
                    write_jsonl(output / "cases.jsonl", cases)
                    manifest["sampling_rounds"].append(
                        dict(seed=sampling_seed, count=deficit, statistics=sampling)
                    )
                    manifest["data_sha256"]["cases.jsonl"] = digest(
                        output / "cases.jsonl"
                    )
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
                                prepared_contexts=sum(
                                    c["ik_available"] for c in contexts
                                ),
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
                f"IK search replenishment budget exhausted after {ik_replenishment_rounds} "
                f"rounds: {sum(c['ik_available'] for c in contexts)}/{target} prepared; "
                "all preparation artifacts preserved"
            )
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
            raise RuntimeError("source/config/resources changed during preparation")
        atomic_json(
            output / "preparation.json",
            dict(
                identity=identity,
                manifest_sha256=digest(output / "manifest.json"),
                elapsed_wall_time=perf_counter() - start,
                sampler_setup_seconds=setup_wall,
                sampling_seconds=sampling_wall,
                parallel_ik_seconds=ik_wall,
                reused_context_validation_seconds=sum(
                    c.get("reuse_validation", {}).get("wall_time", 0.0)
                    for c in contexts
                ),
                data_sha256={
                    str(p.relative_to(output)): digest(p)
                    for p in output.rglob("*")
                    if p.is_file() and p.name not in ("manifest.json", "progress.json")
                },
            ),
        )
        atomic_json(
            output / "progress.json",
            dict(
                phase="prepared",
                prepared_contexts=target,
                completed_contexts=0,
                completed_rollouts=0,
                updated_utc=utc_now(),
            ),
        )
        # Release the sampler's GPU model before the rollout pool starts.
        del kin, self_safety
        if phase == "all":
            return run_prepared_rollouts(experiment, output, visible, workers_per_gpu)

    except BaseException as exc:
        atomic_json(
            output / "failed.json",
            dict(
                error=repr(exc), utc=utc_now(), elapsed_wall_time=perf_counter() - start
            ),
        )
        raise


def verified_preparation(experiment, output):
    prepared = json.loads((output / "preparation.json").read_text())
    expected = fingerprint(
        experiment.config,
        experiment.config_path,
        experiment.main["seeds"],
        experiment.main["cases_per_seed"],
        experiment=experiment,
    )
    if prepared["identity"] != expected:
        raise ValueError("prepared source/config/model identity changed")
    if digest(output / "manifest.json") != prepared["manifest_sha256"]:
        raise ValueError("prepared manifest changed")
    for name, value in prepared["data_sha256"].items():
        if digest(output / name) != value:
            raise ValueError(f"prepared data changed: {name}")
    contexts = [
        json.loads(line)
        for line in (output / "contexts.jsonl").read_text().splitlines()
    ]
    if (
        len({c["case_id"] for c in contexts}) != len(contexts)
        or sum(c["ik_available"] for c in contexts) != experiment.main["cases_per_seed"]
    ):
        raise ValueError("prepared context count/identity mismatch")
    return prepared, contexts


def run_prepared_rollouts(experiment, output, visible, workers_per_gpu):
    prepared, contexts = verified_preparation(experiment, output)
    manifest = json.loads((output / "manifest.json").read_text())
    actions = actions_from_metadata(manifest)
    if actions != LINK_ACTIONS:
        raise ValueError(
            "historical action catalogs cannot be resumed with current code"
        )
    if (
        manifest["gpus"] != len(visible)
        or manifest["workers_per_gpu"] != workers_per_gpu
    ):
        raise ValueError("rollout resources differ from prepared manifest")
    # One-shot claim: interrupted attempts are preserved, never overwritten.
    with (output / ".rollout.claim").open("x") as claim:
        claim.write(str(os.getpid()))
    identity = prepared["identity"]
    started = manifest["started_utc"]
    start = (
        perf_counter()
        - (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds()
    )
    setup_wall = prepared["sampler_setup_seconds"]
    sampling_wall = prepared["sampling_seconds"]
    ik_wall = prepared["parallel_ik_seconds"]
    target = experiment.main["cases_per_seed"]
    try:
        phase_start = perf_counter()
        atomic_json(
            output / "progress.json",
            dict(
                phase="rollout",
                total_rollouts=target * len(actions),
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
            rows = collect_pairs(pool, contexts, output, start, actions)
        rollout_wall = perf_counter() - phase_start
        counts = export_dataset(output, contexts, rows, actions)
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
            run_dir = directory / action_id(row["pair"])
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
            summed_ik_seconds=sum(c["ik_wall_time"] or 0 for c in contexts),
            ik_search_contexts=sum(
                c.get("ik_search_performed", True) for c in contexts
            ),
            reused_contexts=sum(c.get("ik_reused", False) for c in contexts),
            reused_context_validation_seconds=prepared.get(
                "reused_context_validation_seconds", 0.0
            ),
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
                data_sha256=manifest["data_sha256"],
            ),
        )
        print(json.dumps(dict(**counts, **timing)), flush=True)
    except BaseException as exc:
        atomic_json(output / "failed.json", dict(error=repr(exc), utc=utc_now()))
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
    parser.add_argument("--phase", choices=("all", "prepare", "rollout"), default="all")
    parser.add_argument("--ik-context-workers", type=int)
    parser.add_argument(
        "--contexts-file",
        type=Path,
        help="reuse exact saved JSONL scenes and selected IK; validate without an IK search",
    )
    parser.add_argument(
        "--ik-replenishment-rounds",
        type=int,
        default=DEFAULT_IK_REPLENISHMENT_ROUNDS,
        help="maximum sampling rounds to replace no-IK scenes (default: 100)",
    )
    args = parser.parse_args()
    run(
        args.experiment,
        args.output.resolve(),
        args.gpus,
        args.cpus,
        args.workers_per_gpu,
        phase=args.phase,
        ik_context_workers=args.ik_context_workers,
        ik_replenishment_rounds=args.ik_replenishment_rounds,
        contexts_file=args.contexts_file,
    )


if __name__ == "__main__":
    main()

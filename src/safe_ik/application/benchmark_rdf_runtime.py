"""Read-only CPU/CUDA RDF parity and synchronized end-to-end latency benchmark."""

import argparse
import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .panda_configuration import load_panda_config
from .panda_rdf_training import validate_joint_gradients
from .panda_setup import build_panda_components


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/panda.yaml")
    parser.add_argument(
        "--samples",
        help="optional NPZ with q: (N,7); otherwise 64 deterministic joint-limit samples",
    )
    parser.add_argument(
        "--output-dir", required=True, help="new directory; refuses overwriting"
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    config = load_panda_config(args.config)
    before = digest(config.safety.rdf.artifact)
    evaluators, initialization = {}, {}
    with threadpool_limits(limits=1):
        for device in ("cpu", "cuda"):
            configured = replace(
                config,
                safety=replace(
                    config.safety,
                    self_collision_model="rdf",
                    rdf=replace(config.safety.rdf, runtime_device=device),
                ),
            )
            start = perf_counter()
            kin, evaluators[device] = build_panda_components(configured)
            initialization[device] = perf_counter() - start
        if args.samples:
            with np.load(args.samples, allow_pickle=False) as data:
                q = data["q"].copy()
        else:
            q = np.random.default_rng(20261001).uniform(
                kin.lower_limits, kin.upper_limits, (64, 7)
            )
        if q.ndim != 2 or q.shape[1] != 7 or not len(q) or not np.isfinite(q).all():
            raise ValueError("samples must be finite (N,7)")
        import torch

        warmup = {}
        for device, evaluator in evaluators.items():
            start = perf_counter()
            for pose in q[:5]:
                evaluator._q = None
                evaluator.kinematics._cached_q = None
                evaluator.evaluate(pose)
            warmup[device] = perf_counter() - start
        torch.cuda.reset_peak_memory_stats()
        timing = {device: [] for device in evaluators}
        value_error = gradient_error = 0.0
        sign_changes = 0
        counts = []
        for repeat in range(args.repeats):
            for i, pose in enumerate(q):
                result = {}
                for device in (
                    ("cpu", "cuda") if (i + repeat) % 2 else ("cuda", "cpu")
                ):
                    evaluator = evaluators[device]
                    evaluator._q = None
                    evaluator.kinematics._cached_q = None
                    start = perf_counter()
                    result[device] = evaluator.evaluate(pose)
                    timing[device].append(perf_counter() - start)
                cpu, cuda = result["cpu"], result["cuda"]
                assert cpu.labels == cuda.labels
                value_error = max(
                    value_error, float(np.max(np.abs(cpu.values - cuda.values)))
                )
                gradient_error = max(
                    gradient_error,
                    float(np.max(np.abs(cpu.gradients - cuda.gradients))),
                )
                sign_changes += int(
                    np.count_nonzero((cpu.values < 0) != (cuda.values < 0))
                )
                if repeat == 0:
                    counts.append(dict(evaluators["cuda"].last_gate_counts))
        gpu = evaluators["cuda"]
        joint = validate_joint_gradients(
            gpu.kinematics,
            SimpleNamespace(models=gpu.models, queries=gpu.queries, pairs=gpu.pairs),
            gpu.options,
            np.random.default_rng(20261001),
        )
        report = {
            "scope": "offline pose replay; no training/model writes or trajectory simulation",
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "threadpools": threadpool_info(),
                "cuda": gpu.runtime_metadata,
            },
            "poses": len(q),
            "repeats": args.repeats,
            "samples_sha256": digest(args.samples) if args.samples else None,
            "model_sha256": before,
            "model_unchanged": before == digest(config.safety.rdf.artifact),
            "initialization_seconds": initialization,
            "warmup_seconds": warmup,
            "gate_counts": counts,
            "latency": {
                device: {
                    "mean_ms": float(np.mean(t) * 1000),
                    "p95_ms": float(np.percentile(t, 95) * 1000),
                }
                for device, t in timing.items()
            },
            "max_value_difference_m": value_error,
            "max_gradient_difference": gradient_error,
            "barrier_sign_changes": sign_changes,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "joint_validation": joint,
        }
        report["passed"] = bool(
            value_error < 1e-9
            and gradient_error < 1e-8
            and sign_changes == 0
            and joint["passed"]
            and report["model_unchanged"]
        )
        (out / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        np.savez_compressed(
            out / "samples.npz",
            q=q,
            **{device + "_seconds": t for device, t in timing.items()},
        )
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "latency",
                        "max_value_difference_m",
                        "max_gradient_difference",
                        "barrier_sign_changes",
                        "cuda_peak_allocated_bytes",
                        "passed",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
        if not report["passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()

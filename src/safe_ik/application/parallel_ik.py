"""Application-owned process pool for independent Panda IK seed searches."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

from ..domain.interfaces import PoseKinematics
from ..domain.panda_config import PandaControllerConfig
from ..domain.safe_pose_ik import (
    IKSeed,
    SafeIKCandidate,
    SafeIKResult,
    SafePoseIKSolver,
)
from ..domain.spatial_safety import SafetyEvaluator

_worker_solver: SafePoseIKSolver | None = None


def _initialize_worker(config: PandaControllerConfig) -> None:
    # Construct private Pinocchio Data/cache and safety objects once per process.
    # Only the configuration is serialized; no live model, solver or viewer crosses.
    from .panda_setup import build_panda_components

    global _worker_solver
    kinematics, safety = build_panda_components(config)
    _worker_solver = SafePoseIKSolver(config, kinematics, safety)


def _solve_seed(task: tuple[IKSeed, bool]) -> SafeIKCandidate | None:
    if _worker_solver is None:
        raise RuntimeError("IK worker has not been initialized")
    seed, require_safe = task
    return _worker_solver.solve_seed(seed, require_safe=require_safe)


def _solve_seed_with_metrics(task):
    before = dict(getattr(_worker_solver.safety, "total_timings", {}))
    candidate = _solve_seed(task)
    safety = _worker_solver.safety
    return candidate, {
        "timings": {
            key: value - before.get(key, 0.0)
            for key, value in getattr(safety, "total_timings", {}).items()
        },
        "minimum_self_barrier": getattr(safety, "minimum_self_barrier", None),
        "minimum_self_pair": getattr(safety, "minimum_self_pair", None),
    }


class _ProcessIKSearch:
    def __init__(self, local: SafePoseIKSolver, executor: ProcessPoolExecutor):
        self.local = local
        self.executor = executor
        self.rdf_metrics = {
            "timings": {},
            "minimum_self_barrier": None,
            "minimum_self_pair": None,
        }

    def _map_seeds(self, seeds: list[IKSeed], require_safe: bool):
        # map retains input order; each worker takes another seed when available.
        for candidate, metrics in self.executor.map(
            _solve_seed_with_metrics, ((seed, require_safe) for seed in seeds)
        ):
            for key, value in metrics["timings"].items():
                self.rdf_metrics["timings"][key] = (
                    self.rdf_metrics["timings"].get(key, 0.0) + value
                )
            minimum = metrics["minimum_self_barrier"]
            previous = self.rdf_metrics["minimum_self_barrier"]
            if minimum is not None and (previous is None or minimum < previous):
                self.rdf_metrics["minimum_self_barrier"] = minimum
                self.rdf_metrics["minimum_self_pair"] = metrics["minimum_self_pair"]
            yield candidate

    def solve(self, initial_q=None, *, require_safe: bool = True) -> SafeIKResult:
        return self.local.solve(
            initial_q, require_safe=require_safe, seed_map=self._map_seeds
        )


@contextmanager
def panda_ik_search(
    config: PandaControllerConfig,
    kinematics: PoseKinematics,
    safety: SafetyEvaluator,
):
    """Reuse one pool for safe search and optional pose-only diagnosis, then close it.

    Spawn avoids inheriting native library/viewer thread state. Worker exceptions
    propagate to the caller rather than being misreported as an unreachable pose.
    """
    workers = min(config.safe_ik.workers, config.safe_ik.max_restarts)
    local = SafePoseIKSolver(config, kinematics, safety)
    if workers == 1:
        yield local
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(config,),
    ) as executor:
        yield _ProcessIKSearch(local, executor)

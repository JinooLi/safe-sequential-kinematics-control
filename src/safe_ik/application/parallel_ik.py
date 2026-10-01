"""Application-owned process pool for independent Panda IK seed searches."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from queue import Empty
from time import monotonic

from ..domain.interfaces import PoseKinematics
from ..domain.panda_config import PandaControllerConfig
from ..domain.safe_pose_ik import (
    IKProgress,
    IKProgressCallback,
    IKSeed,
    SafeIKCandidate,
    SafeIKResult,
    SafePoseIKSolver,
)
from ..domain.spatial_safety import SafetyEvaluator

_worker_solver: SafePoseIKSolver | None = None
_worker_progress_queue = None


def _initialize_worker(config: PandaControllerConfig, progress_queue=None) -> None:
    # Construct private Pinocchio Data/cache and safety objects once per process.
    # Only the configuration is serialized; no live model, solver or viewer crosses.
    from .panda_setup import build_panda_components

    global _worker_solver, _worker_progress_queue
    _worker_progress_queue = progress_queue
    if progress_queue is not None:
        # Telemetry must not hold up worker exit if a consumer fails. Final
        # completion is also reported authoritatively from each task's Future.
        progress_queue.cancel_join_thread()
    kinematics, safety = build_panda_components(
        config, self_collision_model=config.ik_self_collision_model
    )
    _worker_solver = SafePoseIKSolver(config, kinematics, safety)


def _solve_seed(task: tuple[IKSeed, bool], on_progress=None) -> SafeIKCandidate | None:
    if _worker_solver is None:
        raise RuntimeError("IK worker has not been initialized")
    seed, require_safe = task
    return _worker_solver.solve_seed(
        seed, require_safe=require_safe, on_progress=on_progress
    )


def _solve_seed_with_metrics(task):
    seed, require_safe, search_id, report_progress = task
    last_report = 0.0

    def report(event):
        nonlocal last_report
        now = monotonic()
        if event.finished or now - last_report >= 0.1:
            _worker_progress_queue.put_nowait((search_id, event))
            last_report = now

    before = dict(getattr(_worker_solver.safety, "total_timings", {}))
    candidate = _solve_seed((seed, require_safe), report if report_progress else None)
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
    def __init__(
        self, local: SafePoseIKSolver, executor: ProcessPoolExecutor, progress_queue
    ):
        self.local = local
        self.executor = executor
        self.progress_queue = progress_queue
        self.on_progress = None
        self.search_id = 0
        self.rdf_metrics = {
            "timings": {},
            "minimum_self_barrier": None,
            "minimum_self_pair": None,
        }

    def _drain_progress(self):
        while True:
            try:
                search_id, event = self.progress_queue.get_nowait()
            except Empty:
                return
            if search_id == self.search_id and self.on_progress is not None:
                self.on_progress(event)

    def _map_seeds(self, seeds: list[IKSeed], require_safe: bool):
        pending = {
            self.executor.submit(
                _solve_seed_with_metrics,
                (seed, require_safe, self.search_id, self.on_progress is not None),
            ): seed
            for seed in seeds
        }
        while pending:
            self._drain_progress()
            completed, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            for future in completed:
                seed = pending.pop(future)
                candidate, metrics = future.result()
                self._record_metrics(metrics)
                if self.on_progress is not None:
                    self.on_progress(
                        IKProgress(
                            seed.index,
                            (
                                candidate.iterations
                                if candidate is not None
                                else self.local.config.safe_ik.iteration_number
                            ),
                            True,
                            candidate is not None,
                        )
                    )
                # Completion order improves responsiveness; the domain solver
                # sorts by seed index before selection, preserving tie-breaking.
                yield candidate
        self._drain_progress()

    def _record_metrics(self, metrics):
        for key, value in metrics["timings"].items():
            self.rdf_metrics["timings"][key] = (
                self.rdf_metrics["timings"].get(key, 0.0) + value
            )
        minimum = metrics["minimum_self_barrier"]
        previous = self.rdf_metrics["minimum_self_barrier"]
        if minimum is not None and (previous is None or minimum < previous):
            self.rdf_metrics["minimum_self_barrier"] = minimum
            self.rdf_metrics["minimum_self_pair"] = metrics["minimum_self_pair"]

    def solve(
        self,
        initial_q=None,
        *,
        require_safe: bool = True,
        on_progress: IKProgressCallback | None = None,
    ) -> SafeIKResult:
        self.search_id += 1
        self.on_progress = on_progress
        try:
            return self.local.solve(
                initial_q, require_safe=require_safe, seed_map=self._map_seeds
            )
        finally:
            self.on_progress = None


@contextmanager
def panda_ik_search(
    config: PandaControllerConfig,
    kinematics: PoseKinematics,
    safety: SafetyEvaluator,
):
    """Reuse one pool for safe search and optional pose-only diagnosis, then close it.

    Spawn avoids inheriting native library/viewer thread state. Worker exceptions
    propagate to the caller rather than being misreported as an unreachable pose.
    The supplied components must use config.ik_self_collision_model, as do workers.
    """
    workers = min(config.safe_ik.workers, config.safe_ik.max_restarts)
    local = SafePoseIKSolver(config, kinematics, safety)
    if workers == 1:
        yield local
        return
    context = multiprocessing.get_context("spawn")
    progress_queue = context.Queue()
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(config, progress_queue),
        ) as executor:
            yield _ProcessIKSearch(local, executor, progress_queue)
    finally:
        progress_queue.close()
        progress_queue.join_thread()

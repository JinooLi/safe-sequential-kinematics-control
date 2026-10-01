"""Ordered, CPU-only data queries for offline RDF generation.

Random draws and rendering stay in the parent. Spawn (never fork) also works
after CUDA/EGL initialization. Pools are closed before ridge fitting starts.
Worker count is execution provenance, not a sampling or regression parameter.
"""

import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import copy
from multiprocessing import get_context

import numpy as np
import trimesh
from threadpoolctl import threadpool_limits

from ..adapters.mesh_distance import signed_mesh_distance

_RESOURCE = None
_KIND = None
_THREAD_LIMIT = None


def validate_data_workers(workers):
    if type(workers) is not int or workers < 1:
        raise ValueError("data workers must be a positive integer")
    return workers


def default_data_workers():
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(1, min(4, available))


def data_execution(workers):
    validate_data_workers(workers)
    return {
        "workers": workers,
        "backend": "serial" if workers == 1 else "process_pool",
        "start_method": None if workers == 1 else "spawn",
        "worker_blas_threads": None if workers == 1 else 1,
        "ordered_results": True,
        "random_generation": "parent_only",
        "rendering": "parent_only",
    }


def add_data_workers_argument(parser):
    parser.add_argument(
        "--data-workers",
        type=int,
        default=default_data_workers(),
        help="CPU processes for data queries (default up to 4 available CPUs; 1 is serial); separate from BLAS threads",
    )


def _initialize(kind, resource):
    global _RESOURCE, _KIND, _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)
    _KIND = kind
    if kind == "mesh":
        vertices, faces = resource
        _RESOURCE = trimesh.Trimesh(vertices, faces, process=False)
    else:
        _RESOURCE = resource


def _query(kind, resource, points, sample_count):
    if kind == "mesh":
        return signed_mesh_distance(resource, points)
    # Keep the upstream normal-vote and float32 distance conversion unchanged.
    return resource.get_sdf(points, sample_count=sample_count)


def _worker_query(points, sample_count):
    return _query(_KIND, _RESOURCE, points, sample_count)


class _Evaluator:
    def __init__(self, kind, resource, executor, workers):
        self.kind = kind
        self.resource = resource
        self.executor = executor
        self.workers = workers

    def chunks(self, points, batch_size, sample_count=None):
        """Yield original-size chunks in order, with bounded in-flight copies.

        Exceptions propagate, with no serial fallback or partial-pool acceptance.
        The enclosing context always waits for workers to exit, also on failure.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("query batch size must be a positive integer")
        blocks = iter(
            points[i : i + batch_size] for i in range(0, len(points), batch_size)
        )
        if self.executor is None:
            for block in blocks:
                yield _query(self.kind, self.resource, block, sample_count)
            return
        pending = deque()
        try:
            for _ in range(2 * self.workers):
                block = next(blocks, None)
                if block is None:
                    break
                pending.append(self.executor.submit(_worker_query, block, sample_count))
            while pending:
                yield pending.popleft().result()
                block = next(blocks, None)
                if block is not None:
                    pending.append(
                        self.executor.submit(_worker_query, block, sample_count)
                    )
        finally:
            for future in pending:
                future.cancel()


@contextmanager
def _evaluator(kind, resource, worker_resource, workers):
    validate_data_workers(workers)
    if workers == 1:
        yield _Evaluator(kind, resource, None, workers)
        return
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_initialize,
        initargs=(kind, worker_resource),
    )
    try:
        yield _Evaluator(kind, resource, executor, workers)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def mesh_distance_evaluator(mesh, workers=1):
    # Native R-tree caches must not be pickled. Each worker owns its mesh/tree.
    return _evaluator(
        "mesh", mesh, (np.asarray(mesh.vertices), np.asarray(mesh.faces)), workers
    )


def scan_label_evaluator(cloud, workers=1):
    # Preserve the exact KD-tree (including tie order); don't rebuild/re-render it.
    # Only normal-vote queries are supported: mesh/scans/EGL objects aren't needed.
    snapshot = copy(cloud)
    snapshot.mesh = None
    snapshot.scans = None
    return _evaluator("scan", cloud, snapshot, workers)

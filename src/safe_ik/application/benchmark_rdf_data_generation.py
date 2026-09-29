"""Compare CPU data-worker counts on a saved reference; never train/install RDF."""

import argparse
import json
import os
import platform
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import trimesh
from threadpoolctl import threadpool_limits

from ..adapters.rdf_checkpoint import array_sha256, file_sha256
from .panda_rdf_artifact import read_reconstructed_meshes
from .panda_rdf_training import TrainingOptions
from .rdf_data_workers import data_execution, scan_label_evaluator
from .rdf_exterior_sampling import ExteriorSamplingOptions, sample_exterior_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--link", default="link1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--selection-points", type=int, default=32768)
    parser.add_argument("--label-points", type=int, default=262144)
    args = parser.parse_args()
    if (
        not args.workers
        or args.workers[0] != 1
        or len(set(args.workers)) != len(args.workers)
        or min(args.workers) < 1
        or args.selection_points < 640
        or args.label_points < 1
    ):
        parser.error(
            "require distinct positive worker counts starting with 1, selection >=640 and positive label count"
        )
    # Reserve a new diagnostics directory before doing any work.
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from mesh_to_sdf import get_surface_point_cloud

    meshes, metadata = read_reconstructed_meshes(args.candidate)
    mesh = meshes[args.link]
    with np.load(args.candidate, allow_pickle=False) as arrays:
        center = arrays[f"{args.link}_center"].copy()
        radius = float(arrays[f"{args.link}_radii_max"][1])
    index = [*[f"link{i}" for i in range(8)], "hand"].index(args.link)
    training = TrainingOptions(
        **metadata["training"].get("baseline_options", metadata["training"]["options"])
    )
    options = replace(
        ExteriorSamplingOptions(**metadata["training"]["exterior_sampling"]),
        pool_size=args.selection_points,
    )
    report = {
        "candidate_sha256": file_sha256(args.candidate),
        "link": args.link,
        "reference_mesh_digest": metadata["reconstruction"][args.link]["mesh_digest"],
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "selection_options": asdict(options),
        "label_count": args.label_points,
        "rows": [],
        "scope": "CPU-only diagnostic; no regression, checkpoint or runtime model changes",
    }
    reference = None
    with threadpool_limits(limits=1):
        for workers in args.workers:
            start = perf_counter()
            values = sample_exterior_points(
                mesh, center, radius, options, link_index=index, data_workers=workers
            )
            seconds = perf_counter() - start
            hashes = [array_sha256(v) for v in values[:2]]
            if reference is None:
                reference = (hashes, values[2])
                selected = values[0]
            if (hashes, values[2]) != reference:
                raise ValueError("parallel selection differs from serial")
            report["rows"].append(
                {
                    "workers": workers,
                    "execution": data_execution(workers),
                    "selection_seconds": seconds,
                    "selection_sha256": hashes,
                    "selection_exact": True,
                }
            )
            print(json.dumps(report["rows"][-1]), flush=True)

        # Render ONCE and share exactly that point cloud / KD-tree with each run.
        start = perf_counter()
        cloud = get_surface_point_cloud(
            trimesh.Trimesh(
                (mesh.vertices - center) / radius, mesh.faces, process=False
            ),
            surface_point_method="scan",
            bounding_radius=1.0,
            scan_count=training.scan_count,
            scan_resolution=training.scan_resolution,
            calculate_normals=True,
        )
        report["single_render_seconds"] = perf_counter() - start
        report["scan_cloud_points"] = len(cloud.points)
        pool = args.candidate.with_suffix(".training") / args.link / "training_pool.npz"
        with np.load(pool, allow_pickle=False) as data:
            # Saved near/random points plus new exterior points; no validation reuse.
            candidates = np.vstack(
                (data["near_points"], data["random_points"], selected)
            )
        rng = np.random.default_rng(91027)
        queries = (
            candidates[rng.integers(len(candidates), size=args.label_points)] - center
        ) / radius
        report["label_query_sha256"] = array_sha256(queries)
        baseline = None
        for row in report["rows"]:
            start = perf_counter()
            with scan_label_evaluator(cloud, row["workers"]) as evaluator:
                labels = (
                    np.concatenate(
                        list(evaluator.chunks(queries, 2048, training.normal_count))
                    )
                    * radius
                )
            row["label_seconds"] = perf_counter() - start
            row["label_sha256"] = array_sha256(labels)
            if baseline is None:
                baseline = row["label_sha256"]
            if row["label_sha256"] != baseline:
                raise ValueError("parallel normal labels differ from serial")
            row["labels_exact"] = True
            print(json.dumps(row), flush=True)
    report["passed"] = True
    with (args.output / "benchmark.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

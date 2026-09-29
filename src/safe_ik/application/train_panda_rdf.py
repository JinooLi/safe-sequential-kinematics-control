"""Offline Panda RDF authoring. Failed reports never authorize runtime use."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import trimesh
import yaml
from threadpoolctl import threadpool_limits

from ..adapters.panda_kinematics import PandaKinematics
from ..adapters.panda_rdf_geometry import extract_visual_geometry
from ..domain.panda_rdf import ExtendedRDF
from ..visualization.mesh_inspection import write_inspection_html
from .panda_configuration import load_panda_config
from .panda_mesh_reconstruction import REFERENCE, reconstruct_mesh
from .panda_rdf_artifact import (
    RDFBundle,
    geometry_settings,
    read_candidate_for_validation,
    save_bundle,
)
from .panda_rdf_training import (
    TrainingOptions,
    benchmark_candidate,
    estimate_maximum,
    minimum_sphere,
    surface_queries,
    train_model,
    training_pools,
    training_provenance,
    validate_joint_gradients,
    validate_model,
)
from .paths import default_resource
from .rdf_exterior_sampling import ExteriorSamplingOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_resource("configs/panda.yaml"))
    parser.add_argument(
        "--training-config", default=default_resource("configs/panda_rdf_training.yaml")
    )
    parser.add_argument("--output", help="defaults to safety.rdf.artifact")
    parser.add_argument(
        "--exterior-sampling-config",
        type=Path,
        help="Enriched CUDA fit for every requested link; requires a new --output",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=400)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="ridge training backend; validation and runtime remain CPU",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="reconstruct meshes and export comparison HTML, without fitting/approving RDF",
    )
    parser.add_argument(
        "--links",
        nargs="+",
        help="diagnostic subset; cannot produce an approved full artifact",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="revalidate an existing complete candidate without retraining",
    )
    parser.add_argument(
        "--refresh-queries",
        action="store_true",
        help="resample query density/seed and revalidate, preserving learned coefficients",
    )
    parser.add_argument(
        "--refresh-extension",
        action="store_true",
        help="re-estimate sphere maxima and revalidate without changing coefficients",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU BLAS threads (default 1 avoids oversubscription)",
    )
    parser.add_argument("--gl-platform", default="egl", choices=("egl", "osmesa"))
    args = parser.parse_args()
    if args.refresh_queries or args.refresh_extension:
        args.validate_only = True
    if args.prepare_only and args.validate_only:
        parser.error("--prepare-only cannot be combined with revalidation")
    if args.prepare_only and not args.output:
        parser.error(
            "--prepare-only requires --output to preserve the configured RDF artifact"
        )
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.checkpoint_interval < 1:
        parser.error("checkpoint interval must be positive")
    exterior = None
    if args.exterior_sampling_config:
        if (
            args.prepare_only
            or args.validate_only
            or not args.output
            or args.device != "cuda"
        ):
            parser.error(
                "exterior training requires CUDA, a new --output, and no revalidation/prepare flags"
            )
        exterior = ExteriorSamplingOptions(
            **yaml.safe_load(args.exterior_sampling_config.read_text())
        )
    # Used only by offline official scan generation, before importing pyrender.
    os.environ.setdefault("PYOPENGL_PLATFORM", args.gl_platform)
    config = load_panda_config(args.config)
    training = TrainingOptions(**yaml.safe_load(Path(args.training_config).read_text()))
    compute = {"backend": "numpy/scipy", "device": "cpu", "dtype": "float64"}
    if args.device == "cuda":
        from ..adapters.rdf_torch_training import torch_training_environment

        compute = torch_training_environment("cuda")
    output = Path(args.output) if args.output else config.safety.rdf.artifact
    report_path = output.with_suffix(".report.json")
    state_root = output.with_suffix(".training") if exterior else None
    if exterior and any(
        path.exists()
        for path in (
            output,
            report_path,
            state_root,
            output.with_suffix(".meshes.html"),
        )
    ):
        parser.error("output or training evidence exists; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    geometry = extract_visual_geometry(config.model_path)
    names = args.links or list(geometry.vertices)
    if len(set(names)) != len(names) or any(n not in geometry.vertices for n in names):
        parser.error("links must be unique Panda geometry names")
    metadata = {
        "geometry_fingerprint": geometry.fingerprint,
        "geometry": geometry.provenance,
        "settings": geometry_settings(config.safety.rdf),
        "training": training_provenance(training),
        "training_compute": compute,
        "spheres": {},
        "training_seed": config.safety.rdf.seed,
        "maxima": {},
        "validation": {},
        "validation_reference": REFERENCE,
        "reconstruction": {},
        "joint_validation": {"passed": False},
        "seed_streams": "SeedSequence(seed, spawn_key=(link_index, purpose)); 0=train,1=validation,2=query,3=joint,4=benchmark,5=mesh-comparison",
        "complete": False,
    }
    bundle = RDFBundle({}, {}, geometry.pairs, metadata)
    if exterior:
        if (
            exterior.near_batch > training.near_batch
            or exterior.random_batch > training.random_batch
            or exterior.near_batch + exterior.random_batch + exterior.exterior_batch
            != training.near_batch + training.random_batch
        ):
            parser.error("exterior mixing must preserve baseline batch size")
        state_root.mkdir(parents=True, exist_ok=False)
        metadata["training"]["baseline_options"] = asdict(training)
        metadata["training"]["options"]["near_batch"] = exterior.near_batch
        metadata["training"]["options"]["random_batch"] = exterior.random_batch
        metadata["training"]["exterior_sampling"] = asdict(exterior)
        metadata["link_training"] = {}
        metadata["exterior_seed_streams"] = (
            "SeedSequence(exterior.seed, spawn_key=(link_index, 7=points or 8=batches))"
        )
    if args.validate_only:
        if args.links:
            parser.error("--validate-only requires all links")
        bundle = read_candidate_for_validation(
            output, geometry, config.safety.rdf, refresh_queries=args.refresh_queries
        )
        metadata = bundle.metadata
        # Interrupted revalidation must not retain the previous approval flag.
        metadata["complete"] = False
    started = perf_counter()
    try:
        with threadpool_limits(limits=args.threads):
            for name in names:
                index = list(geometry.vertices).index(name)
                print(
                    f"LINK START: {name} ({names.index(name) + 1}/{len(names)})",
                    flush=True,
                )
                streams = [
                    np.random.default_rng(
                        np.random.SeedSequence(
                            config.safety.rdf.seed, spawn_key=(index, purpose)
                        )
                    )
                    for purpose in range(6)
                ]
                mesh = trimesh.Trimesh(
                    geometry.vertices[name], geometry.faces[name], process=False
                )
                if args.validate_only:
                    reference = bundle.reconstructed[name]
                else:
                    print(
                        f"{name}: reconstructing official voxel reference", flush=True
                    )
                    reference, reconstruction = reconstruct_mesh(
                        mesh, training.reconstruction, streams[5]
                    )
                    bundle.reconstructed[name] = reference
                    metadata["reconstruction"][name] = reconstruction
                    save_bundle(bundle, output)
                if args.prepare_only:
                    continue
                if args.validate_only:
                    if args.refresh_extension:
                        model = bundle.models[name]
                        maximum, maximum_report = estimate_maximum(
                            model.rdf, model.inner_radius, training
                        )
                        bundle.models[name] = ExtendedRDF(
                            model.rdf, model.inner_radius, maximum
                        )
                        metadata["maxima"][name] = maximum_report
                    if args.refresh_queries:
                        r0 = bundle.models[name].inner_radius
                        count = int(
                            np.ceil(config.safety.rdf.query_density * 4 * np.pi * r0**2)
                        )
                        bundle.queries[name] = surface_queries(mesh, count, streams[2])
                    print(f"{name}: revalidating existing coefficients", flush=True)
                    metadata["validation"][name] = validate_model(
                        reference,
                        bundle.models[name],
                        streams[1],
                        config.safety.rdf.error_margin,
                    )
                    metadata["validation"][name]["reference_mesh_digest"] = metadata[
                        "reconstruction"
                    ][name]["mesh_digest"]
                    continue
                center, radius = minimum_sphere(mesh.vertices)
                r0 = radius + config.safety.rdf.effective_margin
                r1 = max((1 + config.safety.rdf.transition_ratio) * r0, radius + 0.060)
                metadata["spheres"][name] = {
                    "center": center.tolist(),
                    "mesh_radius": radius,
                    "r0": r0,
                    "r1": r1,
                    "method": "convex hull + SLSQP, ftol=1e-12",
                }
                print(
                    f"{name}: generating official scan/normal pools (r1={r1:.6f} m)",
                    flush=True,
                )
                if exterior:
                    from .panda_rdf_enriched_training import train_enriched_link

                    rdf, fit_report = train_enriched_link(
                        reference,
                        center,
                        r1,
                        streams[0],
                        training,
                        exterior,
                        link_index=index,
                        directory=state_root / name,
                        context={
                            "link": name,
                            "geometry_fingerprint": geometry.fingerprint,
                            "reference_mesh_digest": reconstruction["mesh_digest"],
                            "settings": geometry_settings(config.safety.rdf),
                            "sphere": metadata["spheres"][name],
                        },
                        checkpoint_interval=args.checkpoint_interval,
                        progress=lambda message: print(
                            f"{name}: {message}", flush=True
                        ),
                    )
                    metadata["link_training"][name] = fit_report
                else:
                    pools = training_pools(reference, center, r1, streams[0], training)
                    print(
                        f"{name}: recursive ridge, {training.updates} updates",
                        flush=True,
                    )
                    rdf = train_model(
                        center, r1, pools, streams[0], training, device=args.device
                    )
                    del pools
                maximum, maximum_report = estimate_maximum(rdf, r0, training)
                model = ExtendedRDF(rdf, r0, maximum)
                bundle.models[name] = model
                count = int(
                    np.ceil(config.safety.rdf.query_density * 4 * np.pi * r0**2)
                )
                bundle.queries[name] = surface_queries(mesh, count, streams[2])
                metadata["maxima"][name] = maximum_report
                print(f"{name}: independent 5000-point triangle validation", flush=True)
                metadata["validation"][name] = validate_model(
                    reference,
                    model,
                    streams[1],
                    config.safety.rdf.error_margin,
                    **(
                        {"evidence_path": state_root / name / "validation.npz"}
                        if exterior
                        else {}
                    ),
                )
                metadata["validation"][name]["reference_mesh_digest"] = metadata[
                    "reconstruction"
                ][name]["mesh_digest"]
                result = metadata["validation"][name]
                print(
                    f"{name}: passed={result['passed']}, near max={result['near_max_error']:.6f} m",
                    flush=True,
                )
                print(
                    f"LINK COMPLETE: {name}; distance={result['distance_passed']}; derivatives={result['derivatives_passed']}",
                    flush=True,
                )
                # Preserve intermediate evidence on interruption; incomplete files fail closed.
                save_bundle(bundle, output)
                report_path.write_text(
                    json.dumps(metadata, indent=2, allow_nan=False) + "\n"
                )
            if not args.prepare_only and set(names) == set(geometry.vertices):
                frames = set(geometry.vertices) | {
                    frame
                    for capsule in config.safety.capsules
                    for frame in (capsule.start_frame, capsule.end_frame)
                }
                kinematics = PandaKinematics(config.model_path, config.ee_frame, frames)
                metadata["joint_validation"] = validate_joint_gradients(
                    kinematics,
                    bundle,
                    config.safety.rdf,
                    np.random.default_rng(
                        np.random.SeedSequence(config.safety.rdf.seed, spawn_key=(0, 3))
                    ),
                )
                metadata["complete"] = True
                metadata["runtime_benchmark"] = benchmark_candidate(
                    config, kinematics, bundle
                )
    except Exception as exc:
        metadata["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        metadata["wall_seconds"] = perf_counter() - started
        save_bundle(bundle, output)
        report_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        if bundle.reconstructed:
            preview = output.with_suffix(".meshes.html")
            write_inspection_html(
                geometry, preview, bundle.reconstructed, metadata["reconstruction"]
            )
            print(f"Mesh comparison: {preview}", flush=True)
    if args.prepare_only:
        print(f"Prepared meshes: {output}; no RDF trained or approved")
        return
    approved = (
        metadata["complete"]
        and metadata["joint_validation"]["passed"]
        and all(v["passed"] for v in metadata["validation"].values())
    )
    print(f"Artifact: {output}; runtime approved: {approved}; report: {report_path}")
    if not approved:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

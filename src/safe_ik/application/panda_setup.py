"""Composition root for the default Panda backends."""

from functools import partial

import numpy as np

from ..adapters.mujoco_plant import MujocoPlant
from ..adapters.osqp_solver import OSQPSolver
from ..adapters.panda_kinematics import PandaKinematics
from ..domain.panda_config import PandaControllerConfig
from ..domain.spatial_safety import PandaCapsuleSafety
from .panda_runtime import PandaRuntime
from .parallel_ik import panda_ik_search


def build_panda_components(config: PandaControllerConfig):
    bundle = None
    if config.safety.self_collision_model == "rdf":
        from ..adapters.panda_rdf_geometry import extract_visual_geometry
        from .panda_rdf_artifact import load_bundle

        geometry = extract_visual_geometry(config.model_path)
        bundle = load_bundle(config.safety.rdf.artifact, geometry, config.safety.rdf)
    frames = {
        frame
        for capsule in config.safety.capsules
        for frame in (capsule.start_frame, capsule.end_frame)
    }
    frames.update(config.trajectory.stage_frames)
    if bundle is not None:
        frames.update(bundle.models)
    kinematics = PandaKinematics(config.model_path, config.ee_frame, frames)
    if np.any(config.initial_q < kinematics.lower_limits) or np.any(
        config.initial_q > kinematics.upper_limits
    ):
        raise ValueError("robot.initial_q violates Panda joint limits")
    safety = PandaCapsuleSafety(
        kinematics,
        config.safety.capsules,
        config.obstacles,
        config.safety.margin,
        config.safety.self_collision_exclusions,
        include_self=bundle is None,
    )
    if bundle is not None:
        from ..domain.panda_rdf_safety import PandaRDFSafety

        safety = PandaRDFSafety(
            kinematics,
            bundle.models,
            bundle.queries,
            bundle.pairs,
            config.safety.rdf,
            safety,
            bundle.identifier,
        )
    return kinematics, safety


def build_panda_runtime(config: PandaControllerConfig) -> PandaRuntime:
    kinematics, safety = build_panda_components(config)
    return PandaRuntime(
        kinematics,
        safety,
        OSQPSolver(
            max_iterations=config.qp.max_iterations,
            absolute_tolerance=config.qp.absolute_tolerance,
            relative_tolerance=config.qp.relative_tolerance,
        ),
        MujocoPlant(config),
        ik_search_factory=partial(panda_ik_search, config, kinematics, safety),
    )

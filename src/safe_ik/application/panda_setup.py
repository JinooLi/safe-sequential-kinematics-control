"""Composition root for the default Panda backends."""

from functools import partial

from ..adapters.mujoco_plant import MujocoPlant
from ..adapters.osqp_solver import OSQPSolver
from ..domain.panda_config import PandaControllerConfig
from .panda_components import build_panda_components
from .panda_runtime import PandaRuntime
from .parallel_ik import panda_ik_search


def build_panda_runtime(config: PandaControllerConfig) -> PandaRuntime:
    kinematics, safety = build_panda_components(config)
    ik_kinematics, ik_safety = kinematics, safety
    if (
        config.trajectory.mode != "direct"
        and config.ik_self_collision_model != config.safety.self_collision_model
    ):
        ik_kinematics, ik_safety = build_panda_components(
            config, self_collision_model=config.ik_self_collision_model
        )
    return PandaRuntime(
        kinematics,
        safety,
        OSQPSolver(
            max_iterations=config.qp.max_iterations,
            absolute_tolerance=config.qp.absolute_tolerance,
            relative_tolerance=config.qp.relative_tolerance,
        ),
        MujocoPlant(config),
        ik_search_factory=partial(panda_ik_search, config, ik_kinematics, ik_safety),
        ik_safety=ik_safety,
    )

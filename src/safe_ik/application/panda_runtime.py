"""Injected runtime dependencies for a Panda rollout."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ContextManager

from ..control.solver import QPSolver
from ..domain.interfaces import JointPlant, PoseKinematics
from ..domain.safe_pose_ik import PoseIKSearch
from ..domain.spatial_safety import SafetyEvaluator


@dataclass
class PandaRuntime:
    kinematics: PoseKinematics
    safety: SafetyEvaluator
    solver: QPSolver
    plant: JointPlant

    ik_search_factory: Callable[[], ContextManager[PoseIKSearch]] | None = None
    # Optional for legacy/injected runtimes sharing a single safety backend.
    ik_safety: SafetyEvaluator | None = None

"""Small dependency boundaries used by the application simulation loop."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import SafetyDiagnostics, SimulationStep


class KinematicsProvider(Protocol):
    def set_end_effector(self, link_index: int): ...

    def update_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def end_effector_yaw(self) -> float: ...


class VelocityController(Protocol):
    def get_velocity(
        self, q: np.ndarray, J: np.ndarray, p_ee: np.ndarray, ee_yaw: float
    ) -> np.ndarray: ...

    def safety_diagnostics(
        self,
        q: np.ndarray,
        J: np.ndarray | None = None,
        p_ee: np.ndarray | None = None,
        ee_yaw: float | None = None,
    ) -> SafetyDiagnostics: ...


class SimulationRecorder(Protocol):
    def record(self, step: SimulationStep): ...

    def save_npz(self, path): ...

    def save_csv(self, path): ...

    def plot(self, path): ...

    def show_gui(self): ...


class PoseKinematics(Protocol):
    @property
    def lower_limits(self) -> np.ndarray: ...
    @property
    def upper_limits(self) -> np.ndarray: ...
    def pose_and_jacobian(
        self, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def frame_point_and_jacobian(
        self, frame_name: str, q: np.ndarray, local_offset=None
    ) -> tuple[np.ndarray, np.ndarray]: ...


class JointPlant(Protocol):
    @property
    def q(self) -> np.ndarray: ...
    @property
    def time(self) -> float: ...
    def advance(self, velocity: np.ndarray, period: float) -> None: ...


class FramePoseKinematics(PoseKinematics, Protocol):
    def frame_pose_and_jacobian(
        self, frame_name: str, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return local-to-world rotation, origin and world-aligned 6D Jacobian."""
        ...

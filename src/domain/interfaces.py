"""Small dependency boundaries used by the application simulation loop."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import SafetyDiagnostics, SimulationStep


class KinematicsProvider(Protocol):
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

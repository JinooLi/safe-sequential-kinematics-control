"""Scene sampling parameters independent of files and execution backends."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PandaSamplingConfig:
    workspace_lower: tuple[float, ...] = (-0.7, -0.7, 0.1)
    workspace_upper: tuple[float, ...] = (0.7, 0.7, 1.0)
    obstacle_count: int = 2
    obstacle_radius_range: tuple[float, ...] = (0.04, 0.10)
    minimum_target_distance: float = 0.15
    joint_limit_margin: float = 0.1
    minimum_barrier: float = 0.005
    max_proposals_per_seed: int = 10000

    def __post_init__(self):
        for name, length in (
            ("workspace_lower", 3),
            ("workspace_upper", 3),
            ("obstacle_radius_range", 2),
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (length,) or not np.isfinite(values).all():
                raise ValueError(f"sampling.{name} must contain {length} finite values")
            object.__setattr__(self, name, tuple(values.tolist()))
        if np.any(np.asarray(self.workspace_lower) >= self.workspace_upper):
            raise ValueError(
                "sampling workspace lower bounds must be below upper bounds"
            )
        if not 0 < self.obstacle_radius_range[0] <= self.obstacle_radius_range[1]:
            raise ValueError("sampling obstacle radii must be positive and ordered")
        for name in ("obstacle_count", "max_proposals_per_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"sampling.{name} must be a positive integer")
        for name in (
            "minimum_target_distance",
            "joint_limit_margin",
            "minimum_barrier",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"sampling.{name} must be finite and nonnegative")

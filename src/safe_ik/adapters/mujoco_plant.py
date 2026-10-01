"""MuJoCo implementation of kinematic joint integration and optional display."""

import time
from contextlib import contextmanager

import mujoco
import numpy as np

from ..domain.panda_config import PandaControllerConfig


class MujocoPlant:
    def __init__(self, config: PandaControllerConfig):
        self.model = mujoco.MjModel.from_xml_path(str(config.model_path))
        goal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_marker")
        if goal_id >= 0:
            self.model.site_pos[goal_id] = config.target.position
        for index, obstacle in enumerate(config.obstacles, start=1):
            geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_geom{index}"
            )
            if geom_id >= 0:
                self.model.geom_pos[geom_id] = obstacle.position
                self.model.geom_size[geom_id, 0] = obstacle.radius
                self.model.geom_rgba[geom_id, 3] = 0.6 if obstacle.enabled else 0.0
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = config.initial_q
        mujoco.mj_forward(self.model, self.data)

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos.copy()

    @property
    def time(self) -> float:
        return float(self.data.time)

    def advance(self, velocity: np.ndarray, period: float) -> None:
        self.data.qvel[:] = velocity
        mujoco.mj_integratePos(self.model, self.data.qpos, self.data.qvel, period)
        self.data.time += period
        mujoco.mj_forward(self.model, self.data)

    @contextmanager
    def viewer(self, *, hold=True):
        from mujoco import viewer

        with viewer.launch_passive(self.model, self.data) as window:
            yield window
            while hold and window.is_running():
                window.sync()
                time.sleep(0.05)

    def replay(self, timestamps: np.ndarray, positions: np.ndarray) -> None:
        """Display recorded poses at 1x speed, then hold the final frame.

        There is no integration, IK, RDF or QP during playback. Closing the
        viewer exits this loop; no computation workers are attached to it.
        Only recorded poses are shown (no unvalidated interpolation).
        """
        timestamps = np.asarray(timestamps, dtype=float)
        positions = np.asarray(positions, dtype=float)
        if (
            timestamps.ndim != 1
            or not timestamps.size
            or positions.shape != (timestamps.size, self.model.nq)
            or not np.all(np.isfinite(timestamps))
            or not np.all(np.isfinite(positions))
            or np.any(np.diff(timestamps) < 0)
        ):
            raise ValueError(
                "replay needs finite ordered timestamps and matching joint poses"
            )
        original_q, original_velocity, original_time = (
            self.q,
            self.data.qvel.copy(),
            self.time,
        )

        def display(index):
            self.data.qpos[:] = positions[index]
            self.data.qvel[:] = 0.0
            self.data.time = timestamps[index]
            mujoco.mj_forward(self.model, self.data)

        try:
            display(0)
            with self.viewer(hold=False) as window:
                start = time.monotonic()
                current = 0
                while window.is_running():
                    sample_time = timestamps[0] + time.monotonic() - start
                    index = min(
                        int(np.searchsorted(timestamps, sample_time, side="right")) - 1,
                        len(timestamps) - 1,
                    )
                    if index != current:
                        with window.lock():
                            display(index)
                        current = index
                    window.sync()
                    time.sleep(1.0 / 60.0)
        finally:
            self.data.qpos[:] = original_q
            self.data.qvel[:] = original_velocity
            self.data.time = original_time
            mujoco.mj_forward(self.model, self.data)

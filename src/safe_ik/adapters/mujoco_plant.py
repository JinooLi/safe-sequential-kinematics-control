"""MuJoCo implementation of kinematic joint integration and optional display."""

import time
from contextlib import contextmanager

import mujoco
import numpy as np

from ..domain.panda_config import PandaControllerConfig


class MujocoPlant:
    def __init__(self, config: PandaControllerConfig):
        spec = mujoco.MjSpec.from_file(str(config.model_path))
        # Replace the XML placeholders in memory so every scene can have its
        # own obstacle count without changing the saved model's file hash.
        for geom in list(spec.worldbody.geoms):
            if geom.name.startswith("obstacle_geom"):
                spec.delete(geom)
        for index, obstacle in enumerate(config.obstacles, start=1):
            spec.worldbody.add_geom(
                name=f"obstacle_geom{index}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=obstacle.position,
                size=[obstacle.radius, 0.0, 0.0],
                rgba=[0.8, 0.1, 0.1, 0.6 if obstacle.enabled else 0.0],
                contype=0,
                conaffinity=0,
            )
        self.model = spec.compile()
        goal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_marker")
        if goal_id >= 0:
            self.model.site_pos[goal_id] = config.target.position
        self.data = mujoco.MjData(self.model)
        scalar = np.isin(
            self.model.jnt_type,
            (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE),
        )
        limited = np.flatnonzero(scalar & self.model.jnt_limited.astype(bool))
        self._limit_qpos = self.model.jnt_qposadr[limited]
        self._limit_dofs = self.model.jnt_dofadr[limited]
        self._lower, self._upper = self.model.jnt_range[limited].T.copy()
        initial = np.asarray(config.initial_q)[self._limit_qpos]
        if np.any(initial < self._lower) or np.any(initial > self._upper):
            raise ValueError("initial joint positions exceed model limits")
        self.last_joint_limit_clipped = np.zeros(self.model.nv, dtype=bool)
        self.last_applied_velocity = np.zeros(self.model.nv)
        self.data.qpos[:] = config.initial_q
        mujoco.mj_forward(self.model, self.data)

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos.copy()

    @property
    def time(self) -> float:
        return float(self.data.time)

    def advance(self, velocity: np.ndarray, period: float) -> None:
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (self.model.nv,) or not np.isfinite(velocity).all():
            raise ValueError("joint velocity must be finite and match model nv")
        if not np.isfinite(period) or period <= 0:
            raise ValueError("integration period must be finite and positive")
        previous = self.data.qpos[self._limit_qpos].copy()
        self.data.qvel[:] = velocity
        mujoco.mj_integratePos(self.model, self.data.qpos, self.data.qvel, period)
        # Kinematic integration does not enforce MuJoCo's dynamic joint limits.
        # Model an ideal position stop independently of the controller's CBFs.
        proposed = self.data.qpos[self._limit_qpos].copy()
        bounded = np.clip(proposed, self._lower, self._upper)
        clipped = proposed != bounded
        self.data.qpos[self._limit_qpos] = bounded
        self.last_joint_limit_clipped.fill(False)
        self.last_joint_limit_clipped[self._limit_dofs] = clipped
        self.last_applied_velocity = velocity.copy()
        self.last_applied_velocity[self._limit_dofs[clipped]] = (
            bounded[clipped] - previous[clipped]
        ) / period
        self.data.qvel[:] = self.last_applied_velocity
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

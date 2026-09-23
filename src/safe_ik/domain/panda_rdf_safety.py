"""Bidirectional visual-surface RDF barriers, shared by IK and trajectory QP."""

from time import perf_counter

import numpy as np

from .interfaces import FramePoseKinematics
from .panda_rdf import relative_point_jacobian, soft_min, top_indices
from .spatial_safety import BarrierEvaluation, PandaCapsuleSafety


class PandaRDFSafety:
    softmin = PandaCapsuleSafety.softmin

    def __init__(
        self,
        kinematics: FramePoseKinematics,
        models,
        queries,
        pairs,
        options,
        obstacles=None,
        artifact_id="",
    ):
        self.kinematics = kinematics
        self.models, self.queries, self.pairs = models, queries, pairs
        self.options, self.obstacles = options, obstacles
        self.artifact_id = artifact_id
        self.last_timings = {}
        self.total_timings = dict(distance=0.0, selection=0.0, gradient=0.0)
        self.minimum_self_barrier = float("inf")
        self.minimum_self_pair = None
        self._q = self._evaluation = None

    def evaluate(self, q):
        q = np.asarray(q, dtype=float)
        if self._q is not None and np.array_equal(q, self._q):
            self.last_timings = dict(distance=0.0, selection=0.0, gradient=0.0)
            return self._evaluation
        timings = dict(distance=0.0, selection=0.0, gradient=0.0)
        start = perf_counter()
        poses = {
            name: self.kinematics.frame_pose_and_jacobian(name, q)
            for name in self.models
        }
        world, point_jac = {}, {}
        for name, (rotation, origin, jac) in poses.items():
            offsets = self.queries[name] @ rotation.T
            world[name] = offsets + origin
            # omega x offset, for every joint and every query point.
            point_jac[name] = jac[:3][None] + np.cross(
                jac[3:].T[None], offsets[:, None]
            ).transpose(0, 2, 1)
        timings["gradient"] += perf_counter() - start
        labels, values, gradients = [], [], []
        for left, right in self.pairs:
            distances, derivatives = [], []
            for target, source in ((left, right), (right, left)):
                rotation, origin, jac = poses[target]
                start = perf_counter()
                local = (world[source] - origin) @ rotation
                distance, _ = self.models[target].query(local, gradient=False)
                timings["distance"] += perf_counter() - start
                start = perf_counter()
                selected = top_indices(distance, self.options.top_k)
                timings["selection"] += perf_counter() - start
                start = perf_counter()
                _, gradient = self.models[target].query(local[selected])
                relative = relative_point_jacobian(
                    rotation,
                    origin,
                    jac[:3],
                    jac[3:],
                    world[source][selected],
                    point_jac[source][selected],
                )
                distances.extend(distance[selected])
                derivatives.extend(np.einsum("bi,bij->bj", gradient, relative))
                timings["gradient"] += perf_counter() - start
            start = perf_counter()
            distance, weights = soft_min(distances, self.options.softmin_error)
            labels.append(f"self_{left}_{right}")
            values.append(distance - self.options.effective_margin)
            gradients.append(weights @ np.asarray(derivatives))
            timings["gradient"] += perf_counter() - start
        if self.obstacles is not None:
            evaluation = self.obstacles.evaluate(q)
            labels.extend(evaluation.labels)
            values.extend(evaluation.values)
            gradients.extend(evaluation.gradients)
        result = BarrierEvaluation(
            tuple(labels), np.asarray(values), np.asarray(gradients).reshape(-1, len(q))
        )
        if (
            not np.isfinite(result.values).all()
            or not np.isfinite(result.gradients).all()
        ):
            raise ValueError("nonfinite Panda RDF barrier/gradient")
        self.last_timings = timings
        index = int(np.argmin(result.values[: len(self.pairs)]))
        if result.values[index] < self.minimum_self_barrier:
            self.minimum_self_barrier = float(result.values[index])
            self.minimum_self_pair = result.labels[index]
        for key, value in timings.items():
            self.total_timings[key] += value
        self._q, self._evaluation = q.copy(), result
        return result

"""Fixed-direction visual-surface RDF barriers for IK and trajectory QP."""

from time import perf_counter

import numpy as np

from .interfaces import FramePoseKinematics
from .panda_rdf import relative_point_jacobian, soft_min, top_indices
from .panda_rdf_gate import SpherePairGate
from .safety_contracts import (
    BarrierEvaluation,
    BarrierValues,
    barrier_values_batch,
    evaluate_softmin,
)

_LINK_ORDER = {f"link{i}": i for i in range(8)} | {"hand": 8}


def _query_pair(left, right):
    """Return (surface source, RDF target), independently of input pair order."""
    try:
        return tuple(sorted((left, right), key=_LINK_ORDER.__getitem__))
    except KeyError as exc:
        raise ValueError(f"unknown Panda RDF link: {exc.args[0]}") from exc


class PandaRDFSafety:
    query_policy = "lower_link_surface_to_higher_link_rdf_v1"

    def __init__(
        self,
        kinematics: FramePoseKinematics,
        models,
        queries,
        pairs,
        options,
        obstacles=None,
        artifact_id="",
        *,
        bounding_vertices=None,
        pair_evaluator=None,
    ):
        self.kinematics = kinematics
        self.models, self.queries, self.pairs = models, queries, tuple(pairs)
        # link0 < ... < link7 < hand. Keep barrier labels/pair coverage unchanged.
        self.query_pairs = tuple(_query_pair(*pair) for pair in self.pairs)
        self.source_links = frozenset(source for source, _ in self.query_pairs)
        self.options, self.obstacles = options, obstacles
        if options.runtime_device != "cpu" and pair_evaluator is None:
            raise ValueError("CUDA RDF requires an explicit tensor pair evaluator")
        self.pair_evaluator = pair_evaluator
        self.runtime_metadata = (
            pair_evaluator.metadata
            if pair_evaluator is not None
            else {"backend": "numpy", "device": "cpu", "dtype": "float64"}
        )
        self.sphere_gate = (
            SpherePairGate(
                models,
                queries,
                self.pairs,
                bounding_vertices,
                options.effective_margin,
                options.sphere_gate_transition_fraction,
            )
            if options.sphere_gate
            else None
        )
        self.gate_metadata = self.sphere_gate.metadata if self.sphere_gate else None
        self.last_gate_counts = {}
        self.artifact_id = artifact_id
        self.last_timings = {}
        self.total_timings = dict(distance=0.0, selection=0.0, gradient=0.0)
        self.minimum_self_barrier = float("inf")
        self.minimum_self_pair = None
        self._q = self._evaluation = None

    def softmin(self, q, temperature: float):
        """Compatibility wrapper; the IK loop aggregates an existing evaluation."""
        return evaluate_softmin(self, q, temperature)

    def evaluate_values_batch(self, states, *, batch_size=8):
        """Value-only endpoint/audit checks; never populate the controller cache."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        result = []
        states = list(states)
        pose = getattr(
            self.kinematics, "frame_pose", self.kinematics.frame_pose_and_jacobian
        )
        labels = tuple(f"self_{a}_{b}" for a, b in self.pairs)
        for offset in range(0, len(states), batch_size):
            chunk = states[offset : offset + batch_size]
            poses = [{name: pose(name, q) for name in self.models} for q in chunk]
            gates = [
                (
                    self.sphere_gate.evaluate_values(p)
                    if self.sphere_gate
                    else (np.zeros(len(self.pairs)), np.ones(len(self.pairs)))
                )
                for p in poses
            ]
            active = [np.flatnonzero(w > 0) for _, w in gates]
            if self.pair_evaluator is not None:
                values = self.pair_evaluator.evaluate_values_batch(poses, active)
            else:
                values = []
                for p, rows in zip(poses, active):
                    current = []
                    for row in rows:
                        source, target = self.query_pairs[row]
                        world = self.queries[source] @ p[source][0].T + p[source][1]
                        local = (world - p[target][1]) @ p[target][0]
                        distance, _ = self.models[target].query(local, gradient=False)
                        chosen = top_indices(distance, self.options.top_k)
                        value, _ = soft_min(
                            distance[chosen], self.options.softmin_error
                        )
                        current.append(value - self.options.effective_margin)
                    values.append(current)
            obstacles = (
                barrier_values_batch(self.obstacles, chunk)
                if self.obstacles
                else [None] * len(chunk)
            )
            for rows, rdf, (sphere, weight), obstacle in zip(
                active, values, gates, obstacles
            ):
                combined = sphere.copy()
                combined[rows] = (1 - weight[rows]) * sphere[rows] + weight[
                    rows
                ] * np.asarray(rdf)
                names = labels
                if obstacle is not None:
                    combined = np.concatenate((combined, obstacle.values))
                    names += obstacle.labels
                if not np.isfinite(combined).all():
                    raise ValueError("nonfinite Panda RDF barrier")
                result.append(BarrierValues(names, combined))
        return result

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
        gate = self.sphere_gate.evaluate(poses) if self.sphere_gate else None
        self.last_gate_counts = {
            "sphere_only": int(np.count_nonzero(gate.weights == 0)) if gate else 0,
            "transition": (
                int(np.count_nonzero((gate.weights > 0) & (gate.weights < 1)))
                if gate
                else 0
            ),
            "rdf_only": (
                int(np.count_nonzero(gate.weights == 1)) if gate else len(self.pairs)
            ),
        }
        world, point_jac = {}, {}
        timings["gradient"] += perf_counter() - start
        batched = {}
        if self.pair_evaluator is not None:
            active = (
                np.flatnonzero(gate.weights > 0) if gate else np.arange(len(self.pairs))
            )
            start = perf_counter()
            batch_values, batch_gradients = self.pair_evaluator.evaluate(poses, active)
            # One synchronized tensor batch includes transform, value, selection,
            # derivative, softmin and transfers. Metadata labels this explicitly.
            timings["distance"] += perf_counter() - start
            batched = dict(zip(active, zip(batch_values, batch_gradients)))
        labels, values, gradients = [], [], []
        for row, ((left, right), (source, target)) in enumerate(
            zip(self.pairs, self.query_pairs)
        ):
            labels.append(f"self_{left}_{right}")
            if gate is not None and gate.weights[row] == 0:
                # Keep the same CBF row/label, but no surface points or RDF calls.
                values.append(gate.values[row])
                gradients.append(gate.gradients[row])
                continue
            if self.pair_evaluator is not None:
                value, derivative = batched[row]
                start = perf_counter()
            else:
                start = perf_counter()
                if source not in world:
                    r, origin, jac = poses[source]
                    offsets = self.queries[source] @ r.T
                    world[source] = offsets + origin
                    point_jac[source] = jac[:3][None] + np.cross(
                        jac[3:].T[None], offsets[:, None]
                    ).transpose(0, 2, 1)
                timings["gradient"] += perf_counter() - start
                rotation, origin, jac = poses[target]
                start = perf_counter()
                # Every source surface point is retained; no hemisphere filtering.
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
                derivatives = np.einsum("bi,bij->bj", gradient, relative)
                timings["gradient"] += perf_counter() - start
                start = perf_counter()
                distance, weights = soft_min(
                    distance[selected], self.options.softmin_error
                )
                value = distance - self.options.effective_margin
                derivative = weights @ derivatives
            if gate is not None and gate.weights[row] < 1:
                w = gate.weights[row]
                derivative = (
                    (1 - w) * gate.gradients[row]
                    + w * derivative
                    + (value - gate.values[row]) * gate.weight_gradients[row]
                )
                value = (1 - w) * gate.values[row] + w * value
            values.append(value)
            gradients.append(derivative)
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

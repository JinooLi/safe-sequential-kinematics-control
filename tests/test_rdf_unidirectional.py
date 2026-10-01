"""Fixed query direction, unchanged aggregation and both-moving-frame gradients."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from safe_ik.domain.panda_config import PandaRDFConfig
from safe_ik.domain.panda_rdf import soft_min, top_indices
from safe_ik.domain.panda_rdf_safety import PandaRDFSafety
from safe_ik.domain.spatial_safety import BarrierEvaluation


class MovingFrames:
    def __init__(self, source):
        self.source = source

    def frame_pose_and_jacobian(self, name, q):
        jac = np.zeros((6, 2))
        if name == self.source:
            rotation = Rotation.from_rotvec([0, 0, q[0]]).as_matrix()
            origin = np.array([0.1 * q[1], 0, 0])
            jac[0, 1], jac[5, 0] = 0.1, 1.0
        else:
            rotation = Rotation.from_rotvec([0, q[1], 0]).as_matrix()
            origin = np.array([0.4, 0.2 * q[0], 0.1])
            jac[1, 0], jac[4, 1] = 0.2, 1.0
        return rotation, origin, jac


class QuadraticField:
    def __init__(self):
        self.calls = []

    def query(self, points, gradient=True):
        self.calls.append((points.copy(), gradient))
        scale = np.array([1.0, 0.4, 0.7])
        values = 0.1 + (points**2 * scale).sum(axis=1) + 0.02 * points[:, 1]
        grad = 2 * points * scale + [0, 0.02, 0]
        return values, grad if gradient else np.empty((0, 3))


class ForbiddenReverseField:
    def query(self, *args, **kwargs):
        raise AssertionError("the reverse RDF direction must never be queried")


class FixedObstacles:
    def evaluate(self, q):
        return BarrierEvaluation(
            ("obstacle",), np.array([0.4]), np.array([[0.2, -0.1]])
        )


class UnidirectionalRDFTest(unittest.TestCase):
    def make_safety(
        self,
        source="link1",
        target="link4",
        *,
        top_k=2,
        reverse_pair=False,
        obstacles=None,
    ):
        field = QuadraticField()
        # Reverse model insertion order on purpose: it must not select direction.
        models = {target: field, source: ForbiddenReverseField()}
        points = np.array(
            [
                [-0.2, 0.1, 0],
                [0.02, 0.12, 0.07],
                [0.15, -0.03, 0.02],
                [-0.1, -0.2, 0.03],
                [0.22, 0.07, -0.1],
            ]
        )
        queries = {source: points, target: np.array([[100.0, 100.0, 100.0]])}
        pair = (target, source) if reverse_pair else (source, target)
        options = PandaRDFConfig(top_k=top_k, physical_margin=0.01, error_margin=0.01)
        safety = PandaRDFSafety(
            MovingFrames(source), models, queries, (pair,), options, obstacles
        )
        return safety, field

    def expected_local(self, safety, q):
        source, target = safety.query_pairs[0]
        rs, ts, _ = safety.kinematics.frame_pose_and_jacobian(source, q)
        rt, tt, _ = safety.kinematics.frame_pose_and_jacobian(target, q)
        return (safety.queries[source] @ rs.T + ts - tt) @ rt

    def test_lower_link_queries_higher_rdf_independent_of_pair_and_model_order(self):
        q = np.array([0.4, -0.2])
        for source, target in (
            ("link1", "link4"),
            ("link7", "hand"),
            ("link0", "link7"),
        ):
            for reverse in (False, True):
                with self.subTest(source=source, target=target, reverse=reverse):
                    safety, field = self.make_safety(
                        source, target, reverse_pair=reverse
                    )
                    result = safety.evaluate(q)
                    self.assertEqual(safety.query_pairs, ((source, target),))
                    self.assertEqual(len(field.calls), 2)
                    points, gradient = field.calls[0]
                    self.assertFalse(gradient)
                    np.testing.assert_array_equal(
                        points, self.expected_local(safety, q)
                    )
                    # All source points survive; no hemisphere filter or target samples.
                    self.assertEqual(points.shape, (5, 3))
                    self.assertTrue(field.calls[1][1])
                    self.assertEqual(field.calls[1][0].shape, (2, 3))
                    pair = (target, source) if reverse else (source, target)
                    self.assertEqual(result.labels, ("self_" + "_".join(pair),))

    def test_topk_singleton_and_all_keep_softmin_and_margin(self):
        q = np.array([0.4, -0.2])
        for top_k in (1, 2, 10, None):
            with self.subTest(top_k=top_k):
                safety, field = self.make_safety(top_k=top_k)
                result = safety.evaluate(q)
                local = self.expected_local(safety, q)
                distance, _ = QuadraticField().query(local, False)
                selected = top_indices(distance, top_k)
                expected, _ = soft_min(distance[selected], safety.options.softmin_error)
                self.assertEqual(result.values[0], expected - 0.02)
                np.testing.assert_array_equal(field.calls[1][0], local[selected])

    def test_relative_gradient_accounts_for_motion_of_both_links(self):
        q, step = np.array([0.4, -0.2]), 1e-6
        for top_k in (1, 2, None):
            with self.subTest(top_k=top_k):
                safety, _ = self.make_safety(top_k=top_k)
                analytic = safety.evaluate(q).gradients
                numeric = np.column_stack(
                    [
                        (
                            safety.evaluate(q + axis).values
                            - safety.evaluate(q - axis).values
                        )
                        / (2 * step)
                        for axis in np.eye(2) * step
                    ]
                )
                np.testing.assert_allclose(analytic, numeric, atol=1e-9, rtol=0)
                self.assertTrue(np.all(np.abs(analytic) > 1e-3))

    def test_obstacle_rows_and_exact_pose_cache_are_unchanged(self):
        obstacles = FixedObstacles()
        safety, field = self.make_safety(obstacles=obstacles)
        q = np.array([0.4, -0.2])
        result = safety.evaluate(q)
        self.assertIs(result, safety.evaluate(q.copy()))
        self.assertEqual(len(field.calls), 2)
        self.assertEqual(
            safety.last_timings, dict(distance=0.0, selection=0.0, gradient=0.0)
        )
        exact = obstacles.evaluate(q)
        self.assertEqual(result.labels[1:], exact.labels)
        np.testing.assert_array_equal(result.values[1:], exact.values)
        np.testing.assert_array_equal(result.gradients[1:], exact.gradients)

    def test_unknown_link_does_not_get_an_arbitrary_direction(self):
        with self.assertRaisesRegex(ValueError, "unknown Panda RDF link"):
            self.make_safety(source="link10")


if __name__ == "__main__":
    unittest.main()

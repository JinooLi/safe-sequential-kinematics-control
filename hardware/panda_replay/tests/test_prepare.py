import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare as p


class SplineTests(unittest.TestCase):
    def test_interpolation_and_rest_boundaries(self):
        time = np.linspace(0, 2, 21)
        q = np.sin(time[:, None] * np.arange(1, 8)[None, :])
        dt, c = p.spline(time, q)
        np.testing.assert_allclose(p.evaluate(dt, c, time), q, atol=1e-12)
        for d in [1, 2]:
            self.assertLess(abs(p.bernstein(c[:1], dt[:1], d)[0, :, 0]).max(), 1e-8)
            self.assertLess(abs(p.bernstein(c[-1:], dt[-1:], d)[0, :, -1]).max(), 1e-8)
        dt, scale = p.retime(dt, c)
        self.assertGreater(scale, 1.)
        for d, limit in [(1, p.VELOCITY), (2, p.ACCELERATION), (3, p.JERK)]:
            self.assertLessEqual(abs(p.bernstein(c, dt, d)).max(), limit)

    def test_invalid_times_or_samples(self):
        for time, q in [(np.arange(5), np.zeros((5, 7))),
                        (np.zeros(7), np.zeros((7, 7))),
                        (np.arange(7), np.full((7, 7), np.nan))]:
            with self.assertRaises(ValueError):
                p.spline(time, q)

    def test_interior_peak_bounds(self):
        c = np.zeros((1, 7, 6))
        c[..., 1:3] = [4., -4.]
        bound = p.bernstein(c, np.ones(1))
        self.assertGreaterEqual(bound.max(), 1.)
        np.testing.assert_allclose(p.evaluate(np.ones(1), c, [.5]), 1.)


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = p.Geometry(p.ROOT / 'assets/robots/panda.xml')
        cls.home = np.array([0, -.77, 0, -2.35, 0, 1.57, .78])
        cls.start = np.array([-1.57079, 0, 0, -1.57079, 0, 1.57079, -.7853])
        cls.options = dict(interval=.01, floor_z=0., floor_margin=.02, self_margin=.003)

    def test_home_is_clear_and_raised_plane_rejected(self):
        floor, own = self.geometry.check(self.home, np.zeros(7), .01, 0, .02, .003)
        self.assertGreater(floor, .02)
        self.assertGreaterEqual(own, .003)
        with self.assertRaisesRegex(ValueError, 'Floor'):
            self.geometry.check(self.home, np.zeros(7), .01, .3, .02, .003)

    def test_finger_envelope_includes_full_opening(self):
        for b in [10, 11]:
            self.assertGreater(np.ptp(self.geometry.points[b][:, 1]), .04)

    @unittest.skipUnless(os.environ.get('PANDA_REPLAY_BIN'), 'Set PANDA_REPLAY_BIN for C++ integration')
    def test_cpp_python_agreement_and_fail_closed(self):
        binary = str(Path(os.environ['PANDA_REPLAY_BIN']).resolve())
        dt, c = p.approach(self.home, self.start)
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / 'test.plan'
            p.write_plan(plan, dt, c, self.geometry, self.options)
            checked = subprocess.run([binary, '--check', str(plan)], text=True, capture_output=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            for t in np.linspace(0, sum(dt), 5):
                result = json.loads(subprocess.check_output([binary, '--sample', str(plan), str(t)], text=True))
                q = p.evaluate(dt, c, np.array([t]))[0]
                np.testing.assert_allclose(result['q'], q, atol=1e-12)
                self.geometry.data.qpos[:] = q
                p.mujoco.mj_fwdPosition(self.geometry.model, self.geometry.data)
                z = min(float((v @ self.geometry.data.xmat[b].reshape(3, 3)[2]
                               + self.geometry.data.xpos[b, 2]).min())
                        for b, v in self.geometry.points.items() if self.geometry.ancestors[b].any())
                self.assertAlmostEqual(result['floor_clearance'], z, places=10)
            # Random postures make different links, rather than the stationary
            # bottom of link1, determine the floor distance. Includes penetration.
            rng = np.random.default_rng(18)
            for _ in range(16):
                q = rng.uniform(p.LOWER + .1, p.UPPER - .1)
                stationary = np.zeros((1, 7, 6))
                stationary[0, :, 0] = q
                p.write_plan(plan, np.ones(1), stationary, self.geometry, self.options)
                result = json.loads(subprocess.check_output([binary, '--sample', str(plan), '0'], text=True))
                self.geometry.data.qpos[:] = q
                p.mujoco.mj_fwdPosition(self.geometry.model, self.geometry.data)
                z = min(float((v @ self.geometry.data.xmat[b].reshape(3, 3)[2]
                               + self.geometry.data.xpos[b, 2]).min())
                        for b, v in self.geometry.points.items() if self.geometry.ancestors[b].any())
                self.assertAlmostEqual(result['floor_clearance'], z, places=10)
            p.write_plan(plan, dt, c, self.geometry, self.options)
            # Checksum failure must happen before trying to connect to this invalid host.
            plan.write_bytes(plan.read_bytes().replace(b'PANDA_REPLAY_V1', b'PANDA_REPLAY_V2'))
            bad = subprocess.run([binary, '--preflight', str(plan), 'invalid.invalid'], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('checksum', bad.stderr)
            p.write_plan(plan, dt, c, self.geometry, {**self.options, 'floor_z': .3})
            bad = subprocess.run([binary, '--check', str(plan)], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('Floor', bad.stderr)
            # Endpoint discontinuity with a valid checksum is still rejected.
            d2, c2 = np.r_[dt, dt], np.concatenate([c, c])
            p.write_plan(plan, d2, c2, self.geometry, self.options)
            bad = subprocess.run([binary, '--check', str(plan)], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('continuous', bad.stderr)


if __name__ == '__main__':
    unittest.main()

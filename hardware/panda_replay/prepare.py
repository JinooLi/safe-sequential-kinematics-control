#!/usr/bin/env python3
"""Prepare a stopped-to-stopped Panda replay. This program never contacts a robot."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.interpolate import PPoly, make_interp_spline
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -.0175, -2.8973])
UPPER = np.array([2.8973, 1.7628, 2.8973, -.0698, 2.8973, 3.7525, 2.8973])
# Deliberately below the manufacturer's limits; these also bound the C++ reader.
VELOCITY = .15
ACCELERATION = 2.
JERK = 100.
TRACKING = .005
START_OFFSET = .0001
JOINT_MARGIN = .02


def bernstein(c, dt, derivative=0):
    """Power coefficients in normalized time -> derivative Bernstein coefficients.

    The convex hull of these coefficients bounds every point in an interval,
    including extrema between samples. Shape: segments x joints x degree.
    """
    n = c.shape[-1] - 1
    a = c[..., derivative:] * np.array([
        math.factorial(k) / math.factorial(k - derivative)
        for k in range(derivative, n + 1)
    ]) / np.asarray(dt)[:, None, None] ** derivative
    n -= derivative
    transform = np.array([[math.comb(i, k) / math.comb(n, k) if k <= i else 0.
                           for k in range(n + 1)] for i in range(n + 1)])
    return np.einsum('ijk,lk->ijl', a, transform)


def spline(time, q):
    """Interpolating quintic B-spline, with zero endpoint velocity/acceleration."""
    time = np.asarray(time, dtype=float)
    q = np.asarray(q, dtype=float)
    if (time.ndim != 1 or len(time) < 6 or q.shape != (len(time), 7)
            or not np.isfinite(time).all() or not np.isfinite(q).all()
            or (np.diff(time) <= 0).any()):
        raise ValueError('Need at least six finite, strictly increasing samples of seven joints')
    pieces = []
    for j in range(7):
        bs = make_interp_spline(time - time[0], q[:, j], k=5,
                               bc_type=([(1, 0), (2, 0)], [(1, 0), (2, 0)]))
        pp = PPoly.from_spline(bs)
        keep = np.diff(pp.x) > 0
        dt = np.diff(pp.x)[keep]
        pieces.append(pp.c[::-1, keep].T * dt[:, None] ** np.arange(6))
    c = np.stack(pieces, axis=1)
    if not np.isfinite(c).all() or dt.min() < 1e-6:
        raise ValueError('Spline is numerically ill-conditioned; check timestamps')
    return dt, c


def retime(dt, c):
    factors = [1.]
    for order, limit in [(1, VELOCITY), (2, ACCELERATION), (3, JERK)]:
        factors.append((np.max(np.abs(bernstein(c, dt, order))) / limit) ** (1 / order))
    scale = max(factors) * 1.02
    return dt * scale, scale


def approach(q0, q1):
    c = np.zeros((1, 7, 6))
    c[0, :, 0] = q0
    c[0, :, 3:] = (q1 - q0)[:, None] * [10., -15., 6.]
    dt, _ = retime(np.ones(1), c)
    return dt, c


def evaluate(dt, c, t):
    ends = np.cumsum(dt)
    t = np.asarray(t)
    idx = np.minimum(np.searchsorted(ends, t, side='right'), len(dt) - 1)
    s = np.clip((t - np.r_[0., ends[:-1]][idx]) / dt[idx], 0., 1.)
    return np.sum(c[idx] * s[..., None, None] ** np.arange(6), axis=-1)


def vertices(model, data, geom):
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        v = model.mesh_vert[start:start + count].astype(float)
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        v = np.array(list(itertools.product([-1, 1], repeat=3))) * model.geom_size[geom]
    else:
        raise ValueError(f'Unsupported robot geometry {geom}: {kind}')
    body = model.geom_bodyid[geom]
    world = v @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    return (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)


def hull(v):
    v = np.unique(np.asarray(v), axis=0)
    return v[ConvexHull(v).vertices]


class Geometry:
    """Full mesh floor envelope; MuJoCo convex collision meshes for self distance.

    Finger envelopes include the entire 0..80 mm gripper opening. Fixed mounting
    link0 is exempt from floor clearance. Adjacent rigid assemblies are excluded
    from self-distance, but no non-adjacent lab-specific exclusions are applied.
    """
    def __init__(self, path):
        path = Path(path).resolve()
        original = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(original)
        mujoco.mj_forward(original, data)
        if original.nq != 7 or original.nv != 7:
            raise ValueError('Expected fixed-finger seven-joint Panda model')
        tree = ET.parse(path)
        compiler = tree.find('compiler')
        compiler.set('meshdir', str((path.parent / compiler.get('meshdir', '')).resolve()))
        asset = tree.find('asset')
        self.points = {}
        for body in range(1, original.nbody):
            name = mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_BODY, body)
            gs = np.flatnonzero(original.geom_bodyid == body)
            v = np.concatenate([vertices(original, data, int(g)) for g in gs])
            if name in ('left_finger', 'right_finger'):
                v = np.concatenate([v, v + [0, .04, 0]])
                # Include the visual finger surfaces, not only the simplified pads.
                hv = hull(v)
                meshname = name + '_opening_envelope'
                ET.SubElement(asset, 'mesh', name=meshname,
                              vertex=' '.join(map(str, hv.ravel())))
                node = tree.find(f".//body[@name='{name}']")
                for g in list(node.findall('geom')):
                    if g.get('class') != 'visual':
                        node.remove(g)
                ET.SubElement(node, 'geom', mesh=meshname, **{'class': 'collision'})
            self.points[body] = hull(v)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding='unicode'))
        self.model.opt.ccd_tolerance = 1e-9
        self.model.opt.ccd_iterations = 200
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.nodes = []
        self.radii = {}
        self.ancestors = {}
        for b in range(1, self.model.nbody):
            m = self.model
            joint = int(m.body_jntadr[b]) if m.body_jntnum[b] else -1
            if joint >= 0 and (not np.allclose(m.jnt_axis[joint], [0, 0, 1])
                               or not np.allclose(m.jnt_pos[joint], 0)):
                raise ValueError('C++ FK requires zero-offset z-axis hinge joints')
            self.nodes.append((int(m.body_parentid[b]), joint, m.body_pos[b], m.body_quat[b]))
            r = np.zeros(7)
            radius = float(np.max(np.linalg.norm(self.points[b], axis=1)))
            ancestor = b
            while ancestor > 0:
                if m.body_jntnum[ancestor]:
                    r[int(m.body_jntadr[ancestor])] = radius
                radius += np.linalg.norm(m.body_pos[ancestor])
                ancestor = int(m.body_parentid[ancestor])
            self.radii[b] = r
            self.ancestors[b] = r > 0
        ids = np.flatnonzero(self.model.geom_group == 3)
        def group(g):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                    int(self.model.geom_bodyid[g]))
            return 'hand' if name.endswith('finger') else name
        adjacent = {frozenset((f'link{i}', f'link{i+1}')) for i in range(7)}
        adjacent.add(frozenset(('link7', 'hand')))
        self.pairs = []
        self.collision_points = {int(g): hull(vertices(self.model, self.data, int(g))) for g in ids}
        self.face_normals = {g: ConvexHull(v).equations[:, :3] for g, v in self.collision_points.items()}
        self.axes = {}
        for i, j in itertools.combinations(ids, 2):
            if group(i) == group(j) or frozenset((group(i), group(j))) in adjacent:
                continue
            a, b = (int(self.model.geom_bodyid[g]) for g in (i, j))
            # Common ancestor rotations cannot change relative distance.
            relative = (self.radii[a] + self.radii[b]) * (self.ancestors[a] ^ self.ancestors[b])
            self.pairs.append((int(i), int(j), relative, group(i) + '/' + group(j)))

    def check(self, q, velocity_bound, interval, floor_z, floor_margin, self_margin):
        self.data.qpos[:] = q
        mujoco.mj_fwdPosition(self.model, self.data)
        uncertainty = TRACKING + START_OFFSET + .5 * interval * velocity_bound
        floor = math.inf
        for body, points in self.points.items():
            if not self.ancestors[body].any():
                continue
            z = np.min(points @ self.data.xmat[body].reshape(3, 3)[2] + self.data.xpos[body, 2])
            clearance = z - floor_z - self.radii[body] @ uncertainty
            floor = min(floor, float(clearance))
        world = {g: v @ self.data.xmat[int(self.model.geom_bodyid[g])].reshape(3, 3).T
                    + self.data.xpos[int(self.model.geom_bodyid[g])]
                 for g, v in self.collision_points.items()}
        own = math.inf
        limiting_pair = ''
        for i, j, radii, name in self.pairs:
            padding = radii @ uncertainty
            a, b = world[i], world[j]
            # A separating projection gives a geometric lower bound, independently
            # of GJK convergence. Some MuJoCo 3.10 mesh queries incorrectly return
            # zero; even a nonzero distance is never accepted without this check.
            candidates = [self.axes.get((i, j), b.mean(axis=0) - a.mean(axis=0))]
            best = -math.inf
            for stage in range(3):
                for axis in candidates:
                    length = np.linalg.norm(axis)
                    if length < 1e-12:
                        continue
                    n = axis / length
                    ap, bp = a @ n, b @ n
                    gap = max(bp.min() - ap.max(), ap.min() - bp.max()) - 1e-7
                    if gap > best:
                        best = float(gap)
                        self.axes[i, j] = n
                    if best >= padding + self_margin:
                        break
                if best >= padding + self_margin:
                    break
                if stage == 0:
                    witness = np.zeros(6)
                    mujoco.mj_geomDistance(self.model, self.data, i, j, .2, witness)
                    candidates = [witness[3:] - witness[:3], b.mean(axis=0) - a.mean(axis=0), *np.eye(3)]
                elif stage == 1:
                    candidates = np.concatenate([
                        self.face_normals[g] @ self.data.xmat[int(self.model.geom_bodyid[g])].reshape(3, 3).T
                        for g in (i, j)])
            clearance = float(best - padding)
            if clearance < own:
                own, limiting_pair = clearance, name
        if floor < floor_margin:
            raise ValueError(f'Floor clearance bound {floor:.6f} < {floor_margin:.6f} m')
        if own < self_margin:
            raise ValueError(f'Self clearance bound {own:.6f} < {self_margin:.6f} m ({limiting_pair})')
        return floor, own

    def validate(self, dt, c, interval, floor_z, floor_margin, self_margin):
        v = np.max(np.abs(bernstein(c, dt, 1)), axis=(0, 2))
        samples = np.linspace(0, sum(dt), math.ceil(sum(dt) / interval) + 1)
        floor, own = math.inf, math.inf
        for t, q in zip(samples, evaluate(dt, c, samples)):
            try:
                f, s = self.check(q, v, interval, floor_z, floor_margin, self_margin)
            except ValueError as error:
                raise ValueError(f't={t:.6f}: {error}') from error
            floor, own = min(floor, f), min(own, s)
        return {'floor_clearance_lower_bound_m': floor,
                'self_clearance_lower_bound_m': own, 'geometry_samples': len(samples)}


def plan_approach(geometry, q0, q1, options):
    """Try direct joint path, then a deterministic collision-checked RRT route.

    Each edge is a stopped quintic, so smoothing cannot cut an unchecked corner.
    This is a bounded planner: it fails closed when no route is found.
    """
    def edge(a, b):
        dt, c = approach(a, b)
        geometry.validate(dt, c, **options)
        return dt, c
    try:
        dt, c = edge(q0, q1)
        return dt, c, [q0.tolist(), q1.tolist()]
    except ValueError as error:
        print(f'Direct approach rejected: {error}; searching a joint-space route.', flush=True)
    # The node/edge budgets keep an unreachable goal from running indefinitely.
    geometry.check(q1, np.zeros(7), 0., options['floor_z'], options['floor_margin'], options['self_margin'])
    rng = np.random.default_rng(20261102)
    nodes, parents = [q0], [-1]
    for attempt in range(1200):
        target = q1 if attempt % 4 == 0 else rng.uniform(LOWER + .1, UPPER - .1)
        nearest = int(np.argmin([np.linalg.norm(q - target) for q in nodes]))
        delta = target - nodes[nearest]
        candidate = nodes[nearest] + delta * min(1., .35 / max(np.linalg.norm(delta), 1e-12))
        try:
            edge(nodes[nearest], candidate)
        except ValueError:
            continue
        nodes.append(candidate)
        parents.append(nearest)
        try:
            edge(candidate, q1)
        except ValueError:
            continue
        route, index = [q1], len(nodes) - 1
        while index >= 0:
            route.append(nodes[index])
            index = parents[index]
        route.reverse()
        # Shortcut only after validating the replacement edge.
        index = 0
        while index < len(route) - 2:
            for far in range(len(route) - 1, index + 1, -1):
                try:
                    edge(route[index], route[far])
                except ValueError:
                    continue
                route[index + 1:far] = []
                break
            index += 1
        parts = [approach(a, b) for a, b in zip(route, route[1:])]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), [q.tolist() for q in route]
    raise ValueError('No approach found in 1200 attempts; no executable plan was written')


def fnv64(data):
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h


def write_plan(path, dt, c, geometry, options):
    def row(items):
        return ' '.join(format(float(x), '.17g') for x in items) + '\n'
    text = 'PANDA_REPLAY_V1\n'
    text += row([len(dt), len(geometry.nodes), options['floor_z'], options['floor_margin'],
                 options['self_margin'], TRACKING, START_OFFSET, options['interval']])
    for index, (parent, joint, pos, quat) in enumerate(geometry.nodes, start=1):
        v = geometry.points[index]
        text += row([parent, joint, len(v), *pos, *quat])
        for point in v:
            text += row(point)
    for duration, coeff in zip(dt, c):
        text += row([duration, *coeff.ravel()])
    data = text.encode('ascii')
    path.write_bytes(data + f'CHECKSUM64 {fnv64(data):016x}\n'.encode('ascii'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True, help='JSON produced by panda_replay --snapshot')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'assets/robots/panda.xml')
    parser.add_argument('--floor-z', type=float, default=0., help='Plane height in robot base coordinates, metres')
    parser.add_argument('--floor-margin', type=float, default=.02)
    parser.add_argument('--self-margin', type=float, default=.003)
    parser.add_argument('--geometry-dt', type=float, default=.01)
    args = parser.parse_args()
    if (args.output / 'replay.plan').exists():
        parser.error('Output already contains a plan; choose a new directory for each snapshot')
    if (not np.isfinite([args.floor_z, args.floor_margin, args.self_margin, args.geometry_dt]).all()
            or args.floor_margin < .01 or args.self_margin < .001 or not 0 < args.geometry_dt <= .01):
        parser.error('Require finite values, floor margin >= .01, self margin >= .001, 0 < geometry dt <= .01')
    state = json.loads(args.state.read_text())
    if state.get('robot_mode') != 'Idle' or state.get('current_errors'):
        raise ValueError('Snapshot must be Idle and error-free')
    q0 = np.asarray(state['q_d'], dtype=float)
    actual = np.asarray(state['q'], dtype=float)
    if (q0.shape != (7,) or actual.shape != (7,) or not np.isfinite(q0).all()
            or not np.isfinite(actual).all() or np.max(np.abs(q0 - actual)) > TRACKING):
        raise ValueError('Invalid snapshot or excessive measured/desired position mismatch')
    for key, limit in [('dq', .01), ('dq_d', 1e-6), ('ddq_d', 1e-5)]:
        value = np.asarray(state[key], dtype=float)
        if value.shape != (7,) or not np.isfinite(value).all() or np.max(np.abs(value)) >= limit:
            raise ValueError('Snapshot must describe a stationary robot')
    with np.load(args.trajectory, allow_pickle=False) as record:
        t, q = record['time'].copy(), record['q'].copy()
    dt, c = spline(t, q)
    dt, scale = retime(dt, c)
    geometry = Geometry(args.model)
    options = dict(interval=args.geometry_dt, floor_z=args.floor_z,
                   floor_margin=args.floor_margin, self_margin=args.self_margin)
    print('Validating the recorded spline...', flush=True)
    replay_result = geometry.validate(dt, c, **options)
    approach_dt, approach_c, waypoints = plan_approach(geometry, q0, q[0], options)
    dt, c = np.r_[approach_dt, .5, dt, .5], np.concatenate([
        approach_c, np.pad(q[0][None, :, None], ((0, 0), (0, 0), (0, 5))), c,
        np.pad(q[-1][None, :, None], ((0, 0), (0, 0), (0, 5)))])
    position = bernstein(c, dt)
    if ((position.min(axis=(0, 2)) < LOWER + JOINT_MARGIN).any()
            or (position.max(axis=(0, 2)) > UPPER - JOINT_MARGIN).any()):
        raise ValueError('Spline can exceed joint position margin')
    print('Validating the full approach + replay...', flush=True)
    result = geometry.validate(dt, c, **options)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = args.output / 'replay.plan'
    report = dict(
        source=str(args.trajectory.resolve()),
        source_sha256=hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
        snapshot=state, approach_waypoints=waypoints,
        approach_duration_s=float(sum(approach_dt)), replay_time_scale=scale,
        original_recorded_duration_s=float(t[-1] - t[0]),
        duration_s=float(sum(dt)), segments=len(dt), options=options,
        tracking_tolerance_rad=TRACKING, start_offset_budget_rad=START_OFFSET,
        maximum_velocity_bound_rad_s=np.max(abs(bernstein(c, dt, 1)), axis=(0, 2)).tolist(),
        maximum_acceleration_bound_rad_s2=np.max(abs(bernstein(c, dt, 2)), axis=(0, 2)).tolist(),
        maximum_jerk_bound_rad_s3=np.max(abs(bernstein(c, dt, 3)), axis=(0, 2)).tolist(),
        replay_validation=replay_result, full_validation=result,
        geometry='Full Panda mesh floor hulls; convex collision mesh separating-plane self bounds; finger opening 0..80 mm; base mount excluded from floor',
        limitations=['Floor is a horizontal plane in robot base coordinates; no external obstacles.',
                     'No attached object, custom tool, cables, base tilt, or unmodelled geometry.',
                     'Bounds cover reference path and stated joint error, not a certified hardware stopping envelope.',
                     'FCI safety settings are retained; offline replay does not run the original CBF controller.'])
    write_plan(plan, dt, c, geometry, options)
    report['plan_sha256'] = hashlib.sha256(plan.read_bytes()).hexdigest()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    # A numerical preview is reusable without the source NPZ or live hardware.
    preview_t = np.linspace(0, sum(dt), math.ceil(sum(dt) / .02) + 1)
    np.savez_compressed(args.output / 'preview.npz', time=preview_t, q=evaluate(dt, c, preview_t))
    print(json.dumps({k: report[k] for k in ['approach_duration_s', 'replay_time_scale', 'duration_s', 'full_validation']}, indent=2))


if __name__ == '__main__':
    main()

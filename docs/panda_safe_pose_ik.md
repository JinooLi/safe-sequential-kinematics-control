# Panda safe pose-IK and trajectory controller

## Pipeline

$$
q_0 \rightarrow \text{safe pose-IK search}
\rightarrow q^*_{safe} \rightarrow \text{CLF-CBF trajectory}
$$

The target orientation is specified in YAML using two non-normalized,
base/world-frame vectors: `orientation.approach` is the desired direction of
the `gripper_tip` local +z axis, and `orientation.up` is the desired direction
of its local +y axis. The up vector is projected orthogonally to approach, so
the two vectors must not be parallel. They form a right-handed rotation matrix
whose columns are the desired local +x, +y, and +z axes; this is converted
internally to a scalar-first quaternion `[w, x, y, z]`. Position, orientation,
capsule endpoints, obstacles, and Jacobians are represented in the base/world
frame.

## Safe pose-IK

The orientation error is the vector part of the relative quaternion specified
in `clarification.md`:

$$
e_O = \eta_e\varepsilon_d-\eta_d\varepsilon_e
-\varepsilon_d\times\varepsilon_e.
$$

The primary inverse is damped while the null projector uses the Moore--Penrose
inverse:

$$
\dot q = J^\#_\lambda k_p
\begin{bmatrix}p_d-p\\2e_O\end{bmatrix}
+\left(I-J^+J\right)k_s\nabla_q\Phi(q).
$$

All obstacle and enabled self-collision barriers are combined by

$$
\Phi(q)=-\tau\log\sum_i\exp\left(-h_i(q)/\tau\right).
$$

The search intentionally has no velocity limit and no hard CBF. Each random
seed gets its own integration and time budget. A candidate is accepted only
when both pose tolerances are met and every margin-adjusted barrier satisfies
$h_i\ge0$. A candidate is accepted only after all these conditions remain
continuously satisfied for `safe_ik.convergence_hold_duration`. If any
condition fails during the hold, its timer resets. The default hold is 3
seconds. Completion ends only the current seed; every configured seed is still
evaluated. After all seeds finish, the safe candidate with the smallest joint
angle distance from the initial configuration is selected. Seed 0 starts at
the initial configuration itself; the remaining seeds are sampled uniformly
inside the Panda joint limits.

## Parallel seed search

```yaml
safe_ik:
  max_restarts: 10
  workers: 4  # 1: serial; >1: independent spawned processes
```

`application/parallel_ik.py` owns the process pool. Each worker constructs its own
Pinocchio model/Data/cache and capsule safety evaluator once and reuses them for
assigned seeds. Live kinematics, MuJoCo state and the viewer are never serialized.
The `spawn` start method avoids inheriting the main process's native/viewer threads.
The pool uses at most `min(workers, max_restarts)` processes and closes before
trajectory control. If safe IK finds no candidate, the same pool evaluates the
pose-only diagnostic search. Worker errors propagate rather than being classified
as an unreachable pose. Direct mode skips both IK and pool creation.

`domain/safe_pose_ik.py` keeps the numerical iteration in `solve_seed()`. The parent
generates every starting configuration in the original RNG order: seed 0 is the
requested initial configuration, followed by uniform joint-limit samples. Results
are sorted by seed index before nearest-candidate selection, preserving the serial
tie rule regardless of scheduling. Hold duration and per-seed iteration budget are
unchanged. A seed's internal iterations remain sequential.

Missing `workers` defaults to 1 for existing YAML files; the editable default config
explicitly selects 4. `workers` must be a positive integer. Applied worker settings
are included in archived YAML and resolved configuration. `safe_ik.wall_time`
includes worker startup during the safe search but, as before, excludes the optional
pose-only diagnostic pass; pool shutdown is also outside that metric.

Use a worker count appropriate to the available CPU cores. Short searches may be
slower because of process startup. If a BLAS runtime starts many native threads per
worker, limiting `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `OMP_NUM_THREADS` to 1
before starting the command can avoid oversubscription. Exact speedup is machine-
and workload-dependent; mathematical equivalence is checked with numerical tolerance,
not promised as bitwise identity across different machines or BLAS implementations.

Python scripts that start parallel search must call it under
`if __name__ == "__main__":`. The provided CLI already has this guard.
Custom injected runtimes can supply `ik_search_factory`; without it the domain solver
runs locally, preserving support for in-memory test backends.

On the development machine (2026-09-23), the current 10-seed configuration took
34.38 s serially and 12.97 s with 4 workers (about 2.65x faster), including pool
startup/shutdown in this benchmark. Both runs found and selected seed 4; candidate
iterations matched and joint values agreed within 1e-11 absolute tolerance. This
is one workload measurement, not a guaranteed speedup.

## Collision model

`PandaCapsuleSafety` evaluates sphere--capsule obstacle barriers and all
non-adjacent capsule pairs. Adjacent chain members and configured fixed-body
exclusions are omitted. Capsule endpoints are frame-local values derived from
the checked-in collision meshes. The controller depends on the safety model's
barrier/gradient interface so a learned distance field can replace it later.

## Trajectory controller

In sequential mode, the selected safe IK configuration supplies each link
endpoint target. For every configured intermediate frame $i$,

$$
V_i(q)=\frac12\lVert p_i(q)-p_i(q^*)\rVert^2.
$$

After each endpoint reaches the configured stage tolerance, control advances
to the next frame. The final stage uses the requested end-effector pose:

$$
V_{pose}(q)=\frac12\lVert p(q)-p_d\rVert^2
+\frac{w_O}{2}\lVert e_O(q)\rVert^2.
$$

The orientation weight $w_O$ is `trajectory.orientation_weight` in YAML.
Direct mode skips the endpoint stages and uses this same pose CLF from the
start. It also skips safe pose-IK search and validation entirely, because its
CLF is defined directly from the requested pose. Consequently, direct mode has
no IK-candidate fallback. Panda trajectory control never uses a
configuration-space CLF.

For the quaternion error scalar/vector pair $(\eta,e_O)$ and the world angular
Jacobian $J_\omega$, the final CLF gradient is

$$
\nabla_q V_{pose}=(p-p_d)^T J_v
-\frac{w_O}{2}\eta e_O^T J_\omega.
$$

The QP applies

$$
\nabla V\dot q-\delta\le-cV,
\qquad
\nabla h_i\dot q\ge-\alpha h_i.
$$

Only the CLF is relaxed. CBF, joint-position, velocity, and command-rate
(acceleration) constraints are hard. QP infeasibility commands zero velocity.
If the moving average of $\lVert\dot q\rVert_2$ over the configured window is
below the stop threshold before success, the next safe IK candidate is used.

## Failure meanings

- `pose_ik_unreachable`: obstacle-free diagnostic search did not reach the
  requested pose within the configured multi-start budget.
- `safe_pose_ik_not_found`: pose IK was found, but no candidate passed every
  safety barrier.
- QP infeasible: safe IK exists, but hard trajectory constraints cannot be
  satisfied at the current state; the controller stops.

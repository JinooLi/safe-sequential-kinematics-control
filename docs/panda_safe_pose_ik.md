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

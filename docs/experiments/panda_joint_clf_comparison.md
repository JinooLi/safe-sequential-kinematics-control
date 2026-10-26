# Fixed-IK joint CLF versus Direct and Sequential

The joint-space baseline uses the saved, validated IK configuration `q_star`
directly:

```text
V(q) = (q - q_star)^T (q - q_star)
dV/dq = 2 (q - q_star)
2 (q - q_star)^T u <= -clf_rate * V(q) + delta
```

There is no factor of 1/2 and no angle wrapping across Panda's physical joint
limits. All other QP rows and objective weights are shared with the existing
controller: obstacle/RDF/joint-limit CBFs, velocity and acceleration bounds,
nonnegative CLF slack, and velocity/slack penalties. `trajectory.mode: joint_clf`
enables the new CLF; its stage name is `joint_configuration` and it has no
intermediate link tasks. Existing Direct and Sequential formulas are unchanged.

The paired comparison command is:

```bash
PYTHONPATH=src .venv/bin/python -m safe_ik.application.panda_joint_clf_comparison \
  --reference-dir /absolute/path/to/saved/link4_link5 \
  --experiment-config configs/experiments/panda_comparison.yaml \
  --count 10 --output /absolute/path/to/new/results
```

It takes the first ten scenes in recorded order, without filtering by outcomes,
and verifies hashes of the saved scenes, configuration and IK records. It never
generates replacement scenes or runs a new IK search. Every selected IK is
revalidated against the current pose tolerances, joint limits and barriers.
It reruns all three methods under the current common settings. Sequential and
joint CLF receive the same single nearest validated IK, without switching;
Direct does not use IK for control. Missing IK stays in the denominator and
counts as failure for both IK-dependent methods. Method execution order rotates
by scene. Output directories must be new; existing results are never overwritten.

To repeat only the joint-CLF method with a different slack weight, append
`--methods joint_clf --joint-clf-slack-weight 1000` and use a new output directory.
The override is applied only to joint CLF, even when other methods are selected.
The common `config.yaml` snapshot retains the base settings; `manifest.json`
records each effective configuration under `method_configs`, and every summary
row records its actual `clf_slack_weight`. Neither the solver settings nor the
CLF/CBF constraint rows change when only this weight is overridden.

The main success definition is the same for all three: first EE pose hit within
the configured position/orientation tolerances, with barrier and joint safety
throughout the sampled trajectory. The hard joint stops and joint-limit CBFs
remain enabled. The comparison stops at this EE hit, so success does not require
an additional, stricter joint-distance tolerance. Each executed trajectory also
records `reference_q_star` and `reference_joint_error`; summaries include the
final `||q-q_star||`, even for Direct, which ignores that reference in control.
Joint-CLF traces are independently checked against squared joint distance.
The existing applied-velocity plus normalized EE/stage progress stopping rule
is retained. Interpolated safety results are reported separately, as before.

Keep interpretation of the gains explicit: joint CLF is measured in squared
radians, while the existing pose CLF mixes squared position and weighted
quaternion-vector error and includes a factor of 1/2. Reusing numerical
`clf_rate` and slack weight is the requested baseline, not a normalization that
makes the tradeoff identical between these CLFs. This comparison changes both
the target-space objective and its numerical scale. Ten scenes are a pilot,
not evidence of a general success-rate advantage.

## Initial ten-scene campaign (2026-10-19)

Server: `lab-server`.
Repository: `/home/cdsl/jinoo_ws/safe-inv-clbf-robot-arm-panda-gpu`.
Reference: `outputs/panda_link_pairs_slack5000_300/20261010T122425_767542Z/link4_link5`.
Cases: `seed2332_0000` through `seed2332_0009`.
Campaign: `outputs/panda_joint_clf/20261019T081303Z`.

Completed at 2026-10-19 08:29:31 UTC (17:29:31 KST), in 979.30 s wall time.
All 30 trajectories, source/resource/input/result hashes, zero IK switches,
joint-CLF values, and common scoring invariants passed the final verification.
The ordinary server checkout's source and configuration hashes stayed unchanged.

| Method | Success /10 | Termination reasons | Mean successful completion (s) |
|---|---:|---|---:|
| Direct | 6 | 6 goal, 4 timeout | 23.42 |
| Sequential | 7 | 7 goal, 2 timeout, 1 QP infeasible status | 42.28 |
| Joint CLF | 0 | 10 initial QP solver failures, before any motion | — |

Direct and Sequential both succeeded in five scenes; Sequential alone in two,
Direct alone in one, and neither in two. The time means condition on different
success subsets and are not paired speed comparisons. Two unsuccessful
Sequential trajectories also violated the 1 mm barrier tolerance. All joint
margin minima were nonnegative. Joint CLF had eight `iteration_limit` statuses
and two `failed` statuses at t=0; frozen initial-QP diagnostics reproduce
`solved inaccurate` for those latter two, which the existing solver rejects.
These results do not establish joint-CLF tracking performance, because it never
produced an accepted motion command in this pilot.

The local copy of the report and per-case CSV is under
`outputs/panda_joint_clf/20261019T081303Z/comparison/`; initial-QP diagnostic
matrices and reports are alongside it in `diagnostics/`. Full trajectories,
snapshots, manifests, and completion hashes remain in the server campaign's
`results/` directory. No additional rollout was launched after the ten scenes.

Conditions: duration 150 s, dt 0.02 s, CLF rate 20, slack weight 5000,
alpha 0.2, RDF CBF gain 1, Sequential link4 then link5 in pose mode,
EE position tolerance 1 cm, quaternion-vector tolerance 0.01, barrier success
tolerance 1 mm, hard joint stops,
and five subdivisions for the existing separate interpolation validation.
The archived runtime is a copy of the server's current source with only the new
CLF/parser/CLI changes and comparison runner applied. Local dataset-generation
and neural-network changes are not included in this controller comparison.

The local controller/configuration/comparison tests pass: 68 tests, including
finite-difference joint-CLF derivatives, unchanged non-CLF QP rows, fixed-IK trace
validation, missing-IK accounting, and an end-to-end three-method export test.
Server preflight and run status are saved in the campaign. No new scheduler is
created; UBAI data generation and existing checkpoints are unaffected.

The first preparation attempt (`20261019T081022Z`) stopped before any rollout
because a check assumed the older 0.5 mm position tolerance. Inspection verified
that both the current server configuration and this 300-scene reference use
1 cm / 0.01. The actual pilot preserves those current tolerances for all three
methods. That preparation log is retained; no scientific parameter was changed
to bypass the check. Server tests pass (67 tests with its archived baseline).

### Initial QP numerical diagnosis

The first joint-CLF rollouts stopped before moving because OSQP reached its
iteration limit. Independently reconstructed initial QPs for all ten scenes
admit the explicit feasible point `u=0, delta=20*V` with zero constraint
violation. An initial solver failure is therefore not evidence of mathematical
infeasibility. Joint-CLF initial values range from 1.340 to 20.379, compared with
0.302 to 1.281 for the Direct pose CLF; the associated slack scale differs.

Frozen-QP diagnostics, saved separately under the campaign's `diagnostics/`,
do not alter or rerun any trajectory. Uniform division of the whole QP objective
by 5000 preserves its mathematical minimizer. With `eps_abs=eps_rel=1e-8` and
polishing, this produced accurate solved initial QPs in eight of ten scenes
within 20,000 iterations; the other two still reached the iteration limit.
This is not a trajectory success count or a validated replacement solver.
Scaling alone at the original tolerances accepted a solution for scene 0000
with maximum constraint violation 0.05277, so a `solved` status alone does not
establish constraint satisfaction. The original pilot's solver settings and
all controller parameters remain unchanged.

## Joint-CLF slack 1000 follow-up (2026-10-19)

Campaign: `outputs/panda_joint_clf_slack1000/20261019T083910Z` on lab-server.
The same ten scenes and fixed IKs are reused. Only joint CLF is rerun, with
slack weight 1000. The already completed Direct/Sequential and joint-CLF-5000
results above are retained as comparison baselines. All other controller,
solver, duration, tolerance and safety settings are unchanged. The run uses
an isolated copy of the previous campaign source with only the comparison
runner updated for method selection and the method-specific weight override.
The launch and exact settings are recorded in `campaign.json`, `preflight.json`
and the result manifest. Local targeted tests passed (39); the same selection
passed on the server's archived baseline (38).

The follow-up completed at 2026-10-19 08:42:22 UTC (17:42:22 KST), taking
136.14 s wall time. All ten result hashes and traces passed independent
verification, including the exact one-parameter configuration difference.

| Joint CLF slack weight | Success /10 | Initial QP solver failure | Timeout |
|---|---:|---:|---:|
| 5000 (previous run) | 0 | 10 | 0 |
| 1000 (new run) | 3 | 6 | 1 |

Successful scenes were `seed2332_0000` (33.32 s), `seed2332_0001` (12.90 s), and
`seed2332_0006` (4.16 s). All three were safe under the common scoring criterion.
Scene `seed2332_0004` timed out at 150 s and also violated the barrier tolerance:
minimum barrier -0.0089936 m versus the allowed -0.001 m. All joint margin minima
were nonnegative. Six remaining scenes stopped at t=0: one `iteration_limit`
and five `failed` statuses; their traces do not store a more detailed solver
message. No solver setting or safety constraint was changed to obtain these
results.

The 40-row combined CSV contains the ten new runs plus the thirty earlier
baselines and identifies reuse explicitly in `executed_this_run` and `source_run`.
It and the report are copied locally under
`outputs/panda_joint_clf_slack1000/20261019T083910Z/comparison/`. Direct and
Sequential retain slack weight 5000 and their previous 6/10 and 7/10 successes;
this follow-up changes the joint method's objective weight as well as the CLF
relative to those methods, so it does not isolate the CLF definition alone.

## Joint-CLF slack 100 follow-up (2026-10-19)

Campaign: `outputs/panda_joint_clf_slack100/20261019T084856Z` on lab-server.
The same ten scenes and fixed IKs are rerun for joint CLF only, using
`--methods joint_clf --joint-clf-slack-weight 100`. The source hashes match
the slack-1000 campaign exactly. Only the effective joint-CLF slack weight
changes; solver settings, safety constraints, duration, tolerances and the
candidate policy remain identical. Previous results are preserved. Preflight
reuses the archived 38-test pass for this unchanged code and records its hash.

The slack-100 run exposed a numerical validity issue in the existing solver
wrapper: scene `seed2332_0003` first commanded joint4 at -0.271695 rad/s despite
the configured speed limit of ±0.145 rad/s. At t=0.02, the acceleration/velocity
box reconstructed from that command has lower=-0.145 and upper=-0.188361, so it
is inconsistent. OSQP logged `ERROR in osqp_update_data_vec: Problem data
validation.`, but the wrapper continued and the trace recorded `solved`.
The original goal/barrier/joint-position score marked this trajectory successful;
that score must not be interpreted as certification of all QP constraints.
No solver change is made during this requested weight-only run.

Checking earlier records also found velocity and acceleration excesses in the
three slack-1000 trajectories previously called successful. Their 3/10 result
remains the original goal/barrier/joint-position score, rather than a claim of
complete velocity/acceleration constraint satisfaction. The slack-100 comparison
adds a separate audit of these limits and rejected updates for all three weights,
while preserving the original data and success labels.

The slack-100 follow-up completed at 2026-10-19 08:56:51 UTC (17:56:51 KST),
in 471.10 s wall time. All ten scenes reached a normal rollout termination:
six EE goal hits and four timeouts. Five satisfy the original
goal/barrier/joint-position score; one goal hit (0000) violates the barrier
tolerance, with a minimum of -4.360 mm. The four timeouts (0002, 0004, 0005,
0008) also violate that tolerance. Joint-position margins remain nonnegative.

| Joint CLF slack weight | Original success /10 | EE goal hits /10 | Initial QP failure | Timeout |
|---|---:|---:|---:|---:|
| 5000 | 0 | 0 | 10 | 0 |
| 1000 | 3 | 3 | 6 | 1 |
| 100 | 5 | 6 | 0 | 4 |

All ten weight-100 trajectories, including all five scored successes, exceed
both velocity and acceleration limits at some point. Among those five,
maximum applied speed excess ranges from 0.004570 to 0.126695 rad/s, and
maximum applied acceleration excess ranges from 0.309521 to 9.418062 rad/s².
These are independently reconstructed from saved joint states. Thus the 5/10
score does not establish successful tracking under all configured constraints.
Zero QP-triggered terminations also does not mean zero solver errors: scene
0003 logged one rejected QP data update as described above.

The final report, 30-row per-case comparison CSV, and separate QP-bound audit
for all three weights are copied locally under
`outputs/panda_joint_clf_slack100/20261019T084856Z/comparison/`.
Input/source/result hashes and the one-parameter configuration difference were
verified; derived report hashes include the additional audit. Raw trajectories
and labels are preserved. No solver fix or further experiment was performed.

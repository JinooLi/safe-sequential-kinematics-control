# QP failure handling

The OSQP adapter rejects a command before Panda advances its simulated state if
the input is malformed, an OSQP update fails, or a purportedly solved result
violates the supplied constraints beyond numerical tolerance. This applies to
Direct, Sequential and any other controller using this adapter.

## Input and update validation

Before setup or reuse, the adapter checks dimensions, finite P/q/A values,
symmetric P, valid finite/infinite bound directions, and `lower <= upper`.
An inconsistent velocity/acceleration interval is an `invalid_qp_bounds`
solver failure, not an accepted solution of the previous problem.

OSQP 1.1.3's public `update()` discards the integer error codes from its native
vector and matrix updates. The adapter calls those two native methods and
checks both codes, preserving workspace reuse for valid updates. A nonzero code,
an exception, or an unavailable checked-update interface fails closed. After
failure the workspace is discarded for the next call. Inaccurate and other
non-solved statuses remain rejected under the previous status policy.

## Independent constraint check

For each finite side `b` of each constraint row `a`, the adapter recomputes
`a @ x` in the original problem units and checks the one-sided violation against

```text
eps_abs + eps_rel * max(abs(a @ x), abs(b))
```

The existing configured OSQP tolerances are used. There is no global scale
shared by different rows: a large CLF slack cannot enlarge the acceptable
residual of a velocity or CBF constraint. This is a numerical acceptance check,
not a promise of exactly zero residual, exact continuous-time safety, or optimality.
It does not change the acceptance tolerances, CLF/CBF rows, relative objective
weights, or experiment success tolerances. It never clips or applies a rejected
command. The numerical solve can perform the bounded refinement described below.

## Numerical accuracy and bounded refinement

OSQP's relative termination criterion uses a global infinity norm of `A @ x`
and its projected constraint vector. Large CLF values/slack can therefore allow
a `solved` vector with excessive velocity/CBF row residuals. In the October 20
3,000-scene campaign, the independent check correctly rejected such vectors;
29,991 of 30,000 rollouts ended with `qp_constraint_violation`.

The adapter now uses the following numerical treatment:

- Divide **both** `P` and the linear objective vector `q` by
  `max(1, max(abs(P)))`, including linear-vector updates during workspace reuse.
  This common positive multiplier preserves the exact minimizer and relative
  slack/velocity weights. It does not reduce the configured slack penalty.
- Enable OSQP polishing and check termination in original constraint units.
  Set the adaptive-rho interval to 50 iterations instead of a timing-dependent
  automatic interval.
- First solve with the configured absolute and relative tolerances. A finite
  `solved` vector must pass the unchanged independent per-row check.
- If that check fails and a positive absolute tolerance and iteration budget
  remain, continue **the same QP**, from its current iterates, with relative
  termination tolerance zero. Only one refinement is allowed. The original
  solve's iterations are subtracted from `max_iterations`; the total budget is
  not increased. Configured settings are restored for the next control call.
- Check the refined vector again. An inaccurate status, exhausted iteration
  budget, invalid vector or remaining constraint violation still returns no
  command. Malformed inputs and rejected data updates are never retried.

This is refinement within one control solve, with no integration, scene change,
candidate change or simulated time advance between attempts. Successful
refinements are identified in the diagnostic message. Numerical acceptance
remains tolerance-based, not exact feasibility or a hardware timing guarantee.

See [OSQP's convergence and polishing definitions](https://osqp.org/docs/solver/index.html)
for the distinction between global solver termination and local row checks.

Rejected results have `QPStatus.FAILED`, no solution vector, and a diagnostic
message identifying the input/update failure or offending row, bound, residual
and tolerance. Panda stops the rollout before integration. Comparisons retain
`reason: qp_solver_failure` and `success: false`, with `qp_failure_message` in
the summary and per-command `qp_message` in the NPZ. Single rollouts retain
`failure.detail` and the same NPZ messages. A failed attempt is retained, rather
than silently removed or retried with a different scene/IK.

These changes affect new executions only. Archived source, completed results,
and old success labels remain intact. An in-progress dataset campaign that
uses a mutable source checkout must finish with its original source identity;
deploy a new version into a separate checkout until that job has stopped.

Regression coverage includes rejected native updates, stale-workspace
prevention, malformed bounds/data, misleading `solved` results, nonfinite
solutions, independent per-row tolerances, valid workspace reuse, and Panda
termination/diagnostic export without applying a rejected command.
Additional coverage exercises an analytically solved large-slack QP, objective
normalization on setup/update/rebuild, bounded refinement, settings restoration,
and rejection of inaccurate or still-violating refined vectors.

## Validation on the 3,000-scene campaign inputs (2026-10-21)

On lab-server, 30 evenly spaced saved scenes supplied 120 initial QPs (four
possible first links per scene). The final adapter solves all 120 and three
additional QPs captured at intermediate failures. Each solution passes the
unchanged row check. The robot/RDF assets and resolved scientific configuration
match the UBAI campaign; the lab baseline and corrected source snapshots differ
only in `src/safe_ik/adapters/osqp_solver.py`.

Six preselected scenes, each with (2,3) and (4,5), were also rerun for the full
120-second simulated limit using the same initial configurations and fixed IK:

| Termination | Baseline adapter | Corrected adapter |
| --- | ---: | ---: |
| QP solver failure | 12 | 0 |
| Successful goal reach | 0 | 4 |
| Timeout | 0 | 6 |
| Stalled | 0 | 2 |

All 53,507 accepted QPs in the corrected rollouts passed a separate residual
audit. The largest raw one-sided residual was 0.0015882682 in its row's units;
every residual stayed below its original per-row absolute-plus-relative
tolerance. This is not a zero-residual assertion. The 73 relevant regression
tests pass (15 solver-contract tests and 58 controller/rollout/dataset/planar
tests). The small selected set validates the numerical fix, not population
success rates.

Evidence is in lab-server's
`outputs/panda_qp_accuracy/20261021T023508Z/`, especially `verification.json`,
`comparison.csv`, and the hashed `complete.json`. Old trajectories and training
labels are preserved. Training remains on hold; the 3,000-scene dataset has not
been regenerated or retrospectively relabeled.

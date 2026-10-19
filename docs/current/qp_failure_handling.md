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
It does not change the solver tolerances, objective, CLF/CBF rows, or experiment
success tolerances. It does not clip, project, retry or apply a rejected command.

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

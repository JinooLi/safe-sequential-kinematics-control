# Re-select links after intermediate target reaches

This opt-in replay evaluates the existing 16-action selector repeatedly along
one trajectory. It does not retrain the selector or search for another IK.

At the initial state and after each successful intermediate stage:

1. Infer the existing action probabilities using current `q`, the original
   selected `q*`, final EE pose, and unchanged obstacles.
2. Take the highest-scoring action from the unchanged catalog (Direct, five
   single links, ten link pairs over `link2`–`link6`).
3. Discard links whose numbers are no greater than the last completed link.
4. Track only the smallest remaining link toward its frame at `q*`.
5. If the chosen action is Direct or no links remain, track the final EE pose.
   This final stage is absorbing.

Reaching requires both existing position and orientation thresholds in pose
mode. Already-reached stages can advance at the same simulation timestamp.
Link numbers strictly increase, bounding intermediate reaches at five. A stall
still ends the fixed-IK rollout; it does not trigger this reselection policy.
The total scene budget is not reset on stage changes.

`PandaTrajectoryController(..., stage_selector=callback)` enables the policy.
The callback receives a copy of the current joint state and returns selected
frame names; an empty sequence means Direct. The controller requires one fixed
IK and an ascending stage catalog. Omitting the callback preserves fixed
Sequential and Direct behavior. `consume_stage_selection_events()` exposes the
input state, action, discarded links, last completed link, and next target.

The replay entry point accepts the exact saved dataset and split:

```bash
PYTHONPATH=src .venv/bin/python -m safe_ik.application.panda_reselect_reached \
  --dataset /absolute/path/to/export \
  --contexts /absolute/path/to/export/contexts.jsonl \
  --split /absolute/path/to/split.json \
  --checkpoint /absolute/path/to/best.pt \
  --experiment configs/experiments/panda_link_dataset.yaml \
  --output /absolute/path/to/new-output \
  --gpus 0 --workers-per-gpu 4 --mode adaptive
```

Use `--mode static --indices 0,1,...` for a matched reproduction pilot. Explicit
indices are positions in the selected saved partition, not source dataset
indices. `--partition test` is the default; `--partition validation` uses the
existing validation IDs and checks disjointness from train and test. Results
record `partition` and `partition_position`. The retained `test_position` field
is a legacy alias of `partition_position`, including validation replays.
Existing output directories are refused. Inputs, source, model resources and
trajectories are hashed. Numerical QP failures are controller failures; runner
or input errors are reported separately and invalidate a clean completion.

The November 24 experiment uses the validation-selected basic model seed 53
(BCE + selection loss with lambda 1, dropout 0.2), the original 1,000 test scenes,
and the production dataset's 120-second configuration. Original static choices
give 581/1,000 successes. Static replay checks precede the new policy, and the
new policy covers each test scene once. Decisions and probabilities are stored
alongside the full trajectories for auditing.

This is an evaluation on a previously inspected test split. The network was
trained on initial-state contexts, so its behavior after a stage is measured
empirically here; its probabilities at those states are not assumed calibrated.

## Collision-constrained intermediate stall advancement

`--mode collision_stall_skip` is a separate, opt-in fixed-route policy. The
model runs once at the initial state and chooses the original Direct, single
link or two-link action. Normal reaches retain this selected route; they never
trigger another model inference. Initial Direct is unchanged.

At an intermediate-stage stall under the existing applied-velocity and progress
criteria, the controller solves that state's existing QP using a **separate
OSQP instance** with unchanged tolerances. The main control solver's warm start
is preserved. A collision CBF is active when its absolute constraint residual
is within the QP tolerance and its lower-bound multiplier is strictly greater
than `1e-6`. Obstacle and self-collision CBFs qualify; joint or velocity limits
alone do not. This identifies an active constraint during a stall, without
claiming that it is the sole cause of that stall.

If the diagnostic solve is valid and at least one collision CBF is active, the
controller skips to the next originally selected link or to the final EE task.
A failed or inactive diagnostic preserves the original stalled termination.
The final EE stage cannot be skipped. The original IK, total simulation budget,
safety thresholds and integration step remain unchanged. No time budget is
reset. A transition can immediately reach an already-satisfied next target in
the same integration step.

Use `--partition validation --mode collision_stall_skip` for the validation
experiment, with the same immutable inputs and 120-second experiment YAML as
the paired `--mode static` run. `collision_stall_events.json` records every
gate check, including rejected checks, its state and command index, simulation
time, stall statistics, diagnostic result, collision residuals and multipliers.
The runner verifies exact state/time identity and the correspondence of accepted
checks to `collision_constrained_stall` transitions. Summaries distinguish gate
checks, skipped stages, diagnostic failures and the number of affected scenes.

## Periodic sequence proposals with consecutive confirmation

`--mode periodic --sequence-period 0.1 --sequence-confirmations 5` enables a
separate policy. The existing network runs once at the initial state and its
argmax selects the initial complete route. During control it runs again at
simulation times `T_seq, 2*T_seq, ...`. The period must be an integer multiple
of the unchanged integration step. Polling is independent of stage reaches and
continues during final EE control, including when the initial action is Direct.
Goal or timeout states that terminate before a control command do not poll.

Each poll uses current `q`, the same saved IK, final EE pose and unchanged
obstacles. The argmax is compared as a **complete original action**, not as its
next link or remaining suffix. An action equal to the active sequence clears
the pending candidate and streak. A different action starts a streak of one;
the same pending action increments that streak. After five consecutive
proposals the controller accepts the new complete route and clears the streak.
Normal stage reaches do not reset a pending streak or alter the active action's
identity. There is no additional neural inference at a stage reach.

An accepted route starts at its first selected link, even if that link was
previously visited. Already-satisfied stages advance immediately under the
existing position/orientation reach criterion. An empty route selects final EE
control, which can subsequently return to intermediate links. This policy does
not combine reached-stage neural reselection or collision-stall skipping.
Existing stall and QP-failure termination remain in effect. A scheduled poll
occurs before the same command state's stall check; an accepted route change
resets the task's progress window, but never the episode clock or 120-second
budget.

Internally, traces retain the stable `link2`–`link6` stage catalog and one saved
IK, even for initially empty routes. This makes later link control possible and
keeps every recorded stage index interpretable; it does not execute unselected
links. No IK search or IK switching occurs.

For the comparison, use the same frozen validation 1,000 IDs and checkpoint for
each period `0.1`, `0.2`, and `0.4` seconds, with five confirmations. Preserve the
original total time, safety settings and fixed IK. Static replay provides the
paired original-policy check; do not compare these validation outcomes with
the separate test partition.

`periodic_selection_events.json` contains one merged record per poll: current
state, command index and simulation time, sixteen probabilities and argmax,
active sequence before/after, pending candidate and streak before/after,
acceptance, and the switch's entry/next stage. An already-reached entry stage
can differ from the next active stage. The runner checks every scheduled poll,
exact state identity, probability argmax, confirmation evidence and recorded
`sequence_switch` transition. `selection_events.json` points to this detailed
file with its hash. Case/phase summaries contain counts and hashes rather than
duplicating large event arrays. Initial inference, subsequent poll count,
accepted switches, affected scenes, and inference wall time are reported
separately; inference wall time is not simulation time or batch elapsed time.

### Limiting accepted sequence changes

`--sequence-max-switches 3` optionally limits a periodic rollout to three
accepted **complete-route changes**. Omitting it leaves the original policy
unlimited. A value of zero keeps the initial route throughout, while still
running and recording every scheduled inference. This option is rejected for
non-periodic policies.

The initial selection and normal reached-stage advances do not count toward
the limit. Every accepted change does count, including changes to/from the
empty EE route and different complete routes that enter the same first link.
After the limit is exhausted, polling and candidate/streak updates continue
unchanged. A proposal with a streak of at least `N_seq` is recorded as blocked
and cannot change the route; its streak may subsequently exceed `N_seq`.
Returning to the current route still clears the streak, and a different
candidate still starts a streak of one. The episode clock, stopping criteria,
and all control and safety settings are unchanged.

Poll records add `max_switches`, `switch_count_before`, `switch_count_after`,
and `blocked_by_switch_limit`. Summaries distinguish accepted switches,
confirmed proposals blocked by the cap, and whether the cap was exhausted.
Blocked-proposal counts count polls, not unique candidate routes. Repeated
confirmed proposals can therefore contribute multiple blocked counts.

The next matched validation comparison uses `(T_seq, N_seq)` of `(0.2, 9)`
and `(0.4, 3)` without a cap, plus `(0.4, 5)` with a cap of three. The latter
can be invoked with:

```bash
--mode periodic --partition validation \
--sequence-period 0.4 --sequence-confirmations 5 --sequence-max-switches 3
```

### Making the first EE entry permanent

`--sequence-lock-on-ee` is an independent, optional periodic-mode flag. The
initial action remains the existing model's argmax. On the first entry into
the final EE stage, the controller permanently stops periodic inference and
route changes. Initial Direct locks at time zero. A normal intermediate reach
locks before a poll scheduled for that same command state; accepting Direct or
a route whose links are already satisfied locks immediately after that accepted
change. The triggering acceptance remains a real poll in the event log.

The flag does not add a switch cap. Before EE entry, the original confirmation
rule can accept any number of route changes unless `--sequence-max-switches`
is also supplied. After entry, ordinary EE control, safety checks and stopping
criteria continue within the same total scene budget. Omitting the flag keeps
the original policy, including switches from EE back to intermediate links.

Case summaries identify `initial_direct`, `periodic_ee_locked`, lock time,
command index and reason. One `periodic_ee_lock_event` records `initial_direct`,
`reached_ee`, or `accepted_sequence_ee`; the event is also referenced in the
selection evidence file. There are no synthetic post-lock poll records. The
runner verifies the first EE entry against transitions and trace stage indices,
complete polling before the lock, and continuous EE control after the lock.

The three independent follow-up validation runs use the same frozen 1,000
validation scenes and 120-second budget:

- `--sequence-period 0.3 --sequence-confirmations 12 --sequence-max-switches 3`
- `--sequence-period 0.3 --sequence-confirmations 12 --sequence-lock-on-ee`
- `--sequence-period 0.3 --sequence-confirmations 15`

All three use `--mode periodic --partition validation`. Each variant starts
from the same initial network choice and original scene, rather than continuing
another variant's trajectory.

## Spatial Gaussian voting

`--mode spatial --spatial-config configs/experiments/panda_spatial_reselection.yaml`
enables a separate policy using the same replay entry point and saved inputs.
It replaces temporal consecutive confirmation with a strict spatial majority.
The initial route is still selected by one inference at the exact initial
configuration. Starting at the first positive polling time, each poll draws:

\[
q^{(m)} = q + \epsilon^{(m)}, \qquad
\epsilon^{(m)} \sim \mathcal{N}(0,\sigma^2 I_7), \qquad
\sigma = r_{3\sigma}/3.
\]

The configurable defaults are:

```yaml
spatial_reselection:
  sample_count: 50
  three_sigma_radius_rad: 0.1
  period_seconds: 0.3
  sampling_seed: 20261126
```

Thus each coordinate has standard deviation 0.033333 rad. The 0.1 rad value
specifies the radius of the three-sigma **isodensity surface** in the isotropic
configuration-space Gaussian. It is not a hard cutoff, nor a claim that 99.7%
of seven-dimensional draws lie inside that radius. Samples are untruncated:
no clipping, joint-limit rejection, or collision filtering is applied. They are
hypothetical network inputs only; the robot is never moved to a sampled state.
Actual QP, joint-limit enforcement, collision checks and success criteria stay
unchanged.

Each sample is passed to the existing predictor with the same saved IK, final
EE pose, and obstacles. Only current `q` changes. Each sample casts one vote
for its original 16-action argmax. A route needs **strictly more than half**
the votes: at least 26 of 50. A different majority route is accepted immediately
(effective temporal confirmations = 1); a current-route majority or no majority
retains the current **complete route and current stage**. Vote fractions measure
local agreement, not predicted success probabilities. Probabilities are not
averaged across samples.

An accepted route starts at its first link; the existing reach test immediately
skips already-satisfied stages. Polling continues during EE control and can
select intermediate links again. There is no additional inference at a normal
stage reach. Optional `--sequence-max-switches` and `--sequence-lock-on-ee` work
as described above; neither is enabled by default. A no-majority decision does
not reset the stall window. An accepted route change uses the existing reset
behavior. IK and total scene time remain fixed.

The period is in simulated seconds and must be an integer multiple of control
`dt`. Change it in the spatial YAML; `--sequence-period` and
`--sequence-confirmations` are rejected in this mode to prevent conflicting
configuration sources. The YAML is copied to the output, hashed in the manifest,
and checked again at completion alongside other inputs. Each scene uses
`PCG64(SeedSequence([sampling_seed, partition_position]))`; worker count and
scheduling do not change its random stream. Initial exact-state inference
consumes no Gaussian samples.

The model is queried serially once per sample on the existing CPU inference
path. The runner logs all sampled configurations, their selected action indices,
vote counts/fractions, majority decision and actual controller switch in one
`periodic_selection_events.json` record per spatial poll. Its audit reproduces
the Gaussian draws and vote decision, checks the current state and polling times
against the trace, and verifies accepted switches. It does not rerun neural
inference during that audit.

Summaries distinguish `spatial_poll_count`, `spatial_neural_evaluations`
(`sample_count` times polls), `total_inference_count` (including the one initial
inference), majority/no-majority polls, and accepted route changes.
`neural_inference_wall_seconds` sums timed predictor calls including the initial
one; `spatial_selection_wall_seconds` sums whole sampling/query/voting rounds,
excluding the initial inference. Neither is simulated time or batch elapsed time.
This implementation does not establish a performance or success-rate result;
those require a separately requested comparison.

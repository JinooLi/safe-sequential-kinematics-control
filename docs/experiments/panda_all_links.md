# All-link Sequential experiments

## Shared 1,000-scene results

The completed UBAI comparison on the same 1,000 validation scenes gives
**458/1,000 Direct successes (45.8%)** and **594/1,000 all-link Sequential
successes (59.4%)** at 150 seconds. The corresponding 120-second counts
are 456 and 590. Success requires both reaching the target and passing the
recorded safety criterion; numerical errors remain failures.

The versioned result package is
[`results/panda_all_links_validation_1000_20261023/`](results/panda_all_links_validation_1000_20261023/):

- [Per-scene comparison CSV](results/panda_all_links_validation_1000_20261023/comparison.csv):
  all 1,000 rows, with success, termination reason, completion time and elapsed
  time for Direct and Sequential at both 120 and 150 seconds.
- [Aggregate analysis](results/panda_all_links_validation_1000_20261023/analysis.json):
  outcome counts, paired comparisons, timing and confidence intervals.
- [Verification record](results/panda_all_links_validation_1000_20261023/completion_verification.json):
  the independent 6,050-file integrity audit and original artifact hashes.
- [Campaign plan](results/panda_all_links_validation_1000_20261023/plan.json)
  and the original [Direct](results/panda_all_links_validation_1000_20261023/direct_150s.yaml)
  and [Sequential](results/panda_all_links_validation_1000_20261023/sequential_150s.yaml)
  configuration snapshots.

The CSV and metadata are byte-for-byte copies of the verified campaign
artifacts. Raw trajectories and saved scene inputs remain in the UBAI
campaign directories listed below. The YAML snapshots retain their original
UBAI asset paths; initial configuration, target and obstacles in those base
files were replaced with each saved scene's values by the evaluation runner.
In particular, the 1,000-scene comparison used safe random initial states
and ten obstacles per scene, not the example state and obstacles in the base
YAML. The snapshots document this experiment and are not portable replay
configurations by themselves.

Both durations use the same archived historical controller/QP runtime.
The repository commit publishing these results is not a claim that its
current controller source reproduces that archived runtime. The separate
ten-scene lab-server pilot below used the corrected solver.

## Ten-scene lab-server pilot

On 2026-10-23, an all-link Sequential pilot was launched on `lab-server`.
The selected stages are `link1` through `link7` in ascending order, followed
by the ordinary end-effector pose task. The ten scenes are the recorded prefix
`seed2332_0000` through `seed2332_0009`, with no selection by previous outcome.
The original scene configurations, obstacles and nearest validated IK are
reused from
`outputs/panda_link_pairs_slack5000_300/20261010T122425_767542Z/link4_link5`.
Each trajectory uses one fixed IK with no candidate switching.

The pilot retains the lab-server comparison settings: 150 s maximum simulated
time, dt 0.02 s, pose stages, CLF rate 20, slack weight 5000, alpha 0.2, RDF
CBF gain 1.0, 1 mm success-barrier tolerance and hard joint stops. Intermediate
pose tolerances are 0.05 m and quaternion-vector norm 0.05; the final EE
tolerances are 0.01 m and quaternion-vector norm 0.01. The existing separate
safety validation uses five subdivisions. This is a 150 s controller pilot,
distinct from the concurrently running 120 s UBAI dataset-generation campaign.

Only Sequential is rerun. The current lab-server source is copied to an
isolated snapshot, with the existing archived saved-scene comparison runner
added unchanged and invoked with `--methods sequential`. All ten saved IKs
and initial states were revalidated before launch. The solver contract and
comparison tests passed. Shared source/configuration files remain unchanged.
Earlier Direct and link4/link5 results used an older solver and must not be
treated as a controlled comparison against this pilot.

Server campaign:
`/home/cdsl/jinoo_ws/safe-inv-clbf-robot-arm-panda-gpu/outputs/panda_all_links/20261023T043243Z`.
Start: 2026-10-23 13:34:58 KST; PID 63658.
The campaign contains `experiment.yaml`, `controller.yaml`, `campaign.json`,
`preflight.json`, `tests.log`, `run.log` and `source/`. Per-case trajectories,
summary CSV/JSON, progress, source/input hashes and final completion validation
are written under `results/`. Local metadata is under the corresponding
`outputs/panda_all_links/20261023T043243Z/` directory.

## Completed results

The ten trajectories completed at **2026-10-23 13:44:34 KST** in 575.30 seconds
of wall time. Independent verification checked all 41 sealed result files,
the exact saved scenes and IKs, every trace's seven configured stage frames,
zero IK switches, success labels, and unchanged shared source/configuration
hashes. Completion and aggregation are recorded in `verification.json` and
`campaign.json`. The user explicitly confirmed the `link1` through `link7` scope.

**6/10 scenes reached the target and passed the safety criterion (60%).**
Their mean completion time is 46.63 simulated seconds. Three scenes timed out
at 150 seconds; one stopped because the numerical QP solution failed constraint
verification. All four failures ended at the `link5_pose` stage.

| Case (`seed2332_`) | Success | Termination | Simulated time (s) | Final stage |
|---|---|---|---:|---|
| 0000 | No | QP constraint verification failure | 76.06 | link5 pose |
| 0001 | Yes | Goal reached | 19.78 | EE pose |
| 0002 | No | Timeout | 150.00 | link5 pose |
| 0003 | Yes | Goal reached | 60.96 | EE pose |
| 0004 | Yes | Goal reached | 50.16 | EE pose |
| 0005 | No | Timeout | 150.00 | link5 pose |
| 0006 | Yes | Goal reached | 43.52 | EE pose |
| 0007 | Yes | Goal reached | 53.32 | EE pose |
| 0008 | No | Timeout | 150.00 | link5 pose |
| 0009 | Yes | Goal reached | 52.06 | EE pose |

No recorded or interpolated trajectory violated the adopted 1 mm barrier
criterion, and no joint position exceeded its limits. The minimum recorded
barrier across the ten trajectories was -0.6097 mm; the minimum joint margin
was zero. Each timeout trajectory touched a joint limit somewhere in its
trace, which alone does not establish the cause of failure.

Case 0000's QP backend reported `solved`, but independent verification rejected
row 129's lower-bound residual of 0.0079284 against tolerance 0.0010084. That
rejected command was not applied. This is recorded as `qp_solver_failure`,
not a proof of mathematical infeasibility.

The per-case export including final stages and solver messages is
`outputs/panda_all_links/20261023T043243Z/case_results.csv` locally and on the
server. Full trajectories remain in the server's `results/` directory.

## All-link Sequential on the existing 1,000 validation scenes

At the user's request, the additional UBAI 4,000-scene generation was stopped
and its outputs preserved. New Slurm array **1039602** was submitted at
**2026-10-23 14:26:26 KST** to evaluate all-link Sequential on exactly the
validation scenes previously evaluated by Direct (job 1025724).

Reference Direct campaign:
`/home1/jinu1423/workspace/safe-inv-clbf-robot-arm/outputs/panda_direct_validation/validation-1000-20261020T020700Z`.
Its verified baseline is **456/1,000 safe successes (45.6%)**, with 465 goal
reaches. The new evaluation reuses the exact validation split, scene order,
initial configurations, targets, obstacle data and each scene's saved selected
IK. There are no new scenes, IK searches, test-set scenes or outcome exclusions.

For a comparison against those existing Direct results, all **95 archived
controller source files and the historical QP solver** are reused unchanged.
Five library versions and all recorded input/resource hashes are checked.
The effective configuration changes only mode and the selected stages to
`link1` through `link7`, then the final EE pose. Each trajectory uses one saved
IK, revalidated for pose, barriers and joint limits before execution, without
candidate switching. Numerical errors remain failures in the denominator.

Conditions match the previous Direct evaluation: **120 s**, dt 0.02 s, slack
weight 5000, alpha 0.2, RDF gain 1.0, 1 mm success-barrier tolerance, hard joint
stops, and safety checks at each integration state (no additional interpolation).
This historical-code comparison is separate from the preceding ten-scene
150 s lab-server pilot, which used the corrected QP solver.

Three groups process strided validation rows (334 / 333 / 333). Each receives
four GPUs and 17 CPUs, with four workers per GPU: **12 GPUs / 51 CPUs / 48
rollout workers** in total. The first 16 trajectories in each group are retained
pilot cases; all 48 must pass input, trace and scoring verification before
production continues. Outcome failures do not trigger exclusion or replacement.
Infrastructure or validation failures stop the submission and preserve artifacts.

After all groups finish, the last group rechecks input/source hashes and all
new result hashes and trajectories, then produces `run/analysis.json`,
`run/comparison.csv` and `run/complete.json`. The paired comparison includes
both-success, Direct-only, Sequential-only and both-failure counts; success
rate difference with a paired scene-bootstrap interval and exact McNemar test;
and completion-time differences restricted to cases both methods solve.

UBAI campaign:
`/home1/jinu1423/workspace/safe-inv-clbf-robot-arm/outputs/panda_all_links_validation/validation-1000-20261023T052037Z`.
Local orchestration and metadata:
`outputs/panda_link_selection/all_links_comparison/validation-1000-20261023T052037Z/`.
The original Direct artifacts, shared runtime/configuration and trained models
are unchanged. No recurring monitor or extra dataset generation was created.

Startup verification passed at **2026-10-23 14:33:01 KST**: all 48 retained
pilot trajectories passed the input, trace and scoring checks, and all three
groups entered production. The allocation is 12 NVIDIA A10 GPUs on n057,
n059 and n060, with four worker processes per GPU. At that snapshot,
375/1,000 scenes had completed (128 / 126 / 121); no infrastructure or
validation error was present. This is an in-progress count, not a final
success-rate estimate. See `startup_verification.json` in the campaign root.

### Completed 1,000-scene comparison

All three tasks completed with exit code 0. The final task ended at
**2026-10-23 14:44:24 KST**, after 17 min 55 s. The merged completion marker
validates all 1,000 scenes. A follow-up audit verified all 3,069 sealed files,
the recorded input hashes, original Direct result hashes, exact scene ordering,
the 1,000 CSV rows, and aggregate paired counts. Audit metadata is saved in
`completion_verification.json`; full trajectories remain on UBAI.

| Outcome | Previous Direct | Sequential, link1–link7 |
|---|---:|---:|
| Safe success | 456 (45.6%) | 590 (59.0%) |
| Goal reached but safety criterion failed | 9 | 7 |
| Timeout | 448 | 254 |
| Stalled | 86 | 144 |
| Numerical QP failure | 1 | 5 |
| Total | 1,000 | 1,000 |

The paired outcomes are 330 both-success, 126 Direct-only, 260 Sequential-only,
and 284 both-failure scenes. The success-rate difference is **+13.4 percentage
points** for all-link Sequential, with a paired scene-bootstrap 95% interval
of **+9.6 to +17.2 points** (5,000 draws, seed 20261023). Exact two-sided
McNemar p = 8.0294e-12. These results apply to this validation set and the
shared historical runtime described above.

Among the 330 scenes both methods solve, mean simulated completion time is
24.2364 s for Direct and 44.4819 s for all-link Sequential: Sequential takes
20.2455 s longer on average. Means over each method's own successful subset
are 25.9581 s and 49.8959 s, respectively; those subsets differ.

Local comparison CSV:
`outputs/panda_link_selection/all_links_comparison/validation-1000-20261023T052037Z/run/comparison.csv`.
`run/analysis.json`, `run/complete.json` and `completion_verification.json`
are also retained locally. No additional experiments were started.

## Paired 150-second rerun on the same validation set

The user subsequently requested increasing both Direct and all-link Sequential
to 150 seconds. The active retry is Slurm array **1039783**, submitted on UBAI
at **2026-10-23 15:06:27 KST**. Both methods rerun all 1,000 validation scenes from
their saved initial states: **2,000 new trajectories**, not a continuation from
saved 120-second endpoints and not restricted to prior failures.

Each method's effective configuration is checked against its own 120-second
manifest: **duration is the only changed field**. The same 95 frozen source
files, historical QP implementation, libraries, assets, split, scenes and
safety scoring are verified. Sequential uses the same saved single IK and
link1–link7 stages without candidate switching; Direct uses no IK candidate.
The 150 seconds are a maximum; existing goal, stall and numerical termination
conditions still apply. Numerical failures remain in the denominator.

Two groups receive 500 paired scenes each (1,000 trajectories per group).
Allocation is 8 GPUs, 34 CPUs and 32 rollout workers, with four workers per
GPU. Each group first executes eight paired scenes. All 32 retained pilot
trajectories must pass identity, duration, trace and scoring checks before
both groups proceed to production. Each allocation must also expose four
working CUDA devices and pass a tensor-operation probe before workers start.

The first submission, **1039735** at 15:01:15 KST, aborted during CUDA
initialization on n057, before any trajectory completed. Diagnostic allocation
1039759 found that Slurm assigned four GPU IDs but NVIDIA and PyTorch exposed
only three devices; those three passed the tensor probe. The retry excludes
n057 and uses the two available healthy four-GPU nodes. This is an execution
resource change; the scientific configuration and frozen runtime are unchanged.
All failed-attempt artifacts remain in the campaign directory without `-r1`,
including `infrastructure_failure.json`. They are not experimental outcomes.

The final merge revalidates all 2,000 trajectories and hashes, and writes
`run/direct_results.json`, `run/sequential_results.json`, `run/analysis.json`,
`run/comparison.csv` and `run/complete.json`. It compares Sequential versus
Direct at 150 seconds and each method against its own 120-second results,
including paired success counts, confidence intervals, paired completion
times, and recovery of previous timeouts. Changes to outcomes previously
ending before the time limit are reported rather than silently excluded.

UBAI campaign:
`/home1/jinu1423/workspace/safe-inv-clbf-robot-arm/outputs/panda_all_links_validation/validation-1000-150s-20261023T055640Z-r1`.
Local orchestration and metadata:
`outputs/panda_link_selection/all_links_comparison/validation-1000-150s-20261023T055640Z-r1/`.
The completed 120-second artifacts and shared source/configuration are preserved.
No new dataset generation, training or recurring monitor was started.

The retry passed startup verification at **2026-10-23 15:08:39 KST**. Both
four-GPU allocations on n059 and n060 passed the CUDA tensor probe; all 32
workers were present. The 16 paired pilot scenes (32 trajectories) passed
validation, including trajectories reaching the new 150-second timeout.
Both groups entered production. At this snapshot, 44/2,000 trajectories and
18/1,000 pairs had completed, with no infrastructure/validation error.
See the retry campaign's `startup_verification.json` for the recorded evidence.

### Completed paired 150-second comparison

Both tasks of array **1039783** completed with exit code 0. The final task
ended at **2026-10-23 16:02:12 KST**, after 55 min 44 s. The merged completion
marker validates 1,000 scenes and 2,000 trajectories. An independent follow-up
audit verified all **6,050 sealed files**, orchestration input hashes, the
120-second reference result hashes, exact scene identities and ordering,
the 1,000 comparison CSV rows, and aggregate paired counts. The audit is
recorded in `completion_verification.json` in the retry campaign directory.

| Outcome | Direct, 150 s | Sequential link1–link7, 150 s |
|---|---:|---:|
| Safe success | 458 (45.8%) | 594 (59.4%) |
| Goal reached but safety criterion failed | 9 | 7 |
| Timeout | 443 | 245 |
| Stalled | 89 | 149 |
| Numerical QP failure | 1 | 5 |
| Total | 1,000 | 1,000 |

The paired counts are 334 both-success, 124 Direct-only, 260 Sequential-only,
and 282 both-failure. Sequential's advantage is **13.6 percentage points**,
with a paired scene-bootstrap 95% interval of **9.8 to 17.4 points** and exact
two-sided McNemar p = 3.355754e-12. On the 334 scenes both methods solve,
mean simulated completion time is 24.5220 s for Direct and 45.2971 s for
Sequential, a difference of 20.7751 s. Means over each method's own successful
subset are 26.4379 s and 50.4221 s; these subsets differ.

Increasing the limit from 120 to 150 seconds recovers only **two Direct**
and **four Sequential** successes (+0.2 and +0.4 percentage points).
All previous successes remain successes, with identical completion times;
no prior non-timeout result changes its success label or termination reason.
Of Direct's 448 previous timeouts, 443 still time out, three become stalled
and two succeed. Of Sequential's 254 previous timeouts, 245 still time out,
five become stalled and four succeed. This supports only a small benefit
from this particular 30-second extension; it does not establish that all
remaining timeouts are permanently unable to reach their targets.

The aggregate analysis, completion marker, verification record and per-scene
comparison CSV are retained locally under
`outputs/panda_link_selection/all_links_comparison/validation-1000-150s-20261023T055640Z-r1/`.
Raw trajectories remain on UBAI. No additional experiment was started.

## Joint-limit behavior in the 260 Sequential-only successes

A retrospective CPU-only analysis of the 150-second run compared the 260
scenes where Direct failed and all-link Sequential succeeded. Job **1040537**
completed on 2026-10-23 in 20 seconds. No new trajectories, controller changes
or QP re-solves were performed. All 520 saved trajectories and their contexts
were checked against the sealed results. Direct's failures comprise 206
timeouts, 44 stalls, nine goal reaches failing the safety criterion and one
initial QP failure. The last case has no applied control command and remains
in the 260-scene denominator, with activation unobservable rather than inferred.

The archived controller imposes
`-0.2*(q-q_lower) <= u <= 0.2*(q_upper-q)`. Approximate binding is reconstructed
from the **commanded** velocity at the corresponding pre-integration state,
using absolute row residual <=0.001 rad/s, only where the joint CBF bound is
stricter than the separate speed/acceleration bounds. Only solved commands
actually applied to the plant are included. Applied velocity is checked
against state differences but is not substituted into the QP residual;
hard-stop clipping would otherwise make activation partly tautological.
"Near a limit" means <=0.02 rad from that same bound. Numerical violations
outside the residual tolerance are counted separately, not called binding.

| Observation, out of 260 scenes | Direct | Sequential |
|---|---:|---:|
| Any approximate joint-CBF binding during the trajectory | 259 | 259 |
| Binding while within 0.02 rad of a joint limit | 245 | 90 |
| Final state within 0.02 rad of any joint limit | 240 | 36 |
| At least one hard-stop clipping step | 197 | 2 |
| Same near-limit joint binding for >=50% of the final 10 s | 209 | 22 |

There are **160** scenes with near-limit binding in Direct but none anywhere
in Sequential; 85 have it in both, five only in Sequential, and ten in neither.
Of the 197 Direct trajectories with clipping, 196 have no clipping in the
matched Sequential trajectory. Same-joint terminal persistence occurs only
in Direct in 190 scenes, in both in 19, and only in Sequential in three.

To control for different trajectory durations, truncate each pair to
`[0, min(T_direct, T_sequential))`. Across the 259 pairs with applied commands
in both methods, mean time with any near-limit binding is **26.1764 s for
Direct and 2.0445 s for Sequential**. In 242/260 scenes the reduction exceeds
0.5 s in this matched window. Thus the contrast is not solely the longer
exposure of Direct trajectories that time out.

| Joint | Direct: near-limit binding ever | Sequential: near-limit binding ever | Direct: terminal persistence | Sequential: terminal persistence |
|---|---:|---:|---:|---:|
| 1 | 51 | 16 | 31 | 6 |
| 2 | 63 | 40 | 33 | 8 |
| 3 | 55 | 15 | 39 | 4 |
| 4 | 79 | 6 | 52 | 0 |
| 5 | 93 | 7 | 78 | 2 |
| 6 | 112 | 11 | 72 | 1 |
| 7 | 85 | 5 | 69 | 2 |

Counts overlap across joints. Joint4 shows a pronounced difference, but the
pattern also affects joints5–7. In **180/250 timeout-or-stall failures (72%)**,
Direct has both terminal same-joint near-limit binding and <=1% reduction in
the final EE pose CLF over the last ten seconds. This diagnostic uses the
recorded `0.5*(position_error^2 + orientation_error^2)` with the archived
quaternion-vector orientation convention; it is not a new controller stop
criterion. In 178 of those 180 scenes, the local EE-CLF descent direction
also points toward a near/binding joint boundary for at least half of the
0.2-second terminal samples. That instantaneous directional check does not
isolate effects of obstacle constraints or establish a counterfactual success.

The overall contrast survives tighter thresholds: at residual tolerance
0.0001 rad/s, near-limit binding is observed in 245 Direct and 88 Sequential
trajectories; terminal persistence counts become 72 and nine. Independently
of velocity residuals, final proximity within 0.001 rad occurs in 230 Direct
and five Sequential trajectories. Exact persistence counts therefore depend
on numerical tolerance, while the direction of the contrast is stable.

These observations support reduced trapping near joint limits in this
outcome-selected cohort, not elimination of all joint-CBF activation. QP
dual variables are unavailable, and the historical solver sometimes produced
commands violating the nominal joint CBF by more than 0.001 rad/s (259 Direct
and 256 Sequential trajectories have at least one such command). The reported
success criterion validates recorded state safety, not strict feasibility
of every QP row. Joint proximity, approximate binding, command violations and
hard-stop clipping are consequently kept as distinct observations. This
analysis alone cannot assign a causal fraction of the full 1,000-scene
success-rate difference to joint limits.

The local derived artifacts are under
`outputs/panda_all_links_analysis/seq_only_260_150s_20261023/`:
`paired_260.csv`, `per_joint_3640.csv`, `analysis.json`, `details.json` and
`complete.json`. The same folder exists on UBAI under the repository's
`outputs/`; raw trajectories were not downloaded. Cohort membership, CSV
counts and output SHA256 values were independently checked after transfer.

## Joint-limit behavior in the 124 Direct-only successes

The reverse cohort from the same sealed 150-second experiment was audited
with the same joint-CBF residual, dominance, proximity and persistence
criteria. CPU-only job **1040973** completed on 2026-10-23 in 12 seconds.
All 248 trajectory/context pairs were verified against their recorded hashes;
the 124 scene IDs match the published comparison CSV. No new rollouts,
QP re-solves or controller changes were made. Both methods have applied
commands in every selected scene. Sequential's failures comprise 72 timeouts,
49 stalls, one goal reach failing the safety criterion and two QP failures
after 6.68 and 21.16 simulated seconds. No IK switches occurred.

| Observation, out of 124 scenes | Direct (successful) | Sequential (failed) |
|---|---:|---:|
| Any approximate joint-CBF binding during the trajectory | 122 | 120 |
| Binding while within 0.02 rad of the same joint limit | 27 | 108 |
| Final state within 0.02 rad of any joint limit | 19 | 106 |
| At least one hard-stop clipping step | 0 | 64 |
| Same near-limit joint binding for >=50% of the final 10 s | 15 | 83 |

Near-limit binding occurs only in Sequential in **85** scenes, in both in
23, only in Direct in four, and in neither in 12. Terminal persistence
occurs only in Sequential in 72 scenes, in both in 11, only in Direct in
four, and in neither in 37. A transient binding constraint is not sufficient
to explain failure: successful Direct also has such events.

| Joint | Direct: near-limit binding ever | Sequential: near-limit binding ever | Direct: terminal persistence | Sequential: terminal persistence |
|---|---:|---:|---:|---:|
| 1 | 5 | 19 | 2 | 13 |
| 2 | 7 | 22 | 3 | 9 |
| 3 | 3 | 34 | 2 | 18 |
| 4 | 0 | 31 | 0 | 23 |
| 5 | 5 | 45 | 4 | 38 |
| 6 | 2 | 16 | 1 | 11 |
| 7 | 8 | 14 | 4 | 8 |

Joint counts overlap. Joint5 is the most frequent terminal near-limit
constraint in failed Sequential; joint4 also shows the opposite contrast
to the Sequential-only-success cohort.

The exposure-time qualification is substantial here. Within each pair's
common interval `[0, min(T_direct, T_sequential))`, mean near-limit-binding
time is **2.0865 s for Direct versus 2.8766 s for Sequential**. Sequential
exceeds Direct by more than 0.5 s in 31 scenes; Direct exceeds Sequential
by that amount in 18. In **66** scenes, Sequential first enters near-limit
binding only at or after Direct has already finished. Thus the large
whole-trajectory contrast partly reflects Sequential's additional time and
path through intermediate stages, not a large disparity from the beginning.
The previous 260-scene cohort's matched-window means were 26.1764 s for
failed Direct and 2.0445 s for successful Sequential (259 observable pairs).

Sequential ends at an intermediate link in 110/124 scenes. Final EE error
is therefore not its active objective for much of this cohort. In addition
to the unchanged EE-based diagnostic (71 timeout/stall cases), this audit
uses the **final active stage's** recorded pose CLF,
`0.5*(position_error^2 + orientation_weight*orientation_error^2)`, restricting
the interval to time after entry to that stage. A trap-pattern flag requires
a full ten-second window, same-joint near-limit binding for at least half
that window, and CLF reduction <=1%. This is a retrospective diagnostic,
not the controller's stop rule.

**61/121 timeout-or-stall failures (50.4%; 61/124 overall)** satisfy that
active-stage pattern: 42 timeouts and 19 stalls. All 61 end at intermediate
links. In **54** of them (38 timeouts, 16 stalls), a 0.2-second FK sampling
also finds the active-stage CLF descent pointing outward through a
persistently near/binding joint boundary for at least half the sampled
window. FK CLF values were checked against both the archived stage errors
and the controller's recorded `clf_value`.

| Sequential final stage | Failed scenes | Active-stage trap pattern | Pattern plus outward-gradient agreement |
|---|---:|---:|---:|
| link1 | 26 | 12 | 9 |
| link2 | 16 | 3 | 1 |
| link3 | 25 | 12 | 11 |
| link4 | 1 | 1 | 0 |
| link5 | 25 | 19 | 19 |
| link6 | 2 | 1 | 1 |
| link7 | 15 | 13 | 13 |
| end effector | 14 | 0 | 0 |

The zero in the EE trap column does not prove absence of joint effects.
Seven EE-stage failures have terminal near-limit persistence, but their
EE CLF reduction exceeds this particular 1% threshold. Conversely, EE
progress can be low while an intermediate-stage objective is progressing,
so the EE-only 71-case count is not a causal classification.

At the tighter residual tolerance of 0.0001 rad/s, near-limit-binding counts
are 27 Direct versus 98 Sequential, and terminal-persistence counts are 12
versus 33. Final geometric proximity within 0.001 rad occurs in five versus
73 scenes. The direction of the contrast survives the threshold change,
but exact persistence counts depend on numerical tolerance. Each method
has 121 trajectories containing at least one nominal joint-CBF violation
larger than 0.001 rad/s; these violations remain separate from approximate
binding and physical hard-stop clipping.

This supports joint-limit trapping as a plausible contributor to a
substantial subset of Sequential failures. It does not establish that the
61 or 54 scenes would succeed if joint constraints were removed: obstacle
constraints, slack and path choice can act together, and QP duals and causal
interventions are unavailable. Taken with the opposite cohort, intermediate
stages can either avoid or lead toward joint-limit traps depending on the
scene; the data do not support unconditional avoidance by Sequential.

Derived artifacts are retained locally and on UBAI under
`outputs/panda_all_links_analysis/direct_only_124_150s_20261023/`:
`paired_124.csv`, `per_joint_1736.csv`, `analysis.json`, `details.json` and
`complete.json`. The local `case_summary_124.csv` is a compact per-scene view;
`local_verification.json` records independent membership, count and hash
checks. Raw trajectories remain on UBAI. The preliminary analysis was
preserved in `preliminary-1040949/`; its supplementary Direct stage indexing
was corrected before final reporting, without changing the joint-binding
metrics. The final script is local `analyze.py` / remote `analyze_v2.py`.

## Initial and stage-entry error audit: all 1,000 scenes, 2026-10-24

This read-only audit uses the same sealed 150-second, all-link Sequential
versus Direct run. No new IK, QP solves, or simulations were performed.
All 2,000 trajectory and context hashes were verified, along with archived
source hashes. Recorded EE errors were independently checked using archived
FK at initial, final and minimum-normalized-error states; intermediate errors
were checked at every stage entry. UBAI CPU job `1045939` completed in 61 s
using one CPU, with 6,433 stage-entry records. An initial audit-only attempt
failed before extraction because the FK reader had not registered link names;
that reader initialization was corrected. Experimental data were untouched.

Artifacts: `outputs/panda_all_links_analysis/error_hypothesis_1000_20261024/`.
`extract.py` and `extract.sbatch` reproduce the remote read-only extraction;
`analyze.py` produces local tables, sensitivity regressions and plots.
`complete.json`, `trace_manifest.json` and `local_verification.json` retain
provenance and hashes. Raw trajectories remain on UBAI.

The orientation quantity recorded by the controller is quaternion error
vector norm `epsilon = sin(theta/2)`; this audit reports the shortest
rotation angle `theta` in degrees for readability. The combined error is
`sqrt(position_error_m^2 + epsilon^2) = sqrt(2*pose_clf)` with weight one.
It is the controller's weighted quantity, not a physical distance.
The simultaneous final-tolerance ratio is
`R = max(position_error_m/0.01, epsilon/0.01)`.
`min(R)` uses one common state; independently minimizing position and
orientation would overstate closeness. Minima are over recorded 0.02 s
states, not continuous-time interpolation.

### Direct: initial error versus success

Each initial-position quintile contains 200 scenes. Edges below are rounded;
exact intervals and Wilson intervals are in `direct_initial_error_bins.csv`.

| Initial position error (cm) | Direct successes | Success rate | 95% Wilson interval |
|---|---:|---:|---:|
| 16.4–50.0 | 117/200 | 58.5% | 51.6–65.1% |
| 50.0–70.8 | 91/200 | 45.5% | 38.7–52.4% |
| 70.8–87.8 | 87/200 | 43.5% | 36.8–50.4% |
| 87.8–106.4 | 88/200 | 44.0% | 37.3–50.9% |
| 106.4–181.9 | 75/200 | 37.5% | 31.1–44.4% |

Orientation-error quintiles, in ascending order, give 50.5%, 44.0%, 47.0%,
42.0%, and 45.5% success. Their boundaries are 11.87, 94.42, 121.49,
142.65, 161.33 and 179.90 degrees. Combined-error quintiles give 57.0%,
46.5%, 41.0%, 47.0%, and 37.5%. Thus the position association is clearer
than the orientation association; neither curve is strictly monotonic.

An exploratory linear-logit model adjusting for initial rotation angle,
joint margin, self-barrier and obstacle-barrier margins estimates an odds
ratio of **0.920 per additional 10 cm** (95% Wald CI 0.883–0.959,
p=0.000096). Rotation per additional 30 degrees has OR 0.943
(0.851–1.046, p=0.268). These are associations, not causal effects or
percentage-point changes. Adding distance to the saved IK as a sensitivity
covariate gives position OR 0.948 (0.906–0.991). Direct does not follow that
IK, so this covariate is only a scene descriptor. Excluding the one initial
QP failure leaves the initial-margin-adjusted position result essentially
unchanged. No selector was trained.

### Sequential: errors at actual stage entry

All medians include immediate passes and failures in the entered stage.
Rows are conditional on reaching that stage, not on eventual success.
The last column is geometric stage completion, not trajectory safety.

| Stage | Entered | Immediate passes | Median position (cm) | Median angle (deg) | Target reached / entered |
|---|---:|---:|---:|---:|---:|
| link1 | 1,000 | 61 | 0.00 | 56.7 | 915/1,000 (91.5%) |
| link2 | 915 | 7 | 0.00 | 44.8 | 872/915 (95.3%) |
| link3 | 872 | 11 | 3.11 | 54.9 | 796/872 (91.3%) |
| link4 | 796 | 31 | 2.03 | 37.8 | 790/796 (99.2%) |
| link5 | 790 | 13 | 4.55 | 66.2 | 720/790 (91.1%) |
| link6 | 720 | 28 | 4.59 | 48.6 | 712/720 (98.9%) |
| link7 | 712 | 11 | 4.54 | 81.5 | 628/712 (88.2%) |
| Final EE | 628 | 0 | 4.76 | 4.49 | 601/628 (95.7%) |

Initial Direct medians across 1,000 scenes were 80.08 cm and 130.49 degrees.
The zero position errors for link1/2 reflect model geometry; they are not
zero pose errors. Intermediate stages use 5 cm / epsilon 0.05 tolerances
(about 5.73 degrees); final tolerances are 1 cm / epsilon 0.01 (1.15 degrees).
Many intermediate stages thus start with already-small position errors but
substantial orientation errors.

For each entered scene, comparing the same link against its original
initial error avoids confusing shorter links with preparation by earlier
stages. Median entry/original combined-error ratios for link2–link7 are
0.772, 0.629, 0.415, 0.530, 0.393 and 0.592; the final EE ratio is 0.051.
At entry, intermediate combined error is below the original EE error in
95.6–99.2% of entered scenes, depending on the link. Relative to the
*contemporaneous* EE error, link7 is smaller in only 53.9%, so a universal
claim that every intermediate objective is smaller than the current EE
objective would be false.

After excluding immediate passes, the lowest/highest entry-combined-error
quintiles have geometric stage completion of 100%/68.6% for link1,
100%/66.7% for link5, and 100%/65.2% for link7. This within-link association
supports the error-size hypothesis, but can also reflect geometry, remaining
time and selection into a stage. Final EE quintiles are not monotonic.

The implementation can terminate on the final goal before all stages
complete, but this did not occur in these 1,000 scenes. Therefore the
observed conditional stage-completion rates telescope to 601/1,000 goal
reaches. Seven of those violate trajectory safety, leaving 594/1,000
successes. Multiplying these observed rates is an accounting identity,
not independent evidence of why the stages work.

### Direct failures: minimum achieved error

Direct has 542 failures: 443 timeouts, 89 stalls, nine unsafe goal reaches,
and one initial QP failure. Classifying by minimum simultaneous ratio:

| Minimum R | Scenes | Fraction of 542 |
|---|---:|---:|
| <=1 | 9 | 1.7% |
| >1 to 2 | 28 | 5.2% |
| >2 to 5 | 55 | 10.1% |
| >5 to 10 | 60 | 11.1% |
| >10 | 390 | 72.0% |

For example, R<=2 means position<=2 cm and angle<=2.29 degrees
simultaneously; R<=5 means <=5 cm and <=5.73 degrees. The nine R<=1
failures reached the goal but violated safety earlier in the trajectory.
All 542 failures have independent minimum-position median 15.91 cm and
minimum-angle median 10.72 degrees, which need not occur at the same time.

Among the 532 timeout/stall cases, 28 ever reach R<=2 and 83 ever reach
R<=5; 26 and 81 respectively do so without an earlier safety violation.
For R<=5, 66 later stall and 17 time out; median observed continuation
after first entering this region is 28.72 s, and 37 continue at least 30 s.
Thus near-goal nonconvergence exists, but it is not the majority of failed
Direct cases in this particular 1,000-scene run.

In the 250 timeout/stall cases rescued by Sequential, 38 Direct traces
reach a lower minimum R than Sequential's final-EE entry R. In **27**,
at Direct's minimum-R state both position and orientation errors are
componentwise no larger than Sequential's final-EE entry errors; 26 also
have a safe trajectory prefix at that state. Their IDs and paired values
are in `direct_failed_despite_smaller_errors.csv`. This is a conservative
snapshot count, not an exhaustive search of every dominated state. Smaller
error alone therefore does not determine convergence.

### Interpretation and limits

The data support decomposing the task into smaller-error stages and
preparing a near-goal final EE state as a plausible contributor. They do
not isolate this contribution from posture, constraint compatibility,
different intermediate tolerances, stage exposure or remaining time.
No initial Direct position error is below 16.39 cm: its initial-error
curve cannot establish a success probability for the 2–5 cm states common
at Sequential stage entry. Final-stage statistics are conditional on the
628 survivors and cannot be compared causally with all 1,000 Direct starts.

These are the historical IK-available validation scenes and archived QP
runtime, not a rerun of the subsequently corrected solver. The historical
definition of success and its 1 mm sampled barrier tolerance were preserved.

## Random-100 EE waypoint intervention (2026-10-24)

To test whether reducing the EE error in stages helps explain Sequential's
advantage, a paired five-arm experiment was submitted to UBAI as job
`1046000`. It uses 100 scenes sampled without replacement from the same
1,000 saved validation contexts with NumPy PCG64 seed `20261024`, sorted by
original index. Selection uses no previous outcomes. All five arms reuse
each scene's initial joints, final pose, obstacles and original saved IK.
Direct and EE waypoint arms do not use an IK target; all-link Sequential
uses the single saved IK without switching.

The arms are Direct, EE Direct with 2/4/8 targets, and Sequential through
link1--link7 then the final EE pose. The target count includes the final
target. EE intermediate positions equally divide the fixed initial-to-final
line, with orientations interpolated by shortest-rotation SLERP. They share
the existing CLF/CBF QP, stage transition and stopping logic. Stage transitions
reset the progress history, retaining the preceding command for acceleration
limits. Targets already within tolerance can pass without motion and are
recorded as such.

Every rollout has one total 150 s budget, dt 0.02 s, CLF rate 20, slack
weight 5000, alpha 0.2, RDF CBF gain 1, hard joint stops and a sampled 1 mm
barrier tolerance. Intermediate position/orientation tolerances are
0.05 m / quaternion-vector norm 0.05; final tolerances are 0.01 m / 0.01.
The independent final-goal criterion is identical across arms. Safety is
evaluated at each integration state, with no interpolated substeps. All
arms rerun the current corrected QP solver, rather than mixing archived
Direct results with new controller code.

After all five trajectories of each scene, a separate endpoint feasibility
audit checks the seven unique intermediate targets at fractions 1/8--7/8.
It first looks for strictly safe recorded trajectory witnesses, then tries
the existing safe IK solver from three deterministic starts for at most
1,000 iterations per start, retaining the existing hold criterion. A missing
witness is not proof of infeasibility. No scene is excluded, and the audit
does not change any rollout. This distinction matters: straight EE waypoint
paths may introduce unreachable or obstructed intermediate tasks.

The first two scenes form a 10-rollout implementation pilot before the
remaining 98 scenes. Six waypoint tests plus the existing pose/stopping/QP/
comparison tests passed (64 total). Runtime checks independently validate
each trace's initial state, final goal, safety, applied velocity, transitions,
time budget and absence of IK switching. Results include all 500 outcomes,
termination reasons, success-only completion times, common-success timing,
paired success differences, bootstrap intervals and exact McNemar tests.
These comparisons are exploratory; raw p-values are not adjusted for the
multiple waypoint variants.

Local campaign metadata:
`outputs/panda_ee_waypoint_comparison/random100-150s-20261024T091721Z/`.
UBAI campaign and raw trajectories:
`/home1/jinu1423/workspace/safe-inv-clbf-robot-arm/outputs/panda_ee_waypoint_comparison/random100-150s-20261024T091721Z/`.
The campaign seals the selected contexts, source snapshot, configuration,
runner, tests and model assets by SHA256. It runs in an isolated source
snapshot on four GPUs with up to 16 simulation workers.

The 10-rollout pilot passed at 09:33:43 UTC. The main batch then started
with 16 initialized workers, four on each of four NVIDIA A10 GPUs on n054.
At the 09:35 UTC check, 43/500 trajectories had finished and passed their
trace checks; four complete scene bundles also included the feasibility
audit. The Slurm job was running without a failure marker. These are launch
checks, not final performance results. `complete.json` and `analysis.json`
are produced only after all 500 outcomes have been collected and validated.

### Completed five-arm comparison

Job `1046000` completed normally (exit 0) at **2026-10-24 19:10:04 KST**;
Slurm elapsed time was **44 min 35 s**. All 100 scene bundles, 500 rollouts
and 700 intermediate-target endpoint checks completed. The separate CPU
audit job `1046106` verified all 1,304 sealed files and all 500 traces,
including initial states, total time, success labels, safety and no IK
switching. The completion marker SHA256 is
`f92de2c162a72b6bc162acbd7dbd2ca57d210c95d728515a0b3d831f2c5f90b4`.

| Method | Safe success / 100 | Mean successful completion (s) | Timeout | Stalled | QP solver failure | QP infeasible | Unsafe goal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 48 | 29.98 | 41 | 4 | 7 | 0 | 0 |
| EE 2 targets | 36 | 35.32 | 58 | 5 | 0 | 1 | 0 |
| EE 4 targets | 32 | 30.27 | 63 | 4 | 0 | 1 | 0 |
| EE 8 targets | 31 | 32.62 | 64 | 4 | 1 | 0 | 0 |
| All-link Sequential | 58 | 54.53 | 27 | 11 | 3 | 0 | 1 |

Means use each method's safe successes, so their scene populations differ.
Sequential reached the final goal in 59 scenes, with one excluded from
success because of the recorded safety criterion. Numerical/QP failures
remain failures; no outcomes were removed.

| Candidate versus Direct | Candidate-only successes | Direct-only successes | Difference (percentage points) | Paired bootstrap 95% CI | Exact McNemar p |
|---|---:|---:|---:|---|---:|
| EE 2 targets | 3 | 15 | -12 | [-20, -4] | 0.00754 |
| EE 4 targets | 5 | 21 | -16 | [-26, -7] | 0.00249 |
| EE 8 targets | 5 | 22 | -17 | [-26, -7] | 0.00151 |
| All-link Sequential | 27 | 17 | +10 | [-3, 23] | 0.17417 |

Sequential has the highest observed success rate here, but this 100-scene
subset alone does not establish a statistically clear advantage over
Direct. On their 31 common safe successes, Direct averaged 27.40 s and
Sequential 48.63 s. Corresponding common-success timings for Direct versus
EE 2/4/8 are 29.24 versus 36.42 s (33 scenes), 26.88 versus 29.17 s
(27 scenes), and 26.86 versus 30.55 s (26 scenes).

The intervention demonstrably reduced the initial active EE error: median
position error was 75.48 cm for Direct, then 37.74/18.87/9.43 cm for
2/4/8 targets. Median quaternion-vector orientation error was
0.908/0.539/0.281/0.142 respectively. Nevertheless, this straight-position /
SLERP waypoint construction reduced success rather than improving it.
This does not support error reduction alone as a sufficient explanation
for the observed all-link Sequential performance.

The experiment does not isolate error magnitude from intermediate task
feasibility or path geometry. Every intermediate endpoint had a strictly
safe witness in 79/100 EE-2 scenes, 66/100 EE-4 scenes and 61/100 EE-8
scenes. The other scenes were retained. Failure to find a witness within
the bounded search does not prove an endpoint impossible, and an endpoint
witness does not prove a feasible connecting path from the current state.
Among failed EE-2/4/8 trajectories, respectively 21/32/32 stopped at an
intermediate target without a witness, 16/23/30 stopped at a certified
intermediate target, and 27/13/7 reached the final-goal stage but failed.
Therefore this result should not be presented as a general refutation of
the smaller-error hypothesis, nor as proof of a specific alternative cause.

The local campaign folder contains `results.csv`, `results.json`,
`analysis.json`, `waypoint_feasibility.json`, `stage_entries.json`,
`verification.json`, `complete.json` and the Slurm logs. The raw 500
trajectories remain under the UBAI campaign's `run/` directory.

### Local descent at Direct's stopped states (2026-10-24)

This diagnostic uses the **same fixed random100 / 150 s comparison** above,
not all 1,000 validation scenes. Its primary population is the four Direct
`stalled` terminations. The 41 `timeout` endpoints form a separate secondary
population; seven QP failures are outside this stagnation analysis. This is
an offline state analysis, not a new rollout or a controller intervention.

At each scene's terminal state, last applied-command state and state 0.5 s
before termination, reconstruct the original QP with the campaign's frozen
runtime, configuration, safety assets and preceding commanded velocity.
Keep the obstacle/self-collision CBFs, joint-limit CBFs, velocity limits and
acceleration limits. Remove the CLF/slack rows only. For each of EE and
link1 through link7, solve `min grad(V_task) @ velocity` over that identical
hard feasible set. Link goals are the FK poses of the original fixed IK.
No IK switching or safety relaxation is introduced by the diagnostic.

The task value is the original pose CLF:
`V = (||position_error||² + ||quaternion_vector_error||²) / 2`.
Define `E = sqrt(2 V)` and report the maximum **instantaneous relative E
decrease**, `100 * (-min(dot(V))) / (2 V)`, in percent per second. These are
local derivative bounds, not predicted reductions over a full second.
Different task errors have different targets and geometry, so a larger
relative rate does not itself establish a better path to the final EE goal.

All 2,160 LPs (1,080 CLF-gradient and 1,080 normalized-progress-gradient
objectives) were feasible and passed primal/dual/gap checks. The maximum
CLF-LP primal residual was `6.68e-17`, dual residual `5.22e-15`, and absolute
duality gap `6.94e-17`. Analytic task gradients matched finite differences
within `1.80e-10`. Four diagnostic solver tests passed. Source/config/asset
and original trace hashes were checked; the refinement job rechecked all
281 files sealed by the first analysis, including 270 state files.

The primary four endpoints all satisfy strict sampled barrier and joint
bounds. Every EE and link task has an infinitesimal decreasing direction.
The following table excludes already-satisfied intermediate targets from
the useful-link comparison. Their tolerances are 5 cm / 0.05 quaternion
vector norm, versus 1 cm / 0.01 for the final EE.

| Case ID (prefix `seed202612`) | Stop (s) | EE position error (mm) | EE (%/s) | Link1 | Link2 | Link3 | Link4 | Link5 | Link6 | Link7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0006_0014 | 60.90 | 34.67 | 0.0665 | 16.2317 | 12.8378 | 13.4554 | 13.5197 | 1.6324 | 1.7065 | Satisfied |
| 0019_0001 | 42.70 | 13.33 | 1.0864 | 0.1751 | 0.0070 | Satisfied | Satisfied | Satisfied | Satisfied | Satisfied |
| 0030_0039 | 70.42 | 13.61 | 3.6475 | 0.2291 | 0.2284 | 42.9450 | 46.8338 | 1.9121 | 1.9140 | Satisfied |
| 0108_0015 | 100.52 | 19.55 | 1.3132 | 3.6066 | 1.9834 | 5.9254 | 5.7657 | 21.5040 | 23.0991 | Satisfied |

Three of four states have an unmet link target with a greater relative
descent bound than EE: link1 / link4 / link6 respectively, by factors
244.26 / 12.84 / 17.59. Case `0019_0001` is a counterexample: its remaining
link1/link2 targets have lower relative descent bounds than EE. This is
evidence of task-dependent local movement opportunities, not proof that
changing the task will recover the failed trajectory.

The primary CBF constraints with the largest absolute normalized LP duals
are listed below. This identifies constraints limiting that local optimum;
it is not a constraint-removal causal experiment or an active-time count.

| Case | Leading CBF constraints limiting EE descent | Minimum joint margin (rad) |
|---|---|---:|
| 0006_0014 | obstacle6–link4; joint5 lower; joint6 upper | 2.47e-6 |
| 0019_0001 | joint1 upper; joint5 upper | 9.45e-5 |
| 0030_0039 | joint1 lower; joint3 lower | 1.16e-5 |
| 0108_0015 | joint7 lower | 9.54e-10 |

The actual preceding Direct command also decreased the CLF in all four
cases and met the reconstructed hard constraints. At its matched state,
its instantaneous EE decrease was 72.0%, 9.10%, 1.79% and 0.500% of the
respective LP bounds. The LP optimizes only descent, whereas the controller
balances velocity cost and CLF slack, so this difference alone is not a
solver defect. The four terminal stop tests also reproduce the low applied
speed and insufficient best-normalized-error improvement over the last 0.5 s.

#### Finite-step checks

An infinitesimal descent direction can increase the nonlinear task value
when applied for 0.02 s. Therefore, additionally test scaled CLF-LP,
normalized-progress-LP and preceding commanded velocities. Every retained
witness obeys the same hard inequalities, stays inside joint bounds, and
meets the sampled barrier threshold at both current and next states.
Check both `V` and the actual stop-progress metric
`R = max(position_error / position_tolerance,
orientation_error / orientation_tolerance)`.

All four stalled EE states and all 20 unmet link targets have a 0.02 s
witness decreasing **both V and R**. These primary witnesses also keep
barriers nonnegative, without relying on the 1 mm tolerance. The best
tested EE witnesses decreasing both metrics reduce E by only
0.000951%, 0.001964%, 0.001296% and 0.000237% per step respectively.
For example, in `0006_0014`, a link1 witness reduces its E by 0.324683%
in one step, while the best tested EE witness reduces EE E by 0.000951%.
These finite candidate searches establish existence, not globally optimal
finite-step velocities or collision-free paths between sampled endpoints.
Failure to find a witness would not establish impossibility.

#### Secondary timeout endpoints

All 41 timeout scenes have insufficient progress over the final 0.5 s but
mean applied speed above the stall threshold. They should not be relabeled
as the four explicitly detected stalls.

| Unmet task at timeout | Count | Infinitesimal descent | Safe sampled V decrease | Safe sampled V and R decrease |
|---|---:|---:|---:|---:|
| EE | 41 | 39 | 34 | 24 |
| Link1 | 38 | 38 | 37 | 37 |
| Link2 | 39 | 39 | 38 | 38 |
| Link3 | 41 | 41 | 40 | 40 |
| Link4 | 41 | 41 | 40 | 39 |
| Link5 | 41 | 41 | 40 | 38 |
| Link6 | 41 | 41 | 40 | 36 |
| Link7 | 41 | 41 | 40 | 24 |

Case `seed2026120084_0038` has no feasible instantaneous EE-CLF decrease
(`min dot(V) = +8.34e-6`), but every link has a decreasing direction and a
sampled V-decreasing witness. Its minimum barrier is -0.1253 mm, inside the
experiment's 1 mm tolerance but outside the strict nonnegative CBF set.
The hard constraints require recovery as well as joint-limit compliance;
this is not evidence of an EE deadlock strictly inside the safe set. Its
dominant EE LP constraints are obstacle3–link6, obstacle9–link3,
obstacle8–hand and joint6 upper. The other no-EE-descent case,
`seed2026120114_0048`, already has a -2.1876 mm barrier and is not counted
as a safe sampled witness. In 40/41 timeout states at least one unmet link
has a safe sampled decreasing witness.

These observations reject the strong claim that Direct's explicit stalls
necessarily have no available EE descent. They support the narrower
mechanism that changing the task gradient can expose more local descent
under the same constraints in some configurations. They do not establish
this as Sequential's dominant success mechanism. Starting from the
original initial states, Sequential succeeded in cases `0006_0014` and
`0030_0039`, and failed in the other two; those are not rollouts switched
at Direct's stalled configurations.

Artifacts:
`outputs/panda_all_links_analysis/direct_stall_descent_20261024T103317Z/`.
`terminal_results.csv` contains all 360 terminal task rows with refined
witness flags and velocities; `stalled_results.csv` contains the primary
32 rows; `results.csv` / `rows.json` retain all 1,080 task-state results.
`summary.json` combines the initial and refined analyses, while
`complete.json`, `refinement_complete.json` and `verification.json` record
hashes and checks. Raw reconstructed constraint problems remain in the
same relative UBAI directory under `states/`. Jobs `1046181` and `1046194`
completed; no simulation campaign, scientific configuration or controller
source was changed for this diagnostic.

## Manuscript synthesis: quantitative evidence and interpretation (2026-10-26)

다음 정리는 기존 결과 파일을 재확인한 논문용 초안이다. 새로운 실험이나 조건 변경은
포함하지 않는다. 모든 Sequential 결과는 link1–link7 전체를 순서대로 추종한 뒤
최종 EE 자세를 추종하는 방법에 해당한다. 학습 모델이 두 링크를 선택하는 결과와는
구분한다. 논문의 핵심 해석은 다음과 같다.

> Sequential은 중간 링크 목표를 통해 오차 감소 방향을 변경한다. 이러한 목표 변경은
> 일부 상태에서 동일한 안전 제약 아래 더 큰 국소적 진행 여지를 제공하며, 관절 한계
> 근처에서의 지속적인 정체를 완화하는 데 기여할 수 있다.

**표 1. IK가 확보된 동일한 1,000장면에서의 150초 비교.**

| 지표 | Direct | 전체 링크 Sequential |
|---|---:|---:|
| 안전 성공 | 458/1,000 (45.8%) | 594/1,000 (59.4%) |
| 해당 방법만 성공 | 124 | 260 |
| 시간 소진 | 443 | 245 |
| 정체 종료 | 89 | 149 |
| 각 방법의 성공 장면 평균 도달시간 | 26.44 s | 50.42 s |
| 공통 성공 334장면의 평균 도달시간 | 24.52 s | 45.30 s |

공통 성공은 334장면, 공통 실패는 282장면이다. 성공률 차이는 +13.6 percentage
points이며, 장면 단위 paired bootstrap 95% 구간은 [9.8, 17.4] points,
양측 exact McNemar p=3.36e-12이다. Sequential의 성공률 향상은 해당 평가 집합에서
확인되지만, 성공 시 도달시간이 더 짧다는 의미는 아니다. 정체 종료 횟수도 Sequential이
더 많으므로 전체 실험에서 정체 자체가 감소했다고 기술해서는 안 된다.

출처: `outputs/panda_link_selection/all_links_comparison/validation-1000-150s-20261023T055640Z-r1/run/analysis.json`.

**표 2. 성공 결과가 다른 두 집합에서의 관절 한계 관련 지표.**

| 지표 | Sequential만 성공: Direct | Sequential만 성공: Sequential | Direct만 성공: Direct | Direct만 성공: Sequential |
|---|---:|---:|---:|---:|
| 집합 크기 | 260 | 260 | 124 | 124 |
| 관절 CBF 근사 활성 경험 장면 수 | 259 | 259 | 122 | 120 |
| 관절 한계 근처에서 근사 활성 경험 | 245 | 90 | 27 | 108 |
| 종료 전 10초 중 절반 이상 동일 관절의 근접 활성 지속 | 209 | 22 | 15 | 83 |
| 최종 관절 한계 여유가 0.02 rad 이하 | 240 | 36 | 19 | 106 |
| 공통 시간 구간의 평균 근접 활성 지속시간 | 26.1764 s | 2.0445 s | 2.0865 s | 2.8766 s |
| hard-stop clipping 경험 장면 수 | 197 | 2 | 0 | 64 |

근사 활성은 적분 직전 상태와 **명령 속도**로 복원한 관절 CBF 경계 잔차가
0.001 rad/s 이하이고, 해당 CBF 경계가 별도 속도·가속도 경계보다 엄격한 경우다.
근접 활성은 같은 관절 경계까지의 거리가 0.02 rad 이하인 경우로 제한한다. 수치적
잔차 위반은 근사 활성과 분리한다. hard-stop clipping은 적분 시 물리적 관절 범위를
벗어나려는 이동을 잘라낸 사건이며, 관절 CBF 활성이나 실제 관절 범위 위반과 동의어가 아니다.
공통 시간 구간은 각 장면에서 [0, min(T_D, T_S))이며, 첫 집합에서는 초기 QP 실패로
명령이 없는 한 쌍을 제외한 259쌍, 둘째 집합에서는 124쌍으로 평균을 계산했다.
첫 집합의 장면 수 열에서 그 초기 실패의 활성 여부는 미관측이며 비활성으로 해석하지 않는다.

Sequential만 성공한 집합에서 평균 근접 활성 지속시간은 92.19% 감소했다. 160장면에서는
Direct에만 근접 활성이 있었고, 242장면에서는 공통 구간의 지속시간이 0.5초 넘게 줄었다.
반대 집합에서는 85장면에 Sequential만 근접 활성이 있었고, 66장면은 Direct가 이미 종료한
후에야 Sequential의 근접 활성이 처음 나타났다. 따라서 전체 궤적의 사건 수와 공통 시간
구간의 지속시간을 함께 제시해야 한다. 두 집합은 성공 결과로 선택되었으므로 이 수치를
전체 1,000장면의 무조건적 활성 감소율이나 인과 효과로 일반화할 수 없다.

잔차 허용치를 0.0001 rad/s로 줄여도 근접 활성 장면 수는 각각 245 대 88,
27 대 98로 나타나 두 집합의 대비 방향은 유지되었다.

출처: `outputs/panda_all_links_analysis/seq_only_260_150s_20261023/` 및
`outputs/panda_all_links_analysis/direct_only_124_150s_20261023/`의 `analysis.json`,
`paired_260.csv`, `paired_124.csv`.

**표 3. 지속적 구속과 진행 부족의 동시 관찰.**

| 분석 대상 | 동일 관절 근접 활성 지속 + 활성 목표 CLF 감소 1% 이하 | 해당 패턴 중 경계 바깥쪽으로 향하는 감소 방향도 관찰 |
|---|---:|---:|
| Sequential만 성공한 집합의 Direct 시간 소진·정체 실패 250건 | 180/250 (72.0%) | 178/180 |
| Direct만 성공한 집합의 Sequential 시간 소진·정체 실패 121건 | 61/121 (50.4%) | 54/61 |

판정 창은 종료 전 10초이고, Sequential은 최종 활성 단계에 진입한 이후 구간만 사용한다.
지속적 구속은 해당 창의 절반 이상에서 같은 관절의 근접 활성이 유지되는 조건이다.
감소 방향의 경계 충돌은 0.2초 간격으로 검사한 상태의 절반 이상에서 관찰되어야 한다.
Sequential의 61건은 모두 중간 링크 단계에서 종료했다. 이 표는 관절 한계와 목표 감소
방향의 충돌이 실패와 동반된다는 근거이며, 그 제약을 제거하면 성공한다는 반사실적
결과를 제공하지는 않는다.

**표 4. 오차 크기 가설을 보완하는 관찰 및 목표 분할 실험.**

| 분석 또는 비교 | 정량 결과 | 해석 범위 |
|---|---|---|
| Direct 초기 위치 오차의 최저·최고 5분위, 각 200장면 | 성공률 58.5% 대 37.5% | 작은 초기 위치 오차와 성공의 연관성 |
| 초기 방향 오차·관절 여유·barrier 여유 보정 회귀, 1,000장면 | 위치 오차 +10 cm당 성공 odds ratio 0.920, 95% CI [0.883, 0.959] | 관찰적 연관성이며 인과 효과나 %p 변화가 아님 |
| Direct 실패·Sequential 성공 중 시간 소진·정체 250장면 | 27장면에서 Direct가 Sequential 최종 EE 단계 진입보다 위치·방향 모두 작은 오차에 도달; 26장면은 그 시점까지 안전 기준 충족 | 오차 크기만으로 수렴을 결정할 수 없음 |
| 고정 무작위 100장면, 동일 150초 | Direct 48%, EE 2단계 36%, 4단계 32%, 8단계 31%, 전체 링크 Sequential 58% | 시험한 EE 직선/SLERP 분할만으로는 성공률이 개선되지 않음 |

100장면 실험의 초기 위치 오차 중앙값은 Direct 75.48 cm에서 EE 2/4/8단계의
37.74/18.87/9.43 cm로 줄었다. 다만 중간 목표의 안전한 IK를 모두 확인한 장면은
각각 79/66/61개였고, 확인 실패가 불가능성의 증명은 아니다. 따라서 이 비교는 오차
크기만의 효과를 고립시킨 실험이 아니다. 이 100장면에서 Sequential–Direct 성공률
차이 +10 points의 paired 95% 구간은 [-3, 23], exact McNemar p=0.174다.
표 1의 1,000장면과 표 4의 100장면 결과를 독립 표본처럼 합산하지 않는다.

출처: `outputs/panda_all_links_analysis/error_hypothesis_1000_20261024/analysis.json` 및
`outputs/panda_ee_waypoint_comparison/random100-150s-20261024T091721Z/analysis.json`.

**표 5. 같은 Direct 정체 상태에서의 국소적 감소 가능성.**

현재 상태와 이전 명령을 고정한 허용 속도 집합을 U(q, u_prev)라 두고, 장애물·자기충돌·
관절 CBF 및 속도·가속도 제한을 모두 유지한다. 활성 목표만 바꾸어 각 목표 k에 대해
다음 값을 구했다.

\[
D_k=\max_{u\in\mathcal U(q,u_{\mathrm{prev}})}[-\nabla V_k(q)^\top u],
\qquad
V_k=\tfrac12(\|e_{p,k}\|^2+\|e_{R,k}\|^2),
\qquad
\rho_k=100\frac{D_k}{2V_k}\quad[\%/\mathrm{s}].
\]

여기서 e_R은 quaternion 오차 벡터이며, rho는 E_k=sqrt(2V_k)의 순간 상대 감소율이다.
E는 제어기의 위치·방향 결합 지표로 물리적 거리 자체가 아니다. 각 목표는 별개의
속도로 최적화한다. 이미 중간목표 허용치에 들어온 링크는 아래 최대 링크 선정에서 제외했다.

| 장면 ID의 `seed202612` 이후 부분 | EE rho (%/s) | 사후 최대 미달성 링크 | 해당 링크 rho (%/s) | 링크/EE 비율 |
|---|---:|---|---:|---:|
| 0006_0014 | 0.06645 | link1 | 16.23167 | 244.26 |
| 0019_0001 | 1.08639 | link1 | 0.17510 | 0.16 |
| 0030_0039 | 3.64753 | link4 | 46.83381 | 12.84 |
| 0108_0015 | 1.31319 | link6 | 23.09910 | 17.59 |

이 네 상태는 수정된 QP를 사용한 최근 100장면 실험의 Direct `stalled` 종료 전부이며,
관절 한계 및 비음수 sampled barrier 기준을 만족했다. 4/4 상태에서 EE 감소 방향이
존재하므로 정체를 EE 감소 방향의 완전 소멸로 해석할 수 없다. 그중 3/4에서는 미달성
링크의 상대 감소 여지가 EE보다 컸고, 1/4에서는 반대였다. 서로 다른 목표의 상대 감소율
비교이며 최종 EE에 대한 기여량이나 성공 가능성의 직접 비교가 아니다. 사후 최대 링크는
진단 목적으로 선택했으며 실제 Sequential 정책이 그 링크로 전환했다는 뜻도 아니다.

0.02초 유한 단계 검사에서도 4/4 EE 상태와 20/20 미달성 링크 목표에서 V와 정체
판정용 R=max(position/tolerance, orientation/tolerance)를 함께 감소시키는 속도를
찾았다. 이때 동일한 hard constraint를 만족하고 다음 상태의 관절 범위 및 비음수
barrier도 확인했다. 예를 들어 0006_0014에서 찾은 한 단계 E 감소는 EE 0.000951%,
link1 0.324683%였다. 이는 유한 후보 탐색에서 확인한 감소이며 전역 최적 유한 단계
감소량, 연속 구간 전체의 충돌 부재 또는 최종 수렴을 증명하지 않는다.

보조 분석의 시간 소진 41건에서는 39건에 EE의 순간 감소 방향이 있었고, 34건에서
안전 기준을 만족하는 한 단계 V 감소, 24건에서 V와 R의 동시 감소를 확인했다. 40건에서는
최소 한 개의 미달성 링크에 안전 기준을 만족하는 한 단계 감소가 있었다. EE 감소 방향이
없는 두 상태의 최소 barrier는 각각 -0.1253 mm와 -2.1876 mm였으므로 strict safe set
내부에서의 정체를 뒷받침하는 사례로 사용하지 않는다. 후보 탐색에서 유한 단계 감소를
찾지 못했다는 사실은 그 감소가 불가능하다는 증명이 아니다.

출처: `outputs/panda_all_links_analysis/direct_stall_descent_20261024T103317Z/`의
`summary.json`, `terminal_results.csv`, `refined_witnesses.json`.

**논문 본문 초안.**

동일한 1,000개의 IK 가용 장면에서 각 방법에 총 150초의 실행 시간을 부여한 결과,
전체 링크 Sequential은 59.4%의 안전 성공률을 보였으며 Direct의 45.8%보다
13.6 percentage points 높았다(paired bootstrap 95% CI: 9.8–17.4 points;
exact McNemar p=3.36e-12). Sequential만 성공한 장면은 260개, Direct만 성공한
장면은 124개였다. 이 결과는 해당 평가 집합에서의 성공률 향상을 보여주지만 모든
장면에서의 우월성이나 더 빠른 도달을 의미하지 않는다. 실제로 공통 성공 334장면의
평균 도달시간은 Direct 24.52초, Sequential 45.30초였다.

성공 결과가 다른 장면들의 궤적을 분석한 결과, Sequential만 성공한 집합에서는
단순한 관절 CBF 활성 경험 횟수보다 관절 한계 근처의 지속적 구속에서 큰 차이가
관찰되었다. 근접 활성을 경험한 장면 수는 Direct 245개, Sequential 90개였으며,
종료 직전 동일 관절의 근접 활성이 지속된 장면 수는 각각 209개와 22개였다. 두 방법의
공통 실행 구간으로 관찰 시간을 맞춘 259쌍에서도 평균 근접 활성 지속시간은
26.18초에서 2.04초로 92.19% 감소하였다. 또한 이 집합의 Direct 시간 소진·정체
실패 250건 중 180건에서 지속적 근접 활성과 낮은 CLF 진행이 함께 나타났으며,
그중 178건에서는 오차 감소 방향이 해당 관절 한계 바깥쪽을 향하는 양상도 관찰되었다.
반대로 Direct만 성공한 집합에서는 Sequential의 근접 활성이 더 자주 나타났다.
따라서 이 결과는 Sequential이 관절 제약을 일괄적으로 회피한다는 주장보다, 목표와
제약의 관계에 따라 지속적 구속을 완화하거나 반대로 유발할 수 있다는 해석을 지지한다.

목표 변경에 따른 차이를 더 직접적으로 조사하기 위해 별도의 고정 100장면 비교에서
발생한 네 Direct 정체 상태를 재구성하였다. 같은 관절각, 이전 명령 및 hard constraint를
유지하고 목표별 CLF 감소율을 최적화한 결과, 네 상태 모두에서 EE 오차를 줄일 수 있는
속도가 존재하였다. 그러나 세 상태에서는 미달성 링크 목표의 최대 순간 상대 감소율이
EE보다 컸으며, 0.02초 후에도 제약을 만족하면서 링크 오차가 감소하는 속도를 확인하였다.
이는 Direct의 정체를 감소 방향의 완전한 부재로 설명하기보다, 현재 목표의 감소 방향과
허용 속도 집합 사이의 제한된 양립 가능성 및 실제 QP의 속도 비용·slack 절충으로
해석해야 함을 시사한다. 중간 링크 목표는 안전 제약을 느슨하게 하지 않고도 오차 감소
방향을 변경하여 일부 구성에서 더 큰 국소적 진행 여지를 제공할 수 있다.

작은 초기 위치 오차와 Direct 성공 사이의 연관성도 관찰되었으나, 시험한 EE 직선/SLERP
목표 분할은 성공률을 높이지 못하였다. 또한 Direct가 Sequential의 최종 EE 단계 진입보다
작은 위치·방향 오차에 도달하고도 실패한 사례가 존재하였다. 따라서 현 결과는 오차 크기
축소만으로 Sequential의 이점을 설명하기보다, 중간 목표가 유도하는 자세와 감소 방향이
제약과 상호작용하는 과정도 함께 고려해야 함을 보여준다. 종합하면, 중간 링크 목표에
의한 감소 방향의 변경과 일부 장면에서 관찰된 관절 한계 근처의 지속적 구속 완화는
Sequential의 성공률 향상에 기여할 수 있는 메커니즘을 뒷받침한다. 다만 이러한 분석은
그 메커니즘의 인과적 기여도나 지배성을 확정하지 않는다.

**Methods/Limitations에 함께 명시할 사항.**

- 성공은 총시간 내 최종 pose 허용치(위치 1 cm, quaternion 오차 벡터 norm 0.01)를
  만족하고 기록된 궤적의 안전 기준을 충족한 경우다. 중간목표 허용치는 5 cm / 0.05로
  다르다. 적분 주기는 0.02초이고 barrier 안전 판정에는 1 mm 허용치가 적용되며,
  관절 범위는 hard stop으로 제한된다. 이는 연속시간 전체의 안전 보장을 검증한 것은 아니다.
- 1,000장면은 IK가 확보된 기존 검증 집합이며, 표 1–3과 표 4의 1,000장면 관찰 통계는 과거 QP 구현의
  기록이다. 해당 기록에는 nominal joint-CBF 잔차를 0.001 rad/s 넘게 위반하는 명령도
  포함된다. 이 사건은 활성과 구분되며, 상태 안전 성공을 모든 QP 명령의 엄밀한 제약
  만족과 동일시하지 않는다. 최근 100장면은 수정된 QP로 모든 방법을 다시 실행했다.
- 260/124장면 집합은 성공 여부로 선택된 집합이다. 두 집합의 구속 차이는 인과 효과의
  추정치가 아니며, 최근 네 정체 상태의 국소 분석을 전체 장면에 대한 빈도로 일반화하지 않는다.
- 국소 비교는 동일한 hard feasible set에서 수행하지만 서로 다른 목표의 상대 오차
  감소율을 비교한다. 별개의 목표별 속도, 사후 최대 링크, 중간목표 허용치 및 실제 QP의
  비용 절충을 구분한다. 정체 상태에서 실제로 Sequential로 전환한 뒤 최종 성공을 확인하는
  개입 실험은 수행하지 않았다.
- EE 분할 실험에서는 모든 중간 목표의 실현 가능성을 보장하지 못했으며, 100장면은
  1,000장면에서 뽑은 부분집합이다. 두 결과를 독립적인 반복 실험이나 단일 코드 버전의
  통합 통계처럼 제시하지 않는다. 통계적 성공률 차이는 이 평가 집합에 대한 결과로 보고한다.

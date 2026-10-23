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

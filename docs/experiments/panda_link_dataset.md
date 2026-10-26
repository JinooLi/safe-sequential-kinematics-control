# Fixed-IK link selection dataset

Submit from the repository root on UBAI:

```bash
uv sync --locked --extra cuda --python 3.11 --no-dev
sbatch scripts/panda_link_dataset.sbatch
```

The pilot requests one node with four available GPUs and 15 CPUs. Preparation
first uses one worker per GPU, each with two CPU IK search subprocesses. It finds
one shared IK per context, then shuts down that pool. Rollouts use four persistent
workers per GPU: up to 16 independent trajectories run concurrently. Each task
is one context/link pair, so ten contexts provide 100 schedulable tasks rather
than limiting concurrency to ten. Every pair reuses its context's fixed IK;
rollout workers never launch additional IK searches. GPU assignments round-robin
over Slurm's allocated devices, including GPU UUIDs.

BLAS/OpenMP and Torch CPU threads are limited to one. The 16 rollout processes
share the 15 allocated CPU cores (plus a mostly idle coordinator); 16 processes
do not imply 16 reserved cores. CPU kinematics/QP work and GPU contention can limit the speedup. GPU utilization alone is not a measure
of end-to-end throughput. The IK seed/iteration budget and control equations are unchanged.
As of 2026-10-16, safety scoring uses every recorded integration state
(`validation.subdivisions: 1`), including the terminal state. There are no
interpolated checks or duplicate post-rollout geometry evaluations.
Use `--workers-per-gpu 1` for the four-worker layout, or `4` (default) for 16.
The Slurm wall limit is four hours; each trajectory's simulated limit is 120 s.

`configs/experiments/panda_link_dataset.yaml` fixes the current controller values
and requests ten eligible contexts. Each context contains a safe randomized
initial configuration, ten obstacles, a goal pose and one validated nearest IK.
All ten ascending link pairs run Sequential from exactly that same state and IK.
No IK switching and no Direct runs are performed. A context without validated IK
is retained with its reason and timing, excluded from supervised inputs, and
replenished. Up to three sampling rounds are allowed; the base seed increments
for each round. No replacement depends on rollout success. A depleted budget
produces `failed.json`, never a false completion. Existing output directories
are refused rather than overwritten; interrupted output is preserved for review.

Outputs under `outputs/panda_link_dataset/pilot-JOBID/`:

- `cases.jsonl`: all sampled scenes, including excluded no-IK scenes.
- `contexts.jsonl`: selected IK and preparation timing for each attempted context.
- `dataset.npz`: arrays `case_id`, `initial_q[N,7]`, `selected_ik[N,7]`,
  `obstacles[N,10,4]` (x,y,z,r), goal pose, `link_pairs[10,2]`,
  `success[N,10]`, and `label_valid[N,10]`. Load with `allow_pickle=False`.
  Invalid numerical solver results are masked and have label -1, not failure 0.
- `results.csv` / `results.json`: one row per context/pair, including termination,
  success, safety, GPU, PID and start/end timestamps for concurrency checks.
  Normal timeout/stall/infeasibility have valid labels.
- `CASE/linkI_linkJ/trajectory.npz`: full trajectory and applied/commanded speeds.
- `manifest.json`: code/model/config/data hashes, exact configuration snapshots,
  software/GPU environment and sampling rejection statistics.
- `timing.json`: total elapsed generation time, serial sampling time, parallel
  IK and rollout phase times, and summed per-task IK/control/interpolation times. Summed
  times overlap across workers and must not be added to elapsed wall time.
- `progress.json`: preparation/rollout phase and counts, updated after every task.
  Only the coordinator writes shared context/progress metadata.
- `complete.json`: validated context/rollout counts and manifest hash.

Primary success is first final EE pose hit within the current position and
orientation tolerances, with all sampled barriers at least -1 mm and joint limits
respected. Legacy `interpolated_*` columns are retained for compatibility: with
subdivisions=1 they equal the sampled metrics, `interpolation_checks` is zero,
and `safety_validation_mode` is `iteration_states`. Historical five-subdivision
outputs remain unchanged. Success does not require matching the final IK joint
configuration. Safe randomized starts require +5 mm **barrier** margin and 0.1 rad
joint margin under the existing model, not a hardware collision guarantee.

For training, keep every pair belonging to a context in the same data split.
The ten labels within one context are not ten independent scenes. The pilot is
for validating data and estimating costs; it is not a success-rate benchmark.

## 1,000-context campaign

```bash
sbatch scripts/panda_link_dataset_1000.sbatch
```

This submits **one** job requesting four GPUs, **25 CPUs** and a 72-hour wall limit.
It creates 1,000 new eligible contexts (10,000 trajectories), in 20
batches of 50 contexts. One batch uses the 16 rollout workers while the next
batch prepares its IK candidates. The pilot YAML/controller/scientific settings are preserved in resolved
input snapshots. New batch seeds begin at 2026101700 and advance by three, reserving
distinct seeds for each batch's three possible IK replenishment rounds. The
earlier ten-context pilot is not included in the new dataset. Neither batching
nor replenishment filters contexts based on trajectory success.

The earlier five-subdivision 1,000-context campaign finished in 30 h 23 min.
That is a historical measurement, not a benchmark of the optimized pipeline.
GPU type, scene difficulty and contention affect throughput.

Outputs are under `outputs/panda_link_dataset/production-1000-JOBID/`:

- `inputs/`: fixed controller and per-batch experiment configurations.
- `campaign.json`: requested size/resources, code/model/input hashes, batch seeds,
  commands, PIDs, job IDs and attempt history.
- `progress.json`: completed totals plus current batch progress, refreshed while
  the child runs. `active_output` identifies that batch's directory.
- `batch-000-attempt-01/`, etc.: independent pilot-format datasets, trajectory
  files and verified completion markers, usable as each batch finishes.
- At full completion, root `dataset.npz`, `contexts.jsonl`, `results.csv` and
  `results.json` combine all batches; `complete.json` records counts and hashes.
  The final success array has shape `[1000, 10]`; numerical errors remain masked.
  Trajectories stay inside their batch directories.

A failed preparation or rollout stops the pipeline and its worker process groups.
At most one additional batch is being prepared; no subsequent rollout is started
after an observed failure. After the previous
Slurm job has stopped, resume explicitly using the **original** output path:

```bash
PANDA_DATASET_OUTPUT=/absolute/path/to/production-1000-ORIGINAL_JOBID \
PANDA_DATASET_RESUME=1 sbatch scripts/panda_link_dataset_1000.sbatch
```

Resume checks the code/config/model identity and every completed batch's data
hashes. Complete batches are reused. An incomplete batch is rerun from its fixed
seed in a new attempt directory; partial files are preserved. Interrupted rollout and lookahead preparation directories are preserved.
On explicit resume those incomplete batches get new attempt directories. There is
no automatic retry, requeue or recurring monitor, and a campaign lock prevents
two coordinators from writing the same campaign. Resources are released when the
Slurm job ends.

## Value-only checks and overlapping preparation

IK search itself still uses its original gradients and candidate budget. Only
candidate endpoint revalidation uses the new float64 value-only path. Multiple
candidate states are batched on the GPU, with state chunks of eight and at most
64 active state/pair rows per tensor chunk. It retains the RDF gate, soft-min,
top-k tie ordering, safety margins and nearest-candidate selection. CPU forward
kinematics skips Jacobians for these checks and lazily computes them again when
control needs gradients. The online CLF/CBF QP still receives full derivatives.

`--prefetch-ik --ik-context-workers 2` overlaps the next batch with the current
rollout. With four GPUs and four rollout workers per GPU the checked CPU budget
is 25: 16 rollout workers, two preparation workers, their four IK subprocesses,
and three coordinator/sampling processes. BLAS/Torch remain single-threaded.
The preparation workers share the first two allocated GPUs only for setup,
sampling and endpoint checks. Their capsule IK searches run on CPU. Preparation
is bounded to one batch ahead, and no IK candidate changes during a trajectory.

The runner supports `--phase prepare` and `--phase rollout`. `preparation.json`
records immutable input/IK hashes; rollout verifies them and claims the output
exclusively. No pair is eligible until its context has a validated shared IK.
`progress.json` reports both the active rollout and lookahead preparation. Batch
wall time includes waiting between preparation and rollout; overlapping batch
wall times must not be added to estimate total campaign elapsed time.

## Multiple independent GPU groups

```bash
sbatch scripts/panda_link_dataset_groups.sbatch
```

The array requests three independent jobs, each with four GPUs and 25 CPUs
(up to 12 GPUs / 75 CPUs concurrently if the scheduler has capacity). Each group
can be placed on a different node; no multi-node GPU communication is required.
The **global** target remains 1,000 contexts, not 1,000 per group. Global batch
indices are partitioned round-robin (350/350/300 contexts for three groups).
Seeds derive from the global batch index, so adding groups never duplicates
scenes. All ten link-pair labels for a scene remain together.

`PANDA_DATASET_CONTEXTS`, `PANDA_DATASET_BATCH_SIZE`, `PANDA_DATASET_SEED` and
`PANDA_DATASET_OUTPUT` configure both production scripts. Choose a fresh seed
range and output directory for each new dataset. For two groups use
`sbatch --array=0-1 scripts/panda_link_dataset_groups.sbatch`; array indices must
start at zero and be contiguous. Outputs use `production-groups-ARRAY_JOBID/group-I`.
After **all** groups finish, merge with:

```bash
.venv/bin/python -m safe_ik.application.panda_link_dataset_merge \
  --groups outputs/panda_link_dataset/production-groups-ARRAY_JOBID/group-{0,1,2} \
  --output outputs/panda_link_dataset/production-groups-ARRAY_JOBID/merged
```

The merger verifies hashes, shared settings/source identity, expected group and
batch coverage, and unique scene/pair keys before writing a new combined dataset.
It refuses missing/duplicate groups and existing output directories. Trajectories
remain in their original group/batch directories. No job is submitted or retried
by the merger. Completed datasets made with the old source/settings are preserved;
source identity checks intentionally prevent resuming them with the new code.

## 7,000-context campaign submitted 2026-10-16

UBAI job **1020552** started on **2026-10-16 19:09:11 KST** on `n013`.
The user expanded the new-data target from 4,000 to **7,000 eligible scenes**
(70,000 Sequential trajectories) with a target of Monday, October 19, 09:00 KST.
These are additional scenes; the two existing 1,000-scene datasets are preserved.

- Resources: four RTX 3090 GPUs, 25 CPUs, up to 16 concurrent rollout workers,
  two IK context workers and one-batch preparation lookahead.
- Size: 140 batches of 50 eligible scenes. Reserved seeds are
  `2026110000` through `2026110419`, including three replenishment seeds per
  batch. This range does not overlap either previous production campaign.
- Settings: 120 s simulated duration, 0.02 s integration period, slack weight
  5,000, alpha 0.2, RDF CBF gain 1.0, 1 mm barrier tolerance, safe random starts,
  ten obstacles, fixed nearest validated IK without switching, and safety
  evaluation at every integration state. All ten ascending link pairs run.
- Before launch, the full generation identity, including 95 source files and
  fixed configuration/model inputs, matched completed job `1020107`. The new
  campaign's recorded identity also matched after launch. No generator source
  or scientific settings were changed for this submission.
- The reference 1,000-scene job took 26,629 s (7 h 23 min 49 s). Linear scaling
  gives **51 h 46 min 43 s**, or approximately **October 18, 22:56 KST**.
  This is an estimate; scene difficulty and throughput can change it.
- The scheduler wall limit is 60 h, ending October 19 at 07:09:11 KST for this
  start time, with a scheduling deadline of October 19 at 09:00 KST. A wall-limit
  termination would preserve partial batches and must not be reported as a
  completed dataset. No automatic retry or monitoring schedule was added.

The existing `scripts/panda_link_dataset_1000.sbatch` was submitted with
`PANDA_DATASET_CONTEXTS=7000`, `PANDA_DATASET_BATCH_SIZE=50`,
`PANDA_DATASET_SEED=2026110000`, `PANDA_DATASET_RESUME=0`, and a fresh output path.
The exact submission is recorded on UBAI in
`outputs/panda_link_dataset/submission-1020552.json`.

Remote repository: `/home1/jinu1423/workspace/safe-inv-clbf-robot-arm`.
Campaign: `outputs/panda_link_dataset/production-7000-20261016T100910Z`.
Log: `panda-link-dataset-1020552.log`. Completion must be checked from the
campaign artifacts and data hashes as well as Slurm state.

### Completion verified 2026-10-19

Job `1020552` completed normally at **2026-10-18 21:21:59 KST**, exit `0:0`,
after **50 h 12 min 48 s**, and released its allocation. All 140 batches and
70,000 trajectories completed. The 7,302 attempted contexts include 302 no-IK
contexts excluded and replenished, leaving the requested 7,000 eligible scenes.
The dataset has 38,088 success labels and 142 numerical-error outcomes across
66 scenes. Those errors remain raw invalid/-1 entries and are treated as
failure/0 during supervised training. Under that policy there are 1,434
all-failure, 1,856 all-success and 3,710 mixed scenes.

The five root artifact hashes, all 140 batch completion-marker and manifest
hashes, scene/output counts, and zero IK-switch counts were verified. The root
NPZ SHA256 is
`477f603323b8bd645d35429e1a95f013c8077f70353ab17ee84b3f9f572437ce`.
On October 19 the six exported data/metadata files were copied to lab-server's
`outputs/panda_link_selection/datasets/production-7000-20261016T100910Z`, where
the hashes were verified again. The full per-trajectory archives remain on UBAI.

## Fresh 2,000-scene evaluation campaign (2026-10-19)

UBAI job **1024550** (`panda-eval-2000`) started on **2026-10-19
12:03:50 KST** on `n004`. It requests four RTX 3090 GPUs, 25 CPUs and
a 24-hour wall limit, with the existing 16 rollout workers, two IK context
workers and one-batch preparation lookahead.

This campaign produces **2,000 new eligible scenes / 20,000 trajectories**,
reserved for **1,000 validation and 1,000 test scenes**, not training.
Before submission, a split plan assigns even zero-based batch indices to
validation and odd indices to test. Each of the 40 batches contains 50 eligible
scenes; all ten link-pair outcomes for a scene stay in its assigned split.
The assignment is independent of outcome labels. Test selection performance
must remain uninspected until the model-selection procedure is fixed.
Numerical-error flags remain in raw data and count as failure/0 in downstream
evaluation, consistent with the previous training policy.

The base seed is **2026120000**, reserving **2026120000–2026120119** including
the three possible replenishment seeds per batch. These seeds do not overlap
any prior production campaign or pilot manifest on UBAI. No-IK attempts are
retained and replenished under the existing policy; rollout outcomes never
determine replenishment.

The complete source/config/model identity, including all 95 source files,
matches the completed 7,000-scene job 1020552. The existing
`scripts/panda_link_dataset_1000.sbatch` was submitted without changing
generator code or scientific settings: 120 s, dt 0.02 s, slack weight 5,000,
alpha 0.2, RDF CBF gain 1.0, 1 mm barrier tolerance, safe random starts,
ten obstacles, fixed nearest validated IK without switching and safety checks
at every integration state. Post-start campaign identity and input hashes match.

UBAI repository: `/home1/jinu1423/workspace/safe-inv-clbf-robot-arm`.

- Output: `outputs/panda_link_dataset/evaluation-2000-20261019T030326Z`.
- Submission record: `outputs/panda_link_dataset/submission-1024550.json`.
- Immutable split plan:
  `outputs/panda_link_dataset/evaluation-2000-20261019T030326Z-split-plan.json`.
- Split-plan SHA256:
  `956e42783f6e728d5749b62a09f1fea9388dfc689114304de4a41dac79045f3a`.
- Log: `panda-link-dataset-1024550.log`.

The submission record and split plan are also copied locally under
`outputs/panda_link_dataset/evaluation-2000-20261019T030326Z/`.
The job was held during submission recording and then released once; no retry
or recurring monitor was created. The output did not exist before launch.

Linear scaling from the 7,000-scene job's 50 h 12 min 48 s gives **14 h 20 min
48 s**, approximately **October 20, 02:25 KST** from this start. This is a
throughput estimate, not a completion claim; runtime can vary by scene.
At submission verification, the job is running and preparing its first batch.

### Completion and evaluation export (2026-10-20)

Job 1024550 completed normally (Slurm COMPLETED, exit 0:0) on October 20 at
02:29:06 KST, after 14 h 25 min 16 s. All 40 batches completed: 2,000 eligible
contexts and 20,000 rollouts. There were 2,090 attempted contexts, including
90 no-IK attempts retained in metadata and replenished. Raw valid-label count
is 19,934; the other 66 numerical-error outcomes count as failure/0 in the
downstream evaluation. There are 1,970 contexts with all ten raw labels valid;
no eligible context was removed from evaluation.

All five data hashes in `complete.json` were independently verified on UBAI,
then again after copying the six exported data/metadata files to lab-server.
The root dataset NPZ SHA256 is
`b5b4196c42ec09a212335257d36d8207c4ad2b5e8b92e72f6f83668127b612f6`.
The predeclared 1,000-validation/1,000-test split is preserved. The export is in
lab-server's `outputs/panda_link_selection/evaluations/evaluation-2000-20261020T003931Z/dataset/`.
Full trajectories remain on UBAI. No simulations were rerun. See
`panda_link_selection.md` for the frozen-checkpoint evaluation and its limitations
under the historical, pre-QP-fix generator.

## New 3,000-scene campaign (2026-10-20)

The user requested 3,000 additional scenes after the fresh evaluation. UBAI job
**1025669** (`panda-data-3000`) is running on **n004** with four RTX 3090 GPUs,
25 CPUs, 64 GiB requested memory and a 36-hour wall limit. It uses the existing
16 rollout workers (four per GPU), two IK context workers and one-batch lookahead.
The target is **3,000 eligible contexts / 30,000 trajectories**, in 60 batches
of 50, not 3,000 contexts per worker or GPU. No training/evaluation job was added.

The seed range **2026130000–2026130179** includes each batch's three possible
IK replenishment rounds and was checked against prior campaign seed ranges.
Scientific settings remain 120 seconds, dt 0.02 seconds, slack weight 5,000,
alpha 0.2, RDF CBF gain 1.0, 1 mm barrier scoring tolerance, hard joint-position
stops, safe random initial configurations, ten obstacles, all ten ascending
link pairs and one nearest validated IK without switching. Safety scoring uses
every integration state. No-IK attempts are retained and replenished; rollout
outcomes do not determine replacement.

This campaign includes the already committed QP failure-handling fix
`7c7e2809200e7d9d5a886eac898d3cfb3f1334fc`. Its isolated runtime is
`/home1/jinu1423/workspace/safe-inv-clbf-robot-arm-dataset3000-20261020T004930Z`.
It combines that commit with the existing UBAI generation optimizations.
Source comparison against the preceding generator found changes only in the
six QP validation/diagnostic propagation files. Configurations, model artifacts
and submission script match the preceding optimized campaign. The original
UBAI checkout and prior datasets remain unchanged. This source change is
recorded explicitly; do not silently pool its labels with pre-fix outcomes.

Before submission, a temporary CPU allocation ran the solver, Panda QP failure,
dataset, campaign, optimization and comparison tests: 55 discovered, 52 passed,
three skipped. Source hashes were frozen after this check. The production job
was submitted held, recorded, then released once with no requeue. At the initial
verification (2026-10-20 09:52 KST), Slurm reported RUNNING on n004, all 60 batches
were registered, the running campaign source hashes matched the prepared
runtime, and the first batch was preparing IK. No failure marker was present.

Output on UBAI:
`/home1/jinu1423/workspace/safe-inv-clbf-robot-arm/outputs/panda_link_dataset/production-3000-20261020T004930Z`.
Submission/configuration/test records and `slurm-1025669.log` are in the sibling
`production-3000-20261020T004930Z-submission/` directory. A local copy of the
submission records is under
`outputs/panda_link_dataset/production-3000-20261020T004930Z/`.

Scaling the previous 2,000-scene wall time gives about 21 h 38 min, excluding
queue time. This is only a historical throughput estimate: the new QP checks
can change rollout termination and hence the actual duration. The source and
settings are fixed for the running campaign; follow-up checks should read its
campaign/progress/log files without altering the runtime.

### Completion and training-data audit (2026-10-21)

Job **1025669** completed with exit **0:0** at **00:49:13 KST** on October 21,
after **14 h 57 min 46 s**. All 60 batches completed: 3,000 eligible contexts,
30,000 rollouts, and 120 additional no-IK attempts retained in metadata.
However, batch/job completion does not establish comparable training labels:

| Outcome | Rollouts |
| --- | ---: |
| QP solution rejected by per-row constraint check | 29,991 |
| Stalled | 6 |
| Goal reached and successful | 3 |

All numerical failures have `qp_constraint_violation` diagnostics after the
backend reported `solved`. Of these, 20,849 occurred at the first command and
27,344 before one simulated second. Their median elapsed simulation time is
zero. All ten choices hit this error in 2,996 scenes. Under the user's existing
numerical-error-as-failure policy, 2,998 scenes have no successful choice; only
two scenes contain a successful choice. The preceding 9,000-scene export had
200 numerical failures among 90,000 rollouts.

The new generator's committed QP validation adds per-row feasibility checks in
original constraint units; the preceding generator accepted backend `solved`
without this check. For example, `seed2026130000_0000`, pair (2,3), stopped at
0.04 s because row 143 had value -0.153757659276 against lower bound -0.145:
residual 0.00875765927634 exceeded tolerance 0.00115375765928. This establishes
the rejection path, not mathematical infeasibility or whether a more accurate
solve would succeed. Solver settings and labels have not been changed.

The five completion-manifest hashes were verified on UBAI, locally, and after
transfer to lab-server. All 95 frozen source hashes match on UBAI. Scene/pair
coverage, fixed shared IK, raw labels and safe randomized starts were checked.
There is no case-ID or exact scene-feature overlap with the old 9,000 training
pool or the separate 2,000 evaluation scenes. The scene fingerprint includes
initial configuration, target position/quaternion and obstacle geometry.

The complete root export is preserved on lab-server at
`outputs/panda_link_selection/datasets/production-3000-20261020T004930Z/`.
Its dataset SHA256 is
`9d2d627d2d1a95a854629bae55424efc3c88b06d143e9f8788b55de6524aee82`.
Local copies are under
`outputs/panda_link_dataset/production-3000-20261020T004930Z/export/`, with the
audit in the parent directory's `data_quality_audit.json`. Full trajectories
remain on UBAI. Existing datasets and checkpoints are unchanged.

The user chose to **hold training and correct/validate QP numerical accuracy
first**. The adapter now normalizes the entire objective, enables polishing,
and, when necessary, refines the same QP within the original iteration budget.
The mathematical QP, relative slack weight and independent row acceptance
thresholds remain unchanged.

On lab-server, all 120 reconstructed initial QPs and three captured intermediate
failure QPs passed with the final adapter. A matched 12-rollout regression using
six saved scenes and pairs (2,3)/(4,5) changed from 12 QP failures to zero QP
failures, four successes, six timeouts and two stalls. All 53,507 accepted QPs
passed an independent per-row residual audit. The 73 relevant regression tests
passed. This small regression is not an estimate of overall success rates.
See [QP failure handling](../current/qp_failure_handling.md) and
`outputs/panda_qp_accuracy/20261021T023508Z/verification.json` on lab-server.

Retraining and full 3,000-scene regeneration have **not started**. The raw export
retains the original invalid labels and validity masks; the incomplete
trajectories cannot be repaired by merely changing their success labels.

## Corrected-QP 6,000-scene campaign (2026-10-21)

The user authorized 6,000 fresh eligible scenes / 60,000 trajectories using
12 UBAI GPUs, with a first-batch quality check and completion before October 22,
08:00 KST. Array job **1030631**, submitted at 12:07 KST, requests three groups
of four GPUs and 25 CPUs (16 rollout workers per group). The global 120 batches
of 50 scenes are partitioned by batch index modulo three: 2,000 scenes per group.
Seeds **2026140000–2026140359**, including no-IK replenishment seeds, do not
overlap preceding production or evaluation campaigns.

The isolated runtime is
`/home1/jinu1423/workspace/safe-inv-clbf-robot-arm-dataset6000-20261021T030100Z`.
It copies the previous 3,000-scene runtime and changes only
`src/safe_ik/adapters/osqp_solver.py` among its 95 source files. The corrected
solver SHA256 is
`b9dc3ca80dfef5f962008def5441aacba605899c3f57717d17398fb336b92ee7`.
All scientific configuration files remain byte-identical: 120-second simulated
limit, dt 0.02, slack weight 5,000, alpha 0.2, RDF gain 1.0, 1 mm scoring tolerance,
joint-position stops, safe random starts, ten obstacles, fixed nearest validated
IK without switching and safety checks at every integration state.

Each allocated node checks all four CUDA devices and the solver regression
tests. Group zero also runs dataset, campaign, optimization and comparison
regressions. A submission wrapper runs the first 50-scene batch in each group
using the existing generator's prepare/rollout commands. It verifies completion
hashes, scene/pair coverage and no IK switches before releasing the remaining
campaign through its existing resume interface. All three first batches must
pass. The operational gate stops the submission if a first batch has more than
5% QP failures or more than 1% initial-step constraint-check failures. This gate
does not discard, relabel or replace any rollout; all first-batch data are part
of the 6,000 requested scenes. Raw numerical-error masks are retained for the
existing downstream error-as-failure policy.

The array has a 19-hour Slurm wall limit, no requeue and a scheduling deadline
of October 22, 07:45 KST. An absolute-time wrapper independently stops at
07:45 regardless of queue delay, leaving 15 minutes before the user's deadline.
These are termination limits, not a guarantee that all scenes finish. On an
execution/quality-gate error the wrapper cancels this array's three allocations.
After all groups complete, the last group verifies and merges their exports
with the existing merge command; it does not submit another job.

UBAI output:
`/home1/jinu1423/workspace/safe-inv-clbf-robot-arm/outputs/panda_link_dataset/production-6000-20261021T030100Z`.
Submission, source hashes, plan and logs are in the sibling
`production-6000-20261021T030100Z-submission/` directory. The root
`first-batch-{0,1,2}.json` files record the quality checks. Final completion
requires all group markers and a verified 6,000-scene / 60,000-rollout export;
Slurm completion alone is insufficient. Local submission copies are under
`outputs/panda_link_dataset/production-6000-20261021T030100Z/submission/`.

An initial launch, job 1030626, stopped before scene generation because the
copied runtime lacked Git metadata used by the manifest. Its allocation was
cancelled, the metadata reference was restored and checked, and the original
artifacts were preserved in `production-6000-20261021T030100Z-launch-failed-1030626/`.
Job 1030631 uses the same seeds and scientific conditions. No training job or
recurring monitor was created.

### First-batch verification and production release

All three groups passed the first-batch gate by **2026-10-21 12:46:41 KST**.
The jobs run on n013/n014 (eight RTX 3090 GPUs) and n060 (four A10 GPUs).
The 58 relevant regression tests passed on group zero; the 15 solver tests
also passed independently on both other nodes. Live batch manifests confirmed
the frozen configuration and corrected solver hash on all three nodes.

| Group | Scenes | Rollouts | Successes | QP failures | IK / rollout wall time |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 50 | 500 | 226 | 7 | 14.44 / 23.30 min |
| 1 | 50 | 500 | 257 | 15 | 15.26 / 20.50 min |
| 2 | 50 | 500 | 272 | 6 | 13.75 / 20.89 min |

An independent metadata audit verified all first-batch manifest/data hashes,
150 disjoint eligible scene IDs, 1,500 unique scene/pair outcomes and zero IK
switches. The 28 numerical failures (1.87%, across 22 scenes) remain preserved
under the existing failure-label policy. These include failures at 0.04 s;
the gate's narrower initial-step counter (at most 0.02 s) is not evidence that
all early numerical failures disappeared. No successful or failed rollout was
replaced. The first batches are retained and skipped by the normal resume path.

The first-batch timing extrapolation, using one additional preparation startup
and 39 overlapping prepare/rollout cycles, places the slowest group near
**October 22, 04:10 KST**. This excludes future scene variability and additional
prefetch contention, so a practical planning range is **04:00–06:00 KST**, not
a completion guarantee. The 19-hour Slurm limits currently end around
07:07–07:08 KST; the 07:45 absolute watchdog remains an additional limit.
The report is in the submission directory as
`first_batch_verification.json`, with a local copy at the same campaign's
`submission/first_batch_verification.json`.

### Interrupted campaign and replenishment correction (2026-10-22)

Job 1030631 stopped at **2026-10-21 20:54:07 KST**. Global batch 72 prepared
47 eligible scenes in its initial 50-scene round, then two of three replacements,
then zero of one replacement. The former three-round sampling limit therefore
left 49/50 scenes and raised `IK search replenishment budget exhausted`. The
submission's execution-error handler cancelled all three allocations. This was
an IK scene-replenishment limit, not a rollout-success requirement.

The original files remain unchanged: 68 completed batches contain 3,400 scenes
and 34,000 outcomes. Interrupted batches contain another 1,002 saved outcomes,
including 23 scenes with all ten pairs finished. The completed-scene recovery
audits completed-batch hashes and those 23 scenes' preparation, trajectory,
shared IK and label consistency, yielding **3,423 scenes / 34,230 outcomes**.
Incomplete scenes are excluded as whole scenes; numerical rollout failures in
completed scenes remain in the dataset and are mapped to zero only for training.

New generator and campaign commands accept `--ik-replenishment-rounds`, default
**100** instead of three. Each round samples only the remaining no-IK deficit;
successful and failed controller rollouts are never resampled. The campaign
reserves the full number of possible sampling seeds per batch, forwards the
same limit in both serial and prefetched execution, and records it in requests
and manifests. Historical request files without the field retain their original
three-seed plan when audited. The bound remains finite and a genuine exhaustion
preserves all preparation artifacts with an explicit count in the error.

The newly requested 9,000-scene campaign uses three groups of 3,000 scenes,
180 batches of 50, and reserved seeds **2026200000–2026217999**. It keeps the
corrected QP and the previous scientific settings, with four GPUs and 25 CPUs
per group, 16 rollout workers and two IK-context workers per group. Its isolated
runtime changes only the three dataset runner/campaign/pipeline source files
relative to the frozen 6,000-scene runtime. The new submission has a 36-hour
wall limit; the earlier October 22 08:00 deadline belonged to the interrupted
6,000-scene request and is not reused for this fresh campaign.

The completed-scene export contains **1,046 `qp_solver_failure` outcomes out
of 34,230 (3.06%)**, affecting 706 of the 3,423 scenes. Sixteen occurred by
simulation time 0.02 seconds; 1,030 occurred later. The separate
`qp_infeasible` count is 17. These counts come from `results.csv` and refer
to the corrected-QP campaign. Raising IK replenishment does not fix these
rollout numerical failures or discard them.

After completed-scene training and artifact verification, fresh array job
**1033054** was submitted at **2026-10-22 07:16:15 KST**. Its three elements
run on n004, n010 and n011 with four RTX 3090 GPUs and 25 CPUs each (12 GPUs,
75 CPUs, up to 48 concurrent rollouts). Runtime hashes for 172 source/config/
test files matched before submission and are checked again on allocated nodes.
Output is `outputs/panda_link_dataset/production-9000-20261021T215851Z` on UBAI;
the sibling `-submission/` directory holds the frozen plan, runtime hashes,
Slurm script, submission record and logs. Group first batches must pass the
same numerical-error checks before subsequent batches run; first-batch
validation was still pending at launch. No recurring monitor was created.
All 61 startup tests passed on group zero; the 15 solver tests also passed
on each other group. By 07:17 KST all three groups had entered first-batch
IK preparation, with no submission error marker. The planning estimate is
21–24 hours from startup, subject to scene difficulty and resource contention.

### Completed 9,000 scenes and transfer (2026-10-23)

All three elements of Slurm array **1033054** completed with exit code 0;
their elapsed times were 21:20:31, 21:49:16 and 21:16:02. The separate export
job 1033053 also completed normally. The merged export finished at
**2026-10-23 05:05:39 KST**, with **9,000 eligible scenes / 90,000 outcomes**.
There were 9,402 attempted contexts: 402 no-IK contexts remain in the raw
`contexts.jsonl` audit but are excluded from the 9,000-row training archive.
All three first-batch gates passed. The frozen runtime still uses the corrected
QP solver and original clipped IK; the later experimental IK barrier was never
used for these data.

The 90,000 outcomes comprise 49,139 goal-reached, 33,317 timeout, 4,622 stalled,
2,884 numerical solver failures and 38 infeasible results. Safety-qualified
successes total 48,976. Numerical failures affect 1,974 scenes and are retained
as failures in training; 1,663 new scenes have no successful link choice under
that policy. There are zero IK candidate switches. CSV labels and all eligible
context inputs were checked against the NPZ arrays, including the fixed IK.

The root completion hash, three group completion hashes and four export data
hashes were checked on UBAI. The five export files were streamed directly to
lab-server without saving a local dataset copy, then all data hashes were
verified again. Raw training export on lab-server:
`outputs/panda_link_selection/datasets/production-9000-20261021T215851Z/`.
Dataset SHA256:
`2f8b53ec896bfa16184088a373287d23a5b76248a02b3d055e3f5e980b54a9e4`.
Export completion SHA256:
`fe7cb2ec05dad239c320bacec32866d875b0bef6960ab8c2334c65e02f96b4e2`.
Full trajectories remain in the completed UBAI campaign directories.

All 9,000 new scenes were added to the previous 12,423 training scenes. The
existing external 1,000 validation and 1,000 test scenes retain their exact
IDs, ordering and arrays. The resulting 23,423 scene IDs and canonical
initial-configuration/goal/obstacle fingerprints are unique across sources.
The training run and comparison are recorded in
`docs/experiments/panda_link_selection.md`.

### Additional 4,000 scenes (2026-10-23)

The requested additional **4,000 eligible scenes / 40,000 link-pair outcomes**
were submitted as Slurm array **1039014** at **2026-10-23 11:39:18 KST**.
The 80 batches of 50 scenes are distributed round-robin across three groups:
1,350 / 1,350 / 1,300 scenes. Reserved sampling seeds are
**2026220000–2026227999**, with zero overlap against 17 existing campaign,
submission-plan and pilot-manifest records. Each batch retains the 100-round
no-IK replenishment limit; controller failures are not resampled.

This campaign reuses the exact frozen runtime
`/home1/jinu1423/workspace/safe-inv-clbf-robot-arm-dataset9000-20261021T215851Z`.
All 172 recorded source/configuration/test hashes matched before submission.
The submission wrapper changes only the scene/outcome counts and corresponding
text. Scientific settings remain those of the completed 9,000-scene run:
120-second rollouts, a single fixed validated IK, safely sampled initial
configurations, all ten link pairs, slack weight 5,000, alpha 0.2, RDF CBF gain
1.0, 1 mm barrier tolerance and hard joint stops. The corrected QP solver and
original clipped IK are unchanged.

Each group has four GPUs, 25 CPUs, 16 rollout workers and two prefetched
IK-context workers. The array started on n012 (four RTX 3090 GPUs), n057 and
n059 (four A10 GPUs each): **12 GPUs / 75 CPUs / up to 48 concurrent rollouts**.
The existing first-batch gate is retained: all three initial batches must
pass the 5% numerical-QP-failure and 1% initial-constraint-failure limits before
the remaining batches proceed. First-batch validation is pending at launch.
The final group automatically verifies and merges the exports; allocations
end on completion, with a 36-hour wall limit. No recurring monitor was created.

UBAI output:
`/home1/jinu1423/workspace/safe-inv-clbf-robot-arm/outputs/panda_link_dataset/production-4000-20261023T023849Z`.
Its sibling `-submission/` directory contains the plan, runtime manifest,
seed-overlap audit, preflight checks, submission record and Slurm logs. Local
copies of submission metadata are under
`outputs/panda_link_dataset/production-4000-20261023T023849Z/submission/`.
The previous 9,000-scene elapsed time extrapolates to about ten hours for this
size; mixed A10/3090 allocation and scene difficulty make this a preliminary
estimate, to be revised from first-batch timings.

At **11:40 KST**, all three array elements were running and preparing their
first batches. All 61 startup tests passed on group zero; the 15 solver tests
also passed on each other group. There were no failure or abort markers.
This launch check does not yet establish that the first-batch numerical-error
gate has passed. The timestamped check is saved as
`submission/launch_verification.json` locally and in the UBAI submission directory.

### User-requested stop of the additional 4,000-scene run (2026-10-23)

At **14:17:48 KST**, the user requested stopping the current generation and
replacing it with all-link Sequential evaluation on the existing 1,000
validation scenes. Only array **1039014** was cancelled. Slurm confirms all
three allocations are cancelled and no allocation from that run remains.
All outputs are preserved, with no resumption or regeneration scheduled.

The final progress records report **872 scenes with all ten outcomes saved**
and **9,033 total saved outcomes**. Seventeen fully verified batches account
for 850 scenes / 8,500 outcomes; the remainder are in interrupted batches.
These progress counts are not a newly merged or fully audited training export.
The campaign's `stopped.json` and submission directory's
`user-stop-request.json` / `user-stop-result.json` record the intentional stop.
Any signal-related failure markers belong to this requested cancellation.
The replacement experiment is documented in `panda_all_links.md`.

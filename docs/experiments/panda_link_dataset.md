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
do not imply 16 reserved cores. CPU kinematics/QP work, offline safety validation
and GPU contention can limit the speedup. GPU utilization alone is not a measure
of end-to-end throughput. This changes execution scheduling only, not the IK
seed/iteration budget, control equations or five-subdivision safety validation.
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
respected. Five-subdivision offline safety validation is also recorded separately,
as in prior experiments. Success does not require matching the final IK joint
configuration. Safe randomized starts require +5 mm **barrier** margin and 0.1 rad
joint margin under the existing model, not a hardware collision guarantee.

For training, keep every pair belonging to a context in the same data split.
The ten labels within one context are not ten independent scenes. The pilot is
for validating data and estimating costs; it is not a success-rate benchmark.

## 1,000-context campaign

```bash
sbatch scripts/panda_link_dataset_1000.sbatch
```

This submits **one** job requesting four GPUs, 15 CPUs and a 72-hour wall limit.
It creates 1,000 new eligible contexts (10,000 trajectories), in 20 consecutive
batches of 50 contexts. Only one batch runs at a time, with the same 16 rollout
workers. The pilot YAML/controller/scientific settings are preserved in resolved
input snapshots. Batch seeds begin at 2026101500 and advance by three, reserving
distinct seeds for each batch's three possible IK replenishment rounds. The
earlier ten-context pilot is not included in the new dataset. Neither batching
nor replenishment filters contexts based on trajectory success.

The 10-context pilot took 21 min 34 s and used about 392 MB, suggesting roughly
36 hours and 39 GB for 1,000 contexts. These are small-sample extrapolations;
scene difficulty, GPU type and scheduling overhead affect actual costs.

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

A failed batch stops the job before the next batch starts. After the previous
Slurm job has stopped, resume explicitly using the **original** output path:

```bash
PANDA_DATASET_OUTPUT=/absolute/path/to/production-1000-ORIGINAL_JOBID \
PANDA_DATASET_RESUME=1 sbatch scripts/panda_link_dataset_1000.sbatch
```

Resume checks the code/config/model identity and every completed batch's data
hashes. Complete batches are reused. An incomplete batch is rerun from its fixed
seed in a new attempt directory; partial files are preserved. Thus an interruption
can cost at most the current 50-context batch, not the entire campaign. There is
no automatic retry, requeue or recurring monitor, and a campaign lock prevents
two coordinators from writing the same campaign. Resources are released when the
Slurm job ends.

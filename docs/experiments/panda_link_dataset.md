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

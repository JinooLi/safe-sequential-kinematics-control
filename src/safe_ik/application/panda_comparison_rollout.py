"""Panda paired rollouts with fresh, stage-independent goal/safety scoring."""

from time import perf_counter

import numpy as np

from ..control.panda_controller import PandaTrajectoryController
from ..experiments.panda_comparison import measure_state


def comparison_rollout(config, runtime, candidates=(), *, subdivisions=5):
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    kin, safety, plant = runtime.kinematics, runtime.safety, runtime.plant
    candidates = () if config.trajectory.mode == "direct" else tuple(candidates)
    if config.trajectory.mode == "sequential" and not candidates:
        raise ValueError("sequential rollout needs prepared candidates")
    controller = PandaTrajectoryController(
        config, kin, safety, candidates, runtime.solver
    )
    period = config.trajectory.integration_period
    maximum_steps = int(round(config.trajectory.duration / period))
    if not np.isclose(maximum_steps * period, config.trajectory.duration):
        raise ValueError("duration must contain an integer number of control periods")
    stage_targets = [
        [
            kin.frame_point_and_jacobian(f, candidate.q)[0]
            for f in config.trajectory.stage_frames
        ]
        for candidate in candidates
    ]
    states, barriers, samples, stage_errors, state_times = [], [], [], [], []
    candidate_indices, candidate_seeds, candidate_events = [], [], []

    def record_candidate_state(q):
        # Record the candidate chosen for this state, including a switch in
        # get_velocity(). Final states without a command keep the active one.
        index = controller.candidate_index if candidates else -1
        candidate_indices.append(index)
        candidate_seeds.append(candidates[index].seed_index if candidates else -1)
        targets = stage_targets[index] if candidates else []
        stage_errors.append(
            [
                float(np.linalg.norm(kin.frame_point_and_jacobian(f, q)[0] - target))
                for f, target in zip(config.trajectory.stage_frames, targets)
            ]
        )

    (
        commands,
        command_times,
        applied,
        stages,
        slacks,
        clf_values,
        qp_times,
        qp_status,
    ) = ([] for _ in range(8))
    rdf_times, events = [], []
    start, initial_time = perf_counter(), plant.time
    steps = 0
    while True:
        q = plant.q
        barrier = safety.evaluate(q)
        sample = measure_state(config, kin, q, barrier)
        states.append(q.copy())
        state_times.append(float(plant.time - initial_time))
        barriers.append(barrier.values.copy())
        samples.append(sample)
        rdf_times.append(
            [
                getattr(safety, "last_timings", {}).get(key, 0.0)
                for key in ("distance", "selection", "gradient")
            ]
        )
        # The final integrated state is evaluated before timeout and without a stage gate.
        if sample["goal_reached"]:
            record_candidate_state(q)
            reason = "goal_reached"
            break
        if steps == maximum_steps:
            record_candidate_state(q)
            reason = "timeout"
            break
        previous_candidate = controller.candidate_index
        velocity = controller.get_velocity(q)
        record_candidate_state(q)
        if controller.candidate_index != previous_candidate:
            candidate_events.append(
                {
                    "time": float(plant.time - initial_time),
                    "previous_index": previous_candidate,
                    "next_index": controller.candidate_index,
                    "previous_seed_index": candidates[previous_candidate].seed_index,
                    "next_seed_index": candidates[
                        controller.candidate_index
                    ].seed_index,
                    "reason": "stalled",
                }
            )
        status = controller.last_status
        commands.append(velocity.copy())
        command_times.append(float(plant.time - initial_time))
        stages.append(controller.stage_index)
        slacks.append(controller.last_clf_slack)
        clf_values.append(controller.safety_diagnostics(q).clf_value)
        qp_times.append(controller.last_qp_solve_time)
        qp_status.append(controller.last_qp_status)
        events.extend(
            {"time": plant.time - initial_time, "previous": a, "next": b, "reason": c}
            for a, b, c in controller.consume_stage_transition_events()
        )
        applied.append(False)
        if status and (status.qp_infeasible or status.qp_solver_failed):
            reason = "qp_infeasible" if status.qp_infeasible else "qp_solver_failure"
            break
        if status and status.candidates_exhausted:
            reason = "stalled"
            break
        if not np.isfinite(velocity).all():
            raise FloatingPointError("nonfinite control command")
        plant.advance(velocity, period)
        applied[-1] = True
        steps += 1
    rollout_wall = perf_counter() - start
    values, q_array = np.asarray(barriers), np.asarray(states)
    times = np.asarray(state_times)
    joint_minimum = min(s["joint_margin"] for s in samples)
    minimum = float(values.min())
    goal_reached = samples[-1]["goal_reached"]
    mask = np.asarray([label.startswith("self_") for label in barrier.labels])
    velocities = np.asarray(commands, dtype=float).reshape(-1, config.dof)
    used = velocities[np.asarray(applied, dtype=bool)]
    acceleration = (
        np.diff(np.vstack([np.zeros((1, config.dof)), used]), axis=0) / period
    )
    summary = {
        "goal_reached": goal_reached,
        "success": bool(goal_reached and minimum >= 0 and joint_minimum >= 0),
        "success_barrier_tolerance_1e6": bool(
            goal_reached and minimum >= -1e-6 and joint_minimum >= 0
        ),
        "safety_violation": bool(minimum < 0 or joint_minimum < 0),
        "reason": reason,
        "steps": steps,
        "state_count": len(states),
        "elapsed": float(times[-1]),
        "completion_time": float(times[-1]) if goal_reached else None,
        "final_q": q_array[-1].tolist(),
        "final_position_error": samples[-1]["position_error"],
        "final_orientation_error": samples[-1]["orientation_error"],
        "minimum_barrier": minimum,
        "minimum_joint_margin": joint_minimum,
        "minimum_self_barrier": float(values[:, mask].min()) if mask.any() else None,
        "minimum_obstacle_barrier": (
            float(values[:, ~mask].min()) if (~mask).any() else None
        ),
        "control_effort": float(np.linalg.norm(used, axis=1).sum() * period),
        "max_velocity_excess": (
            float(max(0.0, np.max(np.abs(used) - config.trajectory.velocity_limits)))
            if used.size
            else 0.0
        ),
        "max_acceleration_excess": (
            float(
                max(
                    0.0,
                    np.max(
                        np.abs(acceleration) - config.trajectory.acceleration_limits
                    ),
                )
            )
            if acceleration.size
            else 0.0
        ),
        "rollout_wall_time": rollout_wall,
        "qp_solve_time": float(sum(qp_times)),
        "rdf_times": np.asarray(rdf_times).sum(axis=0).tolist(),
        "stage_transitions": events,
        "candidate_count": len(candidates),
        "candidate_switch_count": controller.candidate_switch_count,
        "candidate_transitions": candidate_events,
        "final_candidate_seed_index": candidate_seeds[-1] if candidates else None,
    }
    # Offline interpolation is reported separately from sampled primary success.
    validation_start = perf_counter()
    dense_minimum, dense_time, dense_label = minimum, None, None
    checks = 0
    for i in range(len(q_array) - 1):
        for part in range(1, subdivisions):
            fraction = part / subdivisions
            q = q_array[i] + fraction * (q_array[i + 1] - q_array[i])
            evaluation = safety.evaluate(q)
            if not np.isfinite(evaluation.values).all():
                raise FloatingPointError("nonfinite interpolated barrier")
            checks += 1
            if evaluation.minimum < dense_minimum:
                dense_minimum = evaluation.minimum
                dense_time = float(times[i] + fraction * (times[i + 1] - times[i]))
                dense_label = evaluation.labels[int(np.argmin(evaluation.values))]
    summary.update(
        {
            "interpolated_minimum_barrier": dense_minimum,
            "interpolated_worst_time": dense_time,
            "interpolated_worst_pair": dense_label,
            "interpolation_checks": checks,
            "interpolation_subdivisions": subdivisions,
            "interpolated_success": bool(
                goal_reached and dense_minimum >= 0 and joint_minimum >= 0
            ),
            "validation_wall_time": perf_counter() - validation_start,
        }
    )
    trace = {
        "time": times,
        "q": q_array,
        "barrier_values": values,
        "barrier_labels": np.asarray(barrier.labels),
        "command_time": np.asarray(command_times),
        "command": velocities,
        "command_applied": np.asarray(applied, dtype=bool),
        "stage_index": np.asarray(stages, dtype=int),
        "stage_frames": np.asarray(config.trajectory.stage_frames),
        "stage_errors": np.asarray(stage_errors),
        "candidate_index": np.asarray(candidate_indices, dtype=int),
        "candidate_seed_index": np.asarray(candidate_seeds, dtype=int),
        "candidate_seed_indices": np.asarray(
            [candidate.seed_index for candidate in candidates], dtype=int
        ),
        "candidate_q": np.asarray([candidate.q for candidate in candidates]).reshape(
            -1, config.dof
        ),
        "clf_value": np.asarray(clf_values),
        "clf_slack": np.asarray(slacks),
        "qp_time": np.asarray(qp_times),
        "qp_status": np.asarray(qp_status),
        "rdf_times": np.asarray(rdf_times),
        **{key: np.asarray([s[key] for s in samples]) for key in samples[0]},
    }
    return summary, trace

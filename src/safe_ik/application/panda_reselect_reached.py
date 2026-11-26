"""Replay frozen held-out scenes with opt-in link-sequence selection policies.

This runner never searches for IK, changes the data split, or resumes an existing
output.  Static replay is provided to check the simulation against stored labels
before using the adaptive policy.  One process owns each simulation; several
processes may share an explicitly allocated GPU for RDF evaluation.

The threshold_weighted policy keeps initial argmax selection. At each subsequent
reached intermediate stage it samples the original sixteen actions whose score
strictly exceeds the threshold, proportional to their scores; no eligible action
falls back to argmax. Each scene owns PCG64(SeedSequence([seed, test_position])).

The collision_stall_skip policy retains the initially selected static route.
It may advance a stalled intermediate stage only when a separate diagnostic QP
finds an active collision CBF. Reaching or skipping a stage never resets time.

The periodic policy polls at fixed simulation-time intervals, including while
controlling the final EE. Only consecutive identical proposals for a different
complete route trigger a switch. A switch restarts that route at its first link;
ordinary already-reached checks still apply. The episode clock is never reset.
The optional sequence-lock-on-ee flag stops polling permanently at the first
entry into EE, including an initially selected Direct action.

The spatial policy instead queries Gaussian configurations around each polled
state. A strict majority can change the route immediately, without a temporal
confirmation streak. The Gaussian samples are network inputs, not robot states.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import socket
import subprocess
import tarfile
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from ..adapters.mujoco_plant import MujocoPlant
from ..adapters.osqp_solver import OSQPSolver
from ..control.panda_controller import PandaTrajectoryController
from ..domain.ik_contracts import SafeIKCandidate
from ..domain.link_actions import LINK_ACTIONS, action_id, catalog_metadata
from ..experiments.panda_comparison import case_config, measure_state
from ..learning.link_selection import load_dataset, load_model, predict_selection
from ..learning.spatial_selection import (
    SpatialSelectionConfig, SpatialSequenceSelector, load_spatial_selection_config,
)
from .panda_comparison import atomic_json, digest, fingerprint, scene_safety
from .panda_comparison_rollout import comparison_rollout
from .panda_components import build_panda_components
from .panda_experiment_configuration import load_panda_experiment
from .panda_link_dataset import gpu_assignments, save_npz, utc_now, validate_trace
from .panda_runtime import PandaRuntime
from .paths import project_root

_worker = None
STAGE_FRAMES = tuple(f"link{i}" for i in range(2, 7))


class ThresholdWeightedSelector:
    """Sample original actions after each reach, with a scene-owned RNG.

    Filtering already-completed links remains the controller's responsibility.
    Distinct actions keep distinct sampling mass even if that later filtering
    maps them to the same next target. Initial selection and argmax fallback do
    not consume randomness, making all samples reproducible from event logs.
    """

    def __init__(self, sampling_seed, test_position, threshold=0.5):
        if any(type(value) is not int or value < 0 for value in (sampling_seed, test_position)):
            raise ValueError("sampling seed and test position must be nonnegative integers")
        if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("probability threshold must be finite and between zero and one")
        self.seed_entropy = [sampling_seed, test_position]
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed_entropy)))
        self.threshold = float(threshold)
        self.selection_count = 0
        self.draw_count = 0

    def select(self, prediction):
        probabilities = np.asarray(prediction["probabilities"], dtype=np.float64)
        if probabilities.shape != (len(LINK_ACTIONS),) or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("weighted selection needs sixteen finite probability scores")
        argmax = int(probabilities.argmax())
        if prediction["action_index"] != argmax:
            raise ValueError("original model action is not the probability argmax")
        eligible = np.flatnonzero(probabilities > self.threshold)
        weights = np.zeros(len(LINK_ACTIONS), dtype=np.float64)
        initial = self.selection_count == 0
        fallback = not initial and not len(eligible)
        if initial or fallback:
            selected = argmax
            weights[selected] = 1.0
            rule = "initial_argmax" if initial else "no_eligible_argmax_fallback"
        else:
            weights[eligible] = probabilities[eligible] / probabilities[eligible].sum()
            selected = int(self.rng.choice(len(LINK_ACTIONS), p=weights))
            self.draw_count += 1
            rule = "threshold_probability_weighted"
        links = LINK_ACTIONS[selected]
        sampling = dict(
            selection_rule=rule,
            selection_index=self.selection_count,
            probability_threshold=self.threshold,
            threshold_comparison="strictly_greater_than",
            original_argmax_action_index=argmax,
            original_argmax_action_id=action_id(LINK_ACTIONS[argmax]),
            original_argmax_score=float(probabilities[argmax]),
            original_argmax_reason=prediction.get("reason"),
            eligible_action_indices=eligible.tolist(),
            eligible_action_ids=[action_id(LINK_ACTIONS[index]) for index in eligible],
            sampling_weights=weights.tolist(),
            sampled_action_index=selected,
            sampled_action_id=action_id(links),
            sampled_action_score=float(probabilities[selected]),
            sampling_applied=not initial and not fallback,
            fallback_to_argmax=fallback,
            rng_bit_generator="PCG64",
            rng_seed_sequence_entropy=self.seed_entropy.copy(),
            rng_draw_count=self.draw_count,
        )
        self.selection_count += 1
        return dict(
            prediction,
            method="sequential" if links else "direct",
            action_id=action_id(links), action_index=selected,
            pair_index=selected if len(links) == 2 else None,
            link_frames=[f"link{i}" for i in links],
            reason=prediction.get("reason") if initial else rule,
            sampling=sampling,
        )


def select_partition_rows(data, split, *, partition="test", offset=0, limit=None, indices=None):
    """Verify the existing held-out partition without resampling or relabelling."""
    if partition not in ("test", "validation"):
        raise ValueError("replay partition must be test or validation")
    selected = split[partition]
    source = np.asarray(selected["indices"])
    ids = list(selected["case_ids"])
    if (
        source.ndim != 1
        or source.dtype.kind not in "iu"
        or len(source) != len(ids)
        or len(set(source.tolist())) != len(source)
        or len(set(ids)) != len(ids)
        or np.any(source < 0)
        or np.any(source >= len(data["case_id"]))
        or data["case_id"][source].tolist() != ids
    ):
        raise ValueError(f"{partition} split indices and case IDs do not match the dataset")
    for name in ("train", "validation", "test"):
        if name == partition:
            continue
        if set(ids).intersection(split.get(name, {}).get("case_ids", [])):
            raise ValueError(f"{partition} scenes overlap {name}")
    if indices is not None:
        positions = list(indices)
        if offset or limit is not None:
            raise ValueError("explicit indices cannot be combined with offset/limit")
    else:
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("limit must be a positive integer")
        stop = len(source) if limit is None else offset + limit
        if stop > len(source):
            raise ValueError(f"requested slice exceeds the {partition} partition")
        positions = list(range(offset, stop))
    if (
        not positions
        or len(set(positions)) != len(positions)
        or any(type(p) is not int or p < 0 or p >= len(source) for p in positions)
    ):
        raise ValueError(f"{partition} positions must be nonempty, unique and in range")
    return [(position, int(source[position])) for position in positions]


def select_test_rows(data, split, *, offset=0, limit=None, indices=None):
    """Compatibility entry point for existing frozen test replay callers."""
    return select_partition_rows(data, split, offset=offset, limit=limit, indices=indices)


def build_tasks(data, rows, contexts=None, *, partition="test"):
    """Retain float64 arrays and, when available, original candidate metadata."""
    tasks = []
    for position, index in rows:
        case_id = str(data["case_id"][index])
        if Path(case_id).name != case_id or case_id in (".", ".."):
            raise ValueError("case ID is not a safe output directory name")
        case = {
            "case_id": case_id,
            "initial_q": data["initial_q"][index].copy(),
            "target_position": data["target_position"][index].copy(),
            "target_quaternion": data["target_quaternion"][index].copy(),
            # No search uses this placeholder when original contexts are absent.
            "ik_seed": 0,
            "obstacles": [
                dict(name=f"obstacle{i + 1}", position=o[:3].copy(), radius=float(o[3]))
                for i, o in enumerate(data["obstacles"][index])
            ],
        }
        selected = dict(
            q=data["selected_ik"][index].copy(),
            seed_index=0,
            iterations=0,
            position_error=0.0,
            orientation_error=0.0,
            minimum_barrier=0.0,
        )
        if contexts is not None:
            original = contexts[case_id]
            for key in ("initial_q", "target_position", "target_quaternion"):
                np.testing.assert_array_equal(original[key], case[key])
            original_obstacles = np.asarray(
                [list(o["position"]) + [o["radius"]] for o in original["obstacles"]]
            ).reshape(-1, 4)
            np.testing.assert_array_equal(original_obstacles, data["obstacles"][index])
            np.testing.assert_array_equal(original["selected_ik"]["q"], selected["q"])
            case["ik_seed"] = original["ik_seed"]
            case["obstacles"] = original["obstacles"]
            selected = dict(original["selected_ik"])
            selected["q"] = np.asarray(selected["q"], dtype=np.float64)
        tasks.append(
            dict(
                partition=partition,
                partition_position=position,
                # Retained for older replay consumers; explicitly documented
                # as a partition-position alias in new manifests.
                test_position=position,
                dataset_index=index,
                case=case,
                selected_ik=selected,
                original_candidate_metadata=contexts is not None,
                obstacles=data["obstacles"][index].copy(),
                stored_success=data["labels"][index].astype(bool).tolist(),
                stored_label_valid=data["label_valid"][index].tolist(),
            )
        )
    return tasks


def timestamp_collision_stall_events(events, trace, transitions):
    """Validate diagnostic state identity and attach simulation command times.

    A command index is required because the same q can recur at several times.
    Rejected checks are retained, while each successful gate must correspond to
    exactly one recorded skip transition.
    """
    timed, previous_index = [], -1
    skips = [event for event in transitions if event.get("reason") == "collision_constrained_stall"]
    for event in events:
        index = event["control_call_index"]
        if type(index) is not int or index <= previous_index or index >= len(trace["command_time"]):
            raise ValueError("invalid collision-stall diagnostic command index")
        previous_index = index
        np.testing.assert_array_equal(event["q"], trace["q"][index])
        timestamp = float(trace["command_time"][index])
        if timestamp != float(trace["time"][index]):
            raise ValueError("collision-stall diagnostic state and command times differ")
        timed.append(dict(event, time=timestamp))
    accepted = [event for event in timed if event["skipped"]]
    if len(accepted) != len(skips):
        raise ValueError("collision-stall checks and skip transitions differ")
    for check, transition in zip(accepted, skips):
        if (
            check["time"] != transition["time"]
            or check["previous_stage"] != transition["previous"]
            or check["next_stage"] != transition["next"]
            or not check["diagnostic_valid"]
            or not check["active_collision_labels"]
        ):
            raise ValueError("skip transition has no matching valid collision-stall evidence")
    return timed


def periodic_schedule_steps(period, confirmations, integration_period):
    """Reject invalid polling clocks before creating outputs or GPU workers."""
    if (
        type(period) not in (int, float)
        or not np.isfinite(period)
        or period <= 0
        or type(confirmations) is not int
        or confirmations < 1
    ):
        raise ValueError("sequence period and confirmation count must be positive")
    steps = int(round(period / integration_period))
    if steps < 1 or not np.isclose(
        steps * integration_period, period, rtol=0, atol=1e-12
    ):
        raise ValueError("sequence period must be an integer number of control steps")
    return steps


def validate_sequence_max_switches(value, mode="periodic"):
    """The optional cap counts accepted route changes, not polls or reaches."""
    if value is None:
        return None
    if (
        mode not in ("periodic", "spatial")
        or isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError("sequence max switches requires periodic or spatial mode and a nonnegative integer")
    return int(value)


def validate_sequence_lock_on_ee(value, mode="periodic"):
    if not isinstance(value, (bool, np.bool_)) or (value and mode not in ("periodic", "spatial")):
        raise ValueError("sequence lock on EE requires a boolean and periodic or spatial mode")
    return bool(value)


def resolve_sequence_settings(args):
    """Keep temporal flags and spatial YAML settings unambiguous."""
    spatial_path = getattr(args, "spatial_config", None)
    period = getattr(args, "sequence_period", None)
    confirmations = getattr(args, "sequence_confirmations", None)
    if args.mode == "spatial":
        if spatial_path is None:
            raise ValueError("spatial mode requires --spatial-config")
        if period is not None or confirmations is not None:
            raise ValueError("spatial mode uses YAML period_seconds and no temporal confirmation count")
        config = load_spatial_selection_config(spatial_path)
        return config, config.period_seconds, 1
    if spatial_path is not None:
        raise ValueError("--spatial-config requires spatial mode")
    if args.mode == "periodic":
        return None, 0.1 if period is None else period, 5 if confirmations is None else confirmations
    return None, None, None


def _spatial_vote_action(call, current, catalog, config, position, rng, poll_index):
    """Reproduce sampling and strict-majority choice from saved vote evidence."""
    vote = call["spatial_vote"]
    count = config.sample_count
    expected_samples = np.asarray(call["q"]) + rng.normal(0.0, config.sigma_rad, size=(count, 7))
    np.testing.assert_array_equal(vote["samples"], expected_samples)
    actions = np.asarray(vote["sample_actions"])
    if actions.shape != (count,) or actions.dtype.kind not in "iu" or np.any((actions < 0) | (actions >= len(catalog))):
        raise ValueError("spatial evidence has invalid sample actions")
    counts = np.bincount(actions, minlength=len(catalog))
    np.testing.assert_array_equal(vote["vote_counts"], counts)
    np.testing.assert_array_equal(vote["vote_fractions"], counts / count)
    winner = int(counts.argmax())
    majority = winner if 2 * counts[winner] > count else None
    current_action = catalog.index(current)
    selected = current_action if majority is None else majority
    expected_metadata = dict(
        sample_count=count, sigma_rad=config.sigma_rad,
        three_sigma_radius_rad=config.three_sigma_radius_rad,
        sampling_seed=config.sampling_seed, poll_index=poll_index,
        rng_seed_sequence_entropy=[config.sampling_seed, position],
        winner_action_index=winner, majority_action_index=majority,
        current_action_index=current_action,
    )
    if any(vote.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("spatial vote metadata or majority decision is inconsistent")
    return selected


def _periodic_ee_lock_cutoff(trace, transitions, initial_sequence, integration_period, lock_on_ee, event):
    """Verify the one-time lock against actual stage entry, returning poll cutoff."""
    command_count = len(trace["command_time"])
    if not lock_on_ee:
        if event is not None:
            raise ValueError("disabled periodic EE lock has a lock event")
        return command_count
    initial_direct = not initial_sequence
    first_ee = next((i for i, item in enumerate(transitions) if item["next"] == "end_effector_pose"), None)
    if not initial_direct and first_ee is None:
        if event is not None or np.any(trace["stage_index"] == len(STAGE_FRAMES)):
            raise ValueError("periodic EE lock has no matching EE entry")
        return command_count
    if initial_direct:
        index, reason = 0, "initial_direct"
        after_entry = transitions
    else:
        timestamp = transitions[first_ee]["time"]
        index = int(round(timestamp / integration_period))
        if not 0 <= index < command_count or not np.isclose(
            timestamp, trace["command_time"][index], rtol=0, atol=1e-8,
        ):
            raise ValueError("periodic EE entry has no matching command state")
        switched = any(
            item.get("reason") == "sequence_switch" and item["time"] == timestamp
            for item in transitions[:first_ee + 1]
        )
        reason = "accepted_sequence_ee" if switched else "reached_ee"
        after_entry = transitions[first_ee + 1:]
    if (
        event is None
        or type(event.get("control_call_index")) is not int
        or event["control_call_index"] != index
        or event.get("reason") != reason
        or event.get("initial_direct") is not initial_direct
        or not np.isclose(event.get("simulated_time", np.nan), index * integration_period, rtol=0, atol=1e-12)
    ):
        raise ValueError("periodic EE lock evidence differs from the first EE entry")
    if np.any(trace["stage_index"][:index] == len(STAGE_FRAMES)):
        raise ValueError("periodic trace entered EE before the recorded first EE entry")
    if (
        np.any(trace["stage_index"][index:] != len(STAGE_FRAMES))
        or any(item["next"] != "end_effector_pose" for item in after_entry)
    ):
        raise ValueError("periodic route left EE after the absorbing lock")
    # Normal reaches precede polling; an accepted route itself can create the lock.
    return index + (reason == "accepted_sequence_ee")


def timestamp_periodic_events(
    events, calls, trace, transitions, initial_sequence, *, period, confirmations,
    integration_period, max_switches=None, lock_on_ee=False, ee_lock_event=None,
    spatial_config=None, spatial_position=None,
):
    """Certify polling completeness, model action identity and switch timestamps.

    Final goal/timeout states do not call the controller and therefore do not
    poll. Every scheduled command state before termination or an optional EE lock
    must have exactly one inference and one decision. Stable catalog indices retain their meaning even
    when the active route returns to an earlier link or to/from Direct.
    """
    period_steps = periodic_schedule_steps(period, confirmations, integration_period)
    spatial_rng = None
    if spatial_config is not None:
        if (
            not isinstance(spatial_config, SpatialSelectionConfig)
            or confirmations != 1 or period != spatial_config.period_seconds
            or type(spatial_position) is not int or spatial_position < 0
        ):
            raise ValueError("spatial audit requires matching settings, N=1 and a partition position")
        spatial_rng = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([spatial_config.sampling_seed, spatial_position])
        ))
    max_switches = validate_sequence_max_switches(max_switches)
    lock_on_ee = validate_sequence_lock_on_ee(lock_on_ee)
    cutoff = _periodic_ee_lock_cutoff(
        trace, transitions, initial_sequence, integration_period, lock_on_ee, ee_lock_event,
    )
    expected = list(range(period_steps, cutoff, period_steps))
    if len(events) != len(calls) or [e["control_call_index"] for e in events] != expected:
        raise ValueError("periodic decisions do not cover every scheduled command state")
    np.testing.assert_array_equal(trace["stage_frames"], STAGE_FRAMES)
    if np.any((trace["stage_index"] < 0) | (trace["stage_index"] > len(STAGE_FRAMES))):
        raise ValueError("periodic trace has an invalid stable stage index")
    catalog = tuple(tuple(f"link{i}" for i in links) for links in LINK_ACTIONS)
    current = tuple(initial_sequence)
    if current not in catalog:
        raise ValueError("initial periodic route is not an original model action")
    timed, switch_count = [], 0
    for poll_index, (event, call) in enumerate(zip(events, calls)):
        index = event["control_call_index"]
        if type(index) is not int:
            raise ValueError("periodic command index must be an integer")
        locked_after = bool(lock_on_ee and ee_lock_event is not None and index >= ee_lock_event["control_call_index"])
        if event.get("lock_on_ee", False) != lock_on_ee or event.get("ee_locked_after", False) != locked_after:
            raise ValueError("periodic poll EE lock state differs from lock evidence")
        timestamp = float(trace["command_time"][index])
        np.testing.assert_array_equal(event["q"], trace["q"][index])
        np.testing.assert_array_equal(call["q"], trace["q"][index])
        if (
            call["time"] != timestamp
            or timestamp != float(trace["time"][index])
            or not np.isclose(event["simulated_time"], index * integration_period, rtol=0, atol=1e-12)
            or not np.isclose(event["simulated_time"], timestamp, rtol=0, atol=1e-8)
        ):
            raise ValueError("periodic inference and command timestamps differ")
        if spatial_config is None:
            scores = np.asarray(call["probabilities"], dtype=np.float64)
            if scores.shape != (len(catalog),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError("periodic inference has invalid probability scores")
            action = int(scores.argmax())
        else:
            action = _spatial_vote_action(
                call, current, catalog, spatial_config, spatial_position, spatial_rng, poll_index,
            )
        proposal = tuple(event["proposed_sequence"])
        if (
            call["action_index"] != action
            or tuple(call["link_frames"]) != catalog[action]
            or proposal != catalog[action]
            or tuple(event["current_sequence_before"]) != current
        ):
            source = "model argmax" if spatial_config is None else "spatial vote decision"
            raise ValueError(f"periodic controller action differs from the {source} or active route")
        exhausted = max_switches is not None and switch_count >= max_switches
        confirmed = event["confirmation_streak"] >= confirmations
        blocked = confirmed and exhausted
        accepted = confirmed and not exhausted
        if spatial_config is not None and confirmed != (proposal != current):
            raise ValueError("spatial majority must be considered immediately without a temporal streak")
        if (
            event["max_switches"] != max_switches
            or event["switch_count_before"] != switch_count
            or event["blocked_by_switch_limit"] != blocked
            or event["accepted"] != accepted
            or (proposal == current and confirmed)
        ):
            raise ValueError("periodic switch-limit or confirmation evidence is inconsistent")
        if event["accepted"]:
            if (
                proposal == current
                or event["streak_after"] != 0 or event["candidate_after"] is not None
            ):
                raise ValueError("periodic switch lacks the required confirmation evidence")
            current = proposal
            switch_count += 1
        if event["switch_count_after"] != switch_count:
            raise ValueError("periodic accepted-switch count changed without a switch")
        if tuple(event["current_sequence_after"]) != current:
            raise ValueError("periodic route changed without an accepted switch")
        # One authoritative record keeps both inference and controller evidence
        # without repeating q/probabilities in phase-wide summary payloads.
        record = dict(call)
        record.update(event, time=timestamp)
        timed.append(record)
    accepted = [event for event in timed if event["accepted"]]
    switches = [event for event in transitions if event.get("reason") == "sequence_switch"]
    if len(accepted) != len(switches):
        raise ValueError("accepted periodic proposals and sequence switches differ")
    for event, transition in zip(accepted, switches):
        if (
            event["time"] != transition["time"]
            or event["previous_stage"] != transition["previous"]
            or event["entry_stage"] != transition["next"]
        ):
            raise ValueError("periodic switch transition does not match its decision")
    return timed


def initialize_worker(queue, experiment_path, checkpoint, output, mode):
    global _worker
    gpu = queue.get(timeout=30)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    threadpool_limits(limits=1)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each worker must see exactly one allocated GPU")
    started = perf_counter()
    experiment = load_panda_experiment(experiment_path)
    # The kinematics object must know every possible adaptive intermediate frame.
    component_config = replace(
        experiment.config,
        obstacles=(),
        trajectory=replace(experiment.config.trajectory, stage_frames=STAGE_FRAMES),
    )
    kin, self_safety = build_panda_components(component_config)
    model = load_model(checkpoint, device="cpu")
    if model.actions != LINK_ACTIONS:
        raise ValueError("replay requires the existing sixteen-action checkpoint")
    _worker = experiment, kin, self_safety, model, Path(output), mode, gpu
    atomic_json(
        Path(output) / f"worker_{os.getpid()}.json",
        dict(
            gpu=gpu,
            gpu_name=torch.cuda.get_device_name(0),
            inference_device="cpu",
            torch_threads=1,
            pid=os.getpid(),
            ready_utc=utc_now(),
            setup_wall_time=perf_counter() - started,
        ),
    )


def evaluate_case(task):
    experiment, kin, self_safety, model, output, mode, gpu = _worker
    case, selected = task["case"], task["selected_ik"]
    directory = output / case["case_id"]
    directory.mkdir()  # A duplicate task must not overwrite any result.
    started, started_utc = perf_counter(), utc_now()
    atomic_json(directory / "case.json", case)
    atomic_json(
        directory / "ik.json",
        dict(
            selected=selected,
            search_performed=False,
            reused=True,
            original_candidate_metadata=task["original_candidate_metadata"],
        ),
    )
    calls, controllers = [], []
    initial = None
    try:
        cfg = case_config(experiment.config, case)
        inference_started = perf_counter()
        initial = predict_selection(
            model, case["initial_q"], selected["q"], case["target_position"],
            case["target_quaternion"], task["obstacles"],
        )
        initial_inference_wall_seconds = perf_counter() - inference_started
        atomic_json(directory / "initial_selection.json", initial)
        links = tuple(initial["link_frames"]) if mode in ("static", "collision_stall_skip") else STAGE_FRAMES
        method_cfg = replace(
            cfg,
            trajectory=replace(
                cfg.trajectory, mode="sequential" if links else "direct",
                stage_frames=links,
            ),
        )
        safety = scene_safety(method_cfg, kin, self_safety)
        candidate_barrier = safety.evaluate(selected["q"])
        candidate_check = measure_state(method_cfg, kin, selected["q"], candidate_barrier)
        initial_barrier = safety.evaluate(case["initial_q"])
        initial_check = measure_state(method_cfg, kin, case["initial_q"], initial_barrier)
        validation = dict(
            candidate=candidate_check,
            candidate_minimum_barrier=float(candidate_barrier.minimum),
            initial=initial_check,
            initial_minimum_barrier=float(initial_barrier.minimum),
        )
        atomic_json(directory / "input_validation.json", validation)
        if (
            not candidate_check["goal_reached"]
            or candidate_check["joint_margin"] < 0
            or candidate_barrier.minimum < 0
            or initial_check["joint_margin"] < 0
            or initial_barrier.minimum < 0
        ):
            raise ValueError("stored scene or IK does not pass current safety validation")
        plant = MujocoPlant(method_cfg)
        runtime = PandaRuntime(
            kin, safety,
            OSQPSolver(
                max_iterations=cfg.qp.max_iterations,
                absolute_tolerance=cfg.qp.absolute_tolerance,
                relative_tolerance=cfg.qp.relative_tolerance,
            ),
            plant,
        )
        weighted_selector = (
            ThresholdWeightedSelector(
                task["sampling_seed"], task["test_position"], task["probability_threshold"]
            )
            if mode == "threshold_weighted" else None
        )
        spatial_config = SpatialSelectionConfig(**task["spatial_reselection"]) if mode == "spatial" else None
        spatial_selector = (
            SpatialSequenceSelector(
                spatial_config, task["partition_position"], model, selected["q"],
                case["target_position"], case["target_quaternion"], task["obstacles"],
            ) if spatial_config is not None else None
        )

        def select(q):
            if spatial_selector is not None:
                prediction = spatial_selector.select(q, controllers[0].active_sequence)
            else:
                inference_started = perf_counter()
                prediction = predict_selection(
                    model, q, selected["q"], case["target_position"],
                    case["target_quaternion"], task["obstacles"],
                )
                inference_wall_seconds = perf_counter() - inference_started
            original_prediction = prediction
            if weighted_selector is not None:
                prediction = weighted_selector.select(prediction)
            call = dict(time=float(plant.time), q=q.copy(), **prediction)
            if mode == "periodic":
                call["inference_wall_seconds"] = inference_wall_seconds
            calls.append(call)
            if len(calls) == 1 and mode not in ("periodic", "spatial"):
                if original_prediction != initial:
                    raise ValueError("initial adaptive and stored model predictions differ")
                np.testing.assert_array_equal(q, case["initial_q"])
            return prediction["link_frames"]

        def factory(*args):
            collision_options = {}
            if mode == "collision_stall_skip" and links:
                collision_options = dict(
                    advance_on_collision_stall=True,
                    collision_stall_solver=OSQPSolver(
                        max_iterations=cfg.qp.max_iterations,
                        absolute_tolerance=cfg.qp.absolute_tolerance,
                        relative_tolerance=cfg.qp.relative_tolerance,
                    ),
                )
            periodic_options = {}
            if mode in ("periodic", "spatial"):
                periodic_options = dict(
                    periodic_stage_selector=select,
                    periodic_reselection_period=task["sequence_period"],
                    periodic_reselection_confirmations=task["sequence_confirmations"],
                    periodic_reselection_max_switches=task.get("sequence_max_switches"),
                    periodic_reselection_lock_on_ee=task.get("sequence_lock_on_ee", False),
                    initial_stage_sequence=tuple(initial["link_frames"]),
                )
            controller = PandaTrajectoryController(
                *args, stage_selector=select if mode in ("adaptive", "threshold_weighted") else None,
                **collision_options, **periodic_options,
            )
            controllers.append(controller)
            return controller

        candidates = (SafeIKCandidate(**selected),) if links else ()
        result, trace = comparison_rollout(
            method_cfg, runtime, candidates, subdivisions=experiment.subdivisions,
            controller_factory=factory,
        )
        validate_trace(
            method_cfg, case, selected, tuple(int(link[4:]) for link in links),
            result, trace, experiment.subdivisions,
        )
        decisions = controllers[0].consume_stage_selection_events()
        if mode in ("adaptive", "threshold_weighted"):
            if len(decisions) != len(calls):
                raise ValueError("controller decisions and model inference counts differ")
            for decision, call in zip(decisions, calls):
                np.testing.assert_array_equal(decision["q"], call["q"])
                if decision["chosen_frames"] != call["link_frames"]:
                    raise ValueError("controller used a different model action")
                state_index = int(round(call["time"] / cfg.trajectory.integration_period))
                np.testing.assert_array_equal(trace["q"][state_index], call["q"])
            decisions = [dict(**event, time=call["time"]) for event, call in zip(decisions, calls)]
        if mode in ("periodic", "spatial"):
            if decisions:
                raise ValueError("periodic policy unexpectedly ran reached-stage reselection")
            decisions = timestamp_periodic_events(
                controllers[0].consume_periodic_stage_selection_events(), calls,
                trace, result["stage_transitions"], initial["link_frames"],
                period=task["sequence_period"],
                confirmations=task["sequence_confirmations"],
                integration_period=cfg.trajectory.integration_period,
                max_switches=task.get("sequence_max_switches"),
                lock_on_ee=task.get("sequence_lock_on_ee", False),
                ee_lock_event=controllers[0].periodic_ee_lock_event,
                spatial_config=spatial_config,
                spatial_position=task["partition_position"],
            )
            switches = sum(event["accepted"] for event in decisions)
            if controllers[0].periodic_reselection_count != switches:
                raise ValueError("periodic switch count differs from controller decisions")
            atomic_json(directory / "periodic_selection_events.json", decisions)
            lock_event = controllers[0].periodic_ee_lock_event
            result.update(
                sequence_period=task["sequence_period"],
                sequence_confirmations=task["sequence_confirmations"],
                sequence_max_switches=task.get("sequence_max_switches"),
                sequence_lock_on_ee=task.get("sequence_lock_on_ee", False),
                initial_direct=controllers[0].periodic_initial_direct,
                periodic_ee_locked=controllers[0].periodic_ee_locked,
                periodic_ee_lock_time=lock_event["simulated_time"] if lock_event else None,
                periodic_ee_lock_control_call_index=lock_event["control_call_index"] if lock_event else None,
                periodic_ee_lock_reason=lock_event["reason"] if lock_event else None,
                periodic_ee_lock_event=lock_event,
                periodic_poll_count=len(decisions), periodic_switch_count=switches,
                periodic_blocked_switch_proposals=sum(event["blocked_by_switch_limit"] for event in decisions),
                periodic_switch_limit_exhausted=(
                    task.get("sequence_max_switches") is not None
                    and switches >= task["sequence_max_switches"]
                ),
                initial_inference_count=1,
                total_inference_count=1 + len(calls) * (spatial_config.sample_count if spatial_config else 1),
                initial_inference_wall_seconds=initial_inference_wall_seconds,
                neural_inference_wall_seconds=initial_inference_wall_seconds + sum(call["inference_wall_seconds"] for call in calls),
                initial_sequence=initial["link_frames"],
                final_sequence=list(controllers[0].active_sequence),
                periodic_selection_events_sha256=digest(directory / "periodic_selection_events.json"),
            )
            if spatial_config is not None:
                result.update(
                    spatial_reselection=asdict(spatial_config),
                    spatial_poll_count=len(calls),
                    spatial_sample_count=spatial_config.sample_count,
                    spatial_neural_evaluations=len(calls) * spatial_config.sample_count,
                    spatial_majority_polls=sum(call["spatial_vote"]["majority_action_index"] is not None for call in calls),
                    spatial_no_majority_polls=sum(call["spatial_vote"]["majority_action_index"] is None for call in calls),
                    spatial_selection_wall_seconds=sum(call["spatial_selection_wall_seconds"] for call in calls),
                )
        if mode == "collision_stall_skip":
            collision_events = timestamp_collision_stall_events(
                controllers[0].consume_collision_stall_events(), trace, result["stage_transitions"],
            )
            if calls or decisions:
                raise ValueError("collision-stall static-route policy unexpectedly reselected links")
            atomic_json(directory / "collision_stall_events.json", collision_events)
            result.update(
                collision_stall_events=collision_events,
                collision_stall_gate_checks=len(collision_events),
                collision_stall_skipped_stages=sum(event["skipped"] for event in collision_events),
                collision_stall_diagnostic_failures=sum(not event["diagnostic_valid"] for event in collision_events),
                collision_stall_events_sha256=digest(directory / "collision_stall_events.json"),
            )
        save_npz(directory / "trajectory.npz", **trace)
        if mode in ("periodic", "spatial"):
            atomic_json(directory / "selection_events.json", dict(
                initial_inference="initial_selection.json",
                periodic_events="periodic_selection_events.json",
                periodic_events_sha256=result["periodic_selection_events_sha256"],
                periodic_poll_count=len(decisions),
                periodic_ee_lock_event=result["periodic_ee_lock_event"],
            ))
        else:
            atomic_json(directory / "selection_events.json", dict(inferences=calls, decisions=decisions))
        result.update(
            execution_status="completed",
            numerical_failure=result["reason"] == "qp_solver_failure",
            inference_count=len(calls) * (spatial_config.sample_count if spatial_config else 1),
            stage_selection_events=decisions if mode not in ("periodic", "spatial") else [],
            trace_sha256=digest(directory / "trajectory.npz"),
            selection_events_sha256=digest(directory / "selection_events.json"),
            verified_fixed_ik=True,
            verified_time_budget=True,
            verified_outcome_from_trace=True,
        )
    except Exception as error:
        # An infrastructure/input error is visible, never silently relabelled as
        # a controller outcome. It remains in the requested-scene denominator.
        atomic_json(directory / "error.json", dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
        ))
        atomic_json(directory / "selection_events.json", dict(inferences=calls))
        result = dict(
            execution_status="error", reason="runner_error", success=False,
            goal_reached=False, numerical_failure=False,
            error_type=type(error).__name__, error_message=str(error),
        )
    action = initial["action_index"] if initial is not None else None
    stored = task["stored_success"][action] if action is not None else None
    result.update(
        case_id=case["case_id"], test_position=task["test_position"],
        partition=task.get("partition", "test"),
        partition_position=task.get("partition_position", task["test_position"]),
        dataset_index=task["dataset_index"], policy=mode,
        initial_selection=initial, stored_static_success=stored,
        stored_static_label_valid=(task["stored_label_valid"][action] if action is not None else None),
        static_replay_matches=(bool(result["success"]) == stored if mode == "static" and stored is not None and result["execution_status"] == "completed" else None),
        candidate_policy="fixed_saved_ik_no_search_no_switch",
        case_sha256=digest(directory / "case.json"), ik_sha256=digest(directory / "ik.json"),
        task_wall_time=perf_counter() - started, started_utc=started_utc,
        finished_utc=utc_now(), gpu=gpu, worker_pid=os.getpid(),
    )
    if mode == "threshold_weighted":
        sampled = [call["sampling"] for call in calls]
        result.update(
            sampling_seed=task["sampling_seed"],
            probability_threshold=task["probability_threshold"],
            rng_seed_sequence_entropy=[task["sampling_seed"], task["test_position"]],
            weighted_draw_count=sum(event["sampling_applied"] for event in sampled),
            weighted_fallback_count=sum(event["fallback_to_argmax"] for event in sampled),
            selected_nonargmax_count=sum(event["sampled_action_index"] != event["original_argmax_action_index"] for event in sampled),
        )
    atomic_json(directory / "summary.json", result)
    print(f"{case['case_id']} {mode}: {result['reason']} success={result['success']} wall={result['task_wall_time']:.1f}s GPU={gpu}", flush=True)
    return result


def aggregate(rows, requested):
    completed = [r for r in rows if r["execution_status"] == "completed"]
    matched = [r for r in completed if r["stored_static_success"] is not None]
    statistics = dict(
        requested_cases=requested,
        returned_cases=len(rows), completed_simulations=len(completed),
        runner_errors=len(rows) - len(completed), missing_cases=requested - len(rows),
        successes=sum(bool(r["success"]) for r in completed),
        success_rate_requested=sum(bool(r["success"]) for r in completed) / requested,
        goal_reached=sum(bool(r["goal_reached"]) for r in completed),
        numerical_failures=sum(bool(r["numerical_failure"]) for r in completed),
        reasons=dict(Counter(r["reason"] for r in rows)),
        paired_cases=len(matched),
        stored_static_successes=sum(r["stored_static_success"] for r in matched),
        recovered=sum(r["success"] and not r["stored_static_success"] for r in matched),
        lost=sum(not r["success"] and r["stored_static_success"] for r in matched),
        static_replay_mismatches=sum(r["static_replay_matches"] is False for r in completed),
        mean_success_completion_time=(
            float(np.mean([r["completion_time"] for r in completed if r["success"]]))
            if any(r["success"] for r in completed) else None
        ),
    )
    if any(row.get("policy") == "threshold_weighted" for row in rows):
        statistics.update(
            weighted_draw_count=sum(row.get("weighted_draw_count", 0) for row in completed),
            weighted_fallback_count=sum(row.get("weighted_fallback_count", 0) for row in completed),
            selected_nonargmax_count=sum(row.get("selected_nonargmax_count", 0) for row in completed),
            scenes_with_nonargmax_selection=sum(row.get("selected_nonargmax_count", 0) > 0 for row in completed),
        )
    if any(row.get("policy") == "collision_stall_skip" for row in rows):
        statistics.update(
            collision_stall_gate_checks=sum(row.get("collision_stall_gate_checks", 0) for row in completed),
            collision_stall_skipped_stages=sum(row.get("collision_stall_skipped_stages", 0) for row in completed),
            collision_stall_diagnostic_failures=sum(row.get("collision_stall_diagnostic_failures", 0) for row in completed),
            collision_stall_affected_scenes=sum(row.get("collision_stall_skipped_stages", 0) > 0 for row in completed),
        )
    if any(row.get("policy") in ("periodic", "spatial") for row in rows):
        statistics.update(
            periodic_poll_count=sum(row.get("periodic_poll_count", 0) for row in completed),
            periodic_switch_count=sum(row.get("periodic_switch_count", 0) for row in completed),
            periodic_switched_scenes=sum(row.get("periodic_switch_count", 0) > 0 for row in completed),
            periodic_blocked_switch_proposals=sum(row.get("periodic_blocked_switch_proposals", 0) for row in completed),
            periodic_scenes_with_blocked_switch_proposals=sum(row.get("periodic_blocked_switch_proposals", 0) > 0 for row in completed),
            periodic_scenes_at_switch_limit=sum(row.get("periodic_switch_limit_exhausted", False) for row in completed),
            initial_direct_scenes=sum(row.get("initial_direct", False) for row in completed),
            periodic_ee_locked_scenes=sum(row.get("periodic_ee_locked", False) for row in completed),
            periodic_ee_lock_reasons=dict(Counter(row["periodic_ee_lock_reason"] for row in completed if row.get("periodic_ee_locked", False))),
            initial_inference_count=sum(row.get("initial_inference_count", 0) for row in completed),
            total_inference_count=sum(row.get("total_inference_count", 0) for row in completed),
        )
    if any(row.get("policy") == "spatial" for row in rows):
        statistics.update(
            spatial_poll_count=sum(row.get("spatial_poll_count", 0) for row in completed),
            spatial_neural_evaluations=sum(row.get("spatial_neural_evaluations", 0) for row in completed),
            spatial_majority_polls=sum(row.get("spatial_majority_polls", 0) for row in completed),
            spatial_no_majority_polls=sum(row.get("spatial_no_majority_polls", 0) for row in completed),
            spatial_selection_wall_seconds=sum(row.get("spatial_selection_wall_seconds", 0) for row in completed),
        )
    return statistics


def run(args):
    spatial_config, sequence_period, sequence_confirmations = resolve_sequence_settings(args)
    sequence_max_switches = validate_sequence_max_switches(
        getattr(args, "sequence_max_switches", None), args.mode,
    )
    sequence_lock_on_ee = validate_sequence_lock_on_ee(
        getattr(args, "sequence_lock_on_ee", False), args.mode,
    )
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    threadpool_limits(limits=1)
    started, started_utc = perf_counter(), utc_now()
    data = load_dataset(args.dataset, args.expected_dataset_sha256)
    if data["actions"] != LINK_ACTIONS:
        raise ValueError("replay requires the sixteen-action dataset")
    split = json.loads(args.split.read_text())
    partition = getattr(args, "partition", "test")
    rows = select_partition_rows(data, split, partition=partition, offset=args.offset, limit=args.limit, indices=args.indices)
    contexts = None
    if args.contexts is not None:
        records = [json.loads(line) for line in args.contexts.read_text().splitlines() if line.strip()]
        contexts = {record["case_id"]: record for record in records}
        if len(contexts) != len(records):
            raise ValueError("original contexts contain duplicate case IDs")
    tasks = build_tasks(data, rows, contexts, partition=partition)
    if args.mode == "threshold_weighted":
        # Validate before creating output or starting GPU workers.
        ThresholdWeightedSelector(args.sampling_seed, 0, args.probability_threshold)
        for task in tasks:
            task.update(sampling_seed=args.sampling_seed, probability_threshold=args.probability_threshold)
    experiment = load_panda_experiment(args.experiment)
    if experiment.use_all_ik_candidates:
        raise ValueError("experiment must use a single fixed IK candidate")
    if args.mode in ("periodic", "spatial"):
        periodic_schedule_steps(
            sequence_period, sequence_confirmations,
            experiment.config.trajectory.integration_period,
        )
        for task in tasks:
            task.update(
                sequence_period=sequence_period,
                sequence_confirmations=sequence_confirmations,
                sequence_max_switches=sequence_max_switches,
                sequence_lock_on_ee=sequence_lock_on_ee,
            )
            if spatial_config is not None:
                task["spatial_reselection"] = asdict(spatial_config)
    visible = [gpu.strip() for gpu in args.gpus.split(",")]
    assignments = gpu_assignments(visible, args.workers_per_gpu)
    allocated = os.environ.get("CUDA_VISIBLE_DEVICES")
    if allocated is not None and not set(visible).issubset(allocated.split(",")):
        raise ValueError("requested GPUs are outside CUDA_VISIBLE_DEVICES allocation")
    inputs = {
        str(args.dataset.resolve() / "dataset.npz"): data["dataset_sha256"],
        str(args.dataset.resolve() / "complete.json"): digest(args.dataset / "complete.json"),
        str(args.split.resolve()): digest(args.split),
        str(args.checkpoint.resolve()): digest(args.checkpoint),
        str(args.experiment.resolve()): digest(args.experiment),
        str(experiment.config_path): digest(experiment.config_path),
    }
    if args.contexts is not None:
        inputs[str(args.contexts.resolve())] = digest(args.contexts)
    if spatial_config is not None:
        inputs[str(args.spatial_config.resolve())] = digest(args.spatial_config)
    for path, expected in ((args.split, args.expected_split_sha256), (args.checkpoint, args.expected_checkpoint_sha256)):
        if expected is not None and digest(path) != expected:
            raise ValueError(f"input checksum mismatch: {path}")
    # Fail before reserving output or GPU workers if the checkpoint is incompatible.
    model = load_model(args.checkpoint, device="cpu")
    if model.actions != LINK_ACTIONS:
        raise ValueError("checkpoint action catalog differs")
    del model
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = project_root()
    identity = fingerprint(experiment.config, experiment.config_path, experiment.main["seeds"], experiment.main["cases_per_seed"], experiment=experiment)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        commit = None
    manifest = dict(
        policy=args.mode, **catalog_metadata(), identity=identity,
        started_utc=started_utc, hostname=socket.gethostname(), python=platform.python_version(),
        git_commit=commit, input_sha256=inputs,
        data_sha256=experiment.snapshot(output),
        dataset_sha256=data["dataset_sha256"], requested_cases=len(tasks),
        partition=partition, partition_count=len(split[partition]["indices"]),
        partition_positions=[p for p, _ in rows], dataset_indices=[i for _, i in rows],
        legacy_test_position_field="alias_of_partition_position; use_partition_to_identify_split",
        case_ids=[t["case"]["case_id"] for t in tasks],
        gpus=visible, workers_per_gpu=args.workers_per_gpu, workers=len(assignments),
        inference_device="cpu", torch_threads_per_worker=1,
        ik_search_performed=False, candidate_switching=False,
        original_candidate_metadata=args.contexts is not None,
        total_time_budget=experiment.config.trajectory.duration,
        budget_reset_at_stage_transition=False,
        stopping_policy="existing_stage_stall_detection; reselection_only_after_stage_reached",
        numerical_errors_count_as_failures=True,
        runner_errors_reported_separately=True,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    if partition == "test":
        manifest.update(test_partition_count=len(split["test"]["indices"]), test_positions=[p for p, _ in rows])
    if args.mode == "collision_stall_skip":
        manifest["stopping_policy"] = "existing_intermediate_stall_and_active_collision_CBF_skip; otherwise_existing_stop"
        manifest["collision_stall_skip"] = dict(
            initial_selection="original_model_argmax",
            route="initially_selected_links_in_original_order_then_EE",
            inference_after_initial_selection=False,
            gate="existing_stall_criterion_and_active_obstacle_or_self_collision_CBF",
            active_constraint="absolute_residual_within_QP_tolerance_and_lower_multiplier_strictly_above_1e-6",
            dual_threshold=1e-6,
            diagnostic_solver="separate_OSQP_instance; same_QP_and_tolerances; main_solver_warm_start_unmodified",
            diagnostic_failure="do_not_skip; preserve_original_stalled_outcome",
            final_ee_absorbing=True,
            joint_limit_alone_triggers_skip=False,
            rejected_diagnostic_checks_logged=True,
            event_time="trace.command_time[control_call_index]; state_identity_verified",
        )
    if args.mode == "periodic":
        manifest["stopping_policy"] = "existing_stall_and_QP_failure_stop; scheduled_poll_before_same_state_stall_check"
        manifest["periodic_reselection"] = dict(
            sequence_period=sequence_period,
            sequence_confirmations=sequence_confirmations,
            sequence_max_switches=sequence_max_switches,
            sequence_lock_on_ee=sequence_lock_on_ee,
            switch_limit_counts="accepted_complete_route_changes_only; initial_selection_and_reached_stage_advances_excluded",
            switch_limit_exhausted="continue_scheduled_inference_and_streak_update_until_optional_EE_lock; reject_confirmed_route_changes",
            period_control_steps=periodic_schedule_steps(
                sequence_period, sequence_confirmations,
                experiment.config.trajectory.integration_period,
            ),
            initial_selection="one_original_model_argmax_inference_at_t0",
            polling="T_seq,2*T_seq,... on simulation clock; no_poll_at_t0_or_terminal_goal_timeout_state; stop_at_optional_EE_lock",
            proposed_sequence="original_16_action_argmax; complete_route_identity",
            confirmation="N_seq_consecutive_identical_proposals_different_from_active_sequence",
            current_sequence_proposal="clear_candidate_and_streak",
            different_candidate_proposal="replace_candidate_and_set_streak_to_one",
            accepted_switch="restart_new_route_from_first_link; clear_candidate_and_streak",
            already_reached_stages="advance_immediately_using_existing_reach_criterion",
            previously_reached_links_filtered=False,
            reached_stage_resets_proposal_streak=False,
            final_ee_absorbing=sequence_lock_on_ee,
            initial_direct_can_switch_to_links=not sequence_lock_on_ee,
            ee_lock=(dict(
                trigger="first_EE_entry_after_normal_reach_or_accepted_route; initial_Direct_at_t0",
                normal_reach_order="lock_before_same_command_state_poll",
                after_lock="no_further_inference_or_route_change; clear_pending_candidate_and_streak",
                evidence="single_periodic_ee_lock_event_and_case_summary; no_synthetic_poll_records",
            ) if sequence_lock_on_ee else None),
            collision_stall_skip_enabled=False,
            stage_catalog=list(STAGE_FRAMES),
            empty_sequence=(
                "absorbing_EE_control_with_fixed_saved_IK_retained"
                if sequence_lock_on_ee else "EE_control_with_fixed_saved_IK_retained_for_later_link_switch"
            ),
            event_time="trace.command_time[control_call_index]; simulated_index_time_and_state_verified",
        )
    if spatial_config is not None:
        snapshot = output / "spatial_reselection.yaml"
        snapshot.write_bytes(args.spatial_config.read_bytes())
        if (
            digest(snapshot) != inputs[str(args.spatial_config.resolve())]
            or load_spatial_selection_config(snapshot) != spatial_config
        ):
            raise ValueError("spatial configuration changed before snapshot")
        manifest["data_sha256"][snapshot.name] = digest(snapshot)
        manifest["stopping_policy"] = "existing_stall_and_QP_failure_stop; spatial_poll_before_same_state_stall_check"
        manifest["spatial_reselection"] = dict(
            **asdict(spatial_config), sigma_rad=spatial_config.sigma_rad,
            distribution="q_sample = current_q + Normal(0, sigma_rad^2 * I_7)",
            radius_semantics="three_sigma_isodensity_radius; not_a_hard_cutoff_or_99.7_percent_mass_radius",
            joint_limit_sampling="untruncated_no_clipping_or_rejection; hypothetical_network_inputs_only",
            collision_filtered_samples=False,
            context="same_saved_IK_final_EE_pose_and_obstacles; only_current_q_varies",
            initial_selection="one_original_model_argmax_inference_at_t0; no_Gaussian_draw",
            polling="period_seconds,2*period_seconds,... on_simulation_clock; no_t0_or_terminal_poll",
            period_control_steps=periodic_schedule_steps(sequence_period, 1, experiment.config.trajectory.integration_period),
            sample_choice="original_16_action_argmax_for_each_sample",
            selection_rule="strictly_more_than_half_of_samples_vote_for_one_complete_route",
            minimum_majority_votes=spatial_config.sample_count // 2 + 1,
            no_majority="retain_current_complete_route_without_switch_or_progress_window_reset",
            same_route_majority="retain_current_route_and_stage",
            temporal_confirmations_required=1,
            accepted_switch="restart_new_route_at_first_link; skip_already_reached_with_existing_criterion",
            sequence_max_switches=sequence_max_switches,
            sequence_lock_on_ee=sequence_lock_on_ee,
            switch_limit_counts="accepted_complete_route_changes_only; polls_continue_after_cap",
            final_ee_absorbing=sequence_lock_on_ee,
            empty_sequence="Direct_EE_control; fixed_saved_IK_retained",
            rng="numpy.Generator(PCG64(SeedSequence([sampling_seed, partition_position])))",
            numpy_version=np.__version__, scheduling_independent=True,
            inference_execution="sample_count_serial_predict_selection_calls_per_poll; CPU_one_thread",
            vote_fraction_semantics="sample_argmax_agreement_not_success_probability",
            event_artifact="periodic_selection_events.json; spatial_vote contains samples/actions/counts",
            stage_catalog=list(STAGE_FRAMES),
            collision_stall_skip_enabled=False,
        )
    if args.mode == "threshold_weighted":
        manifest["weighted_selection"] = dict(
            sampling_seed=args.sampling_seed,
            probability_threshold=args.probability_threshold,
            initial_selection="original_argmax_without_random_draw",
            reselection="each_reached_intermediate; original_16_actions_strictly_above_threshold; probability_proportional",
            fallback="original_argmax_when_no_action_exceeds_threshold; no_random_draw",
            completed_link_filter="after_sampling_in_existing_controller",
            equivalent_actions_merged=False,
            final_ee_absorbing=True,
            rng="numpy.Generator(PCG64(SeedSequence([sampling_seed, test_position])))",
            numpy_version=np.__version__,
            scheduling_independent=True,
        )
    with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as archive:
        for relative in identity["source_sha256"]:
            archive.add(root / relative, arcname=relative)
    manifest["data_sha256"]["source_snapshot.tar.gz"] = digest(output / "source_snapshot.tar.gz")
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "selected_cases.json", [dict(partition=partition, partition_position=t["partition_position"], test_position=t["test_position"], dataset_index=t["dataset_index"], case_id=t["case"]["case_id"]) for t in tasks])
    results = []

    def progress(status):
        elapsed = perf_counter() - started
        atomic_json(output / "progress.json", dict(
            status=status, updated_utc=utc_now(), elapsed_wall_seconds=elapsed,
            estimated_remaining_seconds=(elapsed * (len(tasks) - len(results)) / len(results) if results else None),
            **aggregate(results, len(tasks)),
        ))

    progress("running")
    spawn = mp.get_context("spawn")
    queue = spawn.Queue()
    for gpu in assignments:
        queue.put(gpu)
    try:
        with ProcessPoolExecutor(
            max_workers=len(assignments), mp_context=spawn, initializer=initialize_worker,
            initargs=(queue, str(args.experiment.resolve()), str(args.checkpoint.resolve()), str(output), args.mode),
        ) as pool:
            futures = [pool.submit(evaluate_case, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                progress("running")
        for path, expected in inputs.items():
            if digest(path) != expected:
                raise ValueError(f"input changed during replay: {path}")
        for relative, expected in identity["source_sha256"].items():
            if digest(root / relative) != expected:
                raise ValueError(f"source changed during replay: {relative}")
        artifacts = {
            **identity["model_resource_sha256"],
            str(experiment.config.model_path): identity["model_sha256"],
            str(experiment.config.safety.rdf.artifact): identity["rdf_sha256"],
            str(root / "uv.lock"): identity["lock_sha256"],
        }
        for path, expected in artifacts.items():
            if digest(path) != expected:
                raise ValueError(f"simulation artifact changed during replay: {path}")
        results.sort(key=lambda row: row["partition_position"])
        atomic_json(output / "summary.json", results)
        statistics = aggregate(results, len(tasks))
        atomic_json(output / "analysis.json", statistics)
        fields = ("case_id", "partition", "partition_position", "test_position", "dataset_index", "execution_status", "success", "goal_reached", "reason", "elapsed", "completion_time", "stored_static_success", "static_replay_matches", "collision_stall_gate_checks", "collision_stall_skipped_stages", "collision_stall_diagnostic_failures", "sequence_period", "sequence_confirmations", "sequence_max_switches", "sequence_lock_on_ee", "initial_direct", "periodic_ee_locked", "periodic_ee_lock_time", "periodic_ee_lock_control_call_index", "periodic_ee_lock_reason", "periodic_poll_count", "periodic_switch_count", "periodic_blocked_switch_proposals", "periodic_switch_limit_exhausted", "initial_inference_count", "total_inference_count", "task_wall_time")
        if spatial_config is not None:
            fields += ("spatial_poll_count", "spatial_sample_count", "spatial_neural_evaluations", "spatial_majority_polls", "spatial_no_majority_polls", "spatial_selection_wall_seconds")
        with (output / "comparison.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        progress("completed" if statistics["runner_errors"] == 0 else "completed_with_errors")
        atomic_json(output / "complete.json", dict(
            complete=True, validated=statistics["runner_errors"] == 0,
            completed_utc=utc_now(), **statistics,
            manifest_sha256=digest(output / "manifest.json"),
            data_sha256={name: digest(output / name) for name in ("summary.json", "analysis.json", "comparison.csv", "selected_cases.json")},
            result_sha256={r["case_id"]: digest(output / r["case_id"] / "summary.json") for r in results},
        ))
        return 0 if statistics["runner_errors"] == 0 else 1
    except BaseException as error:
        progress("failed")
        atomic_json(output / "failure.json", dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc(), timestamp_utc=utc_now()))
        raise
    finally:
        queue.close()
        queue.join_thread()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "split", "checkpoint", "experiment", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--contexts", type=Path)
    parser.add_argument("--gpus", required=True, help="Allocated GPU IDs/UUIDs, comma-separated")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--mode", choices=("adaptive", "static", "threshold_weighted", "collision_stall_skip", "periodic", "spatial"), default="adaptive")
    parser.add_argument("--partition", choices=("test", "validation"), default="test", help="Existing held-out split to replay; never creates a new split")
    parser.add_argument("--sampling-seed", type=int, default=20261125, help="Nonnegative seed for independent per-test-position weighted sampling")
    parser.add_argument("--probability-threshold", type=float, default=0.5, help="Original action scores must strictly exceed this threshold")
    parser.add_argument("--spatial-config", type=Path, help="Spatial mode: YAML with sample count, three-sigma radius, poll period and seed")
    parser.add_argument("--sequence-period", type=float, help="Periodic-mode polling interval in simulated seconds (default 0.1); integer multiple of control dt")
    parser.add_argument("--sequence-confirmations", type=int, help="Consecutive identical proposals required in periodic mode (default 5)")
    parser.add_argument("--sequence-max-switches", type=int, help="Optional nonnegative cap in periodic/spatial mode; polls continue after the cap")
    parser.add_argument("--sequence-lock-on-ee", action="store_true", help="Periodic/spatial mode: permanently stop polling and route changes at first EE entry, including initial Direct")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--indices", type=lambda value: [int(item) for item in value.split(",")], help="Explicit positions within the selected existing partition")
    for name in ("dataset", "split", "checkpoint"):
        parser.add_argument(f"--expected-{name}-sha256")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

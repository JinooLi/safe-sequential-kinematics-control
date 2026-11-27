"""Vote on complete routes using hypothetical nearby configurations.

Samples follow the untruncated isotropic Gaussian exactly: they are not clipped
to joint limits, rejected, or collision-filtered. They are network inputs only;
the real state is never moved to a sample and the controller's safety constraints
remain responsible for physical motion. The configured three-sigma radius is
not a hard bound or a 99.7% probability radius in seven dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from ..domain.link_actions import LINK_ACTIONS, action_id
from .link_selection import predict_selection


@dataclass(frozen=True)
class SpatialSelectionConfig:
    sample_count: int = 50
    three_sigma_radius_rad: float = 0.1
    period_seconds: float = 0.3
    sampling_seed: int = 20261126
    temporal_confirmations: int = 1

    def __post_init__(self):
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        if type(self.temporal_confirmations) is not int or self.temporal_confirmations <= 0:
            raise ValueError("temporal_confirmations must be a positive integer")
        if type(self.sampling_seed) is not int or self.sampling_seed < 0:
            raise ValueError("sampling_seed must be a nonnegative integer")
        for name in ("three_sigma_radius_rad", "period_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def sigma_rad(self):
        return self.three_sigma_radius_rad / 3.0


def load_spatial_selection_config(path: Path) -> SpatialSelectionConfig:
    """Load the explicit spatial_reselection section without silent typos."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "spatial_reselection" not in raw:
        raise ValueError("configuration needs a spatial_reselection mapping")
    section = raw["spatial_reselection"]
    if not isinstance(section, dict):
        raise ValueError("spatial_reselection must be a mapping")
    unknown = set(section) - {field.name for field in fields(SpatialSelectionConfig)}
    if unknown:
        raise ValueError(f"unknown spatial_reselection keys: {sorted(map(str, unknown))}")
    return SpatialSelectionConfig(**section)


def _vector(value, size, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result


def _sequence_index(sequence):
    sequence = tuple(sequence)
    if all(isinstance(frame, str) for frame in sequence):
        for index, links in enumerate(LINK_ACTIONS):
            if sequence == tuple(f"link{link}" for link in links):
                return index
    elif all(type(link) is int or isinstance(link, np.integer) for link in sequence):
        if sequence in LINK_ACTIONS:
            return LINK_ACTIONS.index(sequence)
    raise ValueError("current_sequence must be one complete catalog route")


class SpatialSequenceSelector:
    """Select a strict-majority route, retaining the current route otherwise.

    Each scene owns PCG64(SeedSequence([sampling_seed, position])). Draws depend
    only on that scene's poll order, not the number or scheduling of workers.
    The first poll is numbered zero; initial exact-state selection belongs to
    the caller and does not consume this random stream.
    """

    def __init__(
        self, config, position, model, selected_ik, target_position,
        target_quaternion, obstacles,
    ):
        if not isinstance(config, SpatialSelectionConfig):
            raise ValueError("config must be SpatialSelectionConfig")
        if type(position) is not int or position < 0:
            raise ValueError("partition position must be a nonnegative integer")
        if getattr(model, "actions", None) != LINK_ACTIONS:
            raise ValueError("spatial selection needs the sixteen-action model catalog")
        self.config = config
        self.model = model
        self.selected_ik = _vector(selected_ik, 7, "selected_ik")
        self.target_position = _vector(target_position, 3, "target_position")
        self.target_quaternion = _vector(target_quaternion, 4, "target_quaternion")
        self.obstacles = np.array(obstacles, dtype=np.float64, copy=True)
        if self.obstacles.size == 0:
            self.obstacles = np.empty((0, 4), dtype=np.float64)
        if (
            self.obstacles.ndim != 2 or self.obstacles.shape[1] != 4
            or not np.isfinite(self.obstacles).all() or np.any(self.obstacles[:, 3] < 0)
        ):
            raise ValueError("obstacles must contain finite [x,y,z,r] rows")
        self.seed_entropy = [config.sampling_seed, position]
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed_entropy)))
        self.poll_index = 0

    def select(self, q, current_sequence):
        selection_started = perf_counter()
        q = _vector(q, 7, "q")
        current_index = _sequence_index(current_sequence)
        samples = q + self.rng.normal(
            0.0, self.config.sigma_rad, size=(self.config.sample_count, 7)
        )
        sample_actions = []
        inference_wall_seconds = 0.0
        for sample in samples:
            inference_started = perf_counter()
            prediction = predict_selection(
                self.model, sample, self.selected_ik, self.target_position,
                self.target_quaternion, self.obstacles,
            )
            inference_wall_seconds += perf_counter() - inference_started
            scores = np.asarray(prediction.get("probabilities"), dtype=np.float64)
            if (
                scores.shape != (len(LINK_ACTIONS),)
                or not np.isfinite(scores).all()
                or np.any((scores < 0) | (scores > 1))
            ):
                raise ValueError("spatial voting needs sixteen finite probability scores in [0,1]")
            index = int(scores.argmax())
            if prediction.get("action_index") != index:
                raise ValueError("model action is not the probability argmax")
            sample_actions.append(index)
        counts = np.bincount(sample_actions, minlength=len(LINK_ACTIONS))
        winner = int(counts.argmax())
        majority = winner if 2 * counts[winner] > self.config.sample_count else None
        selected = current_index if majority is None else majority
        links = LINK_ACTIONS[selected]
        reason = (
            "spatial_no_majority" if majority is None
            else "spatial_current_majority" if majority == current_index
            else "spatial_majority"
        )
        vote = dict(
            samples=samples.tolist(),
            sample_actions=sample_actions,
            vote_counts=counts.tolist(),
            vote_fractions=(counts / self.config.sample_count).tolist(),
            majority_action_index=majority,
            winner_action_index=winner,
            current_action_index=current_index,
            sample_count=self.config.sample_count,
            sigma_rad=self.config.sigma_rad,
            three_sigma_radius_rad=self.config.three_sigma_radius_rad,
            poll_index=self.poll_index,
            sampling_seed=self.config.sampling_seed,
            rng_seed_sequence_entropy=self.seed_entropy.copy(),
            fraction_semantics="fraction_of_sample_argmax_votes_not_success_probability",
            majority_comparison="strictly_greater_than_half",
            joint_limit_sampling="untruncated_no_clipping_or_rejection",
        )
        self.poll_index += 1
        return dict(
            action_index=selected,
            action_id=action_id(links),
            pair_index=selected if len(links) == 2 else None,
            method="sequential" if links else "direct",
            link_frames=[f"link{link}" for link in links],
            reason=reason,
            spatial_vote=vote,
            inference_wall_seconds=inference_wall_seconds,
            spatial_selection_wall_seconds=perf_counter() - selection_started,
        )

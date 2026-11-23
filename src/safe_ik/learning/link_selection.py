"""Shared-obstacle encoders and reproducible scene-level supervised inputs.

Only joint angles are standardized. Goal positions, obstacle centers and radii
remain in metres in the dataset's common world/base coordinate system.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from ..domain.link_actions import (
    LINK_ACTIONS,
    LINK_PAIRS,
    action_id,
    actions_from_arrays,
    actions_from_metadata,
    validate_actions,
)

MODEL_SPEC = {
    "architecture": "cross_attention",
    "robot_hidden_dims": [64, 64],
    "obstacle_hidden_dims": [32, 32],
    "attention_dim": 32,
    "attention_heads": 1,
    "prediction_hidden_dims": [64],
    "feature_fusion": "concatenate",
    "mlp_dropout": 0.1,
    "mlp_dropout_scope": "all_hidden",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(directory, expected_sha256=None):
    directory = Path(directory)
    complete = json.loads((directory / "complete.json").read_text())
    actual_hash = sha256(directory / "dataset.npz")
    if actual_hash != complete["data_sha256"]["dataset.npz"] or (
        expected_sha256 is not None and actual_hash != expected_sha256
    ):
        raise ValueError("dataset checksum mismatch")
    with np.load(directory / "dataset.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    actions = actions_from_arrays(data)
    if actions == LINK_ACTIONS or any(
        key in complete for key in ("actions", "link_pairs", "action_schema_version")
    ):
        if actions_from_metadata(complete) != actions:
            raise ValueError("dataset/complete action catalog mismatch")
    size = len(data["case_id"])
    expected_shapes = {
        "initial_q": (size, 7),
        "selected_ik": (size, 7),
        "target_position": (size, 3),
        "target_quaternion": (size, 4),
        "success": (size, len(actions)),
        "label_valid": (size, len(actions)),
    }
    if size != complete["contexts"] or len(set(data["case_id"])) != size:
        raise ValueError("missing or duplicate scene IDs")
    for key, shape in expected_shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"invalid {key}")
    obstacles = data["obstacles"]
    if (
        obstacles.ndim != 3
        or obstacles.shape[0] != size
        or obstacles.shape[-1] != 4
        or not np.isfinite(obstacles).all()
        or np.any(obstacles[..., 3] < 0)
    ):
        raise ValueError("invalid obstacle array")
    if data["label_valid"].dtype != np.bool_:
        raise ValueError("label_valid must be boolean")
    # Numerical failures were executed and remain negative labels. Missing
    # rollouts are not failures and must never be silently trained as zeros.
    observed = data.get("observed")
    if observed is None:
        if actions != LINK_PAIRS:
            raise ValueError("action datasets require an observed outcome mask")
    elif (
        observed.dtype != np.bool_
        or observed.shape != (size, len(actions))
        or not observed.all()
    ):
        raise ValueError("unobserved outcomes cannot be used for training")
    valid = data["label_valid"]
    raw = data["success"]
    if not np.isin(raw[valid], [0, 1]).all() or not np.all(raw[~valid] == -1):
        raise ValueError("unexpected source label encoding")
    if int(valid.sum()) != complete["valid_labels"]:
        raise ValueError("valid-label count mismatch")
    if int((raw[valid] == 1).sum()) != complete["success_labels"]:
        raise ValueError("success count mismatch")
    # User-selected outcome policy: a numerical solver failure is a failure.
    # Preserve source arrays, and produce a separate fully observed target array.
    data["labels"] = np.where(valid, raw, 0).astype(np.float32)
    data["actions"] = actions
    data["dataset_sha256"] = actual_hash
    data["dataset_sources"] = complete.get("sources", [])
    return data


def make_state(
    initial_q,
    selected_ik,
    target_position,
    target_quaternion,
):
    """23 inputs: current joints, selected IK, goal position and 6D rotation."""
    q = np.asarray(initial_q, dtype=np.float64)
    ik = np.asarray(selected_ik, dtype=np.float64)
    position = np.asarray(target_position, dtype=np.float64)
    quaternion = np.asarray(target_quaternion, dtype=np.float64)
    n = len(q)
    if (
        q.shape != (n, 7)
        or ik.shape != (n, 7)
        or position.shape != (n, 3)
        or quaternion.shape != (n, 4)
    ):
        raise ValueError("unexpected state input shapes")
    if not all(np.isfinite(x).all() for x in (q, ik, position, quaternion)):
        raise ValueError("nonfinite state inputs")
    norm = np.linalg.norm(quaternion, axis=1)
    if np.any(norm < 1e-12):
        raise ValueError("zero quaternion")
    matrix = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
    rotation6 = np.concatenate((matrix[:, :, 0], matrix[:, :, 1]), axis=1)
    state = np.concatenate((q, ik, position, rotation6), axis=1).astype(np.float32)
    return state


def joint_statistics(state, train_indices):
    joints = np.asarray(state)[np.asarray(train_indices), :14].astype(np.float64)
    mean, std = joints.mean(axis=0), joints.std(axis=0)
    return mean.astype(np.float32), np.where(std < 1e-8, 1.0, std).astype(np.float32)


def _stratified_take(indices, strata, count, rng):
    """Allocate an exact-size subset proportionally by successful-pair count."""
    groups = [indices[strata[indices] == value] for value in np.unique(strata[indices])]
    quotas = np.array([len(group) for group in groups]) * count / len(indices)
    counts = np.floor(quotas).astype(int)
    remainder_order = np.argsort(-(quotas - counts), kind="stable")
    counts[remainder_order[: count - counts.sum()]] += 1
    taken, remaining = [], []
    for group, selected in zip(groups, counts):
        group = rng.permutation(group)
        taken.extend(group[:selected])
        remaining.extend(group[selected:])
    return rng.permutation(taken).astype(int), rng.permutation(remaining).astype(int)


def split_scenes(
    labels, train_size=700, validation_size=150, test_size=150, seed=20261016
):
    labels = np.asarray(labels)
    sizes = (train_size, validation_size, test_size)
    if (
        any(type(size) is not int for size in sizes)
        or train_size <= 0
        or validation_size <= 0
        or test_size < 0
        or sum(sizes) != len(labels)
    ):
        raise ValueError(
            "positive train/validation and nonnegative test sizes must sum to the scene count"
        )
    if labels.ndim != 2 or labels.shape[1] == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("split requires binary action labels per scene")
    rng = np.random.default_rng(seed)
    strata = labels.sum(axis=1).astype(int)
    remaining = np.arange(len(labels))
    if test_size:
        test, remaining = _stratified_take(remaining, strata, test_size, rng)
    validation, train = _stratified_take(remaining, strata, validation_size, rng)
    split = {"train": train, "validation": validation}
    if test_size:
        split["test"] = test
    return split


def load_scene_split(data, specification, dataset_directory):
    """Reuse hashed scene IDs when growing training data without moving holdouts."""
    if "file" not in specification:
        if "sha256" in specification:
            raise ValueError("a frozen split checksum requires a split file")
        return split_scenes(
            data["labels"],
            specification["train"],
            specification["validation"],
            specification["test"],
            specification["seed"],
        )
    path = Path(dataset_directory) / specification["file"]
    if sha256(path) != specification.get("sha256"):
        raise ValueError("frozen split checksum mismatch")
    record = json.loads(path.read_text())
    lookup = {case_id: index for index, case_id in enumerate(data["case_id"].tolist())}
    split = {}
    for key in ("train", "validation", "test"):
        size = specification[key]
        case_ids = (
            record.get(key, {"case_ids": []})["case_ids"]
            if key == "test" and size == 0
            else record[key]["case_ids"]
        )
        if (
            type(size) is not int
            or size < (0 if key == "test" else 1)
            or len(case_ids) != size
        ):
            raise ValueError(f"frozen split size mismatch: {key}")
        if size == 0:
            continue
        if any(case_id not in lookup for case_id in case_ids):
            raise ValueError("frozen split contains unknown scene IDs")
        split[key] = np.array([lookup[case_id] for case_id in case_ids], dtype=int)
    indices = np.concatenate(list(split.values()))
    if len(indices) != len(lookup) or len(set(indices)) != len(lookup):
        raise ValueError("frozen split must partition all scenes exactly once")
    return split


class LinkSuccessNet(nn.Module):
    """Selected one-head backbone; 16 actions by default, legacy 10 supported."""

    def __init__(self, joint_mean=None, joint_std=None, *, actions=LINK_ACTIONS):
        super().__init__()
        self.actions = validate_actions(actions)
        mean = np.zeros(14) if joint_mean is None else np.asarray(joint_mean)
        std = np.ones(14) if joint_std is None else np.asarray(joint_std)
        if (
            mean.shape != (14,)
            or std.shape != (14,)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
        ):
            raise ValueError("invalid joint standardization statistics")
        self.register_buffer("joint_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("joint_std", torch.tensor(std, dtype=torch.float32))
        # Keep the selected model's weight keys and seeded initialization order.
        self.obstacle_encoder = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(23, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(self.actions))
        )
        self.query_projection = nn.Linear(64, 32)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=1,
            kdim=32,
            vdim=32,
            dropout=0.0,
            batch_first=True,
        )
        self.context_projection = nn.Linear(32, 64)
        for mlp in (self.state_encoder, self.obstacle_encoder, self.head):
            for index, layer in enumerate(mlp):
                if isinstance(layer, nn.ReLU):
                    mlp[index] = nn.Sequential(layer, nn.Dropout(0.1))

    def obstacle_features(self, obstacles, mask=None, state_features=None):
        if obstacles.ndim != 3 or obstacles.shape[-1] != 4:
            raise ValueError("obstacles must have shape [batch, count, 4]")
        batch, count, _ = obstacles.shape
        if mask is None:
            mask = torch.ones((batch, count), dtype=torch.bool, device=obstacles.device)
        if mask.shape != (batch, count) or mask.dtype != torch.bool:
            raise ValueError("obstacle mask must be boolean [batch, count]")
        if count == 0:
            return obstacles.new_zeros((batch, 64))
        if state_features is None or state_features.shape != (batch, 64):
            raise ValueError("cross attention requires [batch, 64] robot features")
        clean = torch.where(mask[..., None], obstacles, torch.zeros_like(obstacles))
        encoded = self.obstacle_encoder(clean)
        has_obstacles = mask.any(dim=1, keepdim=True)
        # Avoid undefined all-masked softmax; discard the dummy row's context.
        attention_mask = mask.clone()
        attention_mask[:, 0] |= ~has_obstacles[:, 0]
        query = self.query_projection(state_features).unsqueeze(1)
        context, _ = self.cross_attention(
            query,
            encoded,
            encoded,
            key_padding_mask=~attention_mask,
            need_weights=False,
        )
        context = self.context_projection(context[:, 0])
        return torch.where(has_obstacles, context, torch.zeros_like(context))

    def forward(self, state, obstacles, mask=None):
        if state.ndim != 2 or state.shape[1] != 23 or len(state) != len(obstacles):
            raise ValueError("state must have shape [batch, 23]")
        normalized = torch.cat(
            ((state[:, :14] - self.joint_mean) / self.joint_std, state[:, 14:]), dim=1
        )
        robot_features = self.state_encoder(normalized)
        context = self.obstacle_features(obstacles, mask, robot_features)
        return self.head(torch.cat((robot_features, context), dim=1))


OBJECTIVE = "negative_log_discounted_success_mass"
SELECTION_METRIC = "validation_selected_time_reward"


class LinkTimeSelectionNet(LinkSuccessNet):
    """Same backbone/initialization; forward returns action selection probabilities."""

    output_activation = "softmax"

    def logits(self, state, obstacles, mask=None):
        return super().forward(state, obstacles, mask)

    def forward(self, state, obstacles, mask=None):
        return self.logits(state, obstacles, mask).softmax(dim=-1)


def validate_time_scale(value):
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise ValueError("time_scale_seconds must be finite and positive")


def predict_selection(
    model, initial_q, selected_ik, target_position, target_quaternion, obstacles
):
    """Offline inference only; a missing validated IK bypasses the network."""
    if selected_ik is None:
        return {
            "method": "direct",
            "action_id": "direct",
            "action_index": None,
            "pair_index": None,
            "link_frames": [],
            "probabilities": None,
            "reason": "no_validated_ik",
        }
    state = make_state(
        [initial_q], [selected_ik], [target_position], [target_quaternion]
    )
    obs = np.asarray(obstacles, dtype=np.float32)
    if obs.size == 0:
        obs = np.empty((0, 4), dtype=np.float32)
    if (
        obs.ndim != 2
        or obs.shape[1] != 4
        or not np.isfinite(obs).all()
        or np.any(obs[:, 3] < 0)
    ):
        raise ValueError("obstacles must contain finite [x,y,z,r] rows")
    device = model.joint_mean.device
    model.eval()
    with torch.no_grad():
        output = model(
            torch.as_tensor(state, device=device),
            torch.as_tensor(obs[None], device=device),
        )
        is_softmax = getattr(model, "output_activation", "sigmoid") == "softmax"
        probabilities = (output if is_softmax else output.sigmoid())[0].cpu().numpy()
    index = int(probabilities.argmax())
    links = model.actions[index]
    legacy = model.actions == LINK_PAIRS
    return {
        "method": "sequential" if links else "direct",
        "action_id": action_id(links),
        "action_index": index,
        # Compatibility for consumers of the original ten-pair API.
        "pair_index": index if len(links) == 2 else None,
        "link_frames": [f"link{i}" for i in links],
        "probabilities": probabilities.tolist(),
        "reason": (
            "highest_time_reward_selection_probability"
            if is_softmax
            else "highest_predicted_success"
        ),
        "probability_semantics": (
            ("link_pair_selection" if legacy else "action_selection")
            if is_softmax
            else ("per_pair_success" if legacy else "per_action_success")
        ),
    }


def load_model(checkpoint_path, device="cpu"):
    """Load the selected model, retaining schema-11/12 Adam inference support."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    version = checkpoint["schema_version"]
    actions = checkpoint_actions(checkpoint)
    if version in (16, 18):
        return load_time_checkpoint(checkpoint, device)
    if version in (12, 13, 17):
        matches = checkpoint.get("model") == MODEL_SPEC
    elif version == 11:
        # Metadata validation only: no implementations of retired variants.
        expected = {
            **{k: v for k, v in MODEL_SPEC.items() if k != "prediction_hidden_dims"},
            "head_hidden_layers": 1,
            "obstacle_self_attention": False,
            "prediction_self_attention": False,
            "fk_features": None,
        }
        matches = all(
            key in checkpoint and checkpoint[key] == value
            for key, value in expected.items()
        )
    else:
        raise ValueError("unsupported checkpoint; only the selected model is supported")
    if version in (13, 17):
        decay = checkpoint.get("weight_decay")
        optimizer_matches = (
            checkpoint.get("optimizer") == "adamw"
            and type(decay) in (int, float)
            and np.isfinite(decay)
            and decay >= 0
        )
    else:
        optimizer_matches = (
            checkpoint.get("optimizer") == "adam" and "weight_decay" not in checkpoint
        )
    if (
        not matches
        or not optimizer_matches
        or checkpoint.get("bce_loss_weight") != 1.0
        or checkpoint.get("selection_loss_weight") != 3.0
        or checkpoint.get("selection_metric") != "validation_selected_success_rate"
        or checkpoint.get("ranking_loss_weight", 0) != 0
        or checkpoint.get("time_loss_weight", 0) != 0
        or "parameter_l2_weight" in checkpoint
    ):
        raise ValueError("checkpoint does not match the selected model and objective")
    model = LinkSuccessNet(actions=actions)
    model.load_state_dict(checkpoint["state_dict"])
    # Loading state_dict replaces the normalization buffers; validate them too.
    if (
        not torch.isfinite(model.joint_mean).all()
        or not torch.isfinite(model.joint_std).all()
        or torch.any(model.joint_std <= 0)
    ):
        raise ValueError("invalid checkpoint joint standardization statistics")
    return model.to(device).eval()


def checkpoint_actions(checkpoint):
    """Never reinterpret legacy checkpoint rows as the expanded action catalog."""
    version = checkpoint.get("schema_version")
    if version in (17, 18):
        if "action_schema_version" not in checkpoint or "actions" not in checkpoint:
            raise ValueError("new checkpoints require an explicit action catalog")
        actions = actions_from_metadata(checkpoint)
        if checkpoint.get("output_dim") != len(actions):
            raise ValueError("checkpoint output dimension/catalog mismatch")
        return actions
    if version in (11, 12, 13, 16):
        actions = actions_from_metadata(checkpoint)
        if actions != LINK_PAIRS or "actions" in checkpoint:
            raise ValueError("legacy checkpoint link mapping mismatch")
        return actions
    raise ValueError("unsupported checkpoint; only the selected model is supported")


def load_time_checkpoint(checkpoint, device="cpu"):
    actions = checkpoint_actions(checkpoint)
    semantics = (
        "action_selection_probability"
        if checkpoint["schema_version"] == 18
        else "link_pair_selection_probability"
    )
    validate_time_scale(checkpoint.get("time_scale_seconds"))
    decay = checkpoint.get("weight_decay")
    if (
        checkpoint.get("schema_version") not in (16, 18)
        or checkpoint.get("model") != MODEL_SPEC
        or checkpoint.get("output_activation") != "softmax"
        or checkpoint.get("output_semantics") != semantics
        or checkpoint.get("optimization_loss") != OBJECTIVE
        or checkpoint.get("selection_metric") != SELECTION_METRIC
        or checkpoint.get("optimizer") != "adamw"
        or type(decay) not in (int, float)
        or not np.isfinite(decay)
        or decay < 0
        or checkpoint.get("bce_loss_weight") != 0.0
        or checkpoint.get("numerical_error_policy") != "failure"
    ):
        raise ValueError(
            "checkpoint does not match the softmax time-selection objective"
        )
    model = LinkTimeSelectionNet(actions=actions)
    model.load_state_dict(checkpoint["state_dict"])
    if (
        not torch.isfinite(model.joint_mean).all()
        or not torch.isfinite(model.joint_std).all()
        or torch.any(model.joint_std <= 0)
    ):
        raise ValueError("invalid checkpoint joint standardization statistics")
    return model.to(device).eval()

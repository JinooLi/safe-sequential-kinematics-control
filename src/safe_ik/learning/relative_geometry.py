"""Ordered action scoring from current-to-target links and obstacle geometry.

All geometry is supplied in metres in the robot's world/base frame. The network
transforms it into each current and target link frame. Only q and q* are
standardized; no future rollout states enter the features.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..domain.link_actions import LINK_ACTIONS, validate_actions


def validate_query_conditioning(value):
    if not isinstance(value, str) or value not in (
        "local",
        "reordered_control",
        "global_query",
    ):
        raise ValueError(
            "query_conditioning must be local, reordered_control or global_query"
        )
    return value


def relative_model_spec(mlp_dropout, geometry, *, query_conditioning="local"):
    from .link_selection import validate_mlp_dropout
    from .relative_geometry_features import relative_geometry_spec

    conditioning = validate_query_conditioning(query_conditioning)
    specification = {
        "architecture": "relative_geometry_cross_attention",
        "geometry": {
            **relative_geometry_spec(geometry),
            "ee_frame": geometry["ee_frame"],
            "capsules": geometry["capsules"],
        },
        "link_hidden_dims": [32, 32],
        "link_id_embedding_dim": 4,
        "link_obstacle_hidden_dims": [32, 32],
        "link_obstacle_input_dim": 9,
        "ee_obstacle_input_dim": 7,
        "attention_dim": 32,
        "attention_heads": 1,
        "attention_dropout": 0.0,
        "token_fusion_hidden_dims": [32],
        "global_hidden_dims": [64, 64],
        "sequential_prediction_hidden_dims": [64],
        "sequential_prediction": "shared_ordered_two_slots_with_presence_masks",
        "direct_prediction_hidden_dims": [64],
        "direct_ik_independent": True,
        "whole_body_clearance": "all_current_capsules_minimum_over_obstacles",
        "empty_obstacles": "zero_context_zero_clearance_and_presence_false",
        "mlp_dropout": validate_mlp_dropout(mlp_dropout),
        "mlp_dropout_scope": "all_hidden",
    }
    # Keep the original schema-20 metadata byte-for-byte compatible. Schema 21
    # explicitly distinguishes the matched forward order from its treatment.
    if conditioning != "local":
        specification["query_conditioning"] = conditioning
    return specification


def _mlp(dimensions, dropout, *, output=False):
    layers = []
    for index, (before, after) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(nn.Linear(before, after))
        if not output or index < len(dimensions) - 2:
            layers.extend((nn.ReLU(), nn.Dropout(dropout)))
    return nn.Sequential(*layers)


def _rotation(pose):
    """The supplied two columns belong to an FK/goal rotation, not a prediction."""
    first, second = pose[..., 3:6], pose[..., 6:9]
    return torch.stack((first, second, torch.linalg.cross(first, second)), dim=-1)


def _local_points(pose, centers):
    # pose [B,L,9], centers [B,N,3] -> [B,L,N,3]. Row vectors multiply R.
    return torch.matmul(centers[:, None] - pose[:, :, None, :3], _rotation(pose))


def _movement_features(current, target):
    rotation = _rotation(current)
    delta_position = torch.matmul(
        (target[..., :3] - current[..., :3]).unsqueeze(-2), rotation
    ).squeeze(-2)
    delta_rotation = rotation.transpose(-1, -2) @ _rotation(target)
    return torch.cat(
        (current, delta_position, delta_rotation[..., :, 0], delta_rotation[..., :, 1]),
        dim=-1,
    )


def capsule_clearances(endpoints, radii, obstacles):
    """Exact sphere/capsule surface signed distance; no safety margin.

    endpoints is [B,J,2,3], obstacles [B,N,4], result [B,J,N]. Endpoints
    may coincide (spherical capsules), and the nearest point is clamped to the
    segment. Masked obstacle rows must be sanitized before this calculation.
    """
    start, end = endpoints[..., 0, :], endpoints[..., 1, :]
    segment = end - start
    offset = obstacles[:, None, :, :3] - start[:, :, None]
    fraction = (offset * segment[:, :, None]).sum(-1) / (
        segment.square().sum(-1, keepdim=True).clamp_min(1e-12)
    )
    closest_offset = offset - fraction.clamp(0, 1)[..., None] * segment[:, :, None]
    return (
        torch.linalg.vector_norm(closest_offset, dim=-1)
        - radii[None, :, None]
        - obstacles[:, None, :, 3]
    )


class RelativeGeometrySuccessNet(nn.Module):
    """Five shared link tokens, a goal EE token, and ordered 16-action logits."""

    def __init__(
        self,
        joint_mean=None,
        joint_std=None,
        *,
        actions=LINK_ACTIONS,
        mlp_dropout=0.2,
        geometry,
        query_conditioning="local",
    ):
        super().__init__()
        from .link_selection import validate_mlp_dropout
        from .relative_geometry_features import relative_geometry_spec

        dropout = validate_mlp_dropout(mlp_dropout)
        self.query_conditioning = validate_query_conditioning(query_conditioning)
        self.actions = validate_actions(actions)
        if self.actions != LINK_ACTIONS:
            raise ValueError("relative geometry requires the 16-action catalog")
        specification = relative_geometry_spec(geometry)
        if any(geometry.get(key) != value for key, value in specification.items()):
            raise ValueError("relative geometry feature conventions do not match")
        self.relative_geometry = geometry
        self.state_input_dim = geometry["state_dim"]
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
        # Geometry is authoritative checkpoint metadata, not trainable state.
        self.register_buffer(
            "capsule_radii",
            torch.tensor(geometry["capsule_radii"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "link_capsule_indices",
            torch.tensor(geometry["link_capsule_indices"], dtype=torch.long),
            persistent=False,
        )
        self.capsule_count = len(geometry["capsule_radii"])
        self.link_embedding = nn.Embedding(5, 4)
        self.link_encoder = _mlp([22, 32, 32], dropout)
        self.link_obstacle_encoder = _mlp([9, 32, 32], dropout)
        self.link_attention = nn.MultiheadAttention(32, 1, dropout=0, batch_first=True)
        self.link_fusion = _mlp([64, 32], dropout)
        self.ee_encoder = _mlp([18, 32, 32], dropout)
        self.ee_obstacle_encoder = _mlp([7, 32, 32], dropout)
        self.ee_attention = nn.MultiheadAttention(32, 1, dropout=0, batch_first=True)
        self.ee_fusion = _mlp([64, 32], dropout)
        self.sequential_global = _mlp([24 + self.capsule_count, 64, 64], dropout)
        self.direct_global = _mlp([17 + self.capsule_count, 64, 64], dropout)
        self.sequential_head = _mlp([162, 64, 1], dropout, output=True)
        self.direct_head = _mlp([96, 64, 1], dropout, output=True)
        stages = [[a[0] - 2, a[1] - 2 if len(a) == 2 else 0] for a in self.actions[:-1]]
        presence = [[1.0, float(len(a) == 2)] for a in self.actions[:-1]]
        self.register_buffer("stage_indices", torch.tensor(stages), persistent=False)
        self.register_buffer("stage_presence", torch.tensor(presence), persistent=False)

    @staticmethod
    def _attention(query, edges, mask, encoder, attention):
        batch, links, _ = query.shape
        count = edges.shape[2]
        if count == 0:
            return query.new_zeros(query.shape)
        encoded = encoder(edges).reshape(batch * links, count, 32)
        has_obstacles = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_mask[:, 0] |= ~has_obstacles
        safe_mask = (
            safe_mask[:, None].expand(-1, links, -1).reshape(batch * links, count)
        )
        context, _ = attention(
            query.reshape(batch * links, 1, 32),
            encoded,
            encoded,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        context = context.reshape(batch, links, 32)
        return torch.where(
            has_obstacles[:, None, None], context, torch.zeros_like(context)
        )

    def forward(self, state, obstacles, mask=None):
        if state.ndim != 2 or state.shape[1] != self.state_input_dim:
            raise ValueError(f"state must have shape [batch, {self.state_input_dim}]")
        if (
            obstacles.ndim != 3
            or obstacles.shape[-1] != 4
            or len(state) != len(obstacles)
        ):
            raise ValueError("obstacles must have shape [batch, count, 4]")
        batch, count, _ = obstacles.shape
        if mask is None:
            mask = torch.ones((batch, count), dtype=torch.bool, device=obstacles.device)
        if mask.shape != (batch, count) or mask.dtype != torch.bool:
            raise ValueError("obstacle mask must be boolean [batch, count]")
        clean = torch.where(mask[..., None], obstacles, torch.zeros_like(obstacles))
        has_obstacles = mask.any(dim=1, keepdim=True)
        current = state[:, 23:68].reshape(batch, 5, 9)
        target = state[:, 68:113].reshape(batch, 5, 9)
        ee_current = state[:, 113:122].reshape(batch, 1, 9)
        ee_target = state[:, 14:23].reshape(batch, 1, 9)
        split = 122 + 6 * self.capsule_count
        current_capsules = state[:, 122:split].reshape(batch, self.capsule_count, 2, 3)
        target_capsules = state[:, split:].reshape(batch, self.capsule_count, 2, 3)
        current_distances = capsule_clearances(
            current_capsules, self.capsule_radii, clean
        )
        target_distances = capsule_clearances(
            target_capsules, self.capsule_radii, clean
        )
        if count:
            minimum = current_distances.masked_fill(~mask[:, None], torch.inf).amin(-1)
            minimum = torch.where(has_obstacles, minimum, torch.zeros_like(minimum))
        else:
            minimum = state.new_zeros((batch, self.capsule_count))
        body_context = torch.cat((minimum, has_obstacles.to(state.dtype)), dim=1)

        if self.query_conditioning != "local":
            # Both experimental modes use exactly the same module/dropout call
            # order. Only the Query supplied to link attention differs below.
            normalized_joints = (state[:, :14] - self.joint_mean) / self.joint_std
            sequential_global = self.sequential_global(
                torch.cat((normalized_joints, state[:, 14:23], body_context), dim=1)
            )
            direct_global = self.direct_global(
                torch.cat(
                    (normalized_joints[:, :7], state[:, 14:23], body_context), dim=1
                )
            )

        link_identity = self.link_embedding.weight[None].expand(batch, -1, -1)
        link_query = self.link_encoder(
            torch.cat((_movement_features(current, target), link_identity), dim=-1)
        )
        radii = clean[:, None, :, 3:4].expand(-1, 5, -1, -1)
        edges = torch.cat(
            (
                _local_points(current, clean[..., :3]),
                _local_points(target, clean[..., :3]),
                radii,
                current_distances[:, self.link_capsule_indices, :, None],
                target_distances[:, self.link_capsule_indices, :, None],
            ),
            dim=-1,
        )
        attention_query = link_query
        if self.query_conditioning == "global_query":
            global_query = (sequential_global[:, :32] + sequential_global[:, 32:]) / 2
            attention_query = link_query + global_query[:, None]
        link_context = self._attention(
            attention_query,
            edges,
            mask,
            self.link_obstacle_encoder,
            self.link_attention,
        )
        link_tokens = self.link_fusion(torch.cat((link_query, link_context), dim=-1))

        ee_query = self.ee_encoder(_movement_features(ee_current, ee_target))
        ee_edges = torch.cat(
            (
                _local_points(ee_current, clean[..., :3]),
                _local_points(ee_target, clean[..., :3]),
                clean[:, None, :, 3:4],
            ),
            dim=-1,
        )
        ee_context = self._attention(
            ee_query, ee_edges, mask, self.ee_obstacle_encoder, self.ee_attention
        )
        ee_token = self.ee_fusion(torch.cat((ee_query, ee_context), dim=-1))[:, 0]
        if self.query_conditioning == "local":
            normalized_joints = (state[:, :14] - self.joint_mean) / self.joint_std
            sequential_global = self.sequential_global(
                torch.cat((normalized_joints, state[:, 14:23], body_context), dim=1)
            )
            # Direct never consumes q*, target link poses or target capsule endpoints.
            direct_global = self.direct_global(
                torch.cat(
                    (normalized_joints[:, :7], state[:, 14:23], body_context), dim=1
                )
            )
        count_actions = len(self.actions) - 1
        stage_tokens = (
            link_tokens[:, self.stage_indices] * self.stage_presence[None, :, :, None]
        )
        sequential_input = torch.cat(
            (
                sequential_global[:, None].expand(-1, count_actions, -1),
                stage_tokens.flatten(2),
                ee_token[:, None].expand(-1, count_actions, -1),
                self.stage_presence[None].expand(batch, -1, -1),
            ),
            dim=-1,
        )
        sequential = self.sequential_head(sequential_input).squeeze(-1)
        direct = self.direct_head(torch.cat((direct_global, ee_token), dim=1))
        return torch.cat((sequential, direct), dim=1)

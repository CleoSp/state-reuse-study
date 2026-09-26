"""Dense graph-aware recurrence; original implementation, not a TRM reproduction.

Each outer cycle computes z = F(e + a + z) inner_cycles times, then
a = F(a + z). F contains two transformer layers and is shared across calls.
No targets, oracle annotations, stopping policy or state adapter enter this API.
"""
from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import Tensor, nn

from state_repair.types import (
    Domain, EncodedObservation, ObservationBatch, PredictionBatch,
    RecurrentState, SolverResult,
)


class GraphTransformerLayer(nn.Module):
    """Pre-LayerNorm attention with per-head observed relation bias.

    Dense mode has global receptive field: absent edges have learned relation 0,
    and open N/E/S/W passages have relations 1/2/3/4. Masked-neighbor mode permits
    only open edges and self, one graph hop per layer. Both execute dense matrix
    operations; the mask makes no sparse-compute claim. No IDs are embedded.
    """

    def __init__(self, width: int, heads: int, attention_mode: str = "dense") -> None:
        super().__init__()
        if attention_mode not in ("dense", "masked_neighbor"):
            raise ValueError("attention_mode must be dense or masked_neighbor")
        self.attention_mode = attention_mode
        self.heads = heads
        self.head_width = width // heads
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.relation_bias = nn.Embedding(5, heads)
        self.out = nn.Linear(width, width)
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width),
        )
        nn.init.normal_(self.relation_bias.weight, std=0.02)

    def forward(self, x: Tensor, relations: Tensor, valid: Tensor) -> Tensor:
        batch, nodes, width = x.shape
        qkv = self.qkv(self.norm_attention(x)).reshape(
            batch, nodes, 3, self.heads, self.head_width,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        scores = scores + self.relation_bias(relations).permute(0, 3, 1, 2)
        if self.attention_mode == "masked_neighbor":
            allowed = (relations > 0) | torch.eye(nodes, dtype=torch.bool, device=x.device)[None]
            allowed = allowed | ~valid[:, :, None]
            scores = scores.masked_fill(~allowed[:, None], -torch.inf)
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        attended = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, nodes, width)
        x = x + self.out(attended)
        x = x + self.feedforward(self.norm_feedforward(x))
        return x.masked_fill(~valid[:, :, None], 0)


class SharedGraphBlock(nn.Module):
    """Two distinct layers whose parameters are reused at every recurrence."""

    def __init__(self, width: int, heads: int, attention_mode: str = "dense") -> None:
        super().__init__()
        self.layers = nn.ModuleList([GraphTransformerLayer(width, heads, attention_mode) for _ in range(2)])
        self.final_norm = nn.LayerNorm(width)

    def forward(self, x: Tensor, relations: Tensor, valid: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, relations, valid)
        return self.final_norm(x).masked_fill(~valid[:, :, None], 0)


class RecursiveSolver(nn.Module):
    """Domain-configured solver preserving gradients through every requested step.

    Supplied state must belong to the current observation frame. Adapters must
    detach previous-frame state before constructing the current state.
    This core intentionally does not detach its input or intermediate latents.
    A new default forward always starts fresh, including across episodes.
    """

    def __init__(self, width: int = 64, heads: int = 2,
                 inner_cycles: int = 1, input_dim: int | None = None,
                 attention_mode: str = "masked_neighbor", domain: Domain = Domain.MAZE) -> None:
        super().__init__()
        if not isinstance(domain, Domain):
            raise TypeError("domain must be Domain")
        expected_dim = 4 if domain == Domain.MAZE else 6
        input_dim = expected_dim if input_dim is None else input_dim
        if any(type(v) is not int for v in (width, heads, inner_cycles, input_dim)):
            raise TypeError("model dimensions and cycle counts must be integers")
        if width < 1 or heads < 1 or width % heads or inner_cycles < 1 or input_dim != expected_dim:
            raise ValueError(f"model needs positive width divisible by heads, inner_cycles >= 1, input_dim={expected_dim}")
        self.domain = domain
        self.width = width
        self.inner_cycles = inner_cycles
        self.attention_mode = attention_mode
        self.encoder = nn.Linear(input_dim, width)
        self.initial_a = nn.Parameter(torch.empty(width))
        self.initial_z = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.initial_a, std=0.02)
        nn.init.normal_(self.initial_z, std=0.02)
        self.block = SharedGraphBlock(width, heads, attention_mode)
        self.head = nn.Linear(width, 6 if domain == Domain.MAZE else 2)

    def _check_observation(self, obs: ObservationBatch) -> None:
        if not isinstance(obs, ObservationBatch):
            raise TypeError("RecursiveSolver accepts only ObservationBatch")
        if obs.domain != self.domain:
            raise ValueError(f"observation domain {obs.domain.value} does not match solver domain {self.domain.value}")
        if obs.node_features.device != self.initial_a.device:
            raise ValueError("observation and model must share a device")
        if obs.node_features.dtype != self.initial_a.dtype:
            raise ValueError("observation and model must share a floating dtype")

    def _check_state(self, obs: ObservationBatch, state: RecurrentState) -> None:
        if not isinstance(state, RecurrentState):
            raise TypeError("state must be RecurrentState")
        state.check_observation(obs)
        if state.a.shape != (*obs.valid_nodes.shape, self.width):
            raise ValueError("state latent dimensions do not match observation/model")
        if state.a.device != obs.node_features.device or state.a.dtype != obs.node_features.dtype or state.z.dtype != state.a.dtype:
            raise ValueError("state and observation must share device and dtype")
        if not torch.isfinite(state.a).all() or not torch.isfinite(state.z).all():
            raise ValueError("state latents must be finite")

    def encode(self, obs: ObservationBatch) -> EncodedObservation:
        self._check_observation(obs)
        features = self.encoder(obs.node_features).masked_fill(~obs.valid_nodes[:, :, None], 0)
        return EncodedObservation(features, obs)

    def fresh_state(self, obs: ObservationBatch) -> RecurrentState:
        self._check_observation(obs)
        shape = (*obs.valid_nodes.shape, self.width)
        a = self.initial_a.expand(shape).clone().masked_fill(~obs.valid_nodes[:, :, None], 0)
        z = self.initial_z.expand(shape).clone().masked_fill(~obs.valid_nodes[:, :, None], 0)
        return RecurrentState(a, z, obs.episode_ids, obs.frame_indices,
                              obs.node_ids.clone(), obs.valid_nodes.clone(), budget=0)

    def step(self, encoded: EncodedObservation, state: RecurrentState) -> RecurrentState:
        if not isinstance(encoded, EncodedObservation):
            raise TypeError("step requires EncodedObservation")
        obs = encoded.observation
        self._check_observation(obs)
        self._check_state(obs, state)
        e = encoded.features
        if e.shape != state.a.shape or e.device != state.a.device or e.dtype != state.a.dtype:
            raise ValueError("encoded feature shape/device/dtype mismatch")
        valid = obs.valid_nodes
        a = state.a.masked_fill(~valid[:, :, None], 0)
        z = state.z.masked_fill(~valid[:, :, None], 0)
        for _ in range(self.inner_cycles):
            z = self.block(e + a + z, obs.edge_types, valid)
        a = self.block(a + z, obs.edge_types, valid)
        return replace(state, a=a, z=z, node_ids=state.node_ids.clone(),
                       valid_nodes=state.valid_nodes.clone(), budget=state.budget + 1)

    def decode(self, obs: ObservationBatch, state: RecurrentState) -> PredictionBatch:
        self._check_observation(obs)
        self._check_state(obs, state)
        return PredictionBatch(self.head(state.a).masked_fill(~obs.valid_nodes[:, :, None], 0))

    def forward(self, obs: ObservationBatch, outer_cycles: int,
                state: RecurrentState | None = None) -> SolverResult:
        if type(outer_cycles) is not int or outer_cycles < 0:
            raise ValueError("outer_cycles must be a nonnegative integer")
        encoded = self.encode(obs)
        if state is None:
            current = self.fresh_state(obs)
        else:
            self._check_state(obs, state)
            current = state.clone()
        for _ in range(outer_cycles):
            current = self.step(encoded, current)
        calls = (self.inner_cycles + 1) * outer_cycles
        return SolverResult(self.decode(obs, current), current, outer_cycles, calls, 2 * calls)

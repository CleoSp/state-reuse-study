"""Observation-only edit-boundary initializers; no oracle imports."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

import torch
from torch import Tensor, nn

from state_repair.models.recursive import SharedGraphBlock
from state_repair.types import Domain, PredictionBatch, RecurrentState, TransitionInput


@dataclass(frozen=True)
class AdapterResult:
    state: RecurrentState
    retain_a: Tensor | None = None
    retain_z: Tensor | None = None
    operations: dict[str, int] = field(default_factory=dict)


def current_state(fresh: RecurrentState, a: Tensor, z: Tensor) -> RecurrentState:
    """Attach current correspondence, reset per-frame budget, isolate storage."""
    mask = ~fresh.valid_nodes[..., None]
    return replace(fresh, a=a.masked_fill(mask, 0), z=z.masked_fill(mask, 0),
                   node_ids=fresh.node_ids.clone(), valid_nodes=fresh.valid_nodes.clone(), budget=0)


class StateAdapter(nn.Module):
    key: str
    is_control = False
    requires_previous_prediction = False

    def initialize(self, transition: TransitionInput, fresh: RecurrentState) -> AdapterResult:
        raise NotImplementedError

    def forward(self, transition: TransitionInput, fresh: RecurrentState,
                *, previous_prediction: PredictionBatch | None = None) -> AdapterResult:
        if not isinstance(transition, TransitionInput):
            raise TypeError("adapter requires TransitionInput")
        fresh.check_observation(transition.new)
        if fresh.budget != 0:
            raise ValueError("fresh state must have budget zero")
        if self.requires_previous_prediction:
            return self.initialize_from_prediction(transition, fresh, previous_prediction)
        return self.initialize(transition, fresh)

    def initialize_from_prediction(self, transition: TransitionInput, fresh: RecurrentState,
                                   prediction: PredictionBatch | None) -> AdapterResult:
        raise NotImplementedError


class RestartAdapter(StateAdapter):
    key = "restart"

    def initialize(self, transition: TransitionInput, fresh: RecurrentState) -> AdapterResult:
        return AdapterResult(fresh.clone())


class CarryAdapter(StateAdapter):
    key = "carry"

    def initialize(self, transition: TransitionInput, fresh: RecurrentState) -> AdapterResult:
        if transition.old_state is None:
            return AdapterResult(fresh.clone())
        old = transition.old_state.detach()
        return AdapterResult(current_state(fresh, old.a, old.z))


class ObservedContext(nn.Module):
    """Identical narrow context design for all learned principal adapters.

    Two layers over NEW relations give exactly two hops of contextual dependence
    from each node's injected observations/edits/state. Prior state can already
    summarize a wider graph. Old wiring is exposed through per-relation degrees;
    edit endpoints through incoming/outgoing changed-edge indicators. No IDs are
    embedded. Each arm owns its own copy with the same configuration and inputs.
    """

    receptive_field_hops = 2

    def __init__(self, width: int = 64, context_width: int = 16,
                 domain: Domain = Domain.MAZE) -> None:
        super().__init__()
        if not isinstance(domain, Domain) or any(type(v) is not int or v < 1 for v in (width, context_width)):
            raise ValueError("context requires Domain and positive integer widths")
        self.width, self.context_width, self.domain = width, context_width, domain
        self.features = 4 if domain == Domain.MAZE else 6
        self.relations = 4 if domain == Domain.MAZE else 2
        self.input_width = 3*self.features + 2 + 2*self.relations + 2*width
        self.project = nn.Linear(self.input_width, context_width)
        self.block = SharedGraphBlock(context_width, 1, "masked_neighbor")

    def forward(self, transition: TransitionInput) -> Tensor:
        old, new, edit, prior = transition.old, transition.new, transition.edit, transition.old_state
        if old is None or edit is None or prior is None:
            raise ValueError("context requires an edited frame")
        if new.domain != self.domain or prior.a.shape[-1] != self.width:
            raise ValueError("context domain/latent width mismatch")
        valid = new.valid_nodes
        h = torch.cat((prior.a.detach(), prior.z.detach()), -1).masked_fill(~valid[...,None], 0)
        endpoint = torch.stack((edit.edge_changed.any(-1), edit.edge_changed.any(-2)), -1).to(h.dtype)
        degree = torch.stack([(obs.edge_types == relation).sum(-1)
                              for obs in (old, new) for relation in range(1, self.relations+1)], -1).to(h.dtype)
        degree = degree / valid.sum(-1)[:,None,None]
        features = torch.cat((old.node_features, new.node_features, edit.node_features_delta,
                              endpoint, degree, h), -1)
        x = self.project(features).masked_fill(~valid[...,None], 0)
        return self.block(x, new.edge_types, valid)

    def operations(self, batch: int, nodes: int) -> dict[str, int]:
        c = self.context_width
        return {"context_calls": 1, "context_block_calls": 1, "context_layer_executions": 2,
                "adapter_linear_calls": 9, "adapter_gru_calls": 0,
                "adapter_macs": batch*nodes*self.input_width*c + 24*batch*nodes*c*c + 4*batch*nodes*nodes*c}


def mix_state(fresh: RecurrentState, old: RecurrentState, retain_a: Tensor,
              retain_z: Tensor, operations: dict[str, int]) -> AdapterResult:
    a = retain_a*old.a.detach() + (1-retain_a)*fresh.a
    z = retain_z*old.z.detach() + (1-retain_z)*fresh.z
    operations = {**operations, "latent_element_operations": 6*old.a.numel() + retain_a.numel() + retain_z.numel()}
    return AdapterResult(current_state(fresh, a, z), retain_a, retain_z, operations)


class LearnedAdapter(StateAdapter):
    def __init__(self, width: int = 64, context_width: int = 16,
                 domain: Domain = Domain.MAZE) -> None:
        super().__init__()
        self.context = ObservedContext(width, context_width, domain)
        self.width, self.context_width = width, context_width


class SpatialGateAdapter(LearnedAdapter):
    key = "spatial_gate"

    def __init__(self, width: int = 64, context_width: int = 16,
                 domain: Domain = Domain.MAZE, retain_override: float | None = None) -> None:
        super().__init__(width, context_width, domain)
        if retain_override is not None and (not math.isfinite(retain_override) or not 0 <= retain_override <= 1):
            raise ValueError("retain_override must be in [0,1]")
        self.retain_override = retain_override
        self.head = nn.Linear(context_width, 2)

    def retention(self, transition: TransitionInput) -> tuple[Tensor, Tensor, dict[str, int]]:
        context = self.context(transition)
        b, n, _ = context.shape
        operations = self.context.operations(b, n)
        if self.key == "global_gate":
            valid = transition.new.valid_nodes[...,None]
            context = (context*valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True)
            operations["pool_element_operations"] = b*n*self.context_width*2 + b*self.context_width
        logits = self.head(context)
        retain = logits.sigmoid()
        if self.retain_override is not None:
            retain = torch.full_like(retain, self.retain_override)
        operations["adapter_linear_calls"] += 1
        operations["adapter_macs"] += logits.numel()*self.context_width
        return retain[..., :1], retain[..., 1:], operations

    def initialize(self, transition: TransitionInput, fresh: RecurrentState) -> AdapterResult:
        if transition.old_state is None:
            return AdapterResult(fresh.clone())
        a, z, operations = self.retention(transition)
        return mix_state(fresh, transition.old_state, a, z, operations)


class GlobalGateAdapter(SpatialGateAdapter):
    key = "global_gate"


class GRUAdapter(LearnedAdapter):
    """PyTorch GRUCell, h=[a,z], x=context; candidate uses reset AFTER W_hn.

    n=tanh(W_in x+b_in + r*(W_hn h+b_hn)); h'=(1-u)*n+u*h.
    r,u=sigmoid(W_ir/iz x+b_ir/iz + W_hr/hz h+b_hr/hz).
    This is the ordinary torch.nn.GRUCell convention (not reset-before-matrix).
    """
    key = "gru_adapter"

    def __init__(self, width: int = 64, context_width: int = 16, domain: Domain = Domain.MAZE) -> None:
        super().__init__(width, context_width, domain)
        self.cell = nn.GRUCell(context_width, 2*width)

    def initialize(self, transition: TransitionInput, fresh: RecurrentState) -> AdapterResult:
        if transition.old_state is None:
            return AdapterResult(fresh.clone())
        context = self.context(transition)
        old = transition.old_state
        h = torch.cat((old.a.detach(), old.z.detach()), -1).masked_fill(~fresh.valid_nodes[...,None], 0)
        result = self.cell(context.flatten(0,1), h.flatten(0,1)).reshape_as(h)
        b, n, hidden = h.shape
        operations = self.context.operations(b, n)
        operations["adapter_gru_calls"] = 1
        operations["adapter_macs"] += 3*b*n*hidden*(hidden+self.context_width)
        operations["latent_element_operations"] = 8*b*n*hidden
        return AdapterResult(current_state(fresh, *result.split(self.width, -1)), operations=operations)


class ResidualAdapter(LearnedAdapter):
    key = "residual_adapter"

    def __init__(self, width: int = 64, context_width: int = 16, domain: Domain = Domain.MAZE) -> None:
        super().__init__(width, context_width, domain)
        self.head = nn.Linear(context_width, 2*width)

    def initialize(self, transition: TransitionInput, fresh: RecurrentState) -> AdapterResult:
        if transition.old_state is None:
            return AdapterResult(fresh.clone())
        context = self.context(transition)
        delta_a, delta_z = self.head(context).split(self.width, -1)
        old = transition.old_state
        b, n, _ = old.a.shape
        operations = self.context.operations(b, n)
        operations["adapter_linear_calls"] += 1
        operations["adapter_macs"] += 2*b*n*self.width*self.context_width
        operations["latent_element_operations"] = 2*old.a.numel()
        return AdapterResult(current_state(fresh, old.a.detach()+delta_a, old.z.detach()+delta_z), operations=operations)


def edited_nodes(transition: TransitionInput) -> Tensor:
    edit = transition.edit
    if edit is None:
        return torch.zeros_like(transition.new.valid_nodes)
    return (edit.node_features_delta.ne(0).any(-1) | edit.edge_changed.any(-1) |
            edit.edge_changed.any(-2)) & transition.new.valid_nodes


class LocalResetAdapter(StateAdapter):
    key = "local_reset"
    is_control = True

    def __init__(self, radius: int = 1) -> None:
        super().__init__()
        if type(radius) is not int or radius not in (1,2,3):
            raise ValueError("local_reset radius must be 1, 2 or 3")
        self.radius = radius

    def initialize(self, transition: TransitionInput, fresh: RecurrentState) -> AdapterResult:
        if transition.old_state is None:
            return AdapterResult(fresh.clone())
        reset = edited_nodes(transition)
        adjacency = (transition.old.edge_types > 0) | (transition.new.edge_types > 0)
        for _ in range(self.radius):
            reset = reset | (adjacency & reset[:,None,:]).any(-1)
        retain = (~reset & fresh.valid_nodes)[...,None].to(fresh.a.dtype)
        b, n = reset.shape
        return mix_state(fresh, transition.old_state, retain, retain,
                         {"adapter_macs": 0, "neighborhood_boolean_ops": self.radius*b*n*n})


class RandomResetAdapter(StateAdapter):
    key = "random_reset"
    is_control = True

    def __init__(self, reset_rate: float = 0.1) -> None:
        super().__init__()
        if not math.isfinite(reset_rate) or not 0 <= reset_rate <= 1:
            raise ValueError("reset_rate must be finite in [0,1]")
        self.reset_rate = reset_rate

    def initialize(self, transition: TransitionInput, fresh: RecurrentState) -> AdapterResult:
        if transition.old_state is None:
            return AdapterResult(fresh.clone())
        retain = fresh.valid_nodes.clone()
        for row, mask in zip(retain, fresh.valid_nodes):
            nodes = mask.nonzero().flatten()
            chosen = nodes[torch.randperm(len(nodes), device=nodes.device)[:round(self.reset_rate*len(nodes))]]
            row[chosen] = False
        retain = retain[...,None].to(fresh.a.dtype)
        return mix_state(fresh, transition.old_state, retain, retain,
                         {"adapter_macs": 0, "random_permutation_items": int(fresh.valid_nodes.sum())})


class NoisyCarryAdapter(StateAdapter):
    key = "noisy_carry"
    is_control = True

    def __init__(self, noise_scale: float = 0.01) -> None:
        super().__init__()
        if not math.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative")
        self.noise_scale = noise_scale

    def initialize(self, transition: TransitionInput, fresh: RecurrentState) -> AdapterResult:
        if transition.old_state is None:
            return AdapterResult(fresh.clone())
        old = transition.old_state
        noise = torch.zeros((*fresh.valid_nodes.shape, 2*fresh.a.shape[-1]), device=fresh.a.device, dtype=fresh.a.dtype)
        count = int(fresh.valid_nodes.sum())
        noise[fresh.valid_nodes] = torch.randn((count, noise.shape[-1]), device=noise.device, dtype=noise.dtype)
        da, dz = (self.noise_scale*noise).split(fresh.a.shape[-1], -1)
        return AdapterResult(current_state(fresh, old.a.detach()+da, old.z.detach()+dz),
                             operations={"adapter_macs": 0, "random_normal_values": count*noise.shape[-1],
                                         "latent_element_operations": 4*fresh.a.numel()})


class ShuffledGateAdapter(StateAdapter):
    """Pass the loaded trained spatial module; paired a/z gates share a shuffle."""
    key = "shuffled_gate"
    is_control = True

    def __init__(self, spatial_gate: SpatialGateAdapter) -> None:
        super().__init__()
        if type(spatial_gate) is not SpatialGateAdapter:
            raise TypeError("shuffled_gate requires a SpatialGateAdapter instance")
        self.spatial_gate = spatial_gate

    def initialize(self, transition: TransitionInput, fresh: RecurrentState) -> AdapterResult:
        if transition.old_state is None:
            return AdapterResult(fresh.clone())
        a, z, operations = self.spatial_gate.retention(transition)
        a, z = a.clone(), z.clone()
        for row, mask in enumerate(fresh.valid_nodes):
            nodes = mask.nonzero().flatten()
            order = nodes[torch.randperm(len(nodes), device=nodes.device)]
            a[row,nodes], z[row,nodes] = a[row,order], z[row,order]
        operations["random_permutation_items"] = int(fresh.valid_nodes.sum())
        return mix_state(fresh, transition.old_state, a, z, operations)


class AnswerOnlyAdapter(StateAdapter):
    key = "answer_only"
    is_control = True
    requires_previous_prediction = True

    def __init__(self, width: int = 64, domain: Domain = Domain.MAZE) -> None:
        super().__init__()
        if not isinstance(domain, Domain) or type(width) is not int or width < 1:
            raise ValueError("answer_only requires Domain and positive width")
        self.domain, self.width = domain, width
        self.project = nn.Linear(6 if domain == Domain.MAZE else 2, 2*width)

    def initialize(self, transition: TransitionInput, fresh: RecurrentState) -> AdapterResult:
        return self.initialize_from_prediction(transition, fresh, None)

    def initialize_from_prediction(self, transition: TransitionInput, fresh: RecurrentState,
                                   prediction: PredictionBatch | None) -> AdapterResult:
        if all(frame == 0 for frame in transition.new.frame_indices):
            return AdapterResult(fresh.clone())
        if not isinstance(prediction, PredictionBatch):
            raise ValueError("answer_only requires previous model prediction; old latents cannot substitute")
        logits = prediction.logits
        if transition.new.domain != self.domain or logits.shape != (*fresh.valid_nodes.shape, self.project.in_features):
            raise ValueError("previous prediction domain/shape mismatch")
        if logits.device != fresh.a.device or logits.dtype != fresh.a.dtype or not torch.isfinite(logits).all():
            raise ValueError("previous prediction must have finite matching device/dtype")
        probabilities = logits.detach().softmax(-1).masked_fill(~fresh.valid_nodes[...,None], 0)
        result = self.project(probabilities)
        return AdapterResult(current_state(fresh, *result.split(self.width, -1)),
                             operations={"adapter_linear_calls": 1, "adapter_macs": result.numel()*self.project.in_features})


ADAPTER_REGISTRY: dict[str, type[StateAdapter]] = {
    "restart": RestartAdapter, "carry": CarryAdapter,
    "spatial_gate": SpatialGateAdapter, "global_gate": GlobalGateAdapter,
    "gru_adapter": GRUAdapter, "residual_adapter": ResidualAdapter,
    "local_reset": LocalResetAdapter, "random_reset": RandomResetAdapter,
    "noisy_carry": NoisyCarryAdapter, "shuffled_gate": ShuffledGateAdapter,
    "answer_only": AnswerOnlyAdapter,
}


def make_adapter(key: str, **kwargs: Any) -> StateAdapter:
    if key not in ADAPTER_REGISTRY:
        raise ValueError(f"unknown deployable adapter {key!r}; choose {sorted(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[key](**kwargs)

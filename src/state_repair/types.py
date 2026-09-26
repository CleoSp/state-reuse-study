"""Explicit policy-visible types. Oracle annotations never enter inference APIs."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import torch
from torch import Tensor


class Domain(str, Enum):
    MAZE = "maze"
    CIRCUIT = "circuit"


def _bookkeeping(episodes: tuple[str, ...], frames: tuple[int, ...], batch: int) -> None:
    if not isinstance(episodes, tuple) or not isinstance(frames, tuple):
        raise ValueError("episode_ids/frame_indices must be immutable tuples")
    if len(episodes) != batch or len(frames) != batch:
        raise ValueError("bookkeeping batch mismatch")
    if any(not isinstance(v, str) or not v for v in episodes):
        raise ValueError("episode IDs must be nonempty strings")
    if any(type(v) is not int or v < 0 for v in frames):
        raise ValueError("frame indices must be nonnegative integers")


def _node_identity(ids: Tensor, valid: Tensor) -> None:
    if (ids[~valid] != -1).any() or not valid.any(dim=1).all():
        raise ValueError("node padding requires ID -1 and at least one valid node per example")
    n = ids.shape[1]
    counts = valid.sum(1, keepdim=True)
    positions = torch.arange(n, device=ids.device)[None].expand_as(ids)
    expected = positions.masked_fill(positions >= counts, n)
    actual = ids.masked_fill(~valid, n).sort(dim=1).values
    if not torch.equal(actual, expected):
        raise ValueError("valid node identities must be a permutation of contiguous stable IDs")


@dataclass(frozen=True)
class ObservationBatch:
    """Observed graphs; edge_types[B, source, target] uses domain vocabulary.

    Maze relations are 0/1N/2E/3S/4W, circuit 0/1 wire/2 reverse relation.
    Node IDs are stable identities, not presentation positions. IDs,
    episode IDs and frame indices are validation/bookkeeping only, never features.
    """
    domain: Domain
    node_features: Tensor
    edge_types: Tensor
    valid_nodes: Tensor
    episode_ids: tuple[str, ...]
    frame_indices: tuple[int, ...]
    node_ids: Tensor
    grid_shapes: tuple[tuple[int, int], ...] = ()
    starts: Tensor | None = None
    goals: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.domain, Domain):
            raise TypeError("domain must be Domain")
        if any(not isinstance(getattr(self, name), Tensor) for name in
               ("node_features", "edge_types", "valid_nodes", "node_ids")):
            raise TypeError("observation tensor fields must be torch.Tensor")
        if not isinstance(self.grid_shapes, tuple) or any(not isinstance(shape, tuple) or len(shape) != 2 for shape in self.grid_shapes):
            raise ValueError("grid_shapes must be a tuple of (height,width) pairs")
        x = self.node_features
        if x.ndim != 3 or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("node_features must be finite float [B,N,F]")
        b, n, f = x.shape
        if b < 1 or n < 1 or f != (4 if self.domain == Domain.MAZE else 6):
            raise ValueError("empty batch or wrong domain feature count")
        if self.valid_nodes.shape != (b,n) or self.valid_nodes.dtype != torch.bool:
            raise ValueError("valid_nodes must be bool [B,N]")
        if not self.valid_nodes.any(dim=1).all():
            raise ValueError("each example needs a valid node")
        if self.edge_types.shape != (b,n,n) or self.edge_types.dtype != torch.long:
            raise ValueError("edge_types must be long [B,N,N]")
        if self.node_ids.shape != (b,n) or self.node_ids.dtype != torch.long:
            raise ValueError("node_ids must be long [B,N]")
        tensors = (self.edge_types,self.valid_nodes,self.node_ids)
        if any(t.device != x.device for t in tensors):
            raise ValueError("observation tensors must share a device")
        if len(self.episode_ids)!=b or len(self.frame_indices)!=b:
            raise ValueError("bookkeeping batch mismatch")
        _bookkeeping(self.episode_ids, self.frame_indices, b)
        pair = self.valid_nodes.unsqueeze(1) & self.valid_nodes.unsqueeze(2)
        if (x[~self.valid_nodes]!=0).any() or (self.edge_types[~pair]!=0).any() or (self.node_ids[~self.valid_nodes]!=-1).any():
            raise ValueError("padding must have zero features/edges and node_id -1")
        if (self.edge_types<0).any() or (self.edge_types>(4 if self.domain == Domain.MAZE else 2)).any():
            raise ValueError("unsupported relation code")
        _node_identity(self.node_ids, self.valid_nodes)
        if self.domain == Domain.CIRCUIT:
            if self.grid_shapes or self.starts is not None or self.goals is not None:
                raise ValueError("circuit observations cannot contain maze metadata")
            self._check_circuit()
            return
        if len(self.grid_shapes) != b:
            raise ValueError("maze grid_shapes batch mismatch")
        if not isinstance(self.starts, Tensor) or not isinstance(self.goals, Tensor):
            raise ValueError("maze starts/goals must be tensors")
        if self.starts.shape!=(b,) or self.goals.shape!=(b,) or self.starts.dtype!=torch.long or self.goals.dtype!=torch.long:
            raise ValueError("starts/goals must be long [B]")
        if self.starts.device != x.device or self.goals.device != x.device:
            raise ValueError("observation tensors must share a device")
        if any(type(h) is not int or type(w) is not int or h < 1 or w < 1 or h*w > n for h, w in self.grid_shapes):
            raise ValueError("grid shape must match valid nodes")
        shapes = torch.tensor(self.grid_shapes, dtype=torch.long, device=x.device)
        counts = self.valid_nodes.sum(1)
        if not torch.equal(shapes.prod(1), counts):
            raise ValueError("grid shape must match valid nodes")
        if ((self.starts < 0) | (self.starts >= counts) | (self.goals < 0) | (self.goals >= counts)).any():
            raise ValueError("start/goal identity outside grid")
        graph, source, target = self.edge_types.nonzero(as_tuple=True)
        codes = self.edge_types[graph, source, target]
        source_ids, target_ids = self.node_ids[graph, source], self.node_ids[graph, target]
        widths = shapes[graph, 1]
        dr = target_ids // widths - source_ids // widths
        dc = target_ids % widths - source_ids % widths
        expected_dr = torch.tensor([0, -1, 0, 1, 0], device=x.device)[codes]
        expected_dc = torch.tensor([0, 0, 1, 0, -1], device=x.device)[codes]
        reverse = torch.tensor([0, 3, 4, 1, 2], device=x.device)[codes]
        if not torch.equal(dr, expected_dr) or not torch.equal(dc, expected_dc):
            raise ValueError("edge direction must match source-to-target maze geometry")
        if not torch.equal(self.edge_types[graph, target, source], reverse):
            raise ValueError("maze passages require opposite reverse edge directions")

    def _check_circuit(self) -> None:
        """Validate observed wiring only; never store a topological trace."""
        x, valid = self.node_features, self.valid_nodes
        operators = x[..., :5]
        if ((operators != 0) & (operators != 1)).any() or not (operators.sum(-1)[valid] == 1).all():
            raise ValueError("circuit operators must be exact one-hot INPUT/AND/OR/XOR/NOT")
        inputs = operators[..., 0].bool() & valid
        if ((x[..., 5] != 0) & (x[..., 5] != 1)).any() or (x[..., 5][~inputs] != 0).any():
            raise ValueError("circuit input bits must be binary and zero at internal/padded nodes")
        if not (valid & ~inputs).any(-1).all():
            raise ValueError("each circuit needs a scored non-input gate")
        wires = self.edge_types == 1
        if not torch.equal(wires.transpose(1, 2), self.edge_types == 2) or wires.diagonal(dim1=1,dim2=2).any():
            raise ValueError("circuit wires require reverse relation 2 and no self edges")
        expected = torch.tensor([0,2,2,2,1], device=x.device)[operators.argmax(-1)]
        if not torch.equal(wires.sum(1)[valid], expected[valid]):
            raise ValueError("circuit wire arity must be 0/2/2/2/1 with distinct parents")
        for graph, mask in zip(wires.detach().cpu(), valid.detach().cpu()):
            indegree = graph.sum(0).tolist()
            pending = [i for i in range(len(mask)) if mask[i] and indegree[i] == 0]
            visited = 0
            while pending:
                u = pending.pop()
                visited += 1
                for v in graph[u].nonzero().flatten().tolist():
                    indegree[v] -= 1
                    if indegree[v] == 0:
                        pending.append(v)
            if visited != int(mask.sum()):
                raise ValueError("circuit wiring contains a cycle")

    def to(self, device: str | torch.device) -> ObservationBatch:
        return replace(self, **{k:getattr(self,k).to(device) if getattr(self,k) is not None else None for k in
            ("node_features","edge_types","valid_nodes","node_ids","starts","goals")})


@dataclass(frozen=True)
class RecurrentState:
    a: Tensor
    z: Tensor
    episode_ids: tuple[str, ...]
    frame_indices: tuple[int, ...]
    node_ids: Tensor
    valid_nodes: Tensor
    budget: int = 0

    def __post_init__(self) -> None:
        if any(not isinstance(getattr(self, name), Tensor) for name in ("a", "z", "node_ids", "valid_nodes")):
            raise TypeError("state tensor fields must be torch.Tensor")
        if self.a.ndim!=3 or self.a.shape!=self.z.shape or not self.a.is_floating_point() or not self.z.is_floating_point():
            raise ValueError("latents must be matching floating [B,N,D]")
        b,n,_=self.a.shape
        if min(self.a.shape) < 1 or self.a.dtype != self.z.dtype:
            raise ValueError("latents require nonempty matching dtype dimensions")
        if not torch.isfinite(self.a).all() or not torch.isfinite(self.z).all():
            raise ValueError("latents must be finite, including padding")
        if self.node_ids.shape!=(b,n) or self.valid_nodes.shape!=(b,n) or self.node_ids.dtype!=torch.long or self.valid_nodes.dtype!=torch.bool:
            raise ValueError("state node metadata mismatch")
        _bookkeeping(self.episode_ids, self.frame_indices, b)
        if type(self.budget) is not int or self.budget<0:
            raise ValueError("state bookkeeping mismatch")
        if any(t.device!=self.a.device for t in (self.z,self.node_ids,self.valid_nodes)):
            raise ValueError("state tensors must share a device")
        _node_identity(self.node_ids, self.valid_nodes)

    def check_observation(self, obs: ObservationBatch, *, same_frame: bool = True) -> None:
        if not isinstance(obs, ObservationBatch):
            raise TypeError("state correspondence requires ObservationBatch")
        if self.a.device != obs.node_features.device or self.a.shape[:2] != obs.node_features.shape[:2]:
            raise ValueError("state observation shape/device correspondence mismatch")
        if self.episode_ids!=obs.episode_ids or not torch.equal(self.node_ids,obs.node_ids) or not torch.equal(self.valid_nodes,obs.valid_nodes):
            raise ValueError("state episode/node correspondence mismatch")
        if same_frame and self.frame_indices!=obs.frame_indices:
            raise ValueError("state frame mismatch")

    def clone(self) -> RecurrentState:
        return replace(self,a=self.a.clone(),z=self.z.clone(),node_ids=self.node_ids.clone(),valid_nodes=self.valid_nodes.clone())

    def detach(self) -> RecurrentState:
        return replace(self,a=self.a.detach().clone(),z=self.z.detach().clone(),node_ids=self.node_ids.clone(),valid_nodes=self.valid_nodes.clone())

    def to(self, device: str | torch.device) -> RecurrentState:
        return replace(self,a=self.a.to(device).clone(),z=self.z.to(device).clone(),node_ids=self.node_ids.to(device).clone(),valid_nodes=self.valid_nodes.to(device).clone())

    def reset(self, mask: Tensor, fresh: RecurrentState) -> RecurrentState:
        if not isinstance(mask, Tensor) or not isinstance(fresh, RecurrentState):
            raise TypeError("reset requires tensor mask and RecurrentState")
        if mask.shape!=(self.a.shape[0],) or mask.dtype!=torch.bool or self.a.shape!=fresh.a.shape:
            raise ValueError("reset needs bool [B] and matching fresh state")
        if mask.device != self.a.device or fresh.a.device != self.a.device or fresh.a.dtype != self.a.dtype:
            raise ValueError("reset mask and states must share device and latent dtype")
        if not mask.any():
            return self.clone()
        if self.budget != fresh.budget and not mask.all():
            raise ValueError("mixed budgets require separate state batches")
        m=mask[:,None,None]
        flags=mask.tolist()
        return RecurrentState(torch.where(m,fresh.a,self.a),torch.where(m,fresh.z,self.z),
            tuple(fresh.episode_ids[i] if v else self.episode_ids[i] for i,v in enumerate(flags)),
            tuple(fresh.frame_indices[i] if v else self.frame_indices[i] for i,v in enumerate(flags)),
            torch.where(mask[:,None],fresh.node_ids,self.node_ids),torch.where(mask[:,None],fresh.valid_nodes,self.valid_nodes),
            fresh.budget if mask.all() else self.budget)


@dataclass(frozen=True)
class ObservedEdit:
    node_features_delta: Tensor
    edge_changed: Tensor

    def __post_init__(self) -> None:
        delta = self.node_features_delta
        if not isinstance(delta, Tensor) or not isinstance(self.edge_changed, Tensor):
            raise TypeError("edit fields must be torch.Tensor")
        if delta.ndim != 3 or min(delta.shape) < 1 or not delta.is_floating_point() or not torch.isfinite(delta).all():
            raise ValueError("edit delta must be finite floating [B,N,F]")
        b, n, _ = delta.shape
        if self.edge_changed.shape != (b,n,n) or self.edge_changed.dtype != torch.bool:
            raise ValueError("edge_changed must be bool [B,N,N]")
        if self.edge_changed.device != delta.device:
            raise ValueError("edit tensors must share a device")

    @classmethod
    def between(cls, old: ObservationBatch, new: ObservationBatch) -> ObservedEdit:
        if not isinstance(old, ObservationBatch) or not isinstance(new, ObservationBatch):
            raise TypeError("edits require ObservationBatch")
        if old.domain != new.domain or old.node_features.shape != new.node_features.shape or old.grid_shapes != new.grid_shapes:
            raise ValueError("edits require matching domain, shape and grid layout")
        if old.node_features.device != new.node_features.device or old.node_features.dtype != new.node_features.dtype:
            raise ValueError("edited observations must share device and feature dtype")
        if old.episode_ids!=new.episode_ids or not torch.equal(old.node_ids,new.node_ids) or not torch.equal(old.valid_nodes,new.valid_nodes):
            raise ValueError("edits require stable episode/node correspondence")
        return cls(new.node_features-old.node_features,new.edge_types!=old.edge_types)


@dataclass(frozen=True)
class TransitionInput:
    new: ObservationBatch
    old: ObservationBatch | None = None
    old_state: RecurrentState | None = None
    edit: ObservedEdit | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.new,ObservationBatch):
            raise TypeError("policy requires ObservationBatch")
        if all(t==0 for t in self.new.frame_indices):
            if any(v is not None for v in (self.old,self.old_state,self.edit)):
                raise ValueError("frame zero requires fresh initialization")
            return
        if self.old is None or self.old_state is None or self.edit is None:
            raise ValueError("edited frame requires old observation/state and observed edit")
        if not isinstance(self.old, ObservationBatch) or not isinstance(self.old_state, RecurrentState) or not isinstance(self.edit, ObservedEdit):
            raise TypeError("transition requires ObservationBatch, RecurrentState and ObservedEdit")
        if any(t == 0 for t in self.new.frame_indices):
            raise ValueError("mixed fresh/edited frames require separate transition batches")
        self.old_state.check_observation(self.old)
        if any(n!=o+1 for o,n in zip(self.old.frame_indices,self.new.frame_indices)):
            raise ValueError("transition frames must be consecutive")
        expected=ObservedEdit.between(self.old,self.new)
        if self.edit.node_features_delta.device != expected.node_features_delta.device or self.edit.node_features_delta.dtype != expected.node_features_delta.dtype:
            raise ValueError("edit device/dtype must match observations")
        if not torch.equal(expected.node_features_delta,self.edit.node_features_delta) or not torch.equal(expected.edge_changed,self.edit.edge_changed):
            raise ValueError("edit must be derived from observations")


@dataclass(frozen=True)
class TargetBatch:
    valid_actions: Tensor | None = None
    values: Tensor | None = None
    scored_mask: Tensor | None = None

    def __post_init__(self) -> None:
        if self.valid_actions is None:
            if not isinstance(self.values, Tensor) or not isinstance(self.scored_mask, Tensor):
                raise TypeError("circuit targets require values and scored_mask tensors")
            if self.values.ndim != 2 or min(self.values.shape) < 1 or self.values.dtype != torch.long:
                raise ValueError("circuit values must be long [B,N]")
            if self.scored_mask.shape != self.values.shape or self.scored_mask.dtype != torch.bool or self.scored_mask.device != self.values.device:
                raise ValueError("scored_mask must be bool [B,N] on target device")
            if not self.scored_mask.any(-1).all():
                raise ValueError("each circuit requires nonempty scored gates")
            if ((self.values[self.scored_mask] < 0) | (self.values[self.scored_mask] > 1)).any() or (self.values[~self.scored_mask] != -1).any():
                raise ValueError("circuit targets require 0/1 scored values and -1 elsewhere")
            return
        if self.values is not None or self.scored_mask is not None:
            raise ValueError("targets require exactly one domain variant")
        if not isinstance(self.valid_actions, Tensor):
            raise TypeError("valid_actions must be torch.Tensor")
        if self.valid_actions.ndim!=3 or self.valid_actions.shape[-1]!=6 or self.valid_actions.dtype!=torch.bool:
            raise ValueError("valid_actions must be bool [B,N,6]")
        if min(self.valid_actions.shape) < 1:
            raise ValueError("target batch cannot be empty")

    def check_observation(self, obs: ObservationBatch) -> None:
        """Trainer/data boundary only; never call from inference."""
        if not isinstance(obs, ObservationBatch):
            raise TypeError("target validation requires ObservationBatch")
        if self.valid_actions is None:
            if obs.domain != Domain.CIRCUIT:
                raise ValueError("circuit targets require circuit observations")
            if self.values.shape != obs.valid_nodes.shape or self.values.device != obs.valid_nodes.device:
                raise ValueError("target shape/device must match observation")
            if not torch.equal(self.scored_mask, obs.valid_nodes & ~obs.node_features[..., 0].bool()):
                raise ValueError("scored_mask must equal valid non-input gates")
            return
        if obs.domain != Domain.MAZE:
            raise ValueError("maze targets require maze observations")
        if self.valid_actions.shape[:2] != obs.valid_nodes.shape or self.valid_actions.device != obs.valid_nodes.device:
            raise ValueError("target shape/device must match observation")
        if not self.valid_actions.any(-1)[obs.valid_nodes].all():
            raise ValueError("every valid node needs at least one target action")
        if self.valid_actions[~obs.valid_nodes].any():
            raise ValueError("padded nodes cannot have target actions")

    def to(self, device: str | torch.device) -> TargetBatch:
        return replace(self, **{name: getattr(self, name).to(device) if getattr(self, name) is not None else None
                               for name in ("valid_actions", "values", "scored_mask")})


@dataclass(frozen=True)
class OracleMetadata:
    distances: Tensor
    action_set_changed: Tensor | None = None
    distance_changed: Tensor | None = None
    old_prediction_now_invalid: Tensor | None = None
    audit: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.distances, Tensor):
            raise TypeError("oracle distances must be torch.Tensor")
        if self.distances.ndim != 2 or min(self.distances.shape) < 1 or self.distances.dtype != torch.long:
            raise ValueError("oracle distances must be long [B,N]; -1 unreachable, -2 padding")
        if (self.distances < -2).any():
            raise ValueError("unsupported oracle distance sentinel")
        for name in ("action_set_changed", "distance_changed", "old_prediction_now_invalid"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, Tensor) or value.shape != self.distances.shape or value.dtype != torch.bool or value.device != self.distances.device):
                raise ValueError(f"{name} must be bool [B,N] on oracle device")
        if self.audit is not None and not isinstance(self.audit, dict):
            raise ValueError("oracle audit must be a mapping")


@dataclass(frozen=True)
class EncodedObservation:
    features: Tensor
    observation: ObservationBatch


@dataclass(frozen=True)
class PredictionBatch:
    logits: Tensor


@dataclass(frozen=True)
class SolverResult:
    prediction: PredictionBatch
    state: RecurrentState
    outer_cycles: int
    block_calls: int
    transformer_layer_executions: int


@dataclass(frozen=True)
class TrainingExample:
    transition: TransitionInput
    target: TargetBatch

    def __post_init__(self) -> None:
        if not isinstance(self.transition, TransitionInput) or not isinstance(self.target, TargetBatch):
            raise TypeError("training example requires separate transition and target types")
        self.target.check_observation(self.transition.new)

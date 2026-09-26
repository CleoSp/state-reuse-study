"""Trainer-only one-edit objective with separately typed observations and labels."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from state_repair.models.adapters import AdapterResult, StateAdapter
from state_repair.models.recursive import RecursiveSolver
from state_repair.train.losses import circuit_value_loss, maze_valid_set_loss
from state_repair.types import Domain, ObservationBatch, ObservedEdit, PredictionBatch, RecurrentState, TargetBatch, TransitionInput


@dataclass(frozen=True)
class FrozenPrior:
    observation: ObservationBatch
    state: RecurrentState
    prediction: PredictionBatch
    checkpoint_sha256: str
    source_policy: str
    budget: int

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ObservationBatch) or not isinstance(self.state, RecurrentState) or not isinstance(self.prediction, PredictionBatch):
            raise TypeError("frozen cache requires typed observed input, model state and model prediction")
        if not isinstance(self.checkpoint_sha256, str) or not self.checkpoint_sha256 or type(self.budget) is not int or self.budget < 1:
            raise ValueError("frozen cache requires checkpoint identity and positive integer budget")

    def check(self, old: ObservationBatch, checkpoint_sha256: str | None, k: int) -> None:
        if not checkpoint_sha256 or self.checkpoint_sha256 != checkpoint_sha256:
            raise ValueError("frozen cache checkpoint provenance mismatch")
        if self.source_policy != "restart" or self.budget != k or self.state.budget != k:
            raise ValueError("frozen cache policy/budget provenance mismatch")
        self.state.check_observation(old)
        for name in ("node_features", "edge_types", "valid_nodes", "node_ids", "starts", "goals"):
            cached, current = getattr(self.observation, name), getattr(old, name)
            if (cached is None) != (current is None) or (cached is not None and not torch.equal(cached, current)):
                raise ValueError("frozen cache observed input mismatch")
        if (self.observation.domain != old.domain or self.observation.grid_shapes != old.grid_shapes
                or self.observation.episode_ids != old.episode_ids or self.observation.frame_indices != old.frame_indices):
            raise ValueError("frozen cache domain/geometry mismatch")
        if self.state.a.requires_grad or self.state.z.requires_grad or self.prediction.logits.requires_grad:
            raise ValueError("frozen cache must not retain autograd graphs")


@dataclass(frozen=True)
class OneEditLoss:
    loss: Tensor
    initial_loss: Tensor
    post_edit_loss: Tensor
    initial_prediction: PredictionBatch
    prediction: PredictionBatch
    prior_state: RecurrentState
    state: RecurrentState
    adapter: AdapterResult
    initial_block_calls_executed: int
    post_edit_block_calls: int


def supervised_loss(obs: ObservationBatch, prediction: PredictionBatch, target: TargetBatch) -> Tensor:
    target.check_observation(obs)
    if obs.domain == Domain.MAZE:
        return maze_valid_set_loss(prediction.logits, target, obs.valid_nodes)
    return circuit_value_loss(prediction.logits, target)


def one_edit_loss(solver: RecursiveSolver, adapter: StateAdapter,
                  old: ObservationBatch, new: ObservationBatch,
                  old_target: TargetBatch, new_target: TargetBatch,
                  k: int, *, track: str, cache: FrozenPrior | None = None,
                  checkpoint_sha256: str | None = None) -> OneEditLoss:
    """No optimizer action here; callers log, backpropagate, clip and update.

    Frozen caches are shared initial solves, created separately and charged once
    in cache-construction records. For a joint step the prior is recomputed from
    current weights, and the loss also supervises that initial prediction.
    """
    if not isinstance(old, ObservationBatch) or not isinstance(new, ObservationBatch):
        raise TypeError("training helper requires separate ObservationBatch objects")
    if track not in ("frozen", "joint") or type(k) is not int or k < 1:
        raise ValueError("require frozen/joint track and positive integer K")
    if not torch.is_grad_enabled() and (track == "joint" or any(p.requires_grad for p in adapter.parameters())):
        raise ValueError("one-edit training requires autograd for the current rollout")
    if any(t != 0 for t in old.frame_indices) or any(t != 1 for t in new.frame_indices):
        raise ValueError("one-edit training requires frame zero followed by frame one")
    if track == "frozen" and any(p.requires_grad for p in solver.parameters()):
        raise ValueError("frozen track requires frozen backbone parameters")
    if track == "joint" and (cache is not None or not all(p.requires_grad for p in solver.parameters())):
        raise ValueError("joint track requires trainable backbone and freshly recomputed prior")
    if cache is not None:
        cache.check(old, checkpoint_sha256, k)
        solver._check_state(old, cache.state)
        prior = cache.state.detach()
        initial_prediction = PredictionBatch(cache.prediction.logits.detach().clone())
        initial_calls = 0
    else:
        with torch.set_grad_enabled(track == "joint"):
            initial = solver(old, k)
        prior = initial.state.detach()
        initial_prediction = initial.prediction
        initial_calls = initial.block_calls
    initial_loss = supervised_loss(old, initial_prediction, old_target)
    transition = TransitionInput(new, old, prior, ObservedEdit.between(old, new))
    fresh = solver.fresh_state(new)
    if adapter.requires_previous_prediction:
        adapted = adapter(transition, fresh, previous_prediction=PredictionBatch(initial_prediction.logits.detach().clone()))
    else:
        adapted = adapter(transition, fresh)
    current = solver(new, k, adapted.state)
    post_loss = supervised_loss(new, current.prediction, new_target)
    return OneEditLoss((initial_loss+post_loss)/2, initial_loss, post_loss, initial_prediction,
                       current.prediction, prior, current.state, adapted, initial_calls, current.block_calls)

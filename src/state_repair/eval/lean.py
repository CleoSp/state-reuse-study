"""Exploratory minimal-history inference; never replaces frozen evaluation."""
from __future__ import annotations

from state_repair.models.adapters import StateAdapter, current_state
from state_repair.models.recursive import RecursiveSolver
from state_repair.types import ObservationBatch, ObservedEdit, PredictionBatch, TransitionInput


class LeanPolicy:
    """Retain only arm-required tensors; no clocks, audit copies or output history.

    Inference only. Caller owns no_grad and whole-stream timing. Existing core
    validation remains active and may itself synchronize tensor checks.
    """

    def __init__(self, solver: RecursiveSolver, adapter: StateAdapter, k: int):
        if type(k) is not int or k < 1:
            raise ValueError("positive fixed budget required")
        self.solver, self.adapter, self.k = solver, adapter, k
        self.reset()

    def reset(self) -> None:
        self.old = self.state = self.prediction = None
        self.identities = self.frame_indices = None

    def __call__(self, obs: ObservationBatch) -> PredictionBatch:
        import torch
        if torch.is_grad_enabled():
            raise ValueError("LeanPolicy is inference-only; use no_grad")
        if not isinstance(obs, ObservationBatch):
            raise TypeError("only observations may enter inference")
        initial = all(f == 0 for f in obs.frame_indices)
        if initial:
            self.reset()
        elif (self.identities != obs.episode_ids or self.frame_indices is None
              or any(f != p + 1 for f, p in zip(obs.frame_indices, self.frame_indices))):
            raise ValueError("stream must start at zero and keep consecutive root identities")
        encoded = self.solver.encode(obs)
        fresh = self.solver.fresh_state(obs)
        if initial or self.adapter.key == "restart":
            state = fresh
        elif self.adapter.key == "answer_only":
            probabilities = self.prediction.logits.softmax(-1).masked_fill(~fresh.valid_nodes[..., None], 0)
            projected = self.adapter.project(probabilities)
            state = current_state(fresh, *projected.split(self.solver.width, -1))
        else:
            transition = TransitionInput(obs, self.old, self.state, ObservedEdit.between(self.old, obs))
            state = self.adapter(transition, fresh).state
        for _ in range(self.k):
            state = self.solver.step(encoded, state)
        prediction = self.solver.decode(obs, state)
        self.identities, self.frame_indices = obs.episode_ids, obs.frame_indices
        if self.adapter.key == "answer_only":
            self.prediction = prediction
        elif self.adapter.key != "restart":
            self.old, self.state = obs, state
        return prediction

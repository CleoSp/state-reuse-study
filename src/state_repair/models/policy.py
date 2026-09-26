"""Fixed-budget observation-only streams with independently owned history."""
from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from uuid import uuid4

import torch
from torch import nn

from state_repair.models.adapters import AdapterResult, StateAdapter
from state_repair.models.recursive import RecursiveSolver
from state_repair.types import ObservationBatch, ObservedEdit, PredictionBatch, RecurrentState, TransitionInput


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def clone_observation(obs: ObservationBatch) -> ObservationBatch:
    return replace(obs, **{name: getattr(obs, name).detach().clone() if getattr(obs, name) is not None else None
                          for name in ("node_features", "edge_types", "valid_nodes", "node_ids", "starts", "goals")})


@dataclass(frozen=True)
class PolicyResult:
    prediction: PredictionBatch
    state: RecurrentState
    observation: ObservationBatch
    stream_token: str
    source_policy: str
    outer_cycles: int
    block_calls: int
    transformer_layer_executions: int
    adapter: AdapterResult
    operations: dict[str, int]
    milliseconds: dict[str, float]


class FixedBudgetPolicy(nn.Module):
    """One K and one adapter per stream; first solve is charged at the same K.

    `previous` is optional explicit continuation of THIS policy's result. It is
    checked even for restart. A new frame-zero observation clears internal state,
    while explicitly supplying history at frame zero is an error. Use a separate
    instance for another K, checkpoint or independently evaluated policy. Reset
    this wrapper after loading/updating model weights; it is not a disk cache.
    """

    def __init__(self, solver: RecursiveSolver, adapter: StateAdapter, outer_cycles: int) -> None:
        super().__init__()
        if not isinstance(solver, RecursiveSolver) or not isinstance(adapter, StateAdapter):
            raise TypeError("policy requires RecursiveSolver and StateAdapter")
        if type(outer_cycles) is not int or outer_cycles < 0:
            raise ValueError("outer_cycles must be a nonnegative integer")
        self.solver, self.adapter = solver, adapter
        self.outer_cycles = outer_cycles
        self._token = uuid4().hex
        self._previous: PolicyResult | None = None

    def reset(self) -> None:
        self._previous = None
        self._token = uuid4().hex

    def forward(self, obs: ObservationBatch, previous: PolicyResult | None = None) -> PolicyResult:
        if not isinstance(obs, ObservationBatch):
            raise TypeError("policy accepts only ObservationBatch")
        device = obs.node_features.device
        synchronize(device)
        started = last = perf_counter()
        times: dict[str, float] = {}

        def mark(stage: str) -> None:
            nonlocal last
            synchronize(device)
            now = perf_counter()
            times[stage] = (now - last) * 1000
            last = now

        if all(frame == 0 for frame in obs.frame_indices):
            if previous is not None:
                raise ValueError("frame zero rejects supplied old state; fresh initialization required")
            self.reset()
            transition = TransitionInput(obs)
        else:
            previous = self._previous if previous is None else previous
            if not isinstance(previous, PolicyResult):
                raise ValueError("edited frame requires previous policy result")
            if previous.outer_cycles != self.outer_cycles or previous.state.budget != self.outer_cycles:
                raise ValueError("budget provenance mismatch: stream must carry its own fixed-K state")
            if previous.stream_token != self._token or previous.source_policy != self.adapter.key:
                raise ValueError("policy/stream provenance mismatch")
            transition = TransitionInput(obs, previous.observation, previous.state,
                                         ObservedEdit.between(previous.observation, obs))
            self.solver._check_state(previous.observation, previous.state)
        mark("edit_and_validation")
        encoded = self.solver.encode(obs)
        mark("encoder")
        fresh = self.solver.fresh_state(obs)
        mark("fresh")
        if self.adapter.requires_previous_prediction:
            adapted = self.adapter(transition, fresh,
                                   previous_prediction=None if previous is None else previous.prediction)
        else:
            adapted = self.adapter(transition, fresh)
        self.solver._check_state(obs, adapted.state)
        if adapted.state.budget != 0:
            raise ValueError("adapter must initialize current-frame budget zero")
        mark("adapter")
        state = adapted.state
        for _ in range(self.outer_cycles):
            state = self.solver.step(encoded, state)
        mark("core")
        prediction = self.solver.decode(obs, state)
        mark("decoder")
        calls = (self.solver.inner_cycles + 1) * self.outer_cycles
        b, n, features = obs.node_features.shape
        d = self.solver.width
        operations = {"encoder_calls": 1, "fresh_calls": 1, "adapter_calls": 1,
                      "decoder_calls": 1, "block_calls": calls,
                      "transformer_layer_executions": 2 * calls,
                      "encoder_macs": b*n*features*d,
                      "core_macs": calls*(24*b*n*d*d + 4*b*n*n*d),
                      "decoder_macs": b*n*d*prediction.logits.shape[-1],
                      "retained_latent_bytes": (state.a.numel()+state.z.numel())*state.a.element_size(),
                      **adapted.operations}
        operations["total_macs"] = sum(operations.get(name, 0) for name in
                                       ("encoder_macs", "core_macs", "decoder_macs", "adapter_macs"))
        result = PolicyResult(prediction, state, clone_observation(obs), self._token,
                              self.adapter.key, self.outer_cycles, calls, 2 * calls,
                              adapted, operations, times)
        saved = state.detach()
        self._previous = replace(result, state=saved,
                                 prediction=PredictionBatch(prediction.logits.detach().clone()),
                                 observation=clone_observation(obs),
                                 adapter=AdapterResult(saved))
        mark("copy")
        times["total"] = (perf_counter() - started) * 1000
        return result

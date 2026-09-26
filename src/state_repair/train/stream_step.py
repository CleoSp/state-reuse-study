"""Memory-bounded gradient accumulation over a policy's own fixed-budget episode."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from state_repair.models.adapters import StateAdapter
from state_repair.models.policy import FixedBudgetPolicy
from state_repair.models.recursive import RecursiveSolver
from state_repair.train.adapter_step import supervised_loss
from state_repair.types import ObservationBatch, TargetBatch


@dataclass(frozen=True)
class FrameTrainingRecord:
    frame: int
    loss: float
    block_calls: int
    backward_executed: bool


def stream_backward(solver: RecursiveSolver, adapter: StateAdapter,
                    observations: Sequence[ObservationBatch], targets: Sequence[TargetBatch],
                    k: int, *, track: str) -> tuple[FrameTrainingRecord, ...]:
    """Accumulate gradients; the caller owns zero_grad, clipping and optimizer.step.

    Weights stay unchanged throughout the episode. Every transition uses this
    policy's detached previous K-step state, including the initial K-step solve.
    No graph-bearing outputs escape this function. Frozen initial-frame loss is
    constant but remains in the episode mean and the returned numerical records.
    """
    if track not in ("frozen", "joint") or type(k) is not int or k < 1:
        raise ValueError("require frozen/joint track and positive integer K")
    if not observations or len(observations) != len(targets):
        raise ValueError("require matching nonempty observations and targets")
    if not torch.is_grad_enabled():
        raise ValueError("stream training requires autograd")
    if track == "frozen" and any(p.requires_grad for p in solver.parameters()):
        raise ValueError("frozen track requires frozen backbone parameters")
    if track == "joint" and not all(p.requires_grad for p in solver.parameters()):
        raise ValueError("joint track requires trainable backbone parameters")
    for frame, (obs, target) in enumerate(zip(observations, targets)):
        if not isinstance(obs, ObservationBatch) or not isinstance(target, TargetBatch):
            raise TypeError("stream training requires separately typed observations and targets")
        if any(t != frame for t in obs.frame_indices):
            raise ValueError("training episode must start at zero and contain consecutive frames")
        target.check_observation(obs)
    policy = FixedBudgetPolicy(solver, adapter, k)
    records = []
    for frame, (obs, target) in enumerate(zip(observations, targets)):
        result = policy(obs)
        loss = supervised_loss(obs, result.prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite stream loss at frame {frame}")
        expected_gradient = track == "joint" or (frame > 0 and any(p.requires_grad for p in adapter.parameters()))
        if expected_gradient and not loss.requires_grad:
            raise RuntimeError("current stream loss lost its gradient path")
        if loss.requires_grad:
            (loss / len(observations)).backward()
        records.append(FrameTrainingRecord(frame, float(loss.detach()), result.block_calls, loss.requires_grad))
    policy.reset()
    return tuple(records)

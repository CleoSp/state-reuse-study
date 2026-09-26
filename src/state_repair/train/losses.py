"""Tie-aware supervised losses; labels stay outside the learned solver."""
from __future__ import annotations

import torch
from torch import Tensor

from state_repair.types import TargetBatch


def circuit_value_loss(logits: Tensor, target: TargetBatch) -> Tensor:
    """Cross entropy only on the explicitly scored non-input gate mask.

    Trainers must first call target.check_observation(obs) at their data boundary.
    Inputs and padding (target value -1) contribute neither loss nor gradients.
    """
    if not isinstance(target, TargetBatch) or target.values is None:
        raise ValueError("circuit loss requires values and scored_mask targets")
    if logits.shape != (*target.values.shape, 2) or logits.device != target.values.device:
        raise ValueError("circuit loss expects matching [B,N,2] logits and target device")
    scores = logits[target.scored_mask]
    if not torch.isfinite(scores).all():
        raise FloatingPointError("nonfinite scored logits")
    return torch.nn.functional.cross_entropy(scores, target.values[target.scored_mask])


def maze_valid_set_loss(logits: Tensor, target: TargetBatch, valid_nodes: Tensor) -> Tensor:
    """Mean negative log mass on all optimal actions, excluding padding."""
    if not isinstance(target, TargetBatch) or target.valid_actions is None:
        raise ValueError("maze loss requires maze valid-action targets, not circuit targets")
    if logits.ndim != 3 or logits.shape[-1] != 6 or logits.shape != target.valid_actions.shape:
        raise ValueError("maze loss expects matching [B,N,6] logits and valid actions")
    if valid_nodes.dtype != torch.bool or valid_nodes.shape != logits.shape[:2]:
        raise ValueError("valid_nodes must be Bool[B,N]")
    if not valid_nodes.any():
        raise ValueError("loss requires at least one scored node")
    labels = target.valid_actions[valid_nodes]
    scores = logits[valid_nodes]
    if not labels.any(dim=-1).all():
        raise ValueError("every scored node must have a valid action")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("nonfinite scored logits")
    return -torch.logsumexp(scores.log_softmax(-1).masked_fill(~labels, -torch.inf), dim=-1).mean()

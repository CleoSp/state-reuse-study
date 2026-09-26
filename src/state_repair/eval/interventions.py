"""Privileged evaluator diagnostics. Never imported by deployable adapters."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from state_repair.models.adapters import current_state
from state_repair.types import RecurrentState


@dataclass(frozen=True)
class InterventionResult:
    state: RecurrentState
    privileged: bool = True
    record_kind: str = "privileged_diagnostic"

    def record(self) -> dict[str, bool | str]:
        return {"privileged": True, "record_kind": "privileged_diagnostic"}


def impact_mask_reset(old: RecurrentState, fresh: RecurrentState, impact_mask: Tensor) -> InterventionResult:
    """Reset at a supplied evaluator impact mask; not an optimal-repair bound."""
    if old.a.shape != fresh.a.shape or old.a.dtype != fresh.a.dtype or old.a.device != fresh.a.device:
        raise ValueError("intervention states must share shape/device/dtype")
    if old.episode_ids != fresh.episode_ids or not torch.equal(old.node_ids, fresh.node_ids) or not torch.equal(old.valid_nodes, fresh.valid_nodes):
        raise ValueError("intervention requires matching episode/node correspondence")
    if impact_mask.dtype != torch.bool or impact_mask.shape != fresh.valid_nodes.shape or impact_mask.device != fresh.a.device:
        raise ValueError("impact mask must be bool [B,N] on state device")
    mask = impact_mask[...,None]
    return InterventionResult(current_state(fresh, torch.where(mask, fresh.a, old.a.detach()),
                                           torch.where(mask, fresh.z, old.z.detach())))

"""Resumable optimizer steps using the existing static and joint stream losses.

Data preparation, backbone eligibility, validation selection and test scoring
are separate jobs. This module accepts typed training observations and labels;
it never changes the learned-policy input boundary.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import fields, replace
from typing import Any

import torch

from state_repair.models.adapters import StateAdapter
from state_repair.models.recursive import RecursiveSolver
from state_repair.train.adapter_step import one_edit_loss, supervised_loss
from state_repair.train.pilot import schedule
from state_repair.train.stream_step import stream_backward
from state_repair.types import ObservationBatch, TargetBatch

Frame = tuple[ObservationBatch, TargetBatch]


def slice_batch(value, start: int, end: int):
    """Slice typed batch fields without passing targets into model APIs."""
    updates = {}
    for field in fields(value):
        item = getattr(value, field.name)
        if isinstance(item, torch.Tensor):
            updates[field.name] = item[start:end]
        elif isinstance(item, tuple) and item:
            updates[field.name] = item[start:end]
    return replace(value, **updates)


class ResearchTrainer:
    """The checkpoint's model includes both backbone and adapter parameters.

    Each update owns a whole detached-version stream. No partially updated
    stream crosses an optimizer boundary, so checkpoints require no graph or
    cached oracle data. One-edit controls use the original two-frame objective.
    """

    def __init__(self, solver: RecursiveSolver, adapter: StateAdapter | None,
                 batches: list[list[Frame]], config: dict):
        self.config = config
        self.recipe = config["recipe"]
        if self.recipe not in ("static", "stream", "one_edit"):
            raise ValueError("unknown training recipe")
        frame_count = {"static": 1, "stream": 5, "one_edit": 2}[self.recipe]
        if not batches or any(len(batch) != frame_count for batch in batches):
            raise ValueError("recipe requires complete training frames")
        if any(not isinstance(o, ObservationBatch) or not isinstance(t, TargetBatch)
               for batch in batches for o, t in batch):
            raise TypeError("separately typed observations and targets required")
        if self.recipe != "static" and adapter is None:
            raise ValueError("adapter required for joint training")
        self.batches = batches
        self.solver, self.adapter = solver, adapter
        self.model = torch.nn.ModuleDict({"backbone": solver, **({"adapter": adapter} if adapter is not None else {})})
        self.model.train()
        self.model.requires_grad_(True)
        groups = [{"params": list(solver.parameters()), "lr": config["learning_rate"]}]
        if adapter is not None and list(adapter.parameters()):
            groups.append({"params": list(adapter.parameters()), "lr": config["adapter_learning_rate"]})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
        self.schedule = schedule(config["seed"], len(batches), config)
        self.counts: Counter[int] = Counter()
        self.examples_forward_calls = 0

    def step(self, index: int, batch: int, k: int) -> dict[str, Any]:
        device = next(self.solver.parameters()).device
        frames = [(o.to(device), t.to(device)) for o, t in self.batches[batch]]
        self.optimizer.zero_grad(set_to_none=True)
        if self.recipe == "static":
            lr = self.config["learning_rate"] if index + 1 <= .75*self.config["steps"] else self.config["final_learning_rate"]
            self.optimizer.param_groups[0]["lr"] = lr
            obs, target = frames[0]
            size = obs.valid_nodes.shape[0]
            micro = self.config.get("microbatch_size", size)
            losses, calls = [], []
            for offset in range(0, size, micro):
                o, t = slice_batch(obs, offset, offset+micro), slice_batch(target, offset, offset+micro)
                result = self.solver(o, k)
                loss = supervised_loss(o, result.prediction, t) * o.valid_nodes.shape[0]/size
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                losses.append(loss.item())
                calls.append(result.block_calls)
                del result, loss
            losses = [sum(losses)]
        elif self.recipe == "stream":
            records = stream_backward(self.solver, self.adapter, [o for o, _ in frames],
                                      [t for _, t in frames], k, track="joint")
            losses, calls = [r.loss for r in records], [r.block_calls for r in records]
        else:
            (old, old_target), (new, new_target) = frames
            size = old.valid_nodes.shape[0]
            micro = self.config.get("microbatch_size", size)
            losses, calls = [0., 0.], []
            for offset in range(0, size, micro):
                o, n = slice_batch(old, offset, offset+micro), slice_batch(new, offset, offset+micro)
                ot, nt = slice_batch(old_target, offset, offset+micro), slice_batch(new_target, offset, offset+micro)
                result = one_edit_loss(self.solver, self.adapter, o, n, ot, nt, k, track="joint")
                if not torch.isfinite(result.loss):
                    raise FloatingPointError("nonfinite training loss")
                weight = o.valid_nodes.shape[0]/size
                (result.loss*weight).backward()
                losses[0] += result.initial_loss.item()*weight
                losses[1] += result.post_edit_loss.item()*weight
                calls.extend([result.initial_block_calls_executed, result.post_edit_block_calls])
                del result
        def norm(module: torch.nn.Module | None) -> float:
            values = [] if module is None else [p.grad.square().sum() for p in module.parameters() if p.grad is not None]
            return float(torch.stack(values).sum().sqrt()) if values else 0.0
        adapter_norm = norm(self.adapter)
        if self.adapter is not None and list(self.adapter.parameters()) and adapter_norm <= 0:
            raise ValueError("missing adapter gradient")
        backbone_norm = norm(self.solver)
        total = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        self.counts[k] += 1
        batch_size = frames[0][0].valid_nodes.shape[0]
        example_calls = ((self.solver.inner_cycles+1)*k*batch_size*(1 if self.recipe == "static" else 2) if self.recipe != "stream"
                         else sum(calls)*batch_size)
        self.examples_forward_calls += example_calls
        return {"loss": sum(losses)/len(losses), "frame_losses": losses, "frame_block_calls": calls,
                "forward_calls": sum(calls), "adapter_gradient_norm": adapter_norm,
                "example_forward_calls": example_calls,
                "backbone_gradient_norm": backbone_norm, "gradient_norm": float(total),
                "learning_rates": [g["lr"] for g in self.optimizer.param_groups]}

    def state_dict(self) -> dict[str, Any]:
        return {"sampled_K_counts": dict(self.counts), "example_forward_calls": self.examples_forward_calls}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.counts = Counter(state["sampled_K_counts"])
        self.examples_forward_calls = state["example_forward_calls"]

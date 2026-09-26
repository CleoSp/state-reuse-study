"""Synchronized measured stages; initial solve is mandatory in amortization."""
from __future__ import annotations

import math
from time import perf_counter
import torch

from state_repair.models.policy import synchronize

STAGES = ("edit_and_validation", "encoder", "fresh", "adapter", "core", "decoder", "copy", "transfer_in", "transfer_out")


def validate_timing(value: dict) -> None:
    if not set(STAGES) <= value.keys() or "total" not in value:
        raise ValueError("timing must cover every stage")
    if any(not math.isfinite(value[k]) or value[k] < 0 for k in (*STAGES, "total")):
        raise ValueError("invalid stage duration")
    if sum(value[k] for k in STAGES) > value["total"] + .01:
        raise ValueError("stage timings exceed total")


def measured_frame(policy, cpu_obs, device: str):
    target = torch.device(device)
    synchronize(target)
    start = perf_counter()
    obs = cpu_obs.to(target)
    synchronize(target)
    moved = perf_counter()
    result = policy(obs)
    decoded = perf_counter()
    actions = result.prediction.logits.argmax(-1).cpu().tolist()
    synchronize(target)
    end = perf_counter()
    milliseconds = {**result.milliseconds, "transfer_in": (moved-start)*1000,
        "transfer_out": (end-decoded)*1000, "total": (end-start)*1000}
    validate_timing(milliseconds)
    return result, actions, milliseconds


def amortized_cost(records: list[dict], edits: int) -> float:
    if sorted(r["frame"] for r in records) != list(range(edits+1)):
        raise ValueError("initial solve and every edit must be included")
    for row in records:
        validate_timing(row["milliseconds"])
    return sum(r["milliseconds"]["total"] for r in records)/(edits+1)

"""Regenerate synthetic=true adapter parameter/work records; no training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from state_repair.accounting.resources import memory_snapshot
from state_repair.data.maze import Maze, observation as maze_observation
from state_repair.data.circuit import Circuit, Operator, observation as circuit_observation
from state_repair.models import RecursiveSolver, FixedBudgetPolicy, make_adapter, ADAPTER_REGISTRY
from state_repair.types import Domain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/prompt04-accounting.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(173)
    started = perf_counter()
    rows = []
    for domain in Domain:
        if domain == Domain.MAZE:
            graph = Maze(2,2,((0,1),(0,2),(2,3)),0,3)
            old = maze_observation(graph, "synthetic=true")
            new = maze_observation(graph.toggle(0,1), "synthetic=true", 1)
        else:
            graph = Circuit((Operator.INPUT,Operator.INPUT,Operator.AND,Operator.NOT),
                            ((),(),(0,1),(2,)), (0,1,0,0))
            old = circuit_observation(graph, "synthetic=true")
            new = circuit_observation(graph.flip_input(0), "synthetic=true", 1)
        model = RecursiveSolver(domain=domain)
        for key in ADAPTER_REGISTRY:
            settings = [{"radius": r} for r in (1,2,3)] if key == "local_reset" else [{}]
            for setting in settings:
                kwargs = dict(setting)
                if key in ("spatial_gate","global_gate","gru_adapter","residual_adapter","answer_only"):
                    kwargs["domain"] = domain
                elif key == "shuffled_gate":
                    kwargs["spatial_gate"] = make_adapter("spatial_gate", domain=domain)
                adapter = make_adapter(key, **kwargs)
                policy = FixedBudgetPolicy(model, adapter, 0)
                with torch.no_grad():
                    initial = policy(old)
                    edited = policy(new)
                rows.append({"synthetic": True, "domain": domain.value, "adapter": key,
                             "control": adapter.is_control, "settings": setting,
                             "parameters": sum(p.numel() for p in adapter.parameters()),
                             "solver_parameters": sum(p.numel() for p in model.parameters()),
                             "initial_operations": initial.operations,
                             "edit_operations": edited.operations,
                             "initial_milliseconds": initial.milliseconds,
                             "edit_milliseconds": edited.milliseconds})
    record = {"synthetic": True, "record_kind": "implementation_accounting", "batch": 1, "nodes": 4,
              "width": 64, "context_width": 16, "outer_cycles": 0,
              "mac_convention": "Dense linear and QK/AV multiply-accumulates; excludes bias, normalization, activation, masking, reductions, validation and copies",
              "timing_note": "One cold synthetic call; correctness accounting, not a latency benchmark",
              "rows": rows, "elapsed_seconds": perf_counter()-started, "memory": memory_snapshot(),
              "external_cost_usd": 0, "electricity_cost_usd": None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "memory": record["memory"],
                      "elapsed_seconds": record["elapsed_seconds"]}))
    for row in rows:
        ops = row["edit_operations"]
        print(row["domain"], row["adapter"], row["settings"], row["parameters"],
              ops.get("adapter_macs",0), ops.get("adapter_linear_calls",0),
              ops.get("adapter_gru_calls",0), ops.get("context_layer_executions",0))


if __name__ == "__main__":
    main()

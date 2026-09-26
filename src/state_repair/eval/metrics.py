"""Raw-action scoring and complete-episode denominators."""
from __future__ import annotations

from collections import defaultdict

from state_repair.data.maze import MazeExample
from state_repair.data.splits import canonical_hash
from state_repair.oracles.maze import score_policy, solve_maze
from state_repair.oracles.circuit import score_circuit
from state_repair.execution.durable import digest_json

MODES = {"fixed_budget_stream", "frozen_state_intervention", "reference_solver", "privileged_diagnostic"}


def input_hash(example) -> str:
    if isinstance(example, MazeExample):
        from dataclasses import asdict
        return digest_json(asdict(example.maze))
    from dataclasses import asdict
    return digest_json({"circuit": asdict(example.circuit), "node_order": example.node_order})


def score(example, actions: list[int]) -> dict:
    if isinstance(example, MazeExample):
        value = score_policy(example.maze, actions)
        unreachable = solve_maze(example.maze)[0][example.maze.start] < 0
        return {**value, "exact_correct": value["route_correct"], "node_accuracy": value["valid_action_accuracy"],
                "unreachable": unreachable, "unreachable_correct": unreachable and value["route_correct"]}
    stable = [0] * example.circuit.n
    if len(actions) != len(stable) or any(type(a) is not int or a not in (0, 1) for a in actions):
        raise ValueError("circuit raw actions must be N binary integers")
    for index, node in enumerate(example.node_order):
        stable[node] = actions[index]
    value = score_circuit(example.circuit, stable)
    return {**value, "exact_correct": value["circuit_correct"]}


def validate_record(row: dict, *, empirical: bool = True) -> None:
    if type(row.get("synthetic")) is not bool or (empirical and row["synthetic"]):
        raise ValueError("empirical records reject synthetic or missing flags")
    mode = row.get("record_kind")
    if mode not in MODES or row.get("privileged") is not (mode == "privileged_diagnostic"):
        raise ValueError("record mode/privilege mismatch")
    if mode == "fixed_budget_stream":
        if row["state_budget"] != row["K"] or row["source_budget"] != (None if row["frame"] == 0 else row["K"]):
            raise ValueError("fixed-budget provenance mismatch")
        if row["block_calls"] != row["K"] * (row["inner_cycles"] + 1):
            raise ValueError("unreported recursive work")
    if digest_json(row["actions"]) != row["prediction_sha256"]:
        raise ValueError("prediction hash mismatch")


def episodes(rows: list[dict], edits: int, *, empirical: bool = True) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        validate_record(row, empirical=empirical)
        if row["record_kind"] not in ("fixed_budget_stream", "reference_solver"):
            raise ValueError("interventions cannot enter deployed stream tables")
        key = tuple(row[k] for k in ("suite", "seed", "policy", "K", "root_id")) + (row.get("repetition", 0),)
        grouped[key].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda r: r["frame"])
        if [r["frame"] for r in values] != list(range(edits + 1)):
            raise ValueError("missing/duplicate initial solve or edit")
        post = values[1:]
        result.append({**dict(zip(("suite", "seed", "policy", "K", "root_id", "repetition"), key)),
            "post_accuracy": sum(r["exact_correct"] for r in post) / edits,
            "node_accuracy": sum(r["node_accuracy"] for r in post) / edits,
            "whole_stream_success": all(r["exact_correct"] for r in values),
            "errors_by_frame": [int(not r["exact_correct"]) for r in post],
            "cumulative_errors": [sum(not r["exact_correct"] for r in post[:i]) for i in range(1, edits+1)],
            "all_node_accuracy": sum(r.get("all_node_correct", r["exact_correct"]) for r in post) / edits,
            "unreachable_count": sum(r.get("unreachable", False) for r in post),
            "unreachable_correct": sum(r.get("unreachable_correct", False) for r in post),
            "initial_ms": values[0]["milliseconds"]["total"],
            "update_ms": sum(r["milliseconds"]["total"] for r in post)/edits,
            "stage_ms": {stage: sum(r["milliseconds"][stage] for r in values)/(edits+1) for stage in values[0]["milliseconds"]},
            "measurement": values[0].get("measurement", "unspecified"),
            "batch_size": values[0].get("batch_size"),
            "amortized_ms": sum(r["milliseconds"]["total"] for r in values) / (edits+1),
            "amortized_macs": sum(r["operations"].get("total_macs", 0) for r in values) / (edits+1),
            "synthetic": not empirical, "record_kind": values[0]["record_kind"]})
    return result

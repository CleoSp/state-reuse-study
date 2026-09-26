"""Bounded fixed-fixture diagnostics; never an empirical repair result.

Uses the production loss and route scorer. Test payloads are never opened.
One invocation is one short adaptive job; all completed jobs count toward 20 min.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import sys
import time

import torch

from state_repair.accounting.resources import memory_snapshot
from state_repair.data.maze import Maze, MazeExample, collate, observation
from state_repair.data.serialization import load_split
from state_repair.eval.smoke import evaluate_examples
from state_repair.models.recursive import RecursiveSolver
from state_repair.oracles.maze import independent_distances, score_policy, solve_maze
from state_repair.provenance import file_hash, source_provenance
from state_repair.train.losses import maze_valid_set_loss

ROOT = Path("runs/static_diagnosis")
DATA = Path("runs/smoke/dataset")


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def fixtures(count: int) -> list[MazeExample]:
    return [e for e in load_split(DATA, "train") if e.frame_index == 0][:count]


def metrics(examples: list[MazeExample], logits: torch.Tensor) -> dict:
    rows = [dict(root_id=e.root_id, **score_policy(e.maze, x.argmax(-1).tolist()))
            for e, x in zip(examples, logits)]
    return {"acceptable_action_accuracy": sum(r["valid_action_accuracy"] for r in rows) / len(rows),
            "complete_route_success": sum(r["route_correct"] for r in rows) / len(rows),
            "failure_reasons": dict(Counter(r["reason"] for r in rows)), "episodes": rows}


def trace(maze: Maze, actions: list[int]) -> list[dict]:
    """Explanatory trace only; score_policy remains the authoritative scorer."""
    distances, valid = solve_maze(maze)
    u, seen, rows = maze.start, set(), []
    for _ in range(maze.n + 1):
        rows.append({"node": u, "action": actions[u], "acceptable": valid[u],
                     "distance": distances[u], "repeated": u in seen})
        if u in seen or actions[u] >= 4:
            break
        seen.add(u)
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[actions[u]]
        row, col = u // maze.width + dr, u % maze.width + dc
        v = row * maze.width + col
        if not (0 <= row < maze.height and 0 <= col < maze.width) or tuple(sorted((u, v))) not in maze.edges:
            break
        u = v
    return rows


def checks(out: Path) -> dict:
    reports = {}
    for count in (1, 4, 32):
        examples = fixtures(count)
        obs, target, meta = collate(examples)
        labels = target.valid_actions
        uniform = torch.zeros_like(labels, dtype=torch.float)
        uniform_loss = maze_valid_set_loss(uniform, target, obs.valid_nodes).item()
        expected = (6 / labels.sum(-1).float()).log().mean().item()
        assert abs(uniform_loss - expected) < 1e-6
        constant = torch.nn.Parameter(torch.zeros(6))
        optimizer = torch.optim.LBFGS([constant], lr=1, max_iter=200, tolerance_grad=1e-9,
                                     tolerance_change=1e-12, line_search_fn="strong_wolfe")
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = maze_valid_set_loss(constant.expand_as(uniform), target, obs.valid_nodes)
            loss.backward()
            return loss
        optimizer.step(closure)
        fitted = maze_valid_set_loss(constant.expand_as(uniform), target, obs.valid_nodes).item()
        oracle_rows = []
        diameters = []
        for e in examples:
            dist, actions = solve_maze(e.maze)
            assert dist == independent_distances(e.maze)
            oracle = [a.index(True) for a in actions]
            for start in range(e.maze.n):
                row = score_policy(replace(e.maze, start=start), oracle)
                assert row["route_correct"] and row["all_node_correct"]
            oracle_rows.append({"root_id": e.root_id, **score_policy(e.maze, oracle)})
            diameters.append(max(max(solve_maze(replace(e.maze, goal=g))[0]) for g in range(e.maze.n)))
        reports[str(count)] = {"uniform_loss": uniform_loss, "constant_loss": fitted,
            "constant_probabilities": constant.detach().softmax(-1).tolist(),
            "constant_metrics": metrics(examples, constant.detach().expand_as(uniform)),
            "uniform_argmax_metrics": metrics(examples, uniform),
            "uniform_expected_action_accuracy": (labels.sum(-1).float() / 6).mean().item(),
            "target_cardinality_counts": dict(Counter(labels.sum(-1).flatten().tolist())),
            "goal_eccentricities": meta.distances.max(-1).values.tolist(),
            "graph_diameters": diameters, "oracle_rollouts": oracle_rows,
            "oracle_all_start_routes_checked": sum(e.maze.n for e in examples)}

    special = [Maze(2, 2, ((0, 1), (0, 2), (1, 3), (2, 3)), 0, 3),
               Maze(1, 2, (), 0, 1), Maze(1, 1, (), 0, 0)]
    reports["synthetic_semantics"] = []
    for maze in special:
        dist, valid = solve_maze(maze)
        row = score_policy(maze, [a.index(True) for a in valid])
        assert row["route_correct"] and row["all_node_correct"]
        reports["synthetic_semantics"].append({"synthetic": True, "maze": asdict(maze), **row})
    reports["objective_infimum"] = 0.0
    reports["objective_explanation"] = "-mean(log(sum of acceptable probabilities)); ties impose no entropy floor. Zero is a limit as invalid probability tends to zero, not attained at finite unconstrained logits."

    obs, target, _ = collate(fixtures(1))
    reports["near_optimal_logit_table_loss"] = maze_valid_set_loss(
        torch.where(target.valid_actions, 30., -30.), target, obs.valid_nodes).item()
    torch.manual_seed(17)
    model = RecursiveSolver(attention_mode="dense")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in model.parameters() if p.requires_grad}
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    result = model(obs, 2)
    maze_valid_set_loss(result.prediction.logits, target, obs.valid_nodes).backward()
    gradients = {n: p.grad.norm().item() if p.grad is not None else None for n, p in model.named_parameters()}
    optimizer.step()
    changes = {n: (p.detach() - before[n]).norm().item() for n, p in model.named_parameters()}
    assert all(v is not None and v > 0 for v in gradients.values())
    assert all(v > 0 for v in changes.values())
    model.eval()
    with torch.no_grad():
        original = model(obs, 2).prediction.logits
        goal_obs = observation(replace(fixtures(1)[0].maze, goal=(fixtures(1)[0].maze.goal + 1) % 64), obs.episode_ids[0])
        edge_obs = observation(fixtures(1)[0].maze.toggle(*fixtures(1)[0].maze.edges[0]), obs.episode_ids[0])
        reports["observation_sensitivity"] = {
            "goal_logit_max_delta": (model(goal_obs, 2).prediction.logits - original).abs().max().item(),
            "edge_logit_max_delta": (model(edge_obs, 2).prediction.logits - original).abs().max().item()}
    actual = evaluate_examples(model, fixtures(1), [2])
    assert actual[0]["valid_action_accuracy"] == metrics(fixtures(1), original)["acceptable_action_accuracy"]
    reports["parameter_check"] = {"optimizer_membership_exact": True, "gradient_norms": gradients, "parameter_delta_norms": changes}
    reports["eval_path_matches_forward"] = True
    reports["record_kind"] = "objective_and_oracle_diagnostic"
    write(out / "checks.json", reports)
    return {"checks_file": str(out / "checks.json")}


def train(args: argparse.Namespace, out: Path, deadline: float) -> dict:
    examples = fixtures(args.count)
    obs, target, _ = collate(examples)
    model = RecursiveSolver(width=args.width, heads=2, inner_cycles=1, attention_mode="dense")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    if args.resume:
        prior = torch.load(ROOT / args.resume / "checkpoint.pt", weights_only=True)
        model.load_state_dict(prior["model"])
        if args.resume_optimizer:
            optimizer.load_state_dict(prior["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        torch.set_rng_state(prior["rng"])
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    steps, training_s = 0, 0.0
    curve = (out / "curve.jsonl").open("x", encoding="utf-8")
    def measure(step: int) -> dict:
        model.eval()
        with torch.no_grad():
            logits = model(obs, args.depth).prediction.logits
            row = {"step": step, "loss": maze_valid_set_loss(logits, target, obs.valid_nodes).item(),
                   **metrics(examples, logits), "training_s": training_s}
        model.train()
        curve.write(json.dumps(row) + "\n")
        curve.flush()
        print(json.dumps({k: v for k, v in row.items() if k != "episodes"}), flush=True)
        return row
    row = measure(0)
    for step in range(1, args.steps + 1):
        if time.perf_counter() >= deadline:
            break
        tick = time.perf_counter()
        if args.count <= 4:
            batch_obs, batch_target = obs, target
        else:
            chosen = [examples[i] for i in torch.randperm(len(examples))[:8].tolist()]
            batch_obs, batch_target, _ = collate(chosen)
        optimizer.zero_grad(set_to_none=True)
        loss = maze_valid_set_loss(model(batch_obs, args.depth).prediction.logits, batch_target, batch_obs.valid_nodes)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        training_s += time.perf_counter() - tick
        steps = step
        if step % 50 == 0:
            row = measure(step)
            if not args.fixed_steps and row["acceptable_action_accuracy"] == 1 and row["complete_route_success"] == 1 and row["loss"] < .02:
                break
    if row["step"] != steps:
        row = measure(steps)
    curve.close()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(),
                "config": vars(args), "steps": steps}, out / "checkpoint.pt")

    restored = RecursiveSolver(width=args.width, heads=2, inner_cycles=1, attention_mode="dense")
    restored.load_state_dict(torch.load(out / "checkpoint.pt", weights_only=True)["model"])
    records = evaluate_examples(restored, examples, sorted(set([1, args.depth, 4, 8])))
    write(out / "evaluation.json", records)
    model.eval()
    with torch.no_grad():
        logits = model(obs, args.depth).prediction.logits
    failures = []
    for e, x in zip(examples, logits):
        actions = x.argmax(-1).tolist()
        failures.append({"root_id": e.root_id, "actions": actions, "trace": trace(e.maze, actions),
            "goal_action": actions[e.maze.goal], **score_policy(e.maze, actions)})
    write(out / "failures.json", failures)
    actual = [r for r in records if r["K"] == args.depth]
    assert sum(r["route_correct"] for r in actual) / len(actual) == row["complete_route_success"]
    assert sum(r["valid_action_accuracy"] for r in actual) / len(actual) == row["acceptable_action_accuracy"]
    return {"final": row, "optimizer_steps": steps, "training_s": training_s,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "training_forward_block_invocations": steps * 2 * args.depth,
            "training_forward_layer_invocations": steps * 4 * args.depth,
            "batch_size": min(args.count, 8), "fit": row["loss"] < .02 and row["acceptable_action_accuracy"] == 1 and row["complete_route_success"] == 1,
            "parameter_delta_norm": sum((p.detach() - initial[n]).square().sum().item() for n, p in model.named_parameters()) ** .5,
            "checkpoint_sha256": file_hash(out / "checkpoint.pt")}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("name")
    parser.add_argument("--checks", action="store_true")
    parser.add_argument("--count", type=int, choices=[1, 4, 32], default=1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seconds", type=float, default=150)
    parser.add_argument("--resume")
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--fixed-steps", action="store_true", help="Disable fit stopping for matched step comparisons")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use the existing virtual environment")
    start = time.perf_counter()
    ROOT.mkdir(exist_ok=True)
    used = sum(json.loads(p.read_text())["wall_s"] for pattern in ("*/summary.json", "*/inspection.json")
               for p in ROOT.glob(pattern))
    allowance = min(args.seconds, 1100 - used - 15)
    if allowance <= 0:
        raise RuntimeError("Cumulative diagnostic allowance exhausted")
    out = ROOT / args.name
    out.mkdir()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)
    config = {**vars(args), "command": sys.argv, "schema_version": 1,
              "seed": 17, "threads": 2, "device": "cpu", "dtype": "float32",
              "optimizer": "AdamW", "weight_decay": 0, "dropout": 0, "augmentation": False,
              "synthetic": False, "record_kind": "static_overfit_diagnostic", "previous_job_wall_s": used,
              "allowed_job_s": allowance, "python": sys.version, "torch": str(torch.__version__),
              "platform": platform.platform(), "source": source_provenance(),
              "script_sha256": file_hash(Path(__file__)), "train_file_sha256": file_hash(DATA / "train.jsonl")}
    write(out / "config.json", config)
    write(out / "fixtures.json", [asdict(e) for e in fixtures(args.count)])
    summary = checks(out) if args.checks else train(args, out, start + allowance)
    summary.update(wall_s=time.perf_counter() - start, memory=memory_snapshot(), config=config, status="completed")
    write(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}), flush=True)


if __name__ == "__main__":
    main()

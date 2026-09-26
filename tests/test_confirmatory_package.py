"""Synthetic CPU tests of production handlers and statistical failure paths."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from state_repair.execution.datasets import identities, prepare_development
from state_repair.execution.durable import atomic_json, atomic_write, digest_json
from state_repair.execution.jobs import build
from state_repair.execution.training import Shutdown, run_training
from state_repair.execution.driver import seal, verify
from state_repair.eval.jobs import saved_records
from state_repair.eval.metrics import episodes, validate_record
from state_repair.eval.statistics import paired_contrast, sample_size
from state_repair.eval.timing import STAGES, amortized_cost, validate_timing


def execute(job, root):
    out = root / job["id"]
    out.mkdir()
    atomic_json(out / "job.json", job)
    with Shutdown() as shutdown:
        summary = run_training(build(job, root), job, out, 60, shutdown, checkpoint_steps=1)
    atomic_write(out / "checkpoint.pt", lambda h: h.write((out / "resume.pt").read_bytes()))
    atomic_json(out / "summary.json", summary)
    if job.get("compress_records"):
        from state_repair.execution.records import compress_steps
        compress_steps(out)
    seal(out)
    assert verify(out, job)["verified"]
    return out


@pytest.mark.parametrize("family,size", [("maze", 3), ("circuit", 8)])
def test_production_training_evaluation_reference(tmp_path, family, size):
    suite = family + str(size)
    spec = {"family": family, "size": size, "inputs": 2, "data_seed": 62017,
            "identities": {s: identities(suite, s, 2) for s in ("train", "val")}}
    data = tmp_path / "data.json"
    prepare_development(spec, data, synthetic=True)
    base = {"family": family, "size": size, "width": 8, "heads": 2, "inner_cycles": 1,
        "context_width": 4, "device": "cpu", "synthetic": True, "dataset": str(data), "seed": 19,
        "batch_size": 2, "steps": 2, "edit_seed": 17, "budgets": [1, 2],
        "learning_rate": .001, "adapter_learning_rate": .001, "final_learning_rate": .0003,
        "gradient_clip": 1., "weight_decay": 0., "suite": suite}
    execute({**base, "id": "static", "kind": "research_training", "recipe": "static"}, tmp_path)
    execute({**base, "id": "stream", "kind": "research_training", "recipe": "stream",
             "source": "static", "arm": "spatial_gate"}, tmp_path)
    eval_job = {**base, "id": "eval", "kind": "evaluation", "source": "stream", "arm": "spatial_gate",
                "split": "val", "edits": 2, "stream_seed": 71, "compress_records": True}
    out = execute(eval_job, tmp_path)
    records = list(saved_records(out))
    assert len(records) == 12
    result = episodes(records, 2, empirical=False)
    assert len(result) == 4
    from state_repair.eval.checker import check_job
    assert check_job(out, eval_job)["raw_actions_rescored"] == 12
    from state_repair.eval.report import summarize_curves
    assert len(summarize_curves(result, empirical=False)) == 2
    altered = {**records[1], "source_budget": 32}
    with pytest.raises(ValueError, match="budget provenance"):
        validate_record(altered, empirical=False)
    with pytest.raises(ValueError, match="synthetic"):
        episodes(records, 2)
    with pytest.raises(ValueError, match="missing/duplicate"):
        episodes(records[1:], 2, empirical=False)
    ref = execute({**eval_job, "id": "reference", "kind": "reference"}, tmp_path)
    assert all(r["exact_correct"] for r in saved_records(ref))
    mechanism = {**eval_job, "id": "mechanism", "mode": "mechanism", "source_K": 2}
    diagnostic = execute(mechanism, tmp_path)
    assert check_job(diagnostic, mechanism)["raw_actions_rescored"] == 56
    dynamics = {**eval_job, "id": "dynamics", "mode": "dynamics", "arm": "carry", "source_K": 2}
    carried = execute(dynamics, tmp_path)
    assert check_job(carried, dynamics)["raw_actions_rescored"] == 4
    with pytest.raises(ValueError, match="interventions"):
        episodes(list(saved_records(diagnostic)), 2, empirical=False)


def test_microbatch_gradient_matches_full_batch(tmp_path):
    from state_repair.execution.research_training import ResearchTrainer
    from state_repair.execution.datasets import collate
    from state_repair.data.maze import MazeExample, generate_maze
    from state_repair.models.recursive import RecursiveSolver
    examples = [MazeExample(generate_maze(3, 3, i), f"synthetic-{i}", 0, "train", True) for i in range(4)]
    batches = [[collate(examples)[:2]]]
    config = {"recipe": "static", "seed": 19, "learning_rate": .001, "weight_decay": 0.,
              "steps": 1, "final_learning_rate": .001, "budgets": [2], "gradient_clip": 1.}
    def create(micro):
        torch.manual_seed(19)
        return ResearchTrainer(RecursiveSolver(width=8, heads=2, inner_cycles=1), None, batches,
                               {**config, "microbatch_size": micro})
    full, small = create(4), create(2)
    a, b = full.step(0, 0, 2), small.step(0, 0, 2)
    assert a["example_forward_calls"] == b["example_forward_calls"]
    assert b["forward_calls"] == 2*a["forward_calls"]
    assert a["loss"] == pytest.approx(b["loss"], abs=2e-6)
    for p, q in zip(full.model.parameters(), small.model.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(p, q, rtol=2e-4, atol=2e-5)


def test_dstar_disconnection_reconnection_undo():
    from state_repair.data.maze import generate_maze, possible_edges
    from state_repair.oracles.incremental_references import DStarLite
    from state_repair.oracles.maze import score_policy
    for seed in range(5):
        maze = generate_maze(4, 4, seed)
        solver = DStarLite(maze)
        assert score_policy(maze, solver.actions())["all_node_correct"]
        for edge in possible_edges(4, 4):
            maze = maze.toggle(*edge)
            assert score_policy(maze, solver.update(maze))["all_node_correct"]
            maze = maze.toggle(*edge)
            assert score_policy(maze, solver.update(maze))["all_node_correct"]


def test_timing_stages_and_original_solve():
    times = {name: 1. for name in STAGES}
    times["total"] = len(STAGES)
    validate_timing(times)
    with pytest.raises(ValueError, match="every stage"):
        validate_timing({k: v for k, v in times.items() if k != "copy"})
    assert amortized_cost([{"frame": f, "milliseconds": times} for f in range(3)], 2) == len(STAGES)
    with pytest.raises(ValueError, match="initial solve"):
        amortized_cost([{"frame": f, "milliseconds": times} for f in (1, 2)], 2)


def test_paired_roots_and_crossed_seeds():
    a = [{"root_id": str(r), "seed": s, "post_accuracy": .2+.1*s, "synthetic": True}
         for s in range(3) for r in range(4)]
    b = [{**row, "post_accuracy": 0.} for row in a]
    result = paired_contrast(a, b, empirical=False, repetitions=1000)
    assert result["roots"] == 4 and result["delta"] == pytest.approx(.3)
    assert result["training_seed_sd"] == pytest.approx(.1)
    assert result["crossed_ci"][1]-result["crossed_ci"][0] > .01
    with pytest.raises(ValueError, match="same roots"):
        paired_contrast(a[1:], b, empirical=False)
    with pytest.raises(ValueError, match="synthetic"):
        paired_contrast(a, b)
    assert 0 < sample_size(.2, .02)["approximate_power"] < 1


def test_full_matrix_is_identity_only_and_dependencies_are_ordered(monkeypatch):
    from state_repair.execution import matrix
    monkeypatch.setattr(matrix, "read_json", lambda _: {"joint-"+a: {"grid": 0}
        for a in ("restart", "carry", "spatial_gate", "answer_only")})
    monkeypatch.setattr(matrix, "file_hash", lambda _: "0"*64)
    value = matrix.build_matrix()
    assert sum(j["kind"] == "research_training" for j in value["jobs"]) == 280
    timing = [j for j in value["jobs"] if j.get("mode") == "latency"]
    assert len(timing) == 37
    assert all(j["seed"] == 29 and j["root_limit"] == 8 and j["edits"] == 32 and j["repetitions"] == 3 for j in timing)
    assert all(j["seed"] in (29,43,71) for j in value["jobs"] if j.get("mode") == "mechanism")
    assert all(j["dataset"].split('/')[-1].removesuffix('-test.json') == j["suite"].split('-on-')[0]
               for j in value["jobs"] if j.get("edits") == 128)
    seen = set()
    for job in value["jobs"]:
        assert job["id"] not in seen
        assert set(job["dependencies"]) <= seen
        seen.add(job["id"])
        assert job["seconds"] <= 14400
    tests = [root for suite in value["suites"].values() for split, names in suite["identities"].items()
             if split in ("test", "challenge") for root in names]
    assert len(tests) == len(set(tests)) == 2176
    assert max(abs(v) for v in value["matched_control"]["relative_example_forward_call_errors"].values()) <= .05

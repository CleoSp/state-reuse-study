"""Synthetic checks of structural generation, budget guards and static gate."""
import importlib.util
import json
from pathlib import Path

import pytest

from state_repair.accounting.gpu_job import remaining_seconds
from state_repair.data.rooms import generate_rooms
from state_repair.oracles.maze import independent_distances, solve_maze


def module():
    spec = importlib.util.spec_from_file_location("scaled", Path("scripts/train_scaled_maze.py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_room_construction_is_connected_on_active_cells_and_has_isolated_walls():
    for seed in range(4):
        maze = generate_rooms(8, 8, seed)
        assert maze == generate_rooms(8, 8, seed)
        distances = solve_maze(maze)[0]
        assert distances == independent_distances(maze)
        active = {u for edge in maze.edges for u in edge}
        assert maze.start in active and maze.goal in active
        assert all((distances[u] >= 0) == (u in active) for u in range(maze.n))
        assert len(active) < maze.n
        assert len(active) == 4*9+3
    with pytest.raises(ValueError, match="room grids"):
        generate_rooms(2, 2, 1)


def test_static_gate_rejects_decline_incompetence_and_incomplete_training():
    m = module()
    config = {"steps": 6000, "minimum_accuracy": .85, "budgets": [1, 2, 4, 8, 16]}
    def rows(values):
        return [{"suite": "ordinary", "K": k, "route_accuracy": v} for k, v in zip(config["budgets"], values)]
    assert m.gate(rows([.2, .4, .8, .85, .9]), 6000, config)["backbone_eligible"]
    assert not m.gate(rows([.2, .4, .8, .9, .89]), 6000, config)["backbone_eligible"]
    assert not m.gate(rows([.2, .4, .6, .8, .84]), 6000, config)["backbone_eligible"]
    assert not m.gate(rows([.2, .4, .8, .85, .9]), 5999, config)["backbone_eligible"]


def test_scaled_split_and_structural_suites_are_reproducible():
    config = json.loads(Path("configs/scaled_maze_v1.json").read_text())
    config.update(train_roots=8, val_roots=2, height=8, width=8, size_shift=12, synthetic=True)
    data = module().make_data(config)
    assert data == module().make_data(config)
    assert [len(data[s]) for s in ("train", "ordinary", "rooms", "size16")] == [8, 2, 2, 2]
    assert all(e.split != "test" for examples in data.values() for e in examples)
    assert all(e.synthetic for examples in data.values() for e in examples)
    assert len({e.root_id for examples in data.values() for e in examples}) == 14
    assert all(e.maze.height == 12 for e in data["size16"])


def test_gpu_budget_reservations_failures_and_overruns_are_counted():
    events = [{"kind": "reserve", "job_id": "a", "seconds": 7200}]
    assert remaining_seconds(events) == 28800
    events.append({"kind": "actual", "job_id": "a", "seconds": 100})
    assert remaining_seconds(events) == 35900
    events.append({"kind": "reserve", "job_id": "b", "seconds": 7000})
    assert remaining_seconds(events) == 28900
    events.append({"kind": "actual", "job_id": "b", "seconds": 40000})
    assert remaining_seconds(events) == -4100
    with pytest.raises(ValueError, match="invalid"):
        remaining_seconds(events + [events[-1]])
    with pytest.raises(ValueError, match="unreserved"):
        remaining_seconds([{ "kind": "actual", "job_id": "x", "seconds": 1}])
    with pytest.raises(ValueError, match="invalid"):
        remaining_seconds([{ "kind": "reserve", "job_id": "x", "seconds": float("nan")}])


def test_artifact_checkers_reject_absent_and_incomplete_runs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from check_frozen_crossover import check_frozen
    from check_scaled_maze import check_scaled
    with pytest.raises(FileNotFoundError):
        check_frozen(tmp_path)
    with pytest.raises(ValueError, match="no completed"):
        check_scaled(tmp_path)
    (tmp_path / "summary.json").write_text(json.dumps({"artifact_hashes": {}}))
    with pytest.raises(ValueError, match="incomplete artifact manifest"):
        check_frozen(tmp_path)
    arm = tmp_path / "seed-29"
    arm.mkdir()
    (tmp_path / "dataset.json").write_text("{}")
    (arm / "summary.json").write_text(json.dumps({"synthetic": True}))
    with pytest.raises(ValueError, match="synthetic run"):
        check_scaled(tmp_path)


def test_frozen_checker_rejects_corrupted_artifact_hash(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from check_frozen_crossover import check_frozen
    names = ["config.json", "dataset.json", "checkpoint.pt", "provenance.json", "selection.json",
             "prior_states.pt", "initial.json", "predictions.jsonl", "preregistered-status.md"]
    (tmp_path / "summary.json").write_text(json.dumps({"artifact_hashes": {name: "bad" for name in names}}))
    (tmp_path / "config.json").write_text('{"synthetic": true}')
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        check_frozen(tmp_path)


def test_prompt05b_allowance_is_separate_and_explicit():
    events = [{"kind": "reserve", "job_id": "stream", "seconds": 7200}]
    assert remaining_seconds(events, 12*3600) == 10*3600
    assert remaining_seconds(events) == 8*3600

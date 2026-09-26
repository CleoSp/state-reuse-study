"""Synthetic inference/statistics controls; never empirical evidence."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from state_repair.data.maze import Maze, observation
from state_repair.eval.crossover import impact_stratum, summarize
from state_repair.eval.frozen import frozen_prediction
from state_repair.models.recursive import RecursiveSolver


def config():
    result = json.loads(Path("configs/frozen_crossover_v1.json").read_text())
    return {**result, "bootstrap_repetitions": 500, "budgets": [1, 2]}


def paired_rows():
    rows = []
    for root in range(8):
        for stratum in ("low", "high"):
            for k in (1, 2):
                for policy in ("restart", "carry"):
                    success = (policy == "carry") == (stratum == "low")
                    rows.append({"root_id": str(root), "branch": stratum, "suite": "challenge",
                        "K": k, "policy": policy, "split": "val", "synthetic": True,
                        "record_kind": "frozen_state_intervention", "stratum": stratum,
                        "input_sha256": f"{root}:{stratum}", "prior_state_sha256": str(root),
                        "checkpoint_sha256": "synthetic", "source_K": 8,
                        "route_correct": success, "all_node_correct": success,
                        "valid_action_accuracy": float(success)})
    return rows


def test_paired_root_bootstrap_crossover_and_branch_dependence():
    result = summarize(paired_rows(), config(), empirical=False)
    assert result["synthetic"] and result["gate"]["crossover"]
    assert len(result["gate"]["crossings"]) == 2
    all_row = next(r for r in result["statistics"] if r["stratum"] == "all")
    assert all_row["roots"] == 8 and all_row["branches"] == 16
    assert all_row["metrics"]["route_correct"]["ci"] == [0, 0]
    low = next(r for r in result["statistics"] if r["stratum"] == "low")
    assert low["metrics"]["route_correct"]["ci"] == [1, 1]
    assert summarize(paired_rows(), config(), empirical=False) == result


def test_root_weighting_with_unequal_branch_counts():
    rows = [r for r in paired_rows() if r["branch"] == "low" and r["root_id"] in ("0", "1")]
    for row in rows:
        if row["root_id"] == "1":
            for metric in ("route_correct", "all_node_correct", "valid_action_accuracy"):
                row[metric] = 1 - row[metric]
    for i in range(9):
        rows.extend([{**r, "branch": f"extra{i}"} for r in rows[:4]])
    result = summarize(rows, config(), empirical=False)
    assert next(r for r in result["statistics"] if r["stratum"] == "all")["metrics"]["route_correct"]["carry_minus_restart"] == 0


@pytest.mark.parametrize("field", ["prior_state_sha256", "input_sha256", "source_K", "checkpoint_sha256", "stratum"])
def test_pair_provenance_mismatch_rejected(field):
    rows = paired_rows()
    rows[0][field] = "bad"
    with pytest.raises(ValueError, match="mismatched paired"):
        summarize(rows, config(), empirical=False)


def test_synthetic_missing_duplicate_and_incomplete_records_rejected():
    rows = paired_rows()
    with pytest.raises(ValueError, match="synthetic"):
        summarize(rows, config())
    with pytest.raises(ValueError, match="missing paired"):
        summarize(rows[1:], config(), empirical=False)
    with pytest.raises(ValueError, match="duplicate"):
        summarize(rows + rows[:1], config(), empirical=False)
    with pytest.raises(ValueError, match="budget coverage"):
        summarize(rows[2:], config(), empirical=False)


def test_crossover_must_share_suite_budget_and_primary_metric():
    for mode in ("different_k", "different_suite", "secondary_only"):
        rows = paired_rows()
        for row in rows:
            if mode == "different_suite":
                row["suite"] = row["stratum"]
            elif mode == "secondary_only" or (mode == "different_k" and
                    ((row["K"] == 1 and row["stratum"] == "high") or
                     (row["K"] == 2 and row["stratum"] == "low"))):
                row["route_correct"] = False
        assert not summarize(rows, config(), empirical=False)["gate"]["crossover"]


def test_inconclusive_and_single_root_do_not_pass():
    rows = paired_rows()
    for row in rows:
        row["route_correct"] = row["policy"] == "carry"
    assert not summarize(rows, config(), empirical=False)["gate"]["crossover"]
    rows = [r for r in paired_rows() if r["root_id"] == "0"]
    assert not summarize(rows, config(), empirical=False)["gate"]["crossover"]
    assert impact_stratum(.1, config()) == "low"
    assert impact_stratum(.4, config()) == "high"
    assert impact_stratum(.2, config()) == "middle"


def test_frozen_fork_matches_manual_and_keeps_source_unchanged(monkeypatch):
    torch.manual_seed(61)
    model = RecursiveSolver(width=8, heads=2)
    maze = Maze(2, 2, ((0, 1), (0, 2), (1, 3)), 0, 3)
    old, new = observation(maze, "synthetic", 0), observation(maze.toggle(2, 3), "synthetic", 1)
    def forbidden(*args, **kwargs):
        raise AssertionError("oracle entered inference")
    monkeypatch.setattr("state_repair.oracles.maze.solve_maze", forbidden)
    with torch.no_grad():
        prior = model(old, 8).state
        saved = prior.clone()
        for policy in ("restart", "carry"):
            output = frozen_prediction(model, old, new, prior, policy, 2)
            initialized = model.fresh_state(new)
            if policy == "carry":
                initialized = replace(initialized, a=prior.a.clone(), z=prior.z.clone())
            expected = model(new, 2, initialized)
            torch.testing.assert_close(output.prediction.logits, expected.prediction.logits, rtol=0, atol=0)
            assert output.state.budget == 2 and output.block_calls == 4
            assert prior.budget == 8
            torch.testing.assert_close(prior.a, saved.a, rtol=0, atol=0)
            torch.testing.assert_close(prior.z, saved.z, rtol=0, atol=0)
        with pytest.raises(ValueError, match="restart/carry"):
            frozen_prediction(model, old, new, prior, "spatial_gate", 2)
        with pytest.raises(TypeError, match="ObservationBatch"):
            frozen_prediction(model, old, {"observation": new, "target": "forbidden"}, prior, "carry", 2)

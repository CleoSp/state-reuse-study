"""Synthetic correctness checks, not empirical learning results."""
import importlib.util
import json
from pathlib import Path

import pytest

from state_repair.data.benchmarks import BenchmarkConfig, generate_benchmarks, load_challenges, load_split
from state_repair.data.challenges import addition_challenge_root, sample_challenges
from state_repair.data.maze import possible_edges
from state_repair.oracles.maze import independent_distances, impact_metadata, solve_maze
from state_repair.types import Domain


@pytest.mark.parametrize("height,width", [(4, 2), (5, 3), (8, 8), (9, 6)])
def test_addition_strata_guaranteed_by_construction(height, width):
    for seed in range(10):
        root = addition_challenge_root(height, width, seed, "synthetic", "val", synthetic=True)
        maze = root.maze
        distances, _ = solve_maze(maze)
        assert distances == independent_distances(maze)
        low, high = 0, 0
        for edge in set(possible_edges(height, width)) - set(maze.edges):
            new = maze.toggle(*edge)
            changed = sum(impact_metadata(maze, new)["action_set_changed"]) / maze.n
            if distances[edge[0]] == distances[edge[1]] == -1:
                assert changed == 0
                low += 1
            if (distances[edge[0]] == -1) != (distances[edge[1]] == -1):
                assert changed >= .5
                high += 1
        assert low and high
        sample = sample_challenges([root], addition_only_roots=frozenset([root.root_id]))
        assert sample.audit["counts"]["addition"]["included_pairs"] == 1


def test_constructed_challenges_roundtrip_and_ordinary_distribution_unchanged(tmp_path):
    config = BenchmarkConfig(roots=16, edits=1, synthetic=True)
    path = generate_benchmarks(tmp_path / "constructed", config)
    legacy = generate_benchmarks(tmp_path / "legacy", BenchmarkConfig(
        roots=16, edits=1, synthetic=True, constructed_additions=False))
    for split in ("train", "val", "test"):
        assert load_split(path, Domain.MAZE, split) == load_split(legacy, Domain.MAZE, split)
        pairs = load_challenges(path, split)
        assert any(p["edit_type"] == "addition" for p in pairs)
        assert all(p["synthetic"] is True for p in pairs)
    assert generate_benchmarks(path, config) == path
    manifest = json.loads((path / "manifest.json").read_text())
    assert all(a["counts"]["addition"]["unmet_pair_quota"] == 0 for a in manifest["challenges"].values())


def test_depth_gate_rejects_decline_flat_and_incompetence():
    spec = importlib.util.spec_from_file_location("generalization", Path("scripts/train_generalization.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def decision(counts):
        return module.gate([{"routes": n, "route_accuracy": n / 128} for n in counts], .5)
    assert not decision([60, 70, 75, 90, 89])["nondecreasing"]
    assert not decision([90] * 5)["prompt04_allowed"]
    assert not decision([1, 2, 3, 4, 5])["prompt04_allowed"]
    assert decision([40, 50, 70, 80, 90])["prompt04_allowed"]


def test_report_creation_frozen(tmp_path):
    from state_repair.eval.smoke import report_run
    with pytest.raises(RuntimeError, match="frozen"):
        report_run(tmp_path)


def test_masked_layer_cannot_see_non_neighbors_and_padding_has_finite_gradients():
    import torch
    from state_repair.models.recursive import GraphTransformerLayer
    torch.manual_seed(19)
    layer = GraphTransformerLayer(16, 2, "masked_neighbor")
    x = torch.randn(1, 5, 16, requires_grad=True)
    valid = torch.tensor([[True, True, True, True, False]])
    edges = torch.zeros(1, 5, 5, dtype=torch.long)
    edges[0, 0, 1], edges[0, 1, 0] = 2, 4
    before = layer(x, edges, valid)
    changed = x.detach().clone()
    changed[0, 2] += torch.arange(16) * 100
    changed[0, 4] -= 1000
    after = layer(changed, edges, valid)
    torch.testing.assert_close(before[0, 0], after[0, 0], atol=0, rtol=0)
    assert torch.isfinite(before).all() and torch.count_nonzero(before[0, 4]) == 0
    before.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters())


@pytest.mark.parametrize("mode", ["dense", "masked_neighbor"])
def test_solver_modes_permutation_padding_and_block_counts(mode):
    import torch
    from dataclasses import replace
    from state_repair.data.maze import Maze, MazeExample, collate
    from state_repair.models.recursive import RecursiveSolver
    torch.manual_seed(19)
    model = RecursiveSolver(width=16, heads=2, attention_mode=mode)
    roots = [MazeExample(Maze(1, 2, (), 0, 1), "synthetic-a", 0, "val", True),
             MazeExample(Maze(2, 2, ((0, 1), (1, 3)), 0, 3), "synthetic-b", 0, "val", True)]
    obs, _, _ = collate(roots)
    calls = []
    handle = model.block.register_forward_hook(lambda *_: calls.append(1))
    result = model(obs, 2)
    handle.remove()
    assert len(calls) == result.block_calls == 4
    order = torch.tensor([3, 0, 2, 1])
    permuted = replace(obs, node_features=obs.node_features[:, order], node_ids=obs.node_ids[:, order],
        valid_nodes=obs.valid_nodes[:, order], edge_types=obs.edge_types[:, order][:, :, order])
    torch.testing.assert_close(model(permuted, 2).prediction.logits, result.prediction.logits[:, order], atol=2e-6, rtol=2e-5)
    plain, _, _ = collate(roots[:1])
    torch.testing.assert_close(model(plain, 2).prediction.logits, result.prediction.logits[:1, :2], atol=2e-6, rtol=2e-5)

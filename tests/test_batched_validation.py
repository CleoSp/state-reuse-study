"""Synthetic differential checks for batching existing validation predicates."""
from dataclasses import replace
import random

import pytest
import torch

from state_repair.data.maze import MazeExample, collate, generate_maze
from state_repair.types import _node_identity


def test_batched_identity_matches_per_row_definition_with_arbitrary_padding():
    rng = random.Random(317)
    for trial in range(100):
        n, b = 11, 7
        valid = torch.zeros((b, n), dtype=torch.bool)
        ids = torch.full((b, n), -1, dtype=torch.long)
        for i in range(b):
            positions = rng.sample(range(n), rng.randint(1, n))
            labels = list(range(len(positions)))
            rng.shuffle(labels)
            valid[i, positions] = True
            ids[i, positions] = torch.tensor(labels)
        if trial % 3:
            ids[rng.randrange(b), rng.randrange(n)] = rng.randrange(-2, n+2)
        expected = all(sorted(row[mask].tolist()) == list(range(int(mask.sum())))
                       and bool((row[~mask] == -1).all()) for row, mask in zip(ids, valid))
        if expected:
            _node_identity(ids, valid)
        else:
            with pytest.raises(ValueError):
                _node_identity(ids, valid)


def test_mixed_maze_geometries_preserve_checks_after_node_permutation():
    examples = [MazeExample(generate_maze(h, w, i), f"synthetic-{i}", 0, "val", True)
                for i, (h, w) in enumerate(((2, 7), (3, 4), (4, 2), (1, 1)))]
    obs, _, _ = collate(examples)
    order = torch.arange(obs.node_features.shape[1]-1, -1, -1)
    permuted = replace(obs, node_features=obs.node_features[:, order],
                       edge_types=obs.edge_types[:, order][:, :, order],
                       valid_nodes=obs.valid_nodes[:, order], node_ids=obs.node_ids[:, order])
    permuted.to("cpu")
    bad = permuted.edge_types.clone()
    graph, source, target = bad.nonzero(as_tuple=True)
    g, s, t = int(graph[0]), int(source[0]), int(target[0])
    bad[g, s, t] = int(bad[g, s, t]) % 4 + 1
    with pytest.raises(ValueError, match="geometry|reverse"):
        replace(permuted, edge_types=bad)
    bad_goal = permuted.goals.clone()
    bad_goal[-1] = 1
    with pytest.raises(ValueError, match="outside grid"):
        replace(permuted, goals=bad_goal)
    with pytest.raises(ValueError, match="grid shape"):
        replace(permuted, grid_shapes=((2**70, 1), *permuted.grid_shapes[1:]))

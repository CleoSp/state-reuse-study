from dataclasses import fields
import json
from pathlib import Path

import pytest
import torch

from state_repair.data.maze import Maze, MazeExample, collate, generate_episode, generate_maze, observation, possible_edges
from state_repair.oracles.maze import independent_distances, impact_metadata, score_policy, solve_maze
from state_repair.types import ObservationBatch


def test_ties_goal_and_start_goal():
    maze = Maze(2, 2, ((0, 1), (0, 2), (1, 3), (2, 3)), 0, 3)
    distances, actions = solve_maze(maze)
    assert distances == [2, 1, 1, 0]
    assert actions[0] == [False, True, True, False, False, False]
    assert score_policy(maze, [1, 2, 1, 4])["route_correct"]
    assert score_policy(maze, [2, 2, 1, 4])["route_correct"]
    assert score_policy(Maze(1, 1, (), 0, 0), [4])["route_correct"]
    assert not score_policy(Maze(1, 1, (), 0, 0), [5])["route_correct"]


def test_serialized_fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "maze_ties.json").read_text())
    raw = fixture["observation"]
    maze = Maze(raw["height"], raw["width"], tuple(tuple(e) for e in raw["edges"]), raw["start"], raw["goal"])
    distances, actions = solve_maze(maze)
    assert fixture["synthetic"] is True
    assert distances == fixture["oracle_metadata"]["distances"] == independent_distances(maze)
    assert actions == fixture["target"]["valid_actions"]


def test_bridge_disconnect_reconnect_undo():
    maze = Maze(1, 3, ((0, 1), (1, 2)), 0, 2)
    edited = maze.toggle(1, 2)
    assert solve_maze(edited)[0] == [-1, -1, 0]
    assert score_policy(edited, [5, 5, 4])["route_correct"]
    assert edited.toggle(2, 1) == maze
    assert impact_metadata(maze, edited)["distance_changed"] == [True, True, False]
    assert impact_metadata(maze, edited, [1, 1, 4])["old_prediction_now_invalid"] == [True, True, False]


def test_invalid_routes_and_borders():
    maze = Maze(2, 2, ((0, 1), (0, 2), (1, 3), (2, 3)), 0, 1)
    assert not score_policy(maze, [2, 4, 1, 0])["route_correct"]
    assert score_policy(maze, [2, 4, 0, 0])["reason"] == "cycle"
    assert score_policy(maze, [0, 4, 1, 0])["reason"] == "illegal_move"
    assert not score_policy(maze, [5, 4, 1, 0])["route_correct"]
    with pytest.raises(ValueError, match="border wrap"):
        Maze(2, 2, ((1, 2),), 0, 3)
    with pytest.raises(ValueError):
        maze.toggle(0, 3)
    assert possible_edges(3, 1) == [(0, 1), (1, 2)]


def test_independent_oracle_many_seeds():
    for seed in range(60):
        maze = generate_maze(2 + seed % 3, 2 + seed % 2, seed)
        assert all(d >= 0 for d in solve_maze(maze)[0])
        for example in generate_episode(maze, f"fixture-{seed}", "train", 4, seed + 100):
            distances, valid = solve_maze(example.maze)
            assert distances == independent_distances(example.maze)
            assert all(any(a) for a in valid)
            optimal = [a.index(True) for a in valid]
            assert score_policy(example.maze, optimal)["route_correct"]


def test_padding_observation_boundary_and_permutation():
    maze = generate_maze(2, 3, 10)
    tiny = Maze(1, 1, (), 0, 0)
    obs, target, metadata = collate([MazeExample(maze, "a", 0, "train", True), MazeExample(tiny, "b", 0, "train", True)])
    assert obs.node_features.shape == (2, 6, 4)
    assert not obs.valid_nodes[1, 1:].any()
    assert not target.valid_actions[1, 1:].any()
    assert (metadata.distances[1, 1:] == -2).all()
    assert not {"distances", "valid_actions", "split", "seed", "synthetic"} & {f.name for f in fields(ObservationBatch)}
    with pytest.raises(TypeError):
        observation(maze, distances=[0])
    order = [5, 0, 3, 1, 4, 2]
    perm = observation(maze, node_order=order)
    base = observation(maze)
    assert torch.equal(perm.node_features, base.node_features[:, order])
    assert torch.equal(perm.edge_types, base.edge_types[:, order][:, :, order])
    for u, v in maze.edges:
        assert int(base.edge_types[0, u, v]) in (2, 3)
        assert int(base.edge_types[0, v, u]) == {2: 4, 3: 1}[int(base.edge_types[0, u, v])]


def test_generator_determinism_and_empty_errors():
    assert generate_maze(4, 4, 12) == generate_maze(4, 4, 12)
    assert len(generate_maze(4, 4, 12, 0).edges) == 15
    assert len(generate_maze(4, 4, 12, 1).edges) == 24
    with pytest.raises(ValueError):
        collate([])
    with pytest.raises(ValueError):
        generate_episode(Maze(1, 1, (), 0, 0), "tiny", "train", 1, 0)

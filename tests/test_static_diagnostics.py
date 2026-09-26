"""Oracle-only path checks; these synthetic fixtures are not solver results."""
from dataclasses import replace

import torch

from state_repair.data.maze import Maze, MazeExample, collate
from state_repair.eval.smoke import evaluate_examples
from state_repair.models.recursive import RecursiveSolver
from state_repair.oracles.maze import score_policy, solve_maze
from state_repair.train.losses import maze_valid_set_loss
from state_repair.types import PredictionBatch


def test_oracle_logits_through_production_evaluator_and_mode_restore():
    mazes = [Maze(2, 2, ((0, 1), (0, 2), (1, 3), (2, 3)), 0, 3),
             Maze(1, 2, (), 0, 1), Maze(1, 1, (), 0, 0)]
    examples = [MazeExample(m, f"oracle-only-{i}", 0, "train", synthetic=True)
                for i, m in enumerate(mazes)]
    tables = {e.root_id: solve_maze(e.maze)[1] for e in examples}

    class OracleOnlyDecoder(RecursiveSolver):
        def decode(self, obs, state):
            assert not self.training
            return PredictionBatch(torch.tensor([tables[obs.episode_ids[0]]], dtype=torch.float))

    model = OracleOnlyDecoder(width=8, heads=2)
    records = evaluate_examples(model, examples, [0, 2])
    assert model.training
    assert len(records) == 6
    assert all(r["synthetic"] and r["route_correct"] and r["all_node_correct"] for r in records)


def test_rollout_rejects_cycles_and_invalid_terminal_actions():
    maze = Maze(1, 3, ((0, 1), (1, 2)), 0, 2)
    assert score_policy(maze, [1, 3, 4])["reason"] == "cycle"
    assert score_policy(maze, [4, 1, 4])["reason"] == "wrong_goal_or_nonoptimal"
    assert score_policy(maze, [5, 1, 4])["reason"] == "incorrect_unreachable"
    assert not score_policy(maze, [1, 1, 3])["route_correct"]
    assert score_policy(replace(maze, start=2), [1, 1, 4])["route_correct"]


def test_ties_have_zero_infimum_not_target_entropy_floor():
    maze = Maze(2, 2, ((0, 1), (0, 2), (1, 3), (2, 3)), 0, 3)
    obs, target, _ = collate([MazeExample(maze, "synthetic-ties", 0, "train", synthetic=True)])
    labels = target.valid_actions
    zeros = torch.zeros_like(labels, dtype=torch.float)
    expected = (6. / labels.sum(-1).float()).log().mean()
    torch.testing.assert_close(maze_valid_set_loss(zeros, target, obs.valid_nodes), expected)
    all_valid = torch.where(labels, 30., -30.)
    one_valid = torch.full_like(zeros, -30.)
    one_valid.scatter_(-1, labels.long().argmax(-1, keepdim=True), 30.)
    for logits in (all_valid, one_valid):
        assert maze_valid_set_loss(logits, target, obs.valid_nodes).abs() < 1e-6

"""Training data and schedules; metadata stays outside inference calls."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import random
from typing import Sequence

from state_repair.data.maze import Maze, MazeExample, generate_episode, possible_edges
from state_repair.data.splits import reject_cross_split_duplicates

LEARNED = ("spatial_gate", "global_gate", "gru_adapter", "residual_adapter", "answer_only")
PRINCIPAL = ("restart", "carry", *LEARNED)
AUXILIARY = ("local_reset_1", "local_reset_2", "local_reset_3", "random_reset", "noisy_carry", "shuffled_gate")


def decode_example(row: dict) -> MazeExample:
    raw = row["maze"]
    maze = Maze(raw["height"], raw["width"], tuple(tuple(e) for e in raw["edges"]), raw["start"], raw["goal"])
    return MazeExample(maze, row["root_id"], row["frame_index"], row["split"], row["synthetic"])


def child_seed(seed: int, root: str, view: int = 0) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{root}:{view}".encode()).digest()[:8], "big")


def training_views(roots: Sequence[MazeExample], config: dict) -> list[list[MazeExample]]:
    views = []
    for view in range(config["training_views"]):
        values = []
        for e in roots:
            rng = random.Random(child_seed(config["edit_seed"], e.root_id, view))
            edge = rng.choice(possible_edges(e.maze.height, e.maze.width))
            values.append(replace(e, maze=e.maze.toggle(*edge), frame_index=1))
        views.append(values)
    reject_cross_split_duplicates([*roots, *(e for v in views for e in v)])
    return views


def streams(roots: Sequence[MazeExample], edits: int, seed: int) -> list[list[MazeExample]]:
    episodes = [[replace(e, synthetic=root.synthetic) for e in generate_episode(
        root.maze, root.root_id, root.split, edits, child_seed(seed, root.root_id))] for root in roots]
    return [[episode[frame] for episode in episodes] for frame in range(edits + 1)]


def schedule(seed: int, batches: int, config: dict) -> list[tuple[int, int]]:
    if batches < 1:
        raise ValueError("training requires batches")
    rng = random.Random(seed)
    order, result = [], []
    for _ in range(config["steps"]):
        if not order:
            order = list(range(batches))
            rng.shuffle(order)
        result.append((order.pop(), rng.choice(config["budgets"])))
    return result


def make_payload(source: dict, config: dict) -> dict:
    train = [decode_example(e) for e in source["train"]]
    val = [decode_example(e) for e in source["ordinary"]]
    if any(e.frame_index or e.split != "train" for e in train) or any(e.frame_index or e.split != "val" for e in val):
        raise ValueError("pilot needs training/validation base roots only")
    if any(e.synthetic != config["synthetic"] for e in [*train, *val]):
        raise ValueError("synthetic flag mismatch")
    new = (streams(train, config["training_views"], config["edit_seed"])[1:]
           if config.get("training_protocol") == "stream" else training_views(train, config))
    episodes = streams(val, config["stream_edits"], config["stream_seed"])
    ordinary = streams(val, 1, config["intervention_seed"])[1]
    reject_cross_split_duplicates([*train, *(e for v in new for e in v), *(e for f in episodes for e in f), *ordinary])
    return {"train": [asdict(e) for e in train], "views": [[asdict(e) for e in v] for v in new],
            "streams": [[asdict(e) for e in f] for f in episodes], "ordinary": [asdict(e) for e in ordinary]}

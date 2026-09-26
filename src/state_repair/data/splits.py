"""Assign roots before generation; reject geometric duplicate problems."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Iterable

from state_repair.data.maze import Maze, MazeExample


def assign_splits(roots: int, seed: int, train_fraction: float, val_fraction: float) -> dict[str, str]:
    if roots < 1 or not 0 < train_fraction <= 1 or not 0 <= val_fraction <= 1 - train_fraction:
        raise ValueError("invalid split configuration")
    indices = list(range(roots))
    random.Random(seed).shuffle(indices)
    train_n, val_n = int(roots * train_fraction), int(roots * val_fraction)
    return {f"root-{i:06d}": "train" if rank < train_n else "val" if rank < train_n + val_n else "test"
            for rank, i in enumerate(indices)}


def canonical_hash(maze: Maze) -> str:
    """Canonical under rectangular rotations/reflections; ignores start.

    Goal is part of the task. Row-major identities are recovered from geometry,
    so tensor presentation permutations cannot create independent problems.
    """
    variants = []
    for transpose in (False, True):
        h, w = (maze.width, maze.height) if transpose else (maze.height, maze.width)
        for flip_row in (False, True):
            for flip_col in (False, True):
                mapping = []
                for u in range(maze.n):
                    r, c = divmod(u, maze.width)
                    if transpose:
                        r, c = c, r
                    if flip_row:
                        r = h - 1 - r
                    if flip_col:
                        c = w - 1 - c
                    mapping.append(r * w + c)
                edges = sorted(tuple(sorted((mapping[u], mapping[v]))) for u, v in maze.edges)
                variants.append(json.dumps([h, w, mapping[maze.goal], edges], separators=(",", ":")))
    return hashlib.sha256(min(variants).encode()).hexdigest()


def reject_cross_split_duplicates(examples: Iterable[MazeExample]) -> None:
    roots: dict[str, str] = {}
    hashes: dict[str, tuple[str, str]] = {}
    for example in examples:
        if example.split not in ("train", "val", "test"):
            raise ValueError("unknown split")
        if example.root_id in roots and roots[example.root_id] != example.split:
            raise ValueError(f"root crosses splits: {example.root_id}")
        roots[example.root_id] = example.split
        key = canonical_hash(example.maze)
        if key in hashes and hashes[key][0] != example.split:
            raise ValueError(f"canonical duplicate crosses splits: {hashes[key][1]} and {example.root_id}")
        if key in hashes and hashes[key][1] != example.root_id:
            raise ValueError("canonical duplicate cannot count as independent roots within one split")
        hashes[key] = (example.split, example.root_id)

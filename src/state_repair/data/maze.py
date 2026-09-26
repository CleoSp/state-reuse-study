"""Fixed-node editable grids and strictly observational tensor conversion."""
from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Sequence

import torch

from state_repair.types import Domain, ObservationBatch, OracleMetadata, TargetBatch


@dataclass(frozen=True)
class Maze:
    height: int
    width: int
    edges: tuple[tuple[int, int], ...]
    start: int
    goal: int

    def __post_init__(self) -> None:
        if any(type(v) is not int for v in (self.height, self.width, self.start, self.goal)):
            raise ValueError("maze dimensions/start/goal must be integers")
        if not isinstance(self.edges, tuple) or any(not isinstance(e, tuple) or len(e) != 2 or
                any(type(v) is not int for v in e) for e in self.edges):
            raise ValueError("passages must be immutable integer pairs")
        if self.height < 1 or self.width < 1:
            raise ValueError("maze dimensions must be positive")
        if not 0 <= self.start < self.n or not 0 <= self.goal < self.n:
            raise ValueError("start/goal outside maze")
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("duplicate passages")
        for u, v in self.edges:
            if not 0 <= u < v < self.n:
                raise ValueError("passages must be canonical pairs 0 <= u < v < N")
            if abs(u // self.width - v // self.width) + abs(u % self.width - v % self.width) != 1:
                raise ValueError("passage must connect four-neighbor cells (no border wrap)")

    @property
    def n(self) -> int:
        return self.height * self.width

    def toggle(self, u: int, v: int) -> Maze:
        """One immutable undirected edit; invalid passages are rejected."""
        edge = (min(u, v), max(u, v))
        edges = set(self.edges)
        if edge in edges:
            edges.remove(edge)
        else:
            edges.add(edge)
        return replace(self, edges=tuple(sorted(edges)))


@dataclass(frozen=True)
class MazeExample:
    maze: Maze
    root_id: str
    frame_index: int
    split: str
    synthetic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.maze, Maze) or not isinstance(self.root_id, str) or not self.root_id:
            raise ValueError("example needs Maze and nonempty root identity")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame index must be a nonnegative integer")
        if self.split not in ("train", "val", "test") or type(self.synthetic) is not bool:
            raise ValueError("example needs a valid split and explicit Boolean synthetic flag")


def possible_edges(height: int, width: int) -> list[tuple[int, int]]:
    edges = []
    for row in range(height):
        for col in range(width):
            u = row * width + col
            if col + 1 < width:
                edges.append((u, u + 1))
            if row + 1 < height:
                edges.append((u, u + width))
    return edges


def generate_maze(height: int, width: int, seed: int, extra_edge_probability: float = 0.15) -> Maze:
    """Randomized Kruskal spanning tree, then independent extra passages.

    This is NOT a uniform sampler over spanning trees.
    """
    if height < 1 or width < 1 or not 0 <= extra_edge_probability <= 1:
        raise ValueError("invalid maze generation parameters")
    rng = random.Random(seed)
    candidates = possible_edges(height, width)
    rng.shuffle(candidates)
    parent = list(range(height * width))

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    tree: set[tuple[int, int]] = set()
    for u, v in candidates:
        a, b = find(u), find(v)
        if a != b:
            tree.add((u, v))
            parent[a] = b
    for edge in sorted(set(candidates) - tree):
        if rng.random() < extra_edge_probability:
            tree.add(edge)
    return Maze(height, width, tuple(sorted(tree)), rng.randrange(height * width), rng.randrange(height * width))


def generate_episode(maze: Maze, root_id: str, split: str, edits: int, seed: int) -> list[MazeExample]:
    """Uniformly toggle a potential passage each frame; repeats can undo.

    Disconnections are retained. No oracle-conditioned rejection or balancing.
    """
    if edits < 0:
        raise ValueError("edit count cannot be negative")
    candidates = possible_edges(maze.height, maze.width)
    if edits and not candidates:
        raise ValueError("cannot edit passages in a one-node maze")
    rng = random.Random(seed)
    examples = [MazeExample(maze, root_id, 0, split)]
    for frame in range(1, edits + 1):
        maze = maze.toggle(*rng.choice(candidates))
        examples.append(MazeExample(maze, root_id, frame, split))
    return examples


def observation(maze: Maze, episode_id: str = "observed", frame_index: int = 0,
                node_order: Sequence[int] | None = None) -> ObservationBatch:
    """Only maze description and bookkeeping enter this conversion."""
    order = list(range(maze.n)) if node_order is None else list(node_order)
    if sorted(order) != list(range(maze.n)):
        raise ValueError("node_order must be a permutation")
    inverse = {u: i for i, u in enumerate(order)}
    features = torch.tensor([[(u // maze.width) / max(1, maze.height - 1),
                              (u % maze.width) / max(1, maze.width - 1),
                              float(u == maze.start), float(u == maze.goal)] for u in order])
    edges = torch.zeros((maze.n, maze.n), dtype=torch.long)
    for u, v in maze.edges:
        code = 2 if u // maze.width == v // maze.width else 3
        edges[inverse[u], inverse[v]] = code
        edges[inverse[v], inverse[u]] = 4 if code == 2 else 1
    return ObservationBatch(Domain.MAZE, features[None], edges[None],
        torch.ones((1, maze.n), dtype=torch.bool), (episode_id,), (frame_index,),
        torch.tensor([order]), ((maze.height, maze.width),),
        torch.tensor([maze.start]), torch.tensor([maze.goal]))


def collate(examples: Sequence[MazeExample], device: str | torch.device = "cpu") -> tuple[ObservationBatch, TargetBatch, OracleMetadata]:
    """Unpack this result before inference; target/metadata are not policy inputs."""
    from state_repair.oracles.maze import solve_maze

    if not examples:
        raise ValueError("cannot collate an empty batch")
    b, n = len(examples), max(e.maze.n for e in examples)
    x = torch.zeros((b, n, 4))
    relations = torch.zeros((b, n, n), dtype=torch.long)
    mask = torch.zeros((b, n), dtype=torch.bool)
    ids = torch.full((b, n), -1, dtype=torch.long)
    targets = torch.zeros((b, n, 6), dtype=torch.bool)
    distances = torch.full((b, n), -2, dtype=torch.long)
    for i, example in enumerate(examples):
        obs = observation(example.maze, example.root_id, example.frame_index)
        count = example.maze.n
        x[i, :count] = obs.node_features[0]
        relations[i, :count, :count] = obs.edge_types[0]
        mask[i, :count] = True
        ids[i, :count] = obs.node_ids[0]
        dist, valid = solve_maze(example.maze)
        targets[i, :count] = torch.tensor(valid)
        distances[i, :count] = torch.tensor(dist)
    obs = ObservationBatch(Domain.MAZE, x, relations, mask,
        tuple(e.root_id for e in examples), tuple(e.frame_index for e in examples), ids,
        tuple((e.maze.height, e.maze.width) for e in examples),
        torch.tensor([e.maze.start for e in examples]), torch.tensor([e.maze.goal for e in examples]))
    return obs.to(device), TargetBatch(targets).to(device), OracleMetadata(distances.to(device),
        audit={"synthetic": [e.synthetic for e in examples], "splits": [e.split for e in examples]})

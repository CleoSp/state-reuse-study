"""Observed room-and-corridor grids for a predeclared structural shift."""
from __future__ import annotations

import random

from state_repair.data.maze import Maze, generate_maze


def generate_rooms(height: int, width: int, seed: int, room_size: int = 3) -> Maze:
    """Open square rooms, one-cell separators, tree-connected random doors.

    Unused grid cells stay present but isolated. Start and goal are uniform over
    the constructed connected cells, without an oracle or solution rejection.
    """
    stride = room_size + 1
    if room_size < 2 or height < stride or width < stride:
        raise ValueError("room grids require room_size >=2 and at least one full room")
    nr, nc = height // stride, width // stride
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    active: set[int] = set()

    def connect(u: int, v: int) -> None:
        edges.add(tuple(sorted((u, v))))
        active.update((u, v))

    for rr in range(nr):
        for cc in range(nc):
            for dr in range(room_size):
                for dc in range(room_size):
                    u = (rr*stride+dr)*width+cc*stride+dc
                    active.add(u)
                    if dr+1 < room_size:
                        connect(u, u+width)
                    if dc+1 < room_size:
                        connect(u, u+1)
    room_graph = generate_maze(nr, nc, seed ^ 0xC011, extra_edge_probability=0)
    for u, v in room_graph.edges:
        rr, cc = divmod(u, nc)
        if v == u+1 and u//nc == v//nc:
            start = (rr*stride+rng.randrange(room_size))*width+cc*stride+room_size-1
            connect(start, start+1)
            connect(start+1, start+2)
        else:
            start = (rr*stride+room_size-1)*width+cc*stride+rng.randrange(room_size)
            connect(start, start+width)
            connect(start+width, start+2*width)
    nodes = sorted(active)
    return Maze(height, width, tuple(sorted(edges)), rng.choice(nodes), rng.choice(nodes))

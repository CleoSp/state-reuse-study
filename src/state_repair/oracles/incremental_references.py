"""Fixed-goal D* Lite reference, with zero heuristic and full policy repair.

The start does not move in these edit streams (km=0). We exhaust the inconsistent
queue to provide optimal actions at ALL nodes, beyond D* Lite's usual start-only
termination. No upstream implementation is copied. This deliberately conservative
reference counts full-policy work and never supplies features to a neural model.
"""
from __future__ import annotations

import heapq
import math

from state_repair.data.maze import Maze


class DStarLite:
    def __init__(self, maze: Maze):
        self.maze = maze
        self.g = [math.inf] * (maze.height*maze.width)
        self.rhs = self.g.copy()
        self.rhs[maze.goal] = 0
        self.queue = [(0., 0., maze.goal)]
        self.pending = {maze.goal: (0., 0.)}
        self._adjacency(maze)
        self._repair()

    def _adjacency(self, maze: Maze) -> None:
        self.neighbors = [set() for _ in self.g]
        for u, v in maze.edges:
            self.neighbors[u].add(v)
            self.neighbors[v].add(u)

    def _key(self, node: int) -> tuple[float, float]:
        value = min(self.g[node], self.rhs[node])
        return value, value

    def _update(self, node: int) -> None:
        if node != self.maze.goal:
            self.rhs[node] = min((self.g[v]+1 for v in self.neighbors[node]), default=math.inf)
        self.pending.pop(node, None)
        if self.g[node] != self.rhs[node]:
            key = self._key(node)
            self.pending[node] = key
            heapq.heappush(self.queue, (*key, node))

    def _repair(self) -> None:
        while self.queue:
            a, b, node = heapq.heappop(self.queue)
            if self.pending.get(node) != (a, b):
                continue
            del self.pending[node]
            if self.g[node] > self.rhs[node]:
                self.g[node] = self.rhs[node]
                for other in self.neighbors[node]:
                    self._update(other)
            else:
                self.g[node] = math.inf
                self._update(node)
                for other in self.neighbors[node]:
                    self._update(other)

    def update(self, maze: Maze) -> list[int]:
        if (maze.height, maze.width, maze.goal) != (self.maze.height, self.maze.width, self.maze.goal):
            raise ValueError("D* Lite stream needs stable nodes and goal")
        changed = set(self.maze.edges) ^ set(maze.edges)
        self.maze = maze
        self._adjacency(maze)
        for node in {u for edge in changed for u in edge}:
            self._update(node)
        self._repair()
        return self.actions()

    def actions(self) -> list[int]:
        actions = []
        for node, distance in enumerate(self.g):
            if node == self.maze.goal:
                actions.append(4)
            elif not math.isfinite(distance):
                actions.append(5)
            else:
                choices = [v for v in self.neighbors[node] if self.g[v] == distance-1]
                if not choices:
                    raise ValueError("inconsistent repaired policy")
                other = min(choices)
                dr, dc = other//self.maze.width-node//self.maze.width, other%self.maze.width-node%self.maze.width
                actions.append({(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}[dr, dc])
        return actions

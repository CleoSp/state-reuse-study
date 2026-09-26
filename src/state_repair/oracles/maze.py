"""Exact maze supervision and scoring, outside the learned policy."""
from __future__ import annotations

from collections import deque
from typing import Sequence

from state_repair.data.maze import Maze


def solve_maze(maze: Maze) -> tuple[list[int], list[list[bool]]]:
    neighbors: list[list[int]] = [[] for _ in range(maze.n)]
    for u, v in maze.edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    dist = [-1] * maze.n
    dist[maze.goal] = 0
    queue = deque([maze.goal])
    while queue:
        u = queue.popleft()
        for v in neighbors[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(v)
    valid = [[False] * 6 for _ in range(maze.n)]
    for u in range(maze.n):
        if u == maze.goal:
            valid[u][4] = True
        elif dist[u] < 0:
            valid[u][5] = True
        else:
            for v in neighbors[u]:
                if dist[v] == dist[u] - 1:
                    dr, dc = v // maze.width - u // maze.width, v % maze.width - u % maze.width
                    action = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}[(dr, dc)]
                    valid[u][action] = True
    return dist, valid


def independent_distances(maze: Maze) -> list[int]:
    """Floyd-Warshall all-pairs check: no BFS/neighbor helper shared."""
    inf = maze.n + 1
    paths = [[0 if i == j else inf for j in range(maze.n)] for i in range(maze.n)]
    for u, v in maze.edges:
        paths[u][v] = paths[v][u] = 1
    for k in range(maze.n):
        for i in range(maze.n):
            for j in range(maze.n):
                paths[i][j] = min(paths[i][j], paths[i][k] + paths[k][j])
    return [row[maze.goal] if row[maze.goal] < inf else -1 for row in paths]


def impact_metadata(old: Maze, new: Maze, old_actions: Sequence[int] | None = None) -> dict[str, list[bool]]:
    if (old.height, old.width) != (new.height, new.width):
        raise ValueError("impact comparison requires fixed grid nodes")
    od, oa = solve_maze(old)
    nd, na = solve_maze(new)
    result = {"action_set_changed": [a != b for a, b in zip(oa, na)],
              "distance_changed": [a != b for a, b in zip(od, nd)]}
    if old_actions is not None:
        if len(old_actions) != old.n or any(type(a) is not int or not 0 <= a < 6 for a in old_actions):
            raise ValueError("invalid old prediction")
        result["old_prediction_now_invalid"] = [not na[u][a] for u, a in enumerate(old_actions)]
    return result


def score_policy(maze: Maze, actions: Sequence[int]) -> dict[str, float | bool | int | str]:
    """Follow a fixed predicted policy without oracle repair or stopping choice."""
    if len(actions) != maze.n or any(type(a) is not int or not 0 <= a < 6 for a in actions):
        raise ValueError("actions must contain N labels in [0,5]")
    distances, valid = solve_maze(maze)
    correct = [valid[u][a] for u, a in enumerate(actions)]
    u, steps, visited = maze.start, 0, set()
    passages = set(maze.edges)
    success, reason = False, "cycle"
    while u not in visited and steps <= maze.n:
        visited.add(u)
        a = actions[u]
        if a == 5:
            success = u == maze.start and distances[u] == -1
            reason = "correct_unreachable" if success else "incorrect_unreachable"
            break
        if a == 4:
            success = u == maze.goal and steps == distances[maze.start]
            reason = "shortest_route" if success else "wrong_goal_or_nonoptimal"
            break
        dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][a]
        row, col = u // maze.width + dr, u % maze.width + dc
        v = row * maze.width + col
        if not (0 <= row < maze.height and 0 <= col < maze.width) or (min(u, v), max(u, v)) not in passages:
            reason = "illegal_move"
            break
        u, steps = v, steps + 1
    return {"route_correct": success, "all_node_correct": all(correct),
            "valid_action_accuracy": sum(correct) / maze.n, "route_steps": steps, "reason": reason}

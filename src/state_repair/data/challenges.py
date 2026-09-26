"""Evaluator-conditioned maze challenges; never an ordinary-prevalence estimate."""
from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Any, Sequence

from state_repair.data.maze import Maze, MazeExample, possible_edges, generate_maze
from state_repair.data.splits import reject_cross_split_duplicates
from state_repair.oracles.maze import impact_metadata


@dataclass(frozen=True)
class ChallengePair:
    root: MazeExample
    edit_type: str
    low: Maze
    high: Maze


@dataclass(frozen=True)
class ChallengeSample:
    pairs: tuple[ChallengePair, ...]
    audit: dict[str, Any]


def sample_challenges(roots: Sequence[MazeExample], *, pairs_per_type: int = 1,
                      seed: int = 17, low_max: float = .1, high_min: float = .4,
                      addition_only_roots: frozenset[str] = frozenset()) -> ChallengeSample:
    """Select paired one-edit branches from the identical base, matched by type.

    Enumerate every potential edge once per supplied root, bounded by input size.
    Shuffle roots; when both types qualify prefer the least-filled quota (random
    tie). Sample uniformly within eligible low/high candidates. At most one pair
    per base root. Return explicit incomplete status if quotas are impossible;
    never resample roots or alter thresholds. All selection data is evaluator-only.
    """
    if type(pairs_per_type) is not int or pairs_per_type < 1 or not 0 <= low_max < high_min <= 1:
        raise ValueError("require positive pair quota and 0 <= low_max < high_min <= 1")
    if any(e.frame_index != 0 for e in roots) or len({e.root_id for e in roots}) != len(roots):
        raise ValueError("challenge inputs must be unique base roots at frame zero")
    reject_cross_split_duplicates(roots)
    rng = random.Random(seed)
    ordered = list(roots)
    rng.shuffle(ordered)
    counts = {kind: {"candidates": 0, "low": 0, "middle": 0, "high": 0,
                     "eligible_roots": 0, "included_pairs": 0} for kind in ("addition", "removal")}
    pairs = []
    for root in ordered:
        pools: dict[str, dict[str, list[Maze]]] = {
            kind: {"low": [], "high": []} for kind in counts}
        for edge in possible_edges(root.maze.height, root.maze.width):
            kind = "removal" if edge in root.maze.edges else "addition"
            new = root.maze.toggle(*edge)
            fraction = sum(impact_metadata(root.maze, new)["action_set_changed"]) / root.maze.n
            stratum = "low" if fraction <= low_max else "high" if fraction >= high_min else "middle"
            counts[kind]["candidates"] += 1
            counts[kind][stratum] += 1
            if stratum != "middle":
                pools[kind][stratum].append(new)
        eligible = [kind for kind, pool in pools.items() if pool["low"] and pool["high"]]
        for kind in eligible:
            counts[kind]["eligible_roots"] += 1
        available = [kind for kind in eligible if counts[kind]["included_pairs"] < pairs_per_type
                     and (root.root_id not in addition_only_roots or kind == "addition")]
        if available:
            rng.shuffle(available)
            kind = min(available, key=lambda k: counts[k]["included_pairs"])
            pairs.append(ChallengePair(root, kind, rng.choice(pools[kind]["low"]), rng.choice(pools[kind]["high"])))
            counts[kind]["included_pairs"] += 1
    for count in counts.values():
        included = count["included_pairs"] * 2
        count["included_candidates"] = included
        count["rejected_candidates"] = count["candidates"] - included
        count["inclusion_rate"] = included / count["candidates"] if count["candidates"] else 0.0
        count["rejection_rate"] = 1 - count["inclusion_rate"] if count["candidates"] else 0.0
        count["pairing_failures"] = len(roots) - count["eligible_roots"]
        count["unmet_pair_quota"] = pairs_per_type - count["included_pairs"]
        for stratum in ("low", "high"):
            total = count[stratum]
            count[f"{stratum}_included"] = count["included_pairs"]
            count[f"{stratum}_rejected"] = total - count["included_pairs"]
            count[f"{stratum}_inclusion_rate"] = count["included_pairs"] / total if total else 0.0
    selected = list(roots)
    for pair in pairs:
        for branch in (pair.low, pair.high):
            selected.append(MazeExample(branch, pair.root.root_id, 1, pair.root.split, pair.root.synthetic))
    reject_cross_split_duplicates(selected)
    return ChallengeSample(tuple(pairs), {
        "suite": "semantic_impact_challenge", "metric": "action_set_changed_fraction",
        "low_max": low_max, "high_min": high_min, "edit_count": 1,
        "requested_pairs_per_type": pairs_per_type, "roots_proposed": len(roots),
        "roots_included": len(pairs), "roots_rejected": len(roots) - len(pairs),
        "complete": all(c["included_pairs"] == pairs_per_type for c in counts.values()),
        "counts": counts, "selection_seed": seed,
        "ordinary_prevalence_estimate": False,
    })


def addition_challenge_root(height: int, width: int, seed: int, root_id: str,
                            split: str, *, synthetic: bool = False) -> MazeExample:
    """Construct a matched-addition base; no oracle rejection or seed search.

    Two rectangular spanning trees are separated by a horizontal cut. The goal
    lies in the smaller/top tree; the bottom tree contains at least half the
    nodes and a closed internal grid edge. Every internal bottom addition has
    zero action-set impact (all nodes stay unreachable). Every cut-edge addition
    reconnects the entire bottom tree, changing at least half the target sets.
    This is a deliberately conditioned challenge, not an ordinary maze sample.
    """
    if height < 4 or width < 2:
        raise ValueError("constructed addition strata require height >= 4 and width >= 2")
    cut = height // 2
    top = generate_maze(cut, width, seed, 0)
    bottom = generate_maze(height - cut, width, seed ^ 0xADD, 0)
    offset = cut * width
    edges = top.edges + tuple((u + offset, v + offset) for u, v in bottom.edges)
    rng = random.Random(seed ^ 0xC07)
    maze = Maze(height, width, tuple(sorted(edges)), rng.randrange(height * width), top.goal)
    if rng.randrange(2):
        mapping = [(height - 1 - u // width) * width + u % width for u in range(maze.n)]
        maze = replace(maze, edges=tuple(sorted(tuple(sorted((mapping[u], mapping[v]))) for u, v in maze.edges)),
                       start=mapping[maze.start], goal=mapping[maze.goal])
    return MazeExample(maze, root_id, 0, split, synthetic)

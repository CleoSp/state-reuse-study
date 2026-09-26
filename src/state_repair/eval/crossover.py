"""Root-paired bootstrap statistics."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

METRICS = ("route_correct", "all_node_correct", "valid_action_accuracy")
STRATA = ("low", "middle", "high")


def impact_stratum(fraction: float, config: dict) -> str:
    return "low" if fraction <= config["low_max"] else "high" if fraction >= config["high_min"] else "middle"


def summarize(rows: list[dict], config: dict, *, empirical: bool = True) -> dict[str, Any]:
    """Resample root clusters jointly across branches, methods, K and metrics.

    Each resampled root contributes its mean over branches in the stratum.
    Empty strata in a bootstrap draw are omitted for that stratum only; fewer
    than two observed roots cannot supply gate evidence. Intervals are pointwise.
    """
    if not rows:
        raise ValueError("no paired records")
    grouped: dict[tuple, dict] = {}
    for row in rows:
        if type(row.get("synthetic")) is not bool or (empirical and row["synthetic"]):
            raise ValueError("empirical statistics reject synthetic/missing flags")
        if row.get("record_kind") != "frozen_state_intervention" or row.get("split") != "val":
            raise ValueError("requires validation frozen-state records")
        if row["policy"] not in config["policies"] or row["K"] not in config["budgets"]:
            raise ValueError("unexpected policy/budget")
        key = (row["suite"], row["root_id"], row["branch"], row["K"])
        pair = grouped.setdefault(key, {})
        if row["policy"] in pair:
            raise ValueError("duplicate paired record")
        pair[row["policy"]] = row
    for pair in grouped.values():
        if set(pair) != {"restart", "carry"}:
            raise ValueError("missing paired method")
        for field in ("stratum", "input_sha256", "prior_state_sha256", "source_K", "checkpoint_sha256"):
            if pair["restart"][field] != pair["carry"][field]:
                raise ValueError(f"mismatched paired {field}")
    branch_budgets: dict[tuple, set] = defaultdict(set)
    for suite, root, branch, k in grouped:
        branch_budgets[suite, root, branch].add(k)
    if any(ks != set(config["budgets"]) for ks in branch_budgets.values()):
        raise ValueError("incomplete budget coverage")
    output = []
    alpha = (1 - config["confidence"]) / 2
    rng = np.random.default_rng(config["bootstrap_seed"])
    for suite in sorted({key[0] for key in grouped}):
        roots = sorted({key[1] for key in grouped if key[0] == suite})
        draws = rng.integers(0, len(roots), size=(config["bootstrap_repetitions"], len(roots)))
        for k in config["budgets"]:
            for stratum in ("all", *STRATA):
                selected = {key: pair for key, pair in grouped.items()
                            if key[0] == suite and key[3] == k
                            and (stratum == "all" or pair["restart"]["stratum"] == stratum)}
                by_root: dict[str, list] = defaultdict(list)
                for key, pair in selected.items():
                    by_root[key[1]].append(pair)
                record = {"suite": suite, "K": k, "stratum": stratum,
                          "roots": len(by_root), "branches": len(selected), "metrics": {}}
                for metric in METRICS:
                    values = np.full((len(roots), 2), np.nan)
                    for i, root in enumerate(roots):
                        if root in by_root:
                            values[i] = [np.mean([pair[p][metric] for pair in by_root[root]])
                                         for p in ("restart", "carry")]
                    valid = np.isfinite(values[:, 0])
                    if not valid.any():
                        record["metrics"][metric] = None
                        continue
                    delta = values[:, 1] - values[:, 0]
                    count = valid[draws].sum(axis=1)
                    resampled = np.nansum(delta[draws], axis=1)[count > 0] / count[count > 0]
                    interval = np.quantile(resampled, [alpha, 1-alpha]).tolist() if valid.sum() >= 2 else None
                    record["metrics"][metric] = {
                        "restart": float(values[valid, 0].mean()), "carry": float(values[valid, 1].mean()),
                        "carry_minus_restart": float(delta[valid].mean()), "ci": interval,
                        "bootstrap_nonempty_draws": len(resampled)}
                output.append(record)
    crossings = []
    for suite in sorted({r["suite"] for r in output}):
        for k in config["budgets"]:
            eligible = [r for r in output if r["suite"] == suite and r["K"] == k
                        and r["stratum"] in STRATA and r["roots"] >= config["minimum_stratum_roots"]]
            positive, negative = [], []
            for record in eligible:
                metric = record["metrics"][config["primary_metric"]]
                if metric and metric["ci"]:
                    if metric["ci"][0] > 0:
                        positive.append(record["stratum"])
                    if metric["ci"][1] < 0:
                        negative.append(record["stratum"])
            if positive and negative:
                crossings.append({"suite": suite, "K": k, "carry_better": positive, "restart_better": negative})
    return {"statistics": output, "gate": {"crossover": bool(crossings), "crossings": crossings,
            "decision": "continue_step2" if crossings else "stop_for_replan"},
            "interval_scope": "pointwise exploratory validation; one training seed",
            "synthetic": not empirical}

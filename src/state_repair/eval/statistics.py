"""Paired roots, separate training-seed variation, and crossed uncertainty."""
from __future__ import annotations

from statistics import NormalDist
import numpy as np


def paired_contrast(left: list[dict], right: list[dict], *, metric: str = "post_accuracy",
                    repetitions: int = 10000, seed: int = 64037, empirical: bool = True) -> dict:
    def table(rows):
        result = {}
        for row in rows:
            if type(row.get("synthetic")) is not bool or (empirical and row["synthetic"]):
                raise ValueError("empirical statistics reject synthetic records")
            key = (row["seed"], row["root_id"])
            if key in result:
                raise ValueError("duplicate seed/root cell")
            result[key] = row[metric]
        return result
    a, b = table(left), table(right)
    if a.keys() != b.keys() or not a:
        raise ValueError("paired methods require the same roots and seeds")
    seeds = sorted({s for s, _ in a})
    roots = sorted({r for _, r in a})
    if len(a) != len(seeds)*len(roots) or len(roots) < 2:
        raise ValueError("incomplete crossed seed/root coverage")
    delta = np.array([[a[s, r]-b[s, r] for r in roots] for s in seeds])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(roots), (repetitions, len(roots)))
    root_mean = delta.mean(0)
    root_samples = root_mean[draws].mean(1)
    seed_draws = rng.integers(0, len(seeds), (repetitions, len(seeds)))
    crossed = np.array([delta[s][:, r].mean() for s, r in zip(seed_draws, draws)])
    return {"delta": float(delta.mean()), "root_ci": np.quantile(root_samples, [.025, .975]).tolist(),
        "crossed_ci": np.quantile(crossed, [.025, .975]).tolist(),
        "per_seed_delta": delta.mean(1).tolist(), "training_seed_sd": float(delta.mean(1).std(ddof=1)) if len(seeds)>1 else None,
        "root_sd": float(root_mean.std(ddof=1)), "roots": len(roots), "seeds": seeds,
        "repetitions": repetitions, "bootstrap_seed": seed, "synthetic": not empirical}


def sample_size(root_sd: float, seed_sd: float, *, roots: int = 256, seeds: int = 5,
                true_delta: float = 0, margin: float = .01, alpha: float = .025) -> dict:
    """Normal approximation, explicitly conditional on development variance."""
    if roots < 2 or seeds < 2 or root_sd < 0 or seed_sd < 0:
        raise ValueError("invalid sample-size inputs")
    se = (root_sd**2/roots + seed_sd**2/seeds)**.5
    power = float(true_delta+margin > 0) if se == 0 else NormalDist().cdf((true_delta+margin)/se-NormalDist().inv_cdf(1-alpha))
    return {"roots": roots, "training_seeds": seeds, "root_sd": root_sd, "seed_sd": seed_sd,
        "assumed_true_delta": true_delta, "margin": margin, "alpha": alpha,
        "approximate_power": power, "standard_error": se,
        "limitation": "normal development-variance approximation; not a guarantee of confirmatory power"}


def paired_cost_ratio(left: list[dict], right: list[dict], *, repetitions: int = 10000,
                      seed: int = 64037, empirical: bool = True) -> dict:
    paired_contrast(left, right, metric="amortized_ms", repetitions=2, empirical=empirical)
    keys = sorted((r["seed"], r["root_id"]) for r in left)
    seeds, roots = sorted({s for s, _ in keys}), sorted({r for _, r in keys})
    def array(rows):
        values = {(r["seed"], r["root_id"]): r["amortized_ms"] for r in rows}
        out = np.array([[values[s, r] for r in roots] for s in seeds])
        if not np.isfinite(out).all() or (out <= 0).any():
            raise ValueError("cost ratio requires positive finite measured costs")
        return out
    a, b = array(left), array(right)
    rng = np.random.default_rng(seed)
    root_draws = rng.integers(0, len(roots), (repetitions, len(roots)))
    seed_draws = rng.integers(0, len(seeds), (repetitions, len(seeds)))
    conditional = a.mean(0)[root_draws].mean(1)/b.mean(0)[root_draws].mean(1)
    crossed = [a[s][:, r].mean()/b[s][:, r].mean() for s, r in zip(seed_draws, root_draws)]
    return {"ratio": float(a.mean()/b.mean()), "root_ci": np.quantile(conditional, [.025, .975]).tolist(),
        "crossed_ci": np.quantile(crossed, [.025, .975]).tolist(), "roots": len(roots), "seeds": seeds,
        "per_seed_ratio": (a.mean(1)/b.mean(1)).tolist(), "bootstrap_seed": seed, "repetitions": repetitions,
        "synthetic": not empirical}


def select_operating_points(curves: list[dict], reuse: str = "answer_only") -> dict:
    if not curves or any(r.get("split") != "val" or r.get("synthetic") is not False for r in curves):
        raise ValueError("operating points require empirical validation curves")
    baseline = max((r for r in curves if r["policy"] == "restart"),
                   key=lambda r: (r["post_accuracy"], -r["amortized_ms"], -r["K"]))
    candidates = [r for r in curves if r["policy"] == reuse]
    eligible = [r for r in candidates if r["post_accuracy"] >= baseline["post_accuracy"]-.01]
    selected = min(eligible, key=lambda r: (r["amortized_ms"], r["K"])) if eligible else max(
        candidates, key=lambda r: (r["post_accuracy"], -r["amortized_ms"], -r["K"]))
    return {"reuse": selected, "comparator": baseline, "validation_noninferiority_feasible": bool(eligible),
        "validation_cost_ratio": selected["amortized_ms"]/baseline["amortized_ms"],
        "rule": "highest-accuracy restart (ties measured cost then K); least-cost reuse within 1 pp, else highest-accuracy reuse; no test selection"}

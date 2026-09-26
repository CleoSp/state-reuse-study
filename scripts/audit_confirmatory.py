"""Audit the confirmatory matrix from sealed raw records using independent scoring."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "confirmatory_v1"
sys.path.insert(0, str(ROOT / "src"))

from state_repair.execution.datasets import decode, frames  # noqa: E402
from state_repair.execution.durable import digest_json, read_json  # noqa: E402
from state_repair.data.circuit import Operator  # noqa: E402

T0 = time.perf_counter()
FINDINGS: list[str] = []


def section(title: str) -> None:
    print(f"\n=== {title}  (t={time.perf_counter()-T0:.1f}s)", flush=True)


def note(kind: str, text: str) -> None:
    FINDINGS.append(f"[{kind}] {text}")
    print(f"  {kind}: {text}", flush=True)


def records(job: str):
    directory = RUN / job
    path = directory / "steps.jsonl.gz"
    opener = gzip.open
    if not path.exists():
        path, opener = directory / "steps.jsonl", open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            unit = json.loads(line)
            for row in unit.get("records", []):
                yield row


def units(job: str):
    directory = RUN / job
    path = directory / "steps.jsonl.gz"
    opener = gzip.open
    if not path.exists():
        path, opener = directory / "steps.jsonl", open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def selection_map(size: str) -> dict:
    return json.loads((RUN / f"selection-{size}" / "steps.jsonl").read_text())["selection"]





def my_bfs(maze) -> list[int]:
    adjacency = [[] for _ in range(maze.n)]
    for u, v in maze.edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    dist = [-1] * maze.n
    dist[maze.goal] = 0
    frontier = [maze.goal]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adjacency[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def my_route(maze, actions: list[int]) -> tuple[bool, str]:
    """Follow argmax actions from start; no repair, no oracle stopping."""
    dist = my_bfs(maze)
    passages = {(min(u, v), max(u, v)) for u, v in maze.edges}
    u, steps, seen = maze.start, 0, set()
    while True:
        if u in seen or steps > maze.n:
            return False, "cycle"
        seen.add(u)
        a = actions[u]
        if a == 5:
            ok = (u == maze.start and dist[u] == -1)
            return ok, "correct_unreachable" if ok else "incorrect_unreachable"
        if a == 4:
            ok = (u == maze.goal and steps == dist[maze.start])
            return ok, "shortest_route" if ok else "wrong_goal_or_nonoptimal"
        dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][a]
        r, c = u // maze.width + dr, u % maze.width + dc
        if not (0 <= r < maze.height and 0 <= c < maze.width):
            return False, "illegal_move"
        v = r * maze.width + c
        if (min(u, v), max(u, v)) not in passages:
            return False, "illegal_move"
        u, steps = v, steps + 1


def my_circuit_eval(circuit) -> list[int]:
    values: list[int | None] = [None] * circuit.n
    remaining = set(range(circuit.n))
    while remaining:
        ready = [u for u in remaining if all(values[p] is not None for p in circuit.parents[u])]
        if not ready:
            raise ValueError("cycle")
        for u in ready:
            op = circuit.operators[u]
            p = [values[q] for q in circuit.parents[u]]
            if op == Operator.INPUT:
                values[u] = circuit.input_bits[u]
            elif op == Operator.AND:
                values[u] = p[0] & p[1]
            elif op == Operator.OR:
                values[u] = p[0] | p[1]
            elif op == Operator.XOR:
                values[u] = p[0] ^ p[1]
            elif op == Operator.NOT:
                values[u] = 1 - p[0]
            else:
                raise ValueError(op)
            remaining.discard(u)
    return values  # type: ignore[return-value]





PRINCIPAL = {"maze": ["restart", "carry", "spatial_gate", "global_gate", "gru_adapter", "residual_adapter", "answer_only"],
             "circuit": ["restart", "carry", "spatial_gate", "answer_only"]}
SEEDS = [29, 43, 71, 101, 137]
SUITES = {"maze12-on-maze12": ("maze", "maze12", 12), "circuit32-on-circuit32": ("circuit", "circuit32", 32),
          "circuit32-on-circuit48": ("circuit", "circuit32", 48)}
OPERATING = {"maze": (("answer_only", 8), ("restart", 8)), "circuit": (("answer_only", 1), ("restart", 2))}
EDITS = 32

section("1. independent regeneration of curves and frozen operating points")
per_cell: dict = {}
per_frame: dict = {}
violations = Counter()
reasons = Counter()
stage_violation = 0
stage_checked = 0
gap_sum = 0.0
gap_max = 0.0
checkpoint_seen = defaultdict(set)
for suite, (family, size, _) in SUITES.items():
    selected = selection_map(size)
    cells = per_cell.setdefault(suite, defaultdict(lambda: [0, 0.0, 0, True]))
    frame_acc = per_frame.setdefault(suite, defaultdict(lambda: [0, 0]))
    for policy in PRINCIPAL[family]:
        for seed in SEEDS:
            job = f"test-{suite}-{policy}-s{seed}-h32"
            for r in records(job):
                if r["record_kind"] != "fixed_budget_stream" or r["synthetic"] is not False or r["split"] != "test":
                    violations["mode/flags"] += 1
                k, f = r["K"], r["frame"]
                if r["state_budget"] != k or r["source_budget"] != (None if f == 0 else k):
                    violations["budget_provenance"] += 1
                if r["block_calls"] != k * (r["inner_cycles"] + 1):
                    violations["block_calls"] += 1
                if r["checkpoint_job"] != selected[policy][str(seed)]:
                    violations["checkpoint_job"] += 1
                if r["measurement"] != "batched_throughput" or r["batch_size"] != 64:
                    violations["measurement_label"] += 1
                checkpoint_seen[suite, policy].add(r["checkpoint_sha256"])
                ms = r["milliseconds"]
                stage_checked += 1
                gap = ms["total"] - sum(v for s, v in ms.items() if s != "total")
                gap_sum += gap
                gap_max = max(gap_max, gap)
                if gap < -1e-6:
                    stage_violation += 1
                cell = cells[policy, k, seed, r["root_id"]]
                cell[1] += ms["total"]
                cell[2] += 1
                if f > 0:
                    cell[0] += int(r["exact_correct"])
                    frame_acc[policy, k, seed, f][0] += int(r["exact_correct"])
                    frame_acc[policy, k, seed, f][1] += 1
                    if family == "maze":
                        reasons[suite, r["reason"]] += 1
                if not r["exact_correct"]:
                    cell[3] = False
    print(f"  {suite}: loaded {len(cells)} seed/root/policy/K cells", flush=True)
print("  record violations:", dict(violations) or "none")
print(f"  stage-sum > total violations: {stage_violation} of {stage_checked} records; mean unmeasured gap (total - stage sum) {gap_sum/stage_checked:.6f} ms, max {gap_max:.6f} ms")
for key, hashes in sorted(checkpoint_seen.items()):
    if len(hashes) != 5:
        note("MAJOR", f"{key}: {len(hashes)} distinct checkpoints across five seeds")
print("  distinct checkpoint hashes per suite/policy all == 5:", all(len(h) == 5 for h in checkpoint_seen.values()))
print("  maze12 post-edit reasons (frames 1..32, all principal policies/K):", dict(sorted(reasons.items())))

curve_mismatch = 0
op_results = {}
for suite, (family, size, _) in SUITES.items():
    report = read_json(RUN / f"report-{suite}-h32" / "results.json")
    cells = per_cell[suite]
    for (policy, k, seed, root), cell in cells.items():
        if cell[2] != EDITS + 1:
            note("BLOCKING", f"{suite} {policy} K{k} s{seed} {root}: {cell[2]} frames, expected {EDITS+1}")
    print(f"-- {suite}")
    for policy in PRINCIPAL[family]:
        for k in (1, 2, 4, 8, 16):
            vals = [cells[policy, k, s, r][0] / EDITS for (p, kk, s, r) in cells if p == policy and kk == k]
            costs = [cells[policy, k, s, r][1] / (EDITS + 1) for (p, kk, s, r) in cells if p == policy and kk == k]
            mine, mine_cost = sum(vals) / len(vals), sum(costs) / len(costs)
            theirs = next(c for c in report["curves"] if c["policy"] == policy and c["K"] == k and c["record_kind"] == "fixed_budget_stream")
            ok = abs(mine - theirs["post_accuracy"]) <= 1e-9 and abs(mine_cost - theirs["amortized_ms"]) <= 1e-6
            curve_mismatch += not ok
            print(f"   {policy:17s} K{k:<2d} acc {mine:.6f} (report {theirs['post_accuracy']:.6f}) amortized {mine_cost:.4f} (report {theirs['amortized_ms']:.4f}) n={len(vals)} {'OK' if ok else 'MISMATCH'}")
    (rp, rk), (cp, ck) = OPERATING[family]
    roots = sorted({r for (_, _, _, r) in cells})
    A = np.array([[cells[rp, rk, s, r][0] / EDITS for r in roots] for s in SEEDS])
    B = np.array([[cells[cp, ck, s, r][0] / EDITS for r in roots] for s in SEEDS])
    CA = np.array([[cells[rp, rk, s, r][1] / (EDITS + 1) for r in roots] for s in SEEDS])
    CB = np.array([[cells[cp, ck, s, r][1] / (EDITS + 1) for r in roots] for s in SEEDS])
    delta = A - B
    rng = np.random.default_rng(64037)
    draws = rng.integers(0, len(roots), (10000, len(roots)))
    root_mean = delta.mean(0)
    samples = root_mean[draws].mean(1)
    lo, hi = np.quantile(samples, [.025, .975])
    seed_draws = rng.integers(0, 5, (10000, 5))
    crossed = np.array([delta[s][:, r].mean() for s, r in zip(seed_draws, draws)])
    clo, chi = np.quantile(crossed, [.025, .975])
    rng2 = np.random.default_rng(64037)
    cdraws = rng2.integers(0, len(roots), (10000, len(roots)))
    ratio = CA.mean() / CB.mean()
    ratio_samples = CA.mean(0)[cdraws].mean(1) / CB.mean(0)[cdraws].mean(1)
    rlo, rhi = np.quantile(ratio_samples, [.025, .975])

    rng3 = np.random.default_rng(1)
    d3 = rng3.integers(0, len(roots), (10000, len(roots)))
    alo, ahi = np.quantile(root_mean[d3].mean(1), [.025, .975])
    op = report["operating_points"][0]
    ra, rc = op["accuracy"], op["cost"]
    match = (abs(ra["delta"] - delta.mean()) < 1e-12 and abs(ra["root_ci"][0] - lo) < 1e-12 and abs(ra["root_ci"][1] - hi) < 1e-12
             and abs(rc["ratio"] - ratio) < 1e-12 and abs(rc["root_ci"][0] - rlo) < 1e-12 and abs(rc["root_ci"][1] - rhi) < 1e-12
             and abs(ra["crossed_ci"][0] - clo) < 1e-12 and abs(ra["crossed_ci"][1] - chi) < 1e-12)
    noninferior = lo > -.01
    saving = rhi < 1
    target25 = rhi <= .75
    op_results[suite] = {"delta": delta.mean(), "root_ci": [lo, hi], "crossed_ci": [clo, chi], "alt_seed_ci": [alo, ahi],
                         "ratio": ratio, "ratio_ci": [rlo, rhi], "noninferior": noninferior, "saving": saving, "target25": target25,
                         "per_seed_delta": delta.mean(1).tolist(), "matches_report": match}
    print(f"   OPERATING {rp} K{rk} vs {cp} K{ck}: delta {delta.mean():+.6f} root CI [{lo:+.6f},{hi:+.6f}] crossed [{clo:+.6f},{chi:+.6f}] alt-seed CI [{alo:+.6f},{ahi:+.6f}]")
    print(f"     cost ratio {ratio:.6f} root CI [{rlo:.6f},{rhi:.6f}]  per-seed delta {[round(x,4) for x in delta.mean(1)]}")
    print(f"     matches results.json to 1e-12: {match}; noninferior(lo>-.01)={noninferior}; saving(hi<1)={saving}; 25% target(hi<=.75)={target25}")
    if not match:
        note("MAJOR", f"{suite} operating point does not reproduce from raw records")
print(f"  curve mismatches: {curve_mismatch}")
if curve_mismatch:
    note("BLOCKING", f"{curve_mismatch} curve cells do not reproduce from raw records")
headline = all(v["noninferior"] and v["target25"] for v in op_results.values())
print(f"  intersection-union headline (all three primary suites noninferior AND 25% saving): {headline}")
print("  batch-one latency operating points (8 roots, seed 29, dedicated latency jobs):")
for suite in ("maze12-on-maze12", "circuit32-on-circuit32", "circuit32-on-circuit48"):
    lat = read_json(RUN / f"report-latency-{suite}" / "results.json")["operating_points"][0]
    print(f"   {suite}: acc delta {lat['accuracy']['delta']:+.4f} [{lat['accuracy']['root_ci'][0]:+.4f},{lat['accuracy']['root_ci'][1]:+.4f}] cost ratio {lat['cost']['ratio']:.4f} [{lat['cost']['root_ci'][0]:.4f},{lat['cost']['root_ci'][1]:.4f}] roots {lat['cost']['roots']} seeds {lat['cost']['seeds']}")




section("2. independent rescoring of sampled raw actions")
rng = random.Random(7)
rescored = Counter()
mismatch = []
for suite, (family, size, eval_size) in SUITES.items():
    data = read_json(RUN / "data" / f"{family}{eval_size}-test.json")
    roots = {e["root_id"]: decode(e) for e in data["test"]}
    frame_cache = {}
    for policy in ("restart", "answer_only", "carry"):
        seed = rng.choice(SEEDS)
        job = f"test-{suite}-{policy}-s{seed}-h32"
        chosen = set(rng.sample(sorted(roots), 12))
        for r in records(job):
            if r["root_id"] not in chosen or r["K"] not in (1, 8):
                continue
            if r["root_id"] not in frame_cache:
                frame_cache[r["root_id"]] = [f[0] for f in frames([roots[r["root_id"]]], EDITS, 64019)]
            example = frame_cache[r["root_id"]][r["frame"]]
            if family == "maze":
                ok, reason = my_route(example.maze, r["actions"])
                dist = my_bfs(example.maze)
                unreachable = dist[example.maze.start] < 0
                agree = ok == r["route_correct"] == r["exact_correct"] and reason == r["reason"] and unreachable == r["unreachable"]
            else:
                truth = my_circuit_eval(example.circuit)
                stable = [0] * example.circuit.n
                for i, node in enumerate(example.node_order):
                    stable[node] = r["actions"][i]
                scored = [stable[u] == truth[u] for u in range(example.circuit.n) if example.circuit.operators[u] != Operator.INPUT]
                agree = all(scored) == r["exact_correct"] and abs(sum(scored) / len(scored) - r["node_accuracy"]) < 1e-12

            from state_repair.eval.metrics import input_hash
            agree = agree and input_hash(example) == r["input_sha256"]
            rescored[suite] += 1
            if not agree:
                mismatch.append((job, r["root_id"], r["frame"], r["K"]))
print("  rescored rows per suite:", dict(rescored), "total", sum(rescored.values()))
print("  mismatches:", len(mismatch), mismatch[:5])
if mismatch:
    note("BLOCKING", f"{len(mismatch)} raw-action rescoring mismatches")




section("5. validation-only grid selection reproduction")
for size in ("maze12", "circuit32"):
    job = read_json(RUN / f"selection-{size}" / "job.json")
    sealed = json.loads((RUN / f"selection-{size}" / "steps.jsonl").read_text())
    per_candidate = {}
    for cand in job["candidates"]:
        tjob = read_json(RUN / cand["evaluation"] / "job.json")
        assert tjob["split"] == "val" and tjob["dataset"].endswith("-development.json"), cand
        summary = read_json(RUN / cand["source"] / "summary.json")
        if summary["steps"] != 1024:
            note("MAJOR", f"{cand['source']} trained {summary['steps']} steps, not 1024")
        ep = defaultdict(lambda: [0, 0])
        for r in records(cand["evaluation"]):
            if r["split"] != "val":
                note("BLOCKING", f"{cand['evaluation']} contains non-validation records")
            if r["frame"] > 0:
                ep[r["K"], r["root_id"]][0] += int(r["exact_correct"])
                ep[r["K"], r["root_id"]][1] += 1
        assert all(v[1] == tjob["edits"] for v in ep.values())
        per_candidate[cand["source"]] = sum(v[0] / v[1] for v in ep.values()) / len(ep)
    ok_all = True
    for arm in job["arms"]:
        means = {}
        for g in (0, 1):
            cands = [c for c in job["candidates"] if c["arm"] == arm and c["grid"] == g]
            means[g] = sum(per_candidate[c["source"]] for c in cands) / len(cands)
        chosen = max((0, 1), key=lambda g: (means[g], -g))
        s = sealed["grids"][arm]
        ok = chosen == s["grid"] and all(abs(means[g] - s["candidate_validation_means"][str(g)]) < 1e-12 for g in (0, 1))
        ok_all &= ok
        print(f"   {size} {arm:17s} grid0 {means[0]:.6f} grid1 {means[1]:.6f} -> chosen g{chosen} (sealed g{s['grid']}) {'OK' if ok else 'MISMATCH'}")
    if not ok_all:
        note("MAJOR", f"{size} selection does not reproduce")




section("6. test-set discipline")
commit_time = subprocess.run(["git", "log", "-1", "--format=%ct", "cdaf843"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
commit_time = int(commit_time)
manifest = read_json(RUN / "data" / "test-manifest.json")
config = read_json(ROOT / "configs" / "confirmatory_tier2_frozen.json")
proto_at_commit = subprocess.run(["git", "show", "cdaf843:reports/FROZEN_PROTOCOL.md"], cwd=ROOT, capture_output=True).stdout
proto_head = (ROOT / "reports" / "FROZEN_PROTOCOL.md").read_bytes()
print("  protocol sha256 @cdaf843:", hashlib.sha256(proto_at_commit).hexdigest())
print("  protocol sha256 @HEAD   :", hashlib.sha256(proto_head).hexdigest())
print("  manifest protocol_sha256:", manifest["protocol_sha256"])
print("  manifest matrix_sha256 == digest_json(config):", manifest["matrix_sha256"] == digest_json(config), (config_matrix := digest_json(config)))
protocol_text = proto_head.decode()
marker_ok = f"<!-- matrix-sha256: {config_matrix} -->" in protocol_text
print("  matrix marker in protocol:", marker_ok)
for name, spec in config["suites"].items():
    test_file = RUN / "data" / f"{name}-test.json"
    claim = RUN / "data" / f"{name}-test-claim.json"
    mt = os.stat(test_file).st_mtime
    ct = os.stat(claim).st_mtime
    data = read_json(test_file)
    ids_from_file = [e["root_id"] for e in data["test"]]
    recomputed = digest_json(ids_from_file)
    in_protocol = spec["identity_hashes"]["test"] in protocol_text
    same = recomputed == spec["identity_hashes"]["test"] == digest_json(spec["identities"]["test"])
    ch_same = True
    if spec["family"] == "maze":
        ch_same = digest_json([e["root_id"] for e in data["challenge"]]) == spec["identity_hashes"]["challenge"] and spec["identity_hashes"]["challenge"] in protocol_text
    claim_ok = read_json(claim)["identities"] == spec["identity_hashes"]
    sha_ok = hashlib.sha256(test_file.read_bytes()).hexdigest() == manifest["suites"][name]["sha256"]
    print(f"   {name}: claim {ct-commit_time:+.0f}s / file {mt-commit_time:+.0f}s after protocol commit; roots {len(ids_from_file)}; test identity hash matches config+file {same}, listed in protocol {in_protocol}; challenge hash ok {ch_same}; claim ok {claim_ok}; manifest sha ok {sha_ok}")
    if not (ct > commit_time and mt > commit_time and same and in_protocol and ch_same and claim_ok and sha_ok):
        note("BLOCKING", f"{name} test-set discipline check failed")
gen_events = [json.loads(l) for l in (RUN / "driver.jsonl").read_text().splitlines() if l.strip()]
first_job = min(e["unix"] for e in gen_events if e.get("event") == "job_started")
print(f"  first driver job started {first_job-commit_time:+.0f}s after protocol commit")
src_diff = subprocess.run(["git", "diff", "--stat", "cdaf843", "HEAD", "--", "src", "configs/confirmatory_tier2_frozen.json", "reports/FROZEN_PROTOCOL.md"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
print("  git diff cdaf843..HEAD on src/, frozen config, protocol:", repr(src_diff) or "(none)")
if src_diff:
    note("MAJOR", "bound sources changed after freeze: " + src_diff.splitlines()[-1])
dev_only = all(read_json(RUN / c["evaluation"] / "job.json")["dataset"].endswith("-development.json")
               for size in ("maze12", "maze16", "circuit32", "circuit64") for c in read_json(RUN / f"selection-{size}" / "job.json")["candidates"])
print("  every selection candidate evaluated on a development dataset:", dev_only)




section("7. carried-state refinement dynamics (carry_refinement forks)")
dyn = defaultdict(lambda: {"n": 0, "changed": 0, "sum_frac": 0.0, "max_frac": 0.0, "exact": 0})
for job in sorted(p.name for p in RUN.glob("dynamics-*") if p.is_dir() and ".incomplete" not in p.name):
    suite = job.split("dynamics-")[1].rsplit("-s", 1)[0]
    for r in records(job):
        assert r["policy"] == "carry_refinement" and r["record_kind"] == "frozen_state_intervention"
        d = dyn[suite, r["stratum"], r["K"]]
        d["n"] += 1
        d["changed"] += r["changed_action_fraction"] > 0
        d["sum_frac"] += r["changed_action_fraction"]
        d["max_frac"] = max(d["max_frac"], r["changed_action_fraction"])
        d["exact"] += int(r["exact_correct"])
for (suite, stratum, k), d in sorted(dyn.items()):
    if k in (1, 8, 16):
        print(f"   {suite:24s} {stratum:6s} K{k:<2d} n={d['n']:4d} rows-with-any-change {d['changed']/d['n']:.3f} mean-changed-fraction {d['sum_frac']/d['n']:.4f} max {d['max_frac']:.3f} exact {d['exact']/d['n']:.4f}")




section("8. per-frame post-edit exact accuracy, maze12-on-maze12 (mean over five seeds and 256 roots)")
fa = per_frame["maze12-on-maze12"]
for policy in ("restart", "carry", "spatial_gate", "answer_only"):
    for k in (1, 8):
        curve = [sum(fa[policy, k, s, f][0] for s in SEEDS) / sum(fa[policy, k, s, f][1] for s in SEEDS) for f in range(1, EDITS + 1)]
        print(f"   {policy:13s} K{k}: f1 {curve[0]:.4f} f4 {curve[3]:.4f} f8 {curve[7]:.4f} f16 {curve[15]:.4f} f24 {curve[23]:.4f} f32 {curve[31]:.4f}  (first-8 mean {np.mean(curve[:8]):.4f}, last-8 mean {np.mean(curve[-8:]):.4f})")

section("12. K16 anomaly characterization")
fc = per_frame["circuit32-on-circuit32"]
for policy in ("answer_only", "spatial_gate", "restart"):
    for k in (8, 16):
        curve = [fc[policy, k, 137, f][0] / fc[policy, k, 137, f][1] for f in range(1, EDITS + 1)]
        print(f"   circuit32 seed137 {policy:13s} K{k:<2d}: frames 1-8 {[round(x,3) for x in curve[:8]]} ... f16 {curve[15]:.3f} f24 {curve[23]:.3f} f32 {curve[31]:.3f}; mean {np.mean(curve):.4f}")

init = Counter()
for r in records("test-circuit32-on-circuit32-answer_only-s137-h32"):
    if r["frame"] == 0:
        init[r["K"]] += int(r["exact_correct"])
print("   circuit32 seed137 answer_only initial-solve (frame 0) correct roots by K:", dict(sorted(init.items())))
init = Counter()
whole = Counter()
frame_curve = defaultdict(lambda: [0, 0])
for r in records("test-maze16-on-maze16-spatial_gate-s71-h32"):
    if r["frame"] == 0:
        init[r["K"]] += int(r["exact_correct"])
    else:
        frame_curve[r["K"], r["frame"]][0] += int(r["exact_correct"])
        frame_curve[r["K"], r["frame"]][1] += 1
print("   maze16 seed71 spatial_gate initial-solve correct roots by K:", dict(sorted(init.items())))
for k in (8, 16):
    curve = [frame_curve[k, f][0] / frame_curve[k, f][1] for f in range(1, EDITS + 1)]
    print(f"   maze16 seed71 spatial_gate K{k:<2d}: frames 1-8 {[round(x,3) for x in curve[:8]]} ... f16 {curve[15]:.3f} f24 {curve[23]:.3f} f32 {curve[31]:.3f}; mean {np.mean(curve):.4f}")




section("9. mechanism adapter sources and privileged labeling")
for job in sorted(p.name for p in RUN.glob("mechanism-*-s29") if p.is_dir()):
    j = read_json(RUN / job / "job.json")
    backbone = selection_map(j["selection"].split("selection-")[1])["spatial_gate"][str(j["seed"])]
    first = next(records(job))
    kinds = Counter()
    for r in records(job):
        kinds[r["policy"], r["record_kind"], r["privileged"]] += 1
    bad = [k for k in kinds if (k[0] == "impact_mask") != (k[1] == "privileged_diagnostic" and k[2] is True)]
    print(f"   {job}: backbone+spatial weights from {backbone}; transferred projections {first['adapter_sources']}; impact_mask labeling ok: {not bad}")
    if bad:
        note("BLOCKING", f"{job}: privileged labeling mismatch {bad}")




section("10. timing integrity and labels")
for job in ("reference-maze12-on-maze12-h32", "reference-circuit32-on-circuit32-h32"):
    labels = Counter((r["record_kind"], r["policy"], r["device"], r["measurement"]) for r in records(job))
    print(f"   {job}: {dict(labels)}")
lat = Counter()
lat_fail = 0
lat_rows = 0
for r in records("latency-maze12-on-maze12-restart-s29-h32"):
    lat[r["measurement"], r["batch_size"], r["warmup_frames"]] += 1
    lat_rows += 1
    lat_fail += not r["exact_correct"]
print(f"   latency-maze12 restart s29: labels {dict(lat)} rows {lat_rows} (expected 8 roots x 33 frames x 3 reps x 5 K = {8*33*3*5}); failed predictions retained: {lat_fail}")
print(f"   stage-sum check on accuracy records (section 1): {stage_violation} violations in {stage_checked}")




section("11. ledger completeness")
journal = [json.loads(l) for l in (RUN / "gpu_time.jsonl").read_text().splitlines() if l.strip()]
reserve = {r["job_id"] for r in journal if r["kind"] == "reserve"}
actual = {r["job_id"]: r for r in journal if r["kind"] == "actual"}
print(f"   journal reserve {len(reserve)} actual {len(actual)} unmatched {len(reserve ^ set(actual))}")
con = sqlite3.connect(f"file:{(RUN/'accounting.sqlite').as_posix()}?mode=ro", uri=True)
db_res = {r[0] for r in con.execute("select job_id from events where kind='reserve'")}
db_act = {r[0] for r in con.execute("select job_id from events where kind='actual'")}
print(f"   sqlite reserve {len(db_res)} actual {len(db_act)} unmatched {len(db_res ^ db_act)}; sqlite ids == journal ids: {db_res == reserve and db_act == set(actual)}")
print(f"   sqlite integrity_check: {con.execute('pragma integrity_check').fetchone()[0]}")
gpu_h = sum(r["wall_seconds"] for r in actual.values() if r["device"] == "cuda") / 3600
cpu_h = sum(r["wall_seconds"] for r in actual.values() if r["device"] != "cuda") / 3600
print(f"   recorded GPU hours {gpu_h:.6f}; CPU-job hours {cpu_h:.6f}; empirical attempts {sum(not r['synthetic'] for r in actual.values())}; synthetic attempts {sum(r['synthetic'] for r in actual.values())}")
for r in actual.values():
    if r["aborted"] or r["error"]:
        print(f"   aborted/error: {r['matrix_job']} aborted={r['aborted']} seconds={r['wall_seconds']} error={r['error']}")
complete = read_json(RUN / "complete.json")
print(f"   complete.json: {complete}")
inc = sorted(p.name for p in RUN.glob("*.incomplete-*"))
print(f"   quarantined directories present: {inc}")
referenced = set()
for p in RUN.glob("report-*/results.json"):
    for c in read_json(p)["checks"]:
        referenced.add(c.get("job", ""))
print(f"   any quarantined directory referenced by a report: {any(any(i in ref for i in inc) for ref in referenced)}")
sealed = sum(1 for p in RUN.iterdir() if p.is_dir() and (p / "seal.json").exists() and ".incomplete" not in p.name)
print(f"   sealed job directories: {sealed} (matrix jobs {complete['jobs']}; capacity/development directories are extra)")




section("13. remaining checks")
cfg_jobs = {j["id"]: j for j in config["jobs"]}
print("   test-split evaluation jobs whose dataset is not a *-test.json:", [j for j, v in cfg_jobs.items() if v.get("split") == "test" and not str(v.get("dataset", "")).endswith("-test.json")])
print("   evaluation jobs referencing a test dataset with split != test:", [j for j, v in cfg_jobs.items() if str(v.get("dataset", "")).endswith("-test.json") and v.get("split") != "test"])
print("   training jobs (research_training) referencing a test dataset:", [j for j, v in cfg_jobs.items() if v.get("kind") == "research_training" and "test" in str(v.get("dataset", ""))])
print("   stream seed shared by test and tuning jobs (edit sequences are deterministic functions of root id, not learned):", cfg_jobs["test-maze12-on-maze12-restart-s29-h32"]["stream_seed"], cfg_jobs["tuning-maze12-restart-g0-s29"]["stream_seed"])
print(f"   audit wall time {time.perf_counter()-T0:.1f}s")
print("\nFINDINGS:")
for f in FINDINGS or ["none"]:
    print("  " + f)

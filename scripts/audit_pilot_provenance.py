"""Supplemental CPU lineage audit and deterministic selected-policy spot replay."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

from state_repair.accounting.resources import memory_snapshot
from state_repair.data.challenges import sample_challenges
from state_repair.data.maze import collate
from state_repair.data.splits import canonical_hash
from state_repair.models.policy import FixedBudgetPolicy
from state_repair.oracles.maze import score_policy
from state_repair.provenance import file_hash
from state_repair.train.pilot import PRINCIPAL, decode_example
from check_adapter_pilot import require
from run_adapter_pilot import (intervention_entries, json_rows, load_selected, read,
                               verify_seal, write)


def main() -> None:
    torch.set_num_threads(2)
    started = time.perf_counter()
    root = Path("runs/adapter_pilot_v1")
    config = read(root / "prepared/config.json")
    payload = read(root / "prepared/dataset.json")
    check = read("runs/prompt05-pilot-check.json")
    require(check["verified"] and check["complete"], "complete primary verification required")
    roots = [decode_example(e) for e in payload["streams"][0]]
    challenge = sample_challenges(roots, pairs_per_type=config["pairs_per_type"], seed=config["challenge_seed"],
        low_max=config["low_max"], high_min=config["high_min"])
    regenerated = [{"root": asdict(p.root), "edit_type": p.edit_type,
        "low": asdict(replace(p.root, maze=p.low, frame_index=1)),
        "high": asdict(replace(p.root, maze=p.high, frame_index=1))} for p in challenge.pairs]
    require(json.loads(json.dumps(regenerated)) == payload["challenge"], "challenge selection replay mismatch")
    require(challenge.audit == read(root / "prepared/challenge_audit.json"), "challenge quota audit mismatch")
    b = config["batch_size"]
    old = [collate([decode_example(e) for e in payload["train"][i:i+b]])[0] for i in range(0, len(payload["train"]), b)]
    cache_count = 0
    for row in read(root / "cache/cache_index.json")["entries"]:
        cache = torch.load(root / "cache" / row["file"], map_location="cpu", weights_only=False)
        cache.check(old[row["batch"]], config["source_checkpoint_sha256"], row["K"])
        require(row["input_sha256"] == [canonical_hash(decode_example(e).maze)
            for e in payload["train"][row["batch"]*b:(row["batch"]+1)*b]], "cache input lineage mismatch")
        cache_count += 1
    entries = intervention_entries(payload, config)
    selection = read(root / "selection.json")
    prior_count = 0
    for track in ("frozen", "joint"):
        for seed in config["seeds"]:
            out = root / "interventions" / f"{track}-seed{seed}"
            verify_seal(out)
            cp_hash = config["source_checkpoint_sha256"] if track == "frozen" else file_hash(
                root / "training" / f"joint-spatial_gate-seed{seed}-grid{selection['joint-spatial_gate']['grid']}" / "checkpoint.pt")
            for offset in range(0, len(entries), b):
                subset = entries[offset:offset+b]
                padded = subset + [subset[-1]]*(b-len(subset))
                obs = collate([decode_example(e["old"]) for e in padded])[0]
                cache = torch.load(out / f"prior-batch{offset//b}.pt", map_location="cpu", weights_only=False)
                cache.check(obs, cp_hash, config["source_K"])
                prior_count += 1
    print(json.dumps({"challenge_replayed": True, "training_cache_batches": cache_count, "intervention_prior_batches": prior_count}), flush=True)



    chosen_ids = {e.root_id for e in roots[:2]}
    frames = [[decode_example(e) for e in frame[:2]] for frame in payload["streams"]]
    observations = [collate(frame)[0] for frame in frames]
    weights = torch.load(config["source_checkpoint"], map_location="cpu", weights_only=False)["model"]
    counts = {"predictions": 0, "action_entries": 0, "action_mismatches": 0, "route_mismatches": 0,
              "all_node_mismatches": 0, "valid_action_metric_mismatches": 0}
    arms = []
    replay_path = Path("runs/prompt05-pilot-cpu-replay.jsonl")
    with torch.no_grad(), replay_path.open("x", encoding="utf-8") as handle:
        for track in ("frozen", "joint"):
            for seed in config["seeds"]:
                for arm in PRINCIPAL:
                    if track == "frozen" and arm in ("restart", "carry") and seed != config["seeds"][0]:
                        continue
                    path = root / "streams" / f"{track}-{arm}-seed{seed}" / "predictions.jsonl.gz"
                    expected = {(r["K"], r["frame"], r["root_id"]): r for r in json_rows(path)
                        if r["K"] in (1, 8) and r["root_id"] in chosen_ids}
                    model, adapter, identity = load_selected(root, config, selection, weights, track, arm, seed, "cpu")
                    model.eval()
                    adapter.eval()
                    mismatches = 0
                    for k in (1, 8):
                        policy = FixedBudgetPolicy(model, adapter, k)
                        for frame, obs in enumerate(observations):
                            result = policy(obs)
                            for e, actions in zip(frames[frame], result.prediction.logits.argmax(-1).tolist()):
                                original = expected[k, frame, e.root_id]
                                score = score_policy(e.maze, actions)
                                different = sum(a != b for a, b in zip(actions, original["actions"]))
                                counts["predictions"] += 1
                                counts["action_entries"] += len(actions)
                                counts["action_mismatches"] += different
                                counts["route_mismatches"] += score["route_correct"] != original["route_correct"]
                                counts["all_node_mismatches"] += score["all_node_correct"] != original["all_node_correct"]
                                counts["valid_action_metric_mismatches"] += score["valid_action_accuracy"] != original["valid_action_accuracy"]
                                mismatches += different
                                handle.write(json.dumps({"track": track, "seed": seed, "arm": arm, "K": k,
                                    "frame": frame, "root_id": e.root_id, "synthetic": False, "device": "cpu",
                                    "checkpoint_sha256": identity["checkpoint_sha256"], "cpu_actions": actions,
                                    "saved_cuda_actions": original["actions"], "action_mismatches": different,
                                    "cpu_score": score, "saved_cuda_route_correct": original["route_correct"]})+"\n")
                        del policy
                    arms.append({"track": track, "seed": seed, "arm": arm, "action_mismatches": mismatches})
                    del model, adapter
                    print(json.dumps(arms[-1]), flush=True)
    write(Path("runs/prompt05-pilot-provenance-audit.json"), {"verified_lineage": True,
        "challenge_selection_replayed": True, "training_cache_batches": cache_count,
        "intervention_prior_batches": prior_count, "cpu_replay": counts, "arms": arms,
        "exact_action_replay": counts["action_mismatches"] == 0,
        "root_ids": sorted(chosen_ids), "budgets": [1, 8], "scope": "first two validation roots, all seeds and deterministic principal arms, all 33 frames",
        "wall_s": time.perf_counter()-started, "memory": memory_snapshot(), "device": "cpu", "torch": str(torch.__version__),
        "primary_check_sha256": file_hash("runs/prompt05-pilot-check.json"), "replay_records_sha256": file_hash(replay_path),
        "script_sha256": file_hash(__file__), "synthetic": False})


if __name__ == "__main__":
    main()

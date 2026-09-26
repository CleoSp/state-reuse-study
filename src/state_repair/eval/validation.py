"""Exact regeneration checks against saved validation evidence."""
from __future__ import annotations

from collections import defaultdict
import gzip
import json
from pathlib import Path

from state_repair.execution.durable import atomic_json
from state_repair.provenance import file_hash


def legacy_episodes(path: Path, *, circuit: bool) -> list[dict]:
    grouped = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["synthetic"] is not False or row["split"] != "val" or row["record_kind"] != "fixed_budget_stream":
                raise ValueError("requires real saved validation stream records")
            if row["state_budget"] != row["K"]:
                raise ValueError("legacy fixed-budget provenance mismatch")
            grouped[row["K"], row["root_id"]].append(row)
    result = []
    for (k, root), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r["frame"])
        if [r["frame"] for r in rows] != list(range(33)):
            raise ValueError("legacy stream has incomplete frames")
        metric = "exact_correct" if circuit else "route_correct"
        value = {"K": k, "root_id": root, "initial_correct": rows[0][metric]}
        for end in (4, 32):
            value[f"post{end}"] = sum(r[metric] for r in rows[1:end+1])/end
            value[f"whole{end}"] = all(r[metric] for r in rows[:end+1])
            if circuit:
                value[f"node{end}"] = sum(r["node_accuracy"] for r in rows[1:end+1])/end
        result.append(value)
    return result


def verify_05b(root: Path) -> dict:
    checks = []
    for family, directory in (("maze", "adapter_stream_v2"), ("circuit", "circuit_stream_v2"),
                              ("maze", "adapter_pilot_v1_longer"), ("circuit", "circuit_stream_v2_depth48")):
        for out in sorted((root / "runs" / directory / "streams").glob("*")):
            if not (out / "summary.json").exists():
                continue
            saved = json.loads((out / "summary.json").read_text())["validation"]["episodes"]
            regenerated = legacy_episodes(out / "predictions.jsonl.gz", circuit=family == "circuit")
            original = {(r["K"], r["root_id"]): r for r in saved}
            if original.keys() != {(r["K"], r["root_id"]) for r in regenerated}:
                raise ValueError("05b episode identities differ")
            for row in regenerated:
                if any(original[row["K"], row["root_id"]][k] != v for k, v in row.items()):
                    raise ValueError("05b regeneration is not exact: " + str(out))
            checks.append({"path": str(out.relative_to(root)), "episodes": len(regenerated),
                           "predictions_sha256": file_hash(out / "predictions.jsonl.gz"), "exact_agreement": True})
    if not checks:
        raise ValueError("05b evidence missing")
    return {"verified": True, "synthetic": False, "checks": checks,
            "episodes": sum(r["episodes"] for r in checks), "scope": "all saved stream episode table inputs"}

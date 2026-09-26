"""Print learning metrics from saved static diagnostics."""
import json
from pathlib import Path

root = Path("runs/static_diagnosis")
checks = json.loads((root / "checks/checks.json").read_text())
print("| Run / fixture size | Width / K (F calls) | Steps / LR | Loss (uniform / constant) | Acceptable actions | Routes | Job s / peak MiB |")
print("|---|---:|---|---|---:|---:|---:|")
for path in sorted(root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime):
    d = json.loads(path.read_text())
    if "final" not in d:
        continue
    c, f = d["config"], d["final"]
    n, k = c["count"], c["depth"]
    prior = ""
    if c["resume"]:
        parent = json.loads((root / c["resume"] / "summary.json").read_text())
        prior = str(parent["optimizer_steps"]) + "+"
    b = checks[str(n)]
    print(f"| {path.parent.name} / {n} | {c['width']} / {k} ({2*k}) | "
          f"{prior}{d['optimizer_steps']} / {c['lr']:g} | {f['loss']:.6f} "
          f"({b['uniform_loss']:.6f} / {b['constant_loss']:.6f}) | "
          f"{f['acceptable_action_accuracy']:.4%} | {round(f['complete_route_success']*n)}/{n} | "
          f"{d['wall_s']:.2f} / {d['memory']['peak_rss_bytes']/2**20:.2f} |")
wall = sum(json.loads(p.read_text())["wall_s"] for pattern in ("*/summary.json", "*/inspection.json")
           for p in root.glob(pattern))
training = sum(json.loads(p.read_text()).get("training_s", 0) for p in root.glob("*/summary.json"))
print(f"\nCumulative completed-job wall time: {wall:.3f}s; optimizer batch time: {training:.3f}s.")

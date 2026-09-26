"""Recompute stream-training tables and verify saved-evidence agreement."""
from pathlib import Path
import json
import subprocess
import sys

from state_repair.eval.validation import verify_05b
from state_repair.execution.durable import atomic_json, atomic_write
from state_repair.provenance import file_hash

root = Path(__file__).resolve().parents[1]
paths = [root / "reports/PILOT_REPORT.md", root / "reports/stream_results.json"]
before = {str(p): file_hash(p) for p in paths}
result = verify_05b(root)
import build_stream_report
generated = root / "runs/prompt06-regenerated-stream-results.json"
def redirect_write(path, value):
    if path.as_posix() != "reports/stream_results.json":
        raise ValueError("unexpected historical report output")
    atomic_json(generated, value)
build_stream_report.write = redirect_write
build_stream_report.main()
if json.loads(paths[1].read_text()) != json.loads(generated.read_text()) or before[str(paths[0])] != file_hash(paths[0]):
    raise ValueError("historical report regeneration changed saved evidence")
result["report_hashes"] = before
atomic_json(root / "runs/prompt06-validation-regeneration.json", result)
print(json.dumps({"verified": True, "exact_report_agreement": True,
                  "directories": len(result["checks"]), "episodes": result["episodes"]}))

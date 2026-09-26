"""Evaluate the maze matrix after the training queue completes."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


def main() -> None:
    root = Path("runs/adapter_stream_v2")
    config = json.loads((root / "prepared/config.json").read_text(encoding="utf-8"))
    expected = [root / "training" / f"joint-{arm}-seed{seed}-grid{grid}"
        for arm in ("restart","carry","spatial_gate","global_gate","gru_adapter","residual_adapter","answer_only")
        for seed in config["seeds"] for grid in range(2)]
    deadline = time.monotonic()+6*3600
    while not all((p / "manifest.json").exists() for p in expected):
        for p in expected:
            if (p / "resources.json").exists() and not json.loads((p / "resources.json").read_text(encoding="utf-8"))["complete"]:
                raise RuntimeError(f"training failed; do not continue: {p}")
        if time.monotonic() >= deadline:
            raise TimeoutError("training completion wait exceeded six hours")
        time.sleep(30)
    gpu_python = "runs/foundation/venv-cuda/Scripts/python.exe"
    base = [gpu_python,"scripts/run_adapter_pilot.py"]
    common = ["--root",str(root),"--config","configs/adapter_stream_v2.json"]
    subprocess.run([*base,"select",*common],check=True)
    for stage in ("streams","interventions"):
        for seed in config["seeds"]:
            subprocess.run([*base,stage,*common,"--track","joint","--seed",str(seed)],check=True)
    subprocess.run([".venv/Scripts/python.exe","scripts/check_adapter_pilot.py","--run",str(root),"--output",str(root / "check.json")],check=True)
    diagnostic = [gpu_python,"scripts/diagnose_stream_reuse.py","dynamics","--family","maze"]
    subprocess.run([*diagnostic,"--version","pilot","--seed",str(config["seeds"][0])],check=True)
    for seed in config["seeds"]:
        for owner in ("carry","spatial_gate"):
            subprocess.run([*diagnostic,"--version","stream","--seed",str(seed),"--owner",owner],check=True)
    subprocess.run([".venv/Scripts/python.exe","scripts/check_stream_diagnostics.py","--family","maze",
        "--dynamics-only","--output",str(root / "g1_check.json")],check=True)
    print("Maze evaluation complete and checked.",flush=True)


if __name__ == "__main__":
    main()

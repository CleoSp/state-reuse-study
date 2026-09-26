"""Run the circuit matrix serially, stopping on any failure."""
from __future__ import annotations

import subprocess


def main() -> None:
    cpu = ".venv/Scripts/python.exe"
    gpu = "runs/foundation/venv-cuda/Scripts/python.exe"
    runner = [gpu,"scripts/run_circuit_stream.py"]
    for seed in (29,43,71):
        subprocess.run([*runner,"train","--seed",str(seed)],check=True)
    subprocess.run([*runner,"select"],check=True)
    for seed in (29,43,71):
        subprocess.run([*runner,"streams","--seed",str(seed)],check=True)
    subprocess.run([cpu,"scripts/check_circuit_stream.py","--output","runs/circuit_stream_v2/check.json"],check=True)
    diagnostic = [gpu,"scripts/diagnose_stream_reuse.py","dynamics","--family","circuit"]
    subprocess.run([*diagnostic,"--version","pilot","--seed","29"],check=True)
    for seed in (29,43,71):
        for owner in ("carry","spatial_gate"):
            subprocess.run([*diagnostic,"--version","stream","--seed",str(seed),"--owner",owner],check=True)
    subprocess.run([cpu,"scripts/check_stream_diagnostics.py","--family","circuit","--dynamics-only",
        "--output","runs/circuit_stream_v2/g1_check.json"],check=True)
    print("Circuit replication and G1 complete and verified; commit/push precedes remaining lesions.",flush=True)


if __name__ == "__main__":
    main()

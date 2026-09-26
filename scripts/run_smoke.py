"""Launch exactly one CPU smoke subprocess with a 20-minute hard timeout."""
from pathlib import Path
import json
import subprocess
import sys
import time


if __name__ == "__main__":
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,"-m","state_repair.cli","train","--config","configs/smoke.yaml","--max-minutes","20"]
    evidence=root/"reports"/"smoke_command.json"
    if evidence.exists():
        raise SystemExit("smoke command evidence already exists; no automatic retry authorized")
    started=time.perf_counter()
    try:
        result=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=1200)
        record={"command":command,"cwd":str(root),"exit_code":result.returncode,"timed_out":False,
                "wall_seconds":time.perf_counter()-started,"stdout":result.stdout,"stderr":result.stderr}
    except subprocess.TimeoutExpired as exc:
        record={"command":command,"cwd":str(root),"exit_code":124,"timed_out":True,
                "wall_seconds":time.perf_counter()-started,"stdout":str(exc.stdout or ""),"stderr":str(exc.stderr or ""),
                "note":"Child terminated by subprocess timeout. Inspect partial artifacts and unreconciled cost reservation; do not retry automatically."}
    with evidence.open("x",encoding="utf-8") as stream:
        json.dump(record,stream,indent=2)
        stream.write("\n")
    print(json.dumps(record,indent=2))
    raise SystemExit(record["exit_code"])

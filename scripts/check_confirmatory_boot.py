"""Read-only acceptance checker, followed by an atomic evidence record."""
import argparse
import json
from pathlib import Path

from state_repair.execution.boot import check_boot
from state_repair.execution.durable import atomic_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observe-automatic-logon", action="store_true")
    args = parser.parse_args()
    audit = json.loads((args.output / "logon-audit.json").read_text(encoding="utf-8-sig"))
    result = check_boot(args.output, audit, observe_automatic_logon=args.observe_automatic_logon)
    atomic_json(args.output / "boot-check.json", result)
    print(json.dumps(result))

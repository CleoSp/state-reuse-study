"""One-time test generation; the committed protocol is a mandatory gate."""
import argparse
from pathlib import Path
import json
from state_repair.execution.datasets import prepare_test_once
from state_repair.execution.durable import read_json

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--protocol", type=Path, required=True)
parser.add_argument("--data", type=Path, default=Path("runs/confirmatory_v1/data"))
args = parser.parse_args()
result = prepare_test_once(read_json(args.config), args.protocol, args.data)
print(json.dumps(result), flush=True)

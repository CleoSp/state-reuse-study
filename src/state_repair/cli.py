"""Command-line interface for configuration, training, and evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(prog="state-repair")
    sub=parser.add_subparsers(dest="command",required=True)
    doctor=sub.add_parser("doctor")
    doctor.add_argument("--device",default="auto",choices=["auto","cpu","cuda","mps"])
    for command in ("generate","train"):
        p=sub.add_parser(command)
        p.add_argument("--config",required=True)
        if command=="train":
            p.add_argument("--max-minutes",type=float,default=20)
    for command in ("evaluate","report"):
        p=sub.add_parser(command)
        p.add_argument("--run",required=True)
        if command=="evaluate":
            p.add_argument("--suite",choices=["smoke"],required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=="doctor":
            from state_repair.devices import doctor
            result=doctor(args.device)
            if any(backend["status"] == "failed" for backend in result["backends"].values()):
                print(json.dumps(result, indent=2, allow_nan=False))
                raise RuntimeError("forward/backward diagnostic failed; see backend report")
        elif args.command=="generate":
            from state_repair.config import load_config, require_static_workflow
            config = load_config(args.config)
            require_static_workflow(config)
            from state_repair.data.serialization import generate_dataset
            result={"dataset_dir":str(generate_dataset(config))}
        elif args.command=="train":
            from state_repair.config import load_config, require_static_workflow
            import math
            if not math.isfinite(args.max_minutes) or not 0<args.max_minutes<=20:
                raise ValueError("train requires --max-minutes > 0 and <= 20")
            config = load_config(args.config)
            require_static_workflow(config)
            from state_repair.train.loop import train_static
            result=train_static(config,max_minutes=args.max_minutes)
        else:
            if not Path(args.run).is_dir():
                raise FileNotFoundError(f"run directory does not exist: {args.run}")
            from state_repair.eval.smoke import evaluate_run, report_run
            result=evaluate_run(Path(args.run)) if args.command=="evaluate" else report_run(Path(args.run))
        print(json.dumps(result,indent=2,default=str,allow_nan=False))
        return 0
    except (ValueError,TypeError,FileNotFoundError,FileExistsError,RuntimeError,NotImplementedError,ModuleNotFoundError) as exc:
        print(f"error: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())

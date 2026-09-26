"""Run stream-training controls with fixed trainers and evaluators."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
from pathlib import Path
import shutil

import torch

import run_adapter_pilot as maze
import run_circuit_stream as circuit
from check_adapter_pilot import require
from state_repair.provenance import file_hash, source_provenance

ROOTS = {"longer": Path("runs/adapter_pilot_v1_longer"), "depth": Path("runs/circuit_stream_v2_depth48")}
CONFIGS = {"longer": Path("configs/adapter_pilot_v1_longer.json"), "depth": Path("configs/circuit_stream_v2_depth48.json")}


def depth_payload(config: dict) -> dict:
    source = maze.read(config["source_dataset"])
    roots = [circuit.decode(r) for r in source[config["source_roots_key"]]]
    require(len(roots) == 256 and all(e.split == "val" and e.frame_index == 0 and
        e.circuit.n == 48 and not e.synthetic for e in roots), "wrong depth validation roots")
    circuit.reject_duplicate_roots(roots)
    return {"streams": [[asdict(e) for e in frame] for frame in
        circuit.stream_frames(roots, config["stream_edits"], config["stream_seed"])]}


def prepare(kind: str) -> None:
    root, config = ROOTS[kind], maze.read(CONFIGS[kind])
    out = root / "prepared"
    out.mkdir(parents=True, exist_ok=False)
    source = Path("runs/adapter_pilot_v1" if kind == "longer" else config["source_run"])
    maze.verify_seal(source / "prepared")
    require(file_hash(config["source_checkpoint"]) == config["source_checkpoint_sha256"], "source checkpoint changed")
    maze.write(out / "config.json", config)
    if kind == "longer":
        shutil.copyfile(source / "prepared/dataset.json", out / "dataset.json")
        selection = {f"joint-{a}": maze.read(source / "selection.json")[f"joint-{a}"] for a in config["arms"]}
    else:
        maze.write(out / "dataset.json", depth_payload(config))
        selection = maze.read(source / "selection.json")
    maze.write(out / "selection.json", selection)
    for path in [Path("REPRODUCING.md"),
                 *Path("src").rglob("*.py"), *Path("scripts").glob("*.py")]:
        target = out / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    maze.write(out / "provenance.json", {**source_provenance(), "synthetic": False,
        "source_dataset_sha256": file_hash(config["source_dataset"]),
        "selection_source": str(source / "selection.json"),
        "selection_source_sha256": file_hash(source / "selection.json"),
        "source_snapshot_hashes": {p.relative_to(out / "source").as_posix(): file_hash(p)
            for p in (out / "source").rglob("*") if p.is_file()}})
    maze.seal(out)


def config_and_payload(kind: str) -> tuple[Path, dict, dict]:
    root = ROOTS[kind]
    maze.verify_seal(root / "prepared")
    config = maze.read(root / "prepared/config.json")
    require(config == maze.read(CONFIGS[kind]), "addendum config changed")

    for name, digest in maze.read(root / "prepared/provenance.json")["source_snapshot_hashes"].items():
        if name.startswith("src/") or name in ("scripts/run_adapter_pilot.py", "scripts/run_circuit_stream.py", "scripts/run_stream_addenda.py"):
            require(file_hash(name) == digest, f"runtime dependency changed: {name}")
    return root, config, maze.read(root / "prepared/dataset.json")


def longer_train(seed: int) -> None:
    root, config, payload = config_and_payload("longer")
    require(seed in config["seeds"], "undeclared seed")
    old, new = maze.training_batches(payload, config)
    tuning = maze.frame_batches(payload, config["tuning_edits"], config)
    weights = torch.load(config["source_checkpoint"], map_location="cpu", weights_only=False)["model"]
    selection = maze.read(root / "prepared/selection.json")
    for arm in config["arms"]:
        grid = selection[f"joint-{arm}"]["grid"]
        maze.train_one(root, config, payload, old, new, tuning, {}, weights, "joint", arm, seed, grid)


def longer_streams(seed: int) -> None:
    root, config, payload = config_and_payload("longer")
    require(seed in config["seeds"], "undeclared seed")
    data = maze.frame_batches(payload, 32, config)
    selection = maze.read(root / "prepared/selection.json")
    for arm in config["arms"]:
        out = root / "streams" / f"joint-{arm}-seed{seed}"
        if out.exists():
            maze.verify_seal(out)
            continue
        with maze.gpu_job(config, out, config["job_seconds"], "prompt05b_matched_compute_streams", synthetic=False) as job:
            model, adapter, identity = maze.load_selected(root, config, selection, {}, "joint", arm, seed, "cuda")
            identity.update(track="joint", arm=arm, seed=seed, config_sha256=file_hash(root / "prepared/config.json"))
            result = maze.evaluate_streams(model, adapter, data, config, out / "predictions.jsonl.gz", identity, job.check_limit, "cuda")
            maze.write(out / "summary.json", {**identity, "synthetic": False, "validation": result})
            del model, adapter
        maze.seal(out)
        gc.collect(); torch.cuda.empty_cache()
        print(out, flush=True)


def depth_streams(seed: int) -> None:
    root, config, payload = config_and_payload("depth")
    require(seed in config["seeds"], "undeclared seed")
    data = circuit.batches(payload["streams"], config, 32)
    source = Path(config["source_run"])
    for arm in config["arms"]:
        for grid in config["checkpoint_grids"]:
            name = f"joint-{arm}-seed{seed}-grid{grid}"
            out = root / "streams" / name
            if out.exists():
                maze.verify_seal(out)
                continue
            checkpoint = source / "training" / name / "checkpoint.pt"
            maze.verify_seal(checkpoint.parent)
            with maze.gpu_job(config, out, config["job_seconds"], "prompt05b_depth48_32_edit_validation", synthetic=False) as job:
                cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = circuit.new_model(config, cp["model"], "cuda", training=False)
                adapter = circuit.adapter_for(arm, config).to("cuda")
                adapter.load_state_dict(cp["adapter"])
                identity = {"track": "joint", "arm": arm, "seed": seed, "grid": grid,
                    "checkpoint_sha256": file_hash(checkpoint), "config_sha256": file_hash(root / "prepared/config.json")}
                result = circuit.evaluate_streams(model, adapter, data, config, out / "predictions.jsonl.gz", identity, job, "cuda")
                maze.write(out / "summary.json", {**identity, "synthetic": False, "validation": result})
                del cp, model, adapter
            maze.seal(out)
            gc.collect(); torch.cuda.empty_cache()
            print(out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare-longer", "prepare-depth", "longer-train", "longer-streams", "depth-streams"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.stage.startswith("prepare-"):
        prepare(args.stage.split("-")[1])
    else:
        {"longer-train": longer_train, "longer-streams": longer_streams, "depth-streams": depth_streams}[args.stage](args.seed)

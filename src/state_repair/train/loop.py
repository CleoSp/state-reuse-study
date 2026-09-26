"""One bounded static CPU run using existing train/validation artifacts only."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any
import uuid

import numpy as np
import torch
import yaml

from state_repair.accounting.ledger import AccountingLedger
from state_repair.accounting.resources import memory_snapshot
from state_repair.config import ExperimentConfig, require_static_workflow
from state_repair.data.maze import collate
from state_repair.data.serialization import load_split
from state_repair.models.recursive import RecursiveSolver
from state_repair.provenance import file_hash, source_provenance
from state_repair.train.checkpoint import save_checkpoint
from state_repair.train.losses import maze_valid_set_loss


class ResourceLimitReached(RuntimeError):
    """The bounded training run stopped before completion."""


def check_limits(deadline: float, memory_limit_mb: int) -> dict[str, Any]:
    if time.perf_counter() >= deadline:
        raise ResourceLimitReached("wall-clock limit reached")
    memory = memory_snapshot()
    observed = memory.get("rss_bytes") or memory.get("peak_rss_bytes")
    if observed is None:
        raise ResourceLimitReached("process memory measurement unavailable; cannot enforce limit")
    if observed > memory_limit_mb * 1024**2:
        raise ResourceLimitReached("process memory limit reached")
    return memory


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _append(stream: Any, record: dict) -> None:
    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def load_static_data(config: ExperimentConfig) -> tuple[dict, dict, dict]:
    """Validate identity and read train/val payloads; test is never opened."""
    dataset_dir = Path(config.output_dir) / "dataset"
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("dataset is missing; run the generate command first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = {"schema_version": 1, "seed": config.seed, "dataset": asdict(config.dataset)}
    if manifest.get("generation_config") != identity:
        raise ValueError("dataset configuration mismatch; use the original config or a new output_dir")
    examples = {split: [e for e in load_split(dataset_dir, split) if e.frame_index == 0]
                for split in ("train", "val")}
    if any(not value for value in examples.values()):
        raise ValueError("static smoke requires nonempty train and validation roots")
    if any(e.synthetic for values in examples.values() for e in values):
        raise ValueError("synthetic fixtures cannot enter this empirical smoke run")
    if {e.root_id for e in examples["train"]} & {e.root_id for e in examples["val"]}:
        raise ValueError("train/validation root overlap")
    hashes = {split: manifest["splits"][split]["sha256"] for split in examples}
    return examples, hashes, manifest


def train_static(config: ExperimentConfig, max_minutes: float = 20) -> dict[str, Any]:
    """Train frame zero with terminal supervision and evaluate fixed budgets.

    Cooperative checks occur before and after each atomic batch/evaluation.
    An external process watchdog is needed for a hard wall-time guarantee.
    Checkpoint includes RNG/optimizer state but exact resumption is not supported.
    """
    require_static_workflow(config)
    if not math.isfinite(max_minutes) or not 0 < max_minutes <= 20:
        raise ValueError("first-session CPU smoke max_minutes must be >0 and <=20")
    if config.device not in ("cpu", "auto"):
        raise ValueError("static smoke is CPU only; accelerators are not tested")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("training must run inside a virtual environment")
    output = Path(config.output_dir)
    artifacts = ("config.json", "provenance.json", "steps.jsonl", "pre_metrics.jsonl", "post_metrics.jsonl", "checkpoint.pt", "summary.json")
    if any((output / name).exists() for name in artifacts):
        raise FileExistsError("training artifacts already exist; inspect them and choose a new output_dir")
    from state_repair.eval.smoke import evaluate_examples
    examples, hashes, _ = load_static_data(config)
    start = time.perf_counter()
    deadline = start + max_minutes * 60
    job_id = "static-" + uuid.uuid4().hex
    ledger = AccountingLedger(Path("runs") / "accounting.sqlite")
    ledger.reserve(job_id, 0, category="local_cpu_smoke")
    status, failure = "failed", None
    steps, block_calls, layer_executions = 0, 0, 0
    training_ms = 0.0
    checkpoint_sha = None
    model = optimizer = None
    provenance: dict[str, Any] = {}
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    summary: dict[str, Any] = {}
    try:
        check_limits(deadline, config.training.memory_limit_mb)
        torch.set_num_threads(config.training.threads)
        torch.use_deterministic_algorithms(True)
        random.seed(config.seed)
        np.random.seed(config.seed % 2**32)
        torch.manual_seed(config.seed)
        provenance = source_provenance()
        provenance.update({"environment": {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "torch": str(torch.__version__), "numpy": np.__version__,
            "pyyaml": yaml.__version__, "device": "cpu", "dtype": "float32", "threads": config.training.threads,
            "deterministic_algorithms": True, "cuda": "not tested", "mps": "not tested"},
            "dataset_dir": str((output / "dataset").resolve()), "split_hashes": hashes})
        _write_json(output / "config.json", config.to_dict())
        _write_json(output / "provenance.json", provenance)
        model = RecursiveSolver(config.model.width, config.model.heads, config.model.inner_cycles,
                                attention_mode="dense")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        summary.update({"parameter_count": sum(p.numel() for p in model.parameters()),
                        "root_counts": {split: len(items) for split, items in examples.items()}})

        def measure(stage: str) -> None:
            nonlocal block_calls, layer_executions
            model.eval()
            with (output / f"{stage}_metrics.jsonl").open("x", encoding="utf-8") as stream:
                for split, items in examples.items():
                    for example in items:
                        for budget in config.training.eval_budgets:
                            check_limits(deadline, config.training.memory_limit_mb)
                            records = evaluate_examples(model, [example], [budget], batch_size=1)
                            for record in records:
                                record.update(stage=stage, split=split, seed=config.seed, run_id=job_id)
                                block_calls += record["block_calls"]
                                layer_executions += record["transformer_layer_executions"]
                                _append(stream, record)
                            check_limits(deadline, config.training.memory_limit_mb)

        measure("pre")
        model.train()
        with (output / "steps.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(config.training.steps):
                check_limits(deadline, config.training.memory_limit_mb)
                tick = time.perf_counter()
                chosen = [examples["train"][i] for i in torch.randint(len(examples["train"]), (config.training.batch_size,)).tolist()]
                obs, target, _ = collate(chosen)
                optimizer.zero_grad(set_to_none=True)
                result = model(obs, config.model.outer_cycles)
                block_calls += result.block_calls
                layer_executions += result.transformer_layer_executions
                loss = maze_valid_set_loss(result.prediction.logits, target, obs.valid_nodes)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite gradient norm")
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise FloatingPointError("nonfinite parameters after optimizer step")
                steps = index + 1
                elapsed = (time.perf_counter() - tick) * 1000
                training_ms += elapsed
                memory = memory_snapshot()
                _append(stream, {"schema_version": 1, "synthetic": False, "record_kind": "static_training_step",
                    "run_id": job_id, "seed": config.seed, "step": steps, "loss": loss.item(),
                    "gradient_norm_before_clipping": norm.item(), "gradient_clip_max_norm": 1.0,
                    "root_ids": [e.root_id for e in chosen], "frame_indices": [0] * len(chosen),
                    "batch_size": len(chosen), "outer_cycles": config.model.outer_cycles,
                    "block_calls": result.block_calls, "transformer_layer_executions": result.transformer_layer_executions,
                    "elapsed_ms_including_collation": elapsed, "memory": memory})
                del result, loss, obs, target
                check_limits(deadline, config.training.memory_limit_mb)
        checkpoint_sha = save_checkpoint(output / "checkpoint.pt", model, optimizer, config, steps, hashes, provenance)
        measure("post")
        status = "completed"
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        status = "stopped" if isinstance(exc, ResourceLimitReached) else "failed"
        raise
    finally:
        cost = ledger.reconcile(job_id, 0)
        summary.update({"schema_version": 1, "synthetic": False, "record_kind": "static_development_summary",
            "run_id": job_id, "status": status, "failure": failure, "finished_at": datetime.now(timezone.utc).isoformat(),
            "seed": config.seed, "optimizer_steps": steps, "requested_optimizer_steps": config.training.steps,
            "training_ms": training_ms, "training_examples_per_second": steps * config.training.batch_size / (training_ms / 1000) if training_ms else None,
            "wall_ms": (time.perf_counter() - start) * 1000, "max_minutes": max_minutes,
            "memory": memory_snapshot(), "memory_limit_mb": config.training.memory_limit_mb,
            "total_forward_block_calls": block_calls, "total_forward_transformer_layer_executions": layer_executions,
            "block_count_convention": "batched forward invocations; backward executes additional uncounted derivative work, included in training wall time",
            "checkpoint_sha256": checkpoint_sha, "config_sha256": config.digest(), "split_hashes": hashes,
            "dataset_dir": str((output / "dataset").resolve()), "source_tree_sha256": provenance.get("source_tree_sha256"),
            "exact_resume_supported": False, "checkpoint_lineage": None, "evaluation": "train and val frame zero only; no test evaluation; no warmup",
            "external_cost_usd": 0, "electricity_cost_usd": None, "ledger": cost,
            "cuda": "not tested", "mps": "not tested"})
        summary["artifact_hashes"] = {name: file_hash(output / name) for name in artifacts
                                     if name != "summary.json" and (output / name).is_file()}
        _write_json(output / "summary.json", summary)
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
    return summary

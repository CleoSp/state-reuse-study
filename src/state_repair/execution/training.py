"""Exact optimizer-boundary resume, independent of the research loss function."""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import random
import signal
import threading
import time
from typing import Any, Protocol

import numpy as np
import torch

from state_repair.provenance import file_hash
from .durable import append_jsonl, atomic_write, digest_json, repair_jsonl


class Trainer(Protocol):
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    schedule: list[tuple[int, int]]

    def step(self, index: int, batch: int, k: int) -> dict[str, Any]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class Shutdown:
    """Request a checkpoint at the next atomic optimizer boundary.

    A Windows close/shutdown callback waits briefly for the main thread to
    persist it. The OS can still impose a shorter shutdown timeout: hard-cut
    recovery remains mandatory. Never serialize tensors concurrently with an
    optimizer update.
    """

    def __init__(self) -> None:
        self.requested = threading.Event()
        self.saved = threading.Event()
        self.previous: dict[int, Any] = {}
        self.callback: Any = None

    def __enter__(self) -> Shutdown:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.signal(signum, lambda *_: self.requested.set())
        if os.name == "nt":
            handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

            def handler(event: int) -> bool:
                if event not in (0, 1, 2, 5, 6):
                    return False
                self.requested.set()
                self.saved.wait(4)
                return True

            self.callback = handler_type(handler)
            if not ctypes.windll.kernel32.SetConsoleCtrlHandler(self.callback, True):
                raise OSError("SetConsoleCtrlHandler failed")
        return self

    def __exit__(self, *args: object) -> None:
        self.saved.set()
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)
        if self.callback is not None:
            ctypes.windll.kernel32.SetConsoleCtrlHandler(self.callback, False)


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA resume requires the original CUDA environment")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def load_resume(path: Path, config: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != 1 or payload["job_sha256"] != digest_json(config):
        raise ValueError("resume checkpoint configuration mismatch")
    if payload["step"] != payload["schedule_position"] or payload["step"] > len(payload["schedule"]):
        raise ValueError("resume checkpoint schedule mismatch")
    return payload


def run_training(trainer: Trainer, config: dict, output: Path, segment_cap: float,
                 shutdown: Shutdown, *, checkpoint_steps: int = 500,
                 checkpoint_seconds: float = 300) -> dict:
    if not 0 < checkpoint_steps <= 500 or not 0 < checkpoint_seconds <= 300:
        raise ValueError("checkpoint interval exceeds five minutes or 500 steps")
    segment_deadline = time.monotonic() + segment_cap
    step_path, checkpoint = output / "steps.jsonl", output / "resume.pt"
    events = output / "execution.jsonl"
    repair_jsonl(step_path, events)
    step, calls, accumulated = 0, 0, 0.0
    if checkpoint.exists():
        payload = load_resume(checkpoint, config)
        if payload["schedule"] != trainer.schedule:
            raise ValueError("deterministic schedule changed")
        raw = step_path.read_bytes() if step_path.exists() else b""
        prefix = raw[:payload["log_bytes"]]
        if len(prefix) != payload["log_bytes"] or hashlib.sha256(prefix).hexdigest() != payload["log_sha256"]:
            raise ValueError("checkpoint step-log prefix mismatch")
        if len(raw) > len(prefix):
            append_jsonl(events, {"event": "rollback_steps_to_checkpoint", "removed_bytes": len(raw)-len(prefix),
                                   "preserved_in_quarantine": True})
            atomic_write(step_path, lambda handle: handle.write(prefix))
        trainer.model.load_state_dict(payload["model"])
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.load_state_dict(payload["trainer"])
        restore_rng(payload["rng"])
        step, calls, accumulated = payload["step"], payload["forward_calls"], payload["training_seconds"]
        append_jsonl(events, {"event": "resume", "checkpoint_sha256": file_hash(checkpoint),
                               "step": step, "unix": time.time()})
    elif step_path.exists() and step_path.stat().st_size:
        raise ValueError("nonempty step log without checkpoint")
    started = time.monotonic()
    last_checkpoint = started

    def save() -> None:
        nonlocal last_checkpoint
        raw = step_path.read_bytes() if step_path.exists() else b""
        payload = {"schema": 1, "job_sha256": digest_json(config), "synthetic": config["synthetic"],
                   "model": trainer.model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
                   "trainer": trainer.state_dict(), "rng": rng_state(), "step": step,
                   "schedule": trainer.schedule, "remaining_batch_order": trainer.schedule[step:],
                   "schedule_position": step, "forward_calls": calls,
                   "training_seconds": accumulated + time.monotonic() - started,
                   "log_bytes": len(raw), "log_sha256": hashlib.sha256(raw).hexdigest()}
        atomic_write(checkpoint, lambda handle: torch.save(payload, handle))
        last_checkpoint = time.monotonic()

    save()
    while step < len(trainer.schedule):
        if shutdown.requested.is_set():
            save()
            shutdown.saved.set()
            raise InterruptedError("graceful shutdown checkpoint saved")
        if time.monotonic() + config.get("unit_reserve_seconds", 0) >= segment_deadline:
            save()
            raise TimeoutError("cumulative job wall-time cap reached")
        if isinstance(trainer, SyntheticTrainer) and not trainer.ready(step):
            if time.monotonic() - last_checkpoint >= checkpoint_seconds:
                save()
                atomic_write(output / "heartbeat", lambda handle: handle.write(str(time.time()).encode()))
            shutdown.requested.wait(.1)
            if (output.parent / "STOP").exists():
                shutdown.requested.set()
            continue
        batch, k = trainer.schedule[step]
        row = trainer.step(step, batch, k)
        step += 1
        calls += row["forward_calls"]
        append_jsonl(step_path, {**row, "step": step, "batch": batch, "K": k, "synthetic": config["synthetic"]})
        if (output.parent / "STOP").exists():
            shutdown.requested.set()
        if step % checkpoint_steps == 0 or time.monotonic() - last_checkpoint >= checkpoint_seconds:
            save()
        atomic_write(output / "heartbeat", lambda handle: handle.write(str(time.time()).encode()))
    save()
    if time.monotonic() >= segment_deadline:
        raise TimeoutError("cumulative job wall-time cap reached")
    shutdown.saved.set()
    return {"steps": step, "forward_calls": calls, "synthetic": config["synthetic"],
            "training_seconds": accumulated + time.monotonic() - started,
            "final_loss": __import__("json").loads(step_path.read_text().splitlines()[-1])["loss"]}


class SyntheticTrainer:
    """Small stochastic CPU regression exercising Adam, dropout and every CPU RNG."""

    def __init__(self, config: dict):
        torch.set_num_threads(1)
        seed = config.get("seed", 17)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.2), torch.nn.Linear(8, 1))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        rng = random.Random(seed)
        self.schedule = [(rng.randrange(4), rng.choice([1, 2, 4])) for _ in range(config["steps"])]
        self.delay = config.get("step_delay", 0)
        self.sum_loss = 0.0
        self.hold_before_step = config.get("hold_before_step")
        self.initial_boot_id = os.environ.get("PAPER1_BOOT_ID")
        if self.hold_before_step is not None and (not config["synthetic"] or not self.initial_boot_id):
            raise ValueError("boot-hold fixture requires synthetic=true and a boot identity")

    def ready(self, step: int) -> bool:
        return (self.hold_before_step is None or step < self.hold_before_step
                or os.environ.get("PAPER1_BOOT_ID") != self.initial_boot_id)

    def step(self, index: int, batch: int, k: int) -> dict[str, Any]:
        x = torch.randn(4, 4) + np.random.normal() + random.random() + batch
        self.optimizer.zero_grad(set_to_none=True)
        loss = sum(self.model(x).square().mean() for _ in range(k)) / k
        loss.backward()
        self.optimizer.step()
        self.sum_loss += loss.item()
        if self.delay:
            time.sleep(self.delay)
        return {"loss": loss.item(), "forward_calls": k}

    def state_dict(self) -> dict[str, Any]:
        return {"sum_loss": self.sum_loss, "initial_boot_id": self.initial_boot_id}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.sum_loss = state["sum_loss"]
        self.initial_boot_id = state.get("initial_boot_id")

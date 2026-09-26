"""Small synthetic backend checks; no training or global changes."""
from __future__ import annotations

import csv
import io
import os
import platform
import sys
import subprocess
import time
from typing import Any

import torch

from state_repair.accounting.resources import memory_snapshot


def _nvidia_inventory() -> dict[str, Any]:
    """A physical GPU can exist even when the installed wheel is CPU only."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10, check=True,
        )
        return {"devices": [
            {"name": name.strip(), "driver_version": driver.strip(),
             "total_memory_bytes": int(total.strip()) * 1024**2,
             "available_memory_bytes": int(free.strip()) * 1024**2}
            for name, driver, total, free in csv.reader(io.StringIO(result.stdout))
        ]}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"devices": [], "measurement_error": str(exc)}


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device == "mps":
        torch.mps.synchronize()


def _probe(device: str) -> dict[str, Any]:
    record: dict[str, Any] = {"device": device, "synthetic": True,
                              "forward_backward": False}
    try:
        if device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info(device)
            capability = torch.cuda.get_device_capability(device)
            arch = f"sm_{capability[0]}{capability[1]}"
            record.update(name=torch.cuda.get_device_name(device),
                          compute_capability=list(capability),
                          compiled_architectures=torch.cuda.get_arch_list(),
                          native_arch_in_build=arch in torch.cuda.get_arch_list(),
                          available_memory_bytes=free, total_memory_bytes=total)
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize(device)
        started = time.perf_counter()
        x = torch.arange(64, dtype=torch.float32, device=device).reshape(8, 8).requires_grad_()
        weight = torch.eye(8, dtype=torch.float32, device=device) * 2
        loss = (x @ weight).sum()
        loss.backward()
        _synchronize(device)
        record["elapsed_ms"] = (time.perf_counter() - started) * 1000
        record["forward_backward"] = bool(
            torch.isfinite(loss) and x.grad is not None
            and torch.equal(x.grad, torch.full_like(x, 2)) and loss.item() == 4032
        )
        record.update(loss=loss.item(), gradient_max_abs_error=(x.grad - 2).abs().max().item(),
                      status="passed" if record["forward_backward"] else "failed")
        if device.startswith("cuda"):
            record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        elif device == "mps":
            record.update(current_allocated_bytes=torch.mps.current_allocated_memory(),
                          driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                          peak_allocated_bytes=None)
            recommended = getattr(torch.mps, "recommended_max_memory", None)
            record["recommended_max_memory_bytes"] = recommended() if recommended else None
    except (RuntimeError, AssertionError, OSError) as exc:
        record.update(status="failed", error=str(exc))
    return record


def doctor(device: str = "auto") -> dict[str, Any]:
    """CPU checks CPU only; auto checks every available backend/device.

    This does not establish solver accuracy or accelerator correctness. Failed
    probes remain in the returned report; the CLI exits nonzero for any failure.
    """
    if device not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda, or mps")
    available = {"cpu": True, "cuda": torch.cuda.is_available(),
                 "mps": torch.backends.mps.is_available()}
    if device in ("cuda", "mps") and not available[device]:
        raise ValueError(f"{device} unavailable to this PyTorch build/platform; not tested; use --device cpu or install a compatible official wheel")
    selected = next(b for b in ("cuda", "mps", "cpu") if available[b]) if device == "auto" else device
    started = time.perf_counter()
    backends: dict[str, Any] = {}
    for backend in ("cpu", "cuda", "mps"):
        record: dict[str, Any] = {"available_to_torch": available[backend],
                                  "status": "not tested", "forward_backward": False}
        if not available[backend]:
            record["reason"] = "backend unavailable to this PyTorch build/platform"
        elif backend == "cpu" or device in ("auto", backend):
            if backend == "cuda":
                probes = [_probe(f"cuda:{i}") for i in range(torch.cuda.device_count())]
                success = bool(probes) and all(p["status"] == "passed" for p in probes)
                record.update(devices=probes, status="passed" if success else "failed",
                              forward_backward=success)
            else:
                record.update(_probe(backend))
        else:
            record["reason"] = "not requested"
        backends[backend] = record
    return {
        "schema_version": 1, "synthetic": True, "record_kind": "compatibility_fixture",
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(), "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "requested_device": device, "selected_device": selected,
        "backends": backends, "nvidia_inventory": _nvidia_inventory(),
        "elapsed_ms": (time.perf_counter() - started) * 1000, "memory": memory_snapshot(),
        "diagnostic_memory_estimate_bytes": {"tensors_upper_bound": 1024**2, "cuda_runtime_allowance": 2 * 1024**3},
        "electricity_cost_usd": None, "incremental_external_charge_usd": 0,
    }

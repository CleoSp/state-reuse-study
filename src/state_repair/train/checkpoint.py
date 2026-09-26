"""Portable static checkpoint payloads; exact resume is not implemented."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from state_repair.config import ExperimentConfig
from state_repair.provenance import file_hash


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    config: ExperimentConfig, steps: int, split_hashes: dict[str, str],
                    provenance: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"checkpoint exists; refusing overwrite: {path}")
    np_state = np.random.get_state()
    payload = {
        "schema_version": 1, "mode": "static", "model_config": asdict(config.model),
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
        "numpy_rng_state": {"algorithm": np_state[0], "keys": np_state[1].tolist(),
                            "position": np_state[2], "has_gauss": np_state[3], "cached_gaussian": np_state[4]},
        "optimizer_steps": steps, "seed": config.seed, "config_sha256": config.digest(),
        "split_hashes": split_hashes, "source_provenance": provenance,
        "lineage": None, "exact_resume_supported": False,
    }
    temporary = path.with_suffix(".partial")
    if temporary.exists():
        raise FileExistsError(f"partial checkpoint exists; inspect it: {temporary}")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
    temporary.replace(path)
    return file_hash(path)

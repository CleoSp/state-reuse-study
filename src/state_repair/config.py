"""Versioned, strict experiment configuration. Paths are project-relative."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import hashlib
import json
import math
from typing import Any

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently changing an experiment."""


def _mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ValueError("configuration keys must be strings")
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


@dataclass(frozen=True)
class DatasetConfig:
    height: int = 8
    width: int = 8
    roots: int = 64
    edits_per_episode: int = 4
    extra_edge_probability: float = 0.15
    train_fraction: float = 0.5
    val_fraction: float = 0.25


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    heads: int = 2
    inner_cycles: int = 1
    outer_cycles: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 200
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    threads: int = 2
    memory_limit_mb: int = 4096
    eval_budgets: tuple[int, ...] = (0, 1, 2, 4)


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int = 1
    seed: int = 17
    device: str = "cpu"
    output_dir: str = "runs/smoke"
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    workflow: str = "static"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def _section(cls: type, value: Any) -> Any:
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**value)


def _integer(name: str, value: Any, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(name: str, value: Any, low: float, high: float | None = None) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"configuration does not exist: {path}")
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML configuration {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    raw = dict(raw)
    for name, cls in (("dataset", DatasetConfig), ("model", ModelConfig), ("training", TrainingConfig)):
        values = raw.get(name, {})
        if name == "training" and isinstance(values, dict) and "eval_budgets" in values:
            values = dict(values)
            if not isinstance(values["eval_budgets"], (list, tuple)):
                raise ValueError("eval_budgets must be a list")
            values["eval_budgets"] = tuple(values["eval_budgets"])
        raw[name] = _section(cls, values)
    cfg = _section(ExperimentConfig, raw)
    if cfg.workflow not in ("static", "repair", "confirmatory"):
        raise ValueError("workflow must be static, repair, or confirmatory")
    if type(cfg.schema_version) is not int or cfg.schema_version != 1:
        raise ValueError("supported schema_version is 1")
    _integer("seed", cfg.seed, 0)
    if cfg.device not in ("cpu", "auto", "cuda", "mps"):
        raise ValueError("device must be cpu, auto, cuda, or mps")
    if not isinstance(cfg.output_dir, str) or not cfg.output_dir.strip():
        raise ValueError("output_dir must be a nonempty string")
    for name in ("height", "width", "roots"):
        _integer(f"dataset.{name}", getattr(cfg.dataset, name), 1)
    _integer("edits_per_episode", cfg.dataset.edits_per_episode, 0)
    for name in ("extra_edge_probability", "train_fraction", "val_fraction"):
        _number(name, getattr(cfg.dataset, name), 0, 1)
    if cfg.dataset.train_fraction + cfg.dataset.val_fraction > 1 or cfg.dataset.train_fraction <= 0:
        raise ValueError("split fractions must leave nonnegative test mass and positive train mass")
    for name in ("width", "heads", "inner_cycles"):
        _integer(f"model.{name}", getattr(cfg.model, name), 1)
    _integer("outer_cycles", cfg.model.outer_cycles, 0)
    if cfg.model.width % cfg.model.heads:
        raise ValueError("model width must be divisible by heads")
    for name in ("steps", "batch_size", "threads", "memory_limit_mb"):
        _integer(f"training.{name}", getattr(cfg.training, name), 1)
    _number("learning_rate", cfg.training.learning_rate, 1e-12)
    _number("weight_decay", cfg.training.weight_decay, 0)
    if not cfg.training.eval_budgets:
        raise ValueError("eval_budgets cannot be empty")
    for budget in cfg.training.eval_budgets:
        _integer("eval_budget", budget, 0)
    if len(set(cfg.training.eval_budgets)) != len(cfg.training.eval_budgets):
        raise ValueError("eval_budgets must be unique")
    return cfg


def require_static_workflow(config: ExperimentConfig) -> None:
    """Fail before side effects for research workflows not yet implemented."""
    if config.workflow != "static":
        raise NotImplementedError(f"{config.workflow} workflow is not implemented; draft configs cannot execute")

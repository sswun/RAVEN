"""Small helpers shared by the offline and online code."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Read a YAML config and apply ``key=value`` overrides (dotted keys allowed)."""
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key=value, got {item!r}")
        value = yaml.safe_load(raw)
        node = config
        *parents, leaf = key.split(".")
        for name in parents:
            node = node.setdefault(name, {})
        node[leaf] = value
    return config


def namespace(config: dict[str, Any]) -> SimpleNamespace:
    """Shallow attribute view of a config dictionary."""
    return SimpleNamespace(**config)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class MetricLogger:
    """Appends ``{"step", "key", "value"}`` records to a JSON-lines file."""

    def __init__(self, path: str | Path | None = None, echo: bool = True):
        self.path = Path(path) if path else None
        self.echo = echo
        self.latest: dict[str, float] = {}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, key: str, value: float, step: int) -> None:
        value = float(value)
        self.latest[key] = value
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps({"step": int(step), "key": key, "value": value}) + "\n")

    def info(self, message: str) -> None:
        if self.echo:
            print(message, flush=True)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

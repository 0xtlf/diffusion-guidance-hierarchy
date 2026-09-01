"""YAML-driven configuration.

Every experiment is fully specified by one YAML file.  Configs may inherit via a
top-level `_base_:` key (relative path), and any leaf may be overridden from the
CLI with `--set a.b.c=value`.  The resolved config is snapshotted into the run
directory so a result can always be traced back to the exact inputs that made it.
"""

from __future__ import annotations

import copy
import json
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _set_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_yaml(path: str | Path) -> dict:
    """Load a YAML config, resolving `_base_` inheritance chains."""
    path = Path(path).resolve()
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw
    base = load_yaml((path.parent / base_ref).resolve())
    return _deep_merge(base, raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a config, apply `key=value` overrides, and return the merged dict."""
    cfg = load_yaml(path)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        _set_path(cfg, key.strip(), _coerce(val.strip()))
    return cfg


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch so a run is reproducible from its seed."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    """Pick a device. ``auto`` prefers CUDA, then MPS, then CPU."""
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class Run:
    """A run directory holding the resolved config, metrics and artefacts."""

    dir: Path
    cfg: dict
    metrics_path: Path = field(init=False)

    def __post_init__(self) -> None:
        """Create the run directory and snapshot the resolved config into it."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.dir / "metrics.jsonl"
        snapshot = {
            "config": self.cfg,
            "git_rev": _git_rev(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.dir / "config.resolved.yaml", "w") as fh:
            yaml.safe_dump(snapshot, fh, sort_keys=False)

    def log(self, **kv: Any) -> None:
        """Append one record to metrics.jsonl. Numpy/torch scalars are coerced."""
        rec = {}
        for k, v in kv.items():
            if isinstance(v, torch.Tensor):
                v = (
                    v.detach().cpu().item()
                    if v.numel() == 1
                    else v.detach().cpu().tolist()
                )
            elif isinstance(v, np.generic):
                v = v.item()
            elif isinstance(v, np.ndarray):
                v = v.tolist()
            rec[k] = v
        with open(self.metrics_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def path(self, *parts: str) -> Path:
        """Path inside the run directory, creating parent directories as needed."""
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def make_run(cfg: dict, default_name: str) -> Run:
    """Create a run directory from the config's ``run`` section."""
    root = Path(cfg.get("run", {}).get("root", "runs"))
    name = cfg.get("run", {}).get("name") or default_name
    if cfg.get("run", {}).get("timestamp", True):
        name = f"{name}-{time.strftime('%m%d-%H%M%S')}"
    return Run(dir=root / name, cfg=cfg)


def setup(
    path: str | Path, overrides: list[str] | None = None, default_name: str = "run"
) -> tuple[dict, Run, torch.device]:
    """One-liner used by every experiment entry point."""
    cfg = load_config(path, overrides)
    seed_everything(int(cfg.get("seed", 0)))
    device = resolve_device(cfg.get("device", "auto"))
    run = make_run(cfg, default_name)
    return cfg, run, device

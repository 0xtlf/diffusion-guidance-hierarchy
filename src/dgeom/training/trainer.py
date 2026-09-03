"""A configurable training loop.

The loop does five things: draw a batch, compute the loss, clip, step, and let
callbacks run. Everything periodic lives in a Callback, and everything about the
objective lives in the model, so this class does not change when either does.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from ..models import TrainableDiffusion
from ..progress import track
from .callbacks import Callback
from .metrics import MetricTracker


@dataclass
class TrainConfig:
    """Optimiser and schedule settings for a training run."""

    n_steps: int = 60_000
    batch_size: int = 1024
    lr: float = 3e-3
    lr_final: float = 4e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    warmup: int = 1_000

    @staticmethod
    def from_cfg(d: dict) -> TrainConfig:
        """Build from a config section."""
        known = TrainConfig.__dataclass_fields__
        alias = {"steps": "n_steps"}
        clean = {alias.get(k, k): v for k, v in d.items()}
        return TrainConfig(**{k: v for k, v in clean.items() if k in known})


class EMA:
    """Exponential moving average of weights. Evaluation uses these."""

    def __init__(self, model: TrainableDiffusion, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: TrainableDiffusion) -> None:
        """Accumulate one step's scalars."""
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)


class Trainer:
    """Runs the training loop. Everything periodic is a Callback."""

    def __init__(
        self,
        model: TrainableDiffusion,
        config: TrainConfig,
        run,
        tag: str,
        callbacks: Sequence[Callback] = (),
    ) -> None:
        self.model = model
        self.config = config
        self.run = run
        self.tag = tag
        self.callbacks = list(callbacks)
        self.tracker = MetricTracker()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.ema = EMA(model, config.ema_decay)
        self.step = 0
        self._t0 = time.time()

    # ------------------------------------------------------------- properties

    @property
    def n_steps(self) -> int:
        """Total number of training steps."""
        return self.config.n_steps

    @property
    def current_lr(self) -> float:
        """Learning rate for the current step (warmup then cosine decay)."""
        c, s = self.config, self.step
        if s < c.warmup:
            return c.lr * (s + 1) / c.warmup
        t = (s - c.warmup) / max(1, c.n_steps - c.warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        return c.lr_final + (c.lr - c.lr_final) * cosine

    @property
    def steps_per_second(self) -> float:
        """Throughput since training began."""
        return (self.step + 1) / max(time.time() - self._t0, 1e-9)

    @contextlib.contextmanager
    def ema_weights(self):
        """Temporarily swap in the EMA weights."""
        backup = {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
            if k in self.ema.shadow
        }
        self.model.load_state_dict(self.ema.shadow, freeze=False)
        try:
            yield
        finally:
            self.model.load_state_dict(backup, freeze=False)

    # ------------------------------------------------------------------- loop

    def fit(self, batch_fn: Callable[[int], torch.Tensor]) -> TrainableDiffusion:
        """batch_fn(n) returns a batch of clean samples."""
        self.model.train()
        self._t0 = time.time()
        for cb in self.callbacks:
            cb.on_train_begin(self)

        steps = track(
            range(self.config.n_steps),
            desc=f"train {self.tag}",
            total=self.config.n_steps,
        )
        for step in steps:
            self.step = step
            for group in self.optimizer.param_groups:
                group["lr"] = self.current_lr

            loss = self.model.loss(batch_fn(self.config.batch_size))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.optimizer.step()
            self.ema.update(self.model)

            self.tracker.update(loss=loss, grad_norm=grad_norm)
            if hasattr(steps, "set_postfix") and step % 50 == 0:
                steps.set_postfix(
                    {"loss": f"{loss.item():.5f}", "lr": f"{self.current_lr:.2e}"},
                    refresh=False,
                )
            for cb in self.callbacks:
                cb.on_step_end(self, step)

        for cb in self.callbacks:
            cb.on_train_end(self)
        return self.model

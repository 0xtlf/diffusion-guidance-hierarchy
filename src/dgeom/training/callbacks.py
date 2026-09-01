"""Trainer callbacks.

Everything periodic -- logging, evaluation, checkpointing -- is a callback, so
the training loop itself stays a loop and nothing else. Adding a new periodic
behaviour means adding a class here, not editing the loop.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


class Callback:
    """Hooks the Trainer calls. Override only the ones you need.

    Not an abstract base class on purpose: every hook has a sensible no-op
    default, so a callback that cares about one moment stays three lines.
    """

    def on_train_begin(self, trainer) -> None:
        """Called once before the first step."""

    def on_step_end(self, trainer, step: int) -> None:
        """Called after every optimiser step."""

    def on_train_end(self, trainer) -> None:
        """Called once after the last step."""

    def _due(self, step: int, every: int, total: int) -> bool:
        """Called after every optimiser step."""
        """Called once after the last step."""
        """Called once before the first step."""
        return every > 0 and ((step + 1) % every == 0 or step + 1 == total)


class MetricLogger(Callback):
    """Print and persist running training metrics at a fixed interval."""

    def __init__(self, every: int = 1000) -> None:
        self.every = every

    def on_step_end(self, trainer, step: int) -> None:
        """Called after every optimiser step."""
        if not self._due(step, self.every, trainer.n_steps):
            return
        stats = trainer.tracker.summary()
        stats.update(
            step=step + 1, lr=trainer.current_lr, steps_per_s=trainer.steps_per_second
        )
        trainer.run.log(stage=f"train_{trainer.tag}", **stats)
        print(
            f"[{trainer.tag}] step {step + 1:>7,}  "
            f"loss {stats.get('loss', float('nan')):.5f} "
            f"+-{stats.get('loss_std', float('nan')):.4f}  "
            f"grad {stats.get('grad_norm', float('nan')):.3f}  "
            f"lr {trainer.current_lr:.2e}  "
            f"{trainer.steps_per_second:.0f} it/s",
            flush=True,
        )
        trainer.tracker.reset()


class Evaluator(Callback):
    """Run an evaluation function on the EMA weights at a fixed interval."""

    def __init__(self, fn: Callable[[], dict], every: int = 10_000) -> None:
        self.fn = fn
        self.every = every

    def on_step_end(self, trainer, step: int) -> None:
        """Called after every optimiser step."""
        if not self._due(step, self.every, trainer.n_steps):
            return
        with trainer.ema_weights():
            trainer.model.eval()
            metrics = self.fn()
            trainer.model.train()
        trainer.run.log(stage=f"eval_{trainer.tag}", step=step + 1, **metrics)
        line = "  ".join(
            f"{k}={v:.4g}" for k, v in metrics.items() if isinstance(v, (int, float))
        )
        print(f"[{trainer.tag}] eval at {step + 1:,}: {line}", flush=True)


class Checkpointer(Callback):
    """Persist model and EMA weights at a fixed interval and at the end."""

    def __init__(self, every: int = 20_000) -> None:
        self.every = every

    def on_step_end(self, trainer, step: int) -> None:
        """Called after every optimiser step."""
        if self._due(step, self.every, trainer.n_steps):
            self.save(trainer, step + 1)

    def on_train_end(self, trainer) -> None:
        """Called once after the last step."""
        self.save(trainer, trainer.n_steps)

    @staticmethod
    def save(trainer, step: int) -> None:
        """Write model and EMA weights to the run directory."""
        torch.save(
            {
                "model": trainer.model.state_dict(),
                "ema": trainer.ema.shadow,
                "step": step,
            },
            trainer.run.path("ckpt", f"{trainer.tag}.pt"),
        )

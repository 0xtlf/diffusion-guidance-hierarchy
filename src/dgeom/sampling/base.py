"""Sampler interface.

A sampler consumes a DiffusionModel and produces samples. It never inspects how
the model computes its score, so the same sampler drives a trained network, a
closed-form reference, or a conditional model, unchanged.

Tempering lives HERE and not in the model, because it is a sampling choice: the
same diffusion model run with alpha = 0 targets p_data and with 0 < alpha < 2
targets the uniform measure on the manifold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from ..models import DiffusionModel


@dataclass
class Trace:
    """Diagnostics recorded along a run."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, **kv: Any) -> None:
        """Record one diagnostic entry, converting scalars off the device."""
        self.records.append(
            {
                k: (v.item() if torch.is_tensor(v) and v.numel() == 1 else v)
                for k, v in kv.items()
            }
        )

    def last(self) -> dict[str, Any]:
        """Most recent entry, or an empty dict."""
        return self.records[-1] if self.records else {}

    def column(self, key: str) -> list[Any]:
        """All values recorded under `key`."""
        return [r[key] for r in self.records if key in r]


class Sampler(ABC):
    """Base class for sampling schemes."""

    @abstractmethod
    def sample(
        self,
        model: DiffusionModel,
        x: torch.Tensor,
        *,
        probe: Callable[[torch.Tensor, int], dict] | None = None,
        generator=None,
    ) -> tuple[torch.Tensor, Trace]:
        """Evolve x under the model. Returns the final state and a trace.

        ``probe`` is called as ``probe(x, step)`` whenever the sampler records a
        trace entry, and whatever it returns is merged into that entry. The step
        is passed so a probe can write step-stamped artifacts of its own.
        """

    def __repr__(self) -> str:
        """Readable one-line summary."""
        fields = ", ".join(
            f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_")
        )
        return f"{type(self).__name__}({fields})"

"""Noise schedules.

The schedule owns everything about sigma: its range, how training draws it, and
nothing else. Keeping it separate from the model means an experiment can change
the noise regime without touching the network or the objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class NoiseSchedule:
    """Log-spaced sigma, optionally biased toward the small end.

    bias = 1 is log-uniform. bias > 1 concentrates draws near sigma_min, which is
    where the quality gates live and where the score is hardest to learn; a
    log-uniform schedule spends most of its budget on easy noise levels that no
    downstream result depends on.
    """

    sigma_min: float = 0.01
    sigma_max: float = 10.0
    bias: float = 1.0

    def sample(
        self, n: int, *, generator=None, dtype=torch.float32, device=None
    ) -> torch.Tensor:
        """Draw samples."""
        u = torch.rand(n, dtype=dtype, generator=generator)
        if self.bias != 1.0:
            u = u**self.bias
        lo, hi = math.log(self.sigma_min), math.log(self.sigma_max)
        return (hi + u * (lo - hi)).exp().to(device)

    def clamp(self, sigma: float) -> float:
        """Warn-free clip into the trained range.

        Evaluating the model outside [sigma_min, sigma_max] means extrapolating
        the sigma-embedding, which is not a small error: annealing a corrector
        below sigma_min once drove the distance-to-manifold from 0.028 to 0.652.
        """
        return float(min(max(sigma, self.sigma_min), self.sigma_max))

    def contains(self, sigma: float) -> bool:
        """True if sigma lies inside the trained range."""
        return self.sigma_min <= sigma <= self.sigma_max

    @staticmethod
    def from_cfg(cfg: dict) -> NoiseSchedule:
        """Build from a config section."""
        d = cfg.get("diffusion", cfg)
        return NoiseSchedule(
            sigma_min=float(d["sigma_min"]),
            sigma_max=float(d["sigma_max"]),
            bias=float(d.get("sigma_bias", 1.0)),
        )

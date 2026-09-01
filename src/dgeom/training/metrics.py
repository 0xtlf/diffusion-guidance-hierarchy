"""Running metric accumulation for the training loop."""

from __future__ import annotations

from collections import defaultdict

import torch


class MetricTracker:
    """Accumulates scalars between logging intervals.

    Reports mean and standard deviation rather than the latest value: a single
    step's loss is dominated by which sigmas the batch happened to draw, and the
    spread is itself informative about gradient noise.
    """

    def __init__(self) -> None:
        self._sum: dict[str, float] = defaultdict(float)
        self._sq: dict[str, float] = defaultdict(float)
        self._n: dict[str, int] = defaultdict(int)

    def update(self, **kv: float) -> None:
        """Accumulate one step's scalars."""
        for k, v in kv.items():
            v = float(v.detach()) if torch.is_tensor(v) else float(v)
            self._sum[k] += v
            self._sq[k] += v * v
            self._n[k] += 1

    def mean(self, key: str) -> float:
        """Running mean of a tracked scalar."""
        return self._sum[key] / max(self._n[key], 1)

    def std(self, key: str) -> float:
        """Running standard deviation of a tracked scalar."""
        n = max(self._n[key], 1)
        var = self._sq[key] / n - (self._sum[key] / n) ** 2
        return max(var, 0.0) ** 0.5

    def summary(self) -> dict[str, float]:
        """Means and standard deviations of everything tracked."""
        out: dict[str, float] = {}
        for k in self._sum:
            out[k] = self.mean(k)
            out[f"{k}_std"] = self.std(k)
        return out

    def reset(self) -> None:
        """Clear all accumulators."""
        self._sum.clear()
        self._sq.clear()
        self._n.clear()

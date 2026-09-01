"""Reusable building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Pre-norm residual block: LN -> SiLU -> Linear -> SiLU -> Linear."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return h + self.net(h)

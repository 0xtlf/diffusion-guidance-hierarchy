"""MLP backbone: maps a conditioned input vector to features."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..layers import ResidualBlock


class MLPBackbone(nn.Module):
    """Residual MLP trunk. Returns features, not predictions."""

    def __init__(self, in_dim: int, width: int = 256, depth: int = 4) -> None:
        super().__init__()
        self.width = width
        self.stem = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(depth))
        self.norm = nn.Sequential(nn.LayerNorm(width), nn.SiLU())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = self.stem(z)
        for block in self.blocks:
            h = block(h)
        return self.norm(h)

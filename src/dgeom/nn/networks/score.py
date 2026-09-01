"""Score network: predicts the hat-space score of a noised sample.

Output is shat = sigma^2 * grad log p_sigma(x), which by Tweedie equals
E[x0 | x] - x -- the displacement from x to its denoised point. That target is
bounded by the tube radius instead of blowing up like sigma^-1, which is what
keeps training stable across several decades of sigma.

The network is deliberately generic: it is given no hint that the data lies on a
manifold (no normalisation of the output, no geometric parameterisation), because
learning the manifold is exactly what is being tested.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..backbones import MLPBackbone
from ..layers import FourierSigmaEmbedding


class ScoreNetwork(nn.Module):
    """MLP that predicts the hat-space score of a noised sample."""

    def __init__(
        self, dim: int = 4, width: int = 256, depth: int = 4, emb_dim: int = 64
    ) -> None:
        super().__init__()
        self.dim = dim
        self.embed = FourierSigmaEmbedding(emb_dim)
        self.backbone = MLPBackbone(dim + self.embed.out_dim, width, depth)
        self.head = nn.Linear(width, dim)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if sigma.ndim == 0:
            sigma = sigma.expand(x.shape[0])
        h = self.backbone(torch.cat([x, self.embed(sigma)], dim=-1))
        return self.head(h)

    @property
    def n_params(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

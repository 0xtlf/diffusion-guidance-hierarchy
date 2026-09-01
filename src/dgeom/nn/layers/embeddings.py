"""Conditioning embeddings."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierSigmaEmbedding(nn.Module):
    """Random Fourier features of log(sigma).

    log-sigma rather than sigma: the schedule is log-uniform over several
    decades, so features of sigma itself would spend all their resolution at the
    top of the range and none at sigma_min, which is the regime that matters.
    """

    def __init__(self, dim: int = 64, scale: float = 4.0) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("embedding dim must be even")
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    @property
    def out_dim(self) -> int:
        """Width of the output this layer produces."""
        return self.freqs.numel() * 2

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        ang = 2 * math.pi * sigma.reshape(-1, 1).log() * self.freqs.unsqueeze(0)
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

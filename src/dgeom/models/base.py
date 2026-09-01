"""The diffusion model interface.

A diffusion model is three things: a forward noising process, a way to evaluate
the score of the noised density, and (if trainable) an objective. Everything in
this project talks to a model through this interface, so samplers, metrics and
experiments never care whether the score came from a network, a closed form, or
a numerical quadrature.

Everything is expressed in HAT SPACE:

    shat(x, sigma) = sigma^2 * grad_x log p_sigma(x) = E[x0 | x] - x

the Tweedie displacement. It is O(1) near the manifold where the raw score blows
up like sigma^-2, which keeps float32 safe at sigma_min, and it makes the
theory's tolerance directly readable -- raw score error o(sigma^-2) is exactly
hat error o(1).

Subclassing contract: a subclass implements `shat` and nothing else is required.
That is deliberate. Conditional variants (classifier guidance, for instance) add
a term to the score and therefore need only override `shat`, inheriting the
forward process, Tweedie denoising and every sampler unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .schedule import NoiseSchedule


def broadcast_sigma(sigma, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar or per-sample sigma to shape (B, 1)."""
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, dtype=x.dtype, device=x.device)
    sigma = sigma.to(device=x.device, dtype=x.dtype)
    if sigma.ndim == 0:
        sigma = sigma.expand(x.shape[0])
    return sigma.reshape(-1, 1)


class DiffusionModel(ABC):
    """Variance-exploding diffusion over a d-dimensional space."""

    def __init__(self, dim: int, schedule: NoiseSchedule) -> None:
        self.dim = int(dim)
        self.schedule = schedule

    # ------------------------------------------------------------ the contract

    @abstractmethod
    def shat(self, x: torch.Tensor, sigma) -> torch.Tensor:
        """sigma^2 * grad_x log p_sigma(x). The only method a subclass must give."""

    # ------------------------------------------------ derived, never overridden

    def score(self, x: torch.Tensor, sigma) -> torch.Tensor:
        """The raw score. Diverges like sigma^-2; prefer shat."""
        return self.shat(x, sigma) / broadcast_sigma(sigma, x) ** 2

    def denoise(self, x: torch.Tensor, sigma) -> torch.Tensor:
        """E[x0 | x_sigma = x], by Tweedie's identity."""
        return x + self.shat(x, sigma)

    def add_noise(self, x0: torch.Tensor, sigma, noise: torch.Tensor | None = None):
        """Forward process x_sigma = x0 + sigma * z. Returns (x, noise)."""
        if noise is None:
            noise = torch.randn_like(x0)
        return x0 + broadcast_sigma(sigma, x0) * noise, noise

    def __repr__(self) -> str:
        """Readable one-line summary."""
        s = self.schedule
        return (
            f"{type(self).__name__}(dim={self.dim}, "
            f"sigma=[{s.sigma_min:g}, {s.sigma_max:g}])"
        )


class TrainableDiffusion(DiffusionModel):
    """A diffusion model backed by parameters, with a training objective."""

    @abstractmethod
    def parameters(self):
        """Parameters to optimise."""

    @abstractmethod
    def loss(self, x0: torch.Tensor, *, generator=None) -> torch.Tensor:
        """Scalar training loss for a batch of clean samples."""

    def train(self) -> None:
        """Switch to training mode. Stateless models may ignore this."""

    def eval(self) -> None:
        """Switch to evaluation mode. Stateless models may ignore this."""

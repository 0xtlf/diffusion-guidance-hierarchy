"""Densities on a manifold.

A density is defined by its log-density with respect to the manifold's own
VOLUME measure, so "uniform" is log p = 0 and nothing else needs to know about
Jacobians.

The mixture used throughout is von Mises-Fisher in the AMBIENT space, restricted
to the manifold:

    log p(x) = logsumexp_k [ log w_k + kappa_k <mu_k, x> ]

On the sphere this is exactly the standard vMF, and it keeps the closed form that
makes the analytic reference score possible. On any other manifold it is still
smooth and well defined -- which matters on the Klein bottle, where a density
written directly in chart coordinates would generically be DISCONTINUOUS across
the identification Phi(u + 2pi, v) = Phi(u, -v). Defining it on the embedded
point sidesteps that entirely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Density(ABC):
    """log-density with respect to the manifold's volume measure, up to a constant."""

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log-density at each point, up to an additive constant.

        Args:
            x: ``(B, d)`` points lying on the manifold.

        Returns:
            ``(B,)`` log-densities with respect to the volume measure.
        """

    @property
    def is_uniform(self) -> bool:
        """True if this is the volume measure, which loaders can sample directly."""
        return False

    def __repr__(self) -> str:
        """Short name of the density."""
        return f"{type(self).__name__}()"


class UniformDensity(Density):
    """The volume measure itself: constant log-density."""

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Zero everywhere, since the density is constant by definition."""
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

    @property
    def is_uniform(self) -> bool:
        """Always True."""
        return True


class VonMisesFisherMixture(Density):
    """Mixture of ambient von Mises-Fisher kernels restricted to a manifold."""

    def __init__(
        self,
        means: torch.Tensor,
        concentrations: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        self.means = means / means.norm(dim=-1, keepdim=True)
        self.concentrations = concentrations
        w = (
            torch.ones(len(concentrations), dtype=means.dtype)
            if weights is None
            else weights
        )
        self.weights = w / w.sum()

    @property
    def n_components(self) -> int:
        """Number of mixture components."""
        return self.means.shape[0]

    @property
    def d(self) -> int:
        """Ambient dimension of the mixture means."""
        return self.means.shape[-1]

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log-density of the mixture at each point, up to a constant.

        Args:
            x: ``(B, d)`` points on the manifold.

        Returns:
            ``(B,)`` values of ``logsumexp_k[log w_k + kappa_k <mu_k, x>]``.
        """
        terms = self.weights.log() + self.concentrations * (
            x.unsqueeze(-2) * self.means
        ).sum(-1)
        return torch.logsumexp(terms, dim=-1)

    def to(self, dtype=None, device=None) -> VonMisesFisherMixture:
        """Return a copy with parameters moved to a dtype and/or device."""
        return VonMisesFisherMixture(
            self.means.to(dtype=dtype, device=device),
            self.concentrations.to(dtype=dtype, device=device),
            self.weights.to(dtype=dtype, device=device),
        )

    @staticmethod
    def random(
        n_components: int,
        d: int,
        kappa_range: tuple[float, float] = (1.0, 3.0),
        *,
        dtype=torch.float64,
        generator=None,
    ) -> VonMisesFisherMixture:
        """Draw a random instance, reproducible from the generator."""
        means = torch.randn(n_components, d, dtype=dtype, generator=generator)
        lo, hi = kappa_range
        kappa = lo + (hi - lo) * torch.rand(
            n_components, dtype=dtype, generator=generator
        )
        weights = torch.rand(n_components, dtype=dtype, generator=generator) + 0.5
        return VonMisesFisherMixture(means, kappa, weights)

    def __repr__(self) -> str:
        """Component count and concentrations."""
        k = [round(float(v), 2) for v in self.concentrations]
        return f"VonMisesFisherMixture(K={self.n_components}, kappa={k})"

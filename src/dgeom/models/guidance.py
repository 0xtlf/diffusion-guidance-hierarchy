"""Classifier guidance for a linear constraint, in closed form.

Conditioning on ``<w, x_0> = 0`` is the tractable model of a conditional
submanifold. The constraint is LINEAR, and that is what makes the guidance exact
rather than learned:

    m_c = E[<w, x_0> | x_t] = <w, E[x_0 | x_t]> = <w, x_t + shat(x_t, sigma)>

because expectation commutes with a linear map, and Tweedie gives the posterior
mean as the hat displacement the base model already returns. Nothing is trained.
Under the usual Gaussian-posterior approximation ``p(c | x_t) ~ N(0; m_c, v_c)``,

    grad log p(c | x_t) = -(m_c / v_c) * grad_x m_c

Two approximations remain, and both are switchable so their cost is measured
rather than assumed: ``v_c`` (the posterior variance of the constraint) and the
Jacobian in ``grad_x m_c = (I + d shat/dx)^T w``, dropped to ``w`` by default.

Note the scale. With ``v_c = sigma^2`` the hat-space guidance is just ``-m_c w``,
which is ``Theta(1)`` -- the SAME order as the geometry term. That is the claim
the experiment tests: conditioning on a measure-zero linear constraint adds
codimension and behaves like geometry, not like density. A learned softmax head
cannot produce this scale, which is why an exact reference matters.
"""

from __future__ import annotations

import torch

from .base import DiffusionModel, broadcast_sigma


class GuidedDiffusion(DiffusionModel):
    """A base model plus the exact guidance toward ``<w, x> = 0``.

    Only ``shat`` is overridden, so every sampler, metric and figure that works
    on an unconditional model works on this unchanged.
    """

    def __init__(
        self,
        base: DiffusionModel,
        hyperplane,
        gamma: float = 1.0,
        variance: str = "sigma2",
        normalise: bool = False,
        manifold=None,
    ) -> None:
        super().__init__(base.dim, base.schedule)
        self.base, self.hyperplane = base, hyperplane
        self.gamma, self.variance = float(gamma), str(variance)
        self.normalise, self.manifold = bool(normalise), manifold
        if normalise and manifold is None:
            raise ValueError("normalise=True needs a manifold to build P_T w")

    def constraint_mean(self, x: torch.Tensor, sigma, base=None) -> torch.Tensor:
        """``m_c = <w, x + shat>``: the exact posterior mean of ``<w, x_0>``.

        ``base`` lets a caller pass an already-computed ``shat``; the guided score
        needs it anyway, and evaluating the network twice per step halved the
        sampling rate before this was threaded through.
        """
        w = self.hyperplane.w.to(device=x.device, dtype=x.dtype)
        b = self.base.shat(x, sigma) if base is None else base
        return (x + b) @ w - self.hyperplane.b

    def constraint_variance(self, x: torch.Tensor, sigma) -> torch.Tensor:
        """``v_c = Var[<w, x_0> | x_t]``, approximated.

        ``sigma2`` takes the posterior covariance as ``sigma^2 I``, which is exact
        for a flat manifold and the standard first-order choice. ``tangent`` scales
        it by ``|P_T w|^2``, recognising that the posterior is concentrated ALONG
        the manifold, so only the tangential part of ``w`` carries variance.
        """
        s = broadcast_sigma(sigma, x).squeeze(-1)
        if self.variance == "sigma2":
            return s**2
        if self.variance == "tangent":
            return (s**2 * self._tangential(x) ** 2).clamp_min(1e-30)
        raise ValueError(f"unknown variance model {self.variance!r}")

    def _tangential(self, x: torch.Tensor) -> torch.Tensor:
        """``|P_{T_x M} w|``, the co-area factor. Constant 1 on a sphere section."""
        w = self.hyperplane.w.to(device=x.device, dtype=x.dtype)
        t = self.manifold.tangent_basis(x)
        return torch.einsum("bnd,d->bn", t, w).norm(dim=-1).clamp_min(1e-12)

    def ghat(self, x: torch.Tensor, sigma, base=None) -> torch.Tensor:
        """Hat-space guidance ``sigma^2 * grad log p(c | x)``.

        Returned separately from ``shat`` so the two can be compared directly --
        their ratio is the number that decides whether guidance sits at the
        geometry rate or the density rate.
        """
        w = self.hyperplane.w.to(device=x.device, dtype=x.dtype)
        s2 = broadcast_sigma(sigma, x) ** 2
        m = self.constraint_mean(x, sigma, base=base)
        v = self.constraint_variance(x, sigma)
        coef = -(m / v)
        if self.normalise:
            # equalise the restoring stiffness along N: without this the drift
            # toward N is proportional to |P_T w|^2, which varies 7x on the Klein
            # bottle and is constant on the sphere
            coef = coef / self._tangential(x) ** 2
        return s2 * coef.unsqueeze(-1) * w

    def shat(self, x: torch.Tensor, sigma) -> torch.Tensor:
        """Guided hat score: the base score plus ``gamma`` times the guidance."""
        base = self.base.shat(x, sigma)
        return base + self.gamma * self.ghat(x, sigma, base=base)


__all__ = ["GuidedDiffusion"]

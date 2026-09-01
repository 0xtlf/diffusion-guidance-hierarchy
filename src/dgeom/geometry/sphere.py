"""The sphere S^{d-1} embedded in R^d.

Everything is exact and closed form, which is what lets the validation gates run
with no learned components at all. Wood's rejection algorithm for vMF sampling
lives here too, since it is specific to spheres.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.special import ive

from .base import Manifold


def _log_iv(order: int, x: torch.Tensor) -> torch.Tensor:
    """Log I_nu(x), stable for large x via the exponentially scaled ive."""
    xn = x.detach().cpu().double().numpy()
    val = torch.from_numpy(ive(order, xn)).to(x.device, x.dtype)
    return val.log() + x


def bessel_ratio_i0_i1(r: torch.Tensor) -> torch.Tensor:
    """I_0(r) / I_1(r), computed with exponentially scaled Bessels.

    At sigma_min the argument reaches ~1e4, where the unscaled I_nu overflows
    float64.  ive strips the common exp(r) factor so the ratio stays finite.
    """
    rn = r.detach().cpu().double().numpy()
    ratio = ive(0, rn) / ive(1, rn)
    return torch.from_numpy(ratio).to(r.device, r.dtype)


class Sphere(Manifold):
    """The unit sphere S^{d-1}."""

    name = "sphere"

    def __init__(self, dim_ambient: int = 4) -> None:
        self._d = int(dim_ambient)

    @property
    def d(self) -> int:
        """Ambient dimension."""
        return self._d

    @property
    def n(self) -> int:
        """Intrinsic dimension: a sphere has codimension 1."""
        return self._d - 1

    def project(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """P_M(x) = x / |x|, the nearest point on the manifold."""
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    def dist_sq(self, x: torch.Tensor) -> torch.Tensor:
        """dist(x, M)^2 = (|x| - 1)^2."""
        return (x.norm(dim=-1) - 1.0) ** 2

    def tangent_project(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project v onto T_{P_M(x)} M."""
        u = self.project(x)
        return v - (v * u).sum(-1, keepdim=True) * u

    def sample_uniform(
        self, n: int, *, device=None, dtype=torch.float64, generator=None
    ) -> torch.Tensor:
        """Draw exactly uniformly on the sphere by normalising Gaussians.

        Args:
            n: number of samples.
            device: destination device.
            dtype: floating point type.
            generator: RNG for reproducibility.

        Returns:
            ``(n, d)`` points with unit norm.
        """
        g = torch.randn(n, self.d, device=device, dtype=dtype, generator=generator)
        return g / g.norm(dim=-1, keepdim=True)

    def tangent_basis(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n, d) orthonormal basis of the tangent space at P_M(x).

        The tangent space of a sphere is just the orthogonal complement of the
        radius, so a QR of I - u u^T gives it directly.
        """
        u = self.project(x)
        eye = torch.eye(self.d, dtype=x.dtype, device=x.device).expand(
            x.shape[0], self.d, self.d
        )
        proj = eye - torch.einsum("bi,bj->bij", u, u)
        q, _ = torch.linalg.qr(proj.transpose(-1, -2), mode="complete")
        overlap = torch.einsum("bi,bid->bd", u, q).abs()
        idx = overlap.argsort(dim=-1)[:, : self.n]
        return torch.gather(
            q.transpose(-1, -2), 1, idx.unsqueeze(-1).expand(-1, -1, self.d)
        )

    def uniform_marginals(self) -> list[tuple]:
        """Projections onto fixed axes.

        For uniform on ``S^{d-1}`` the marginal of ``<u, x>`` has density
        proportional to ``(1 - t^2)^{(d-3)/2}``; for ``d = 4`` that is
        ``(2/pi) sqrt(1 - t^2)``, independent of the direction ``u``.
        """
        import math

        power = (self.d - 3) / 2
        norm = math.pi / 2 if self.d == 4 else None

        def make(k: int):
            def project(x: torch.Tensor) -> torch.Tensor:
                return self.project(x)[..., k]

            def pdf(t):
                import numpy as np

                dens = np.clip(1 - t**2, 0, None) ** power
                return dens / norm if norm else dens

            return (f"<e{k}, x>", project, pdf, (-1.0, 1.0))

        return [make(k) for k in range(min(3, self.d))]

    def chart_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Hyperspherical angles of P_M(x). For d=4: (theta1, theta2, phi).

        Only used for coverage binning, so the coordinate singularities at the
        poles are harmless.
        """
        u = self.project(x)
        d = self.d
        angs = []
        for k in range(d - 2):
            r = u[..., k:].norm(dim=-1).clamp_min(1e-30)
            angs.append(torch.arccos((u[..., k] / r).clamp(-1, 1)))
        angs.append(torch.atan2(u[..., -1], u[..., -2]) % (2 * torch.pi))
        return torch.stack(angs, dim=-1)


@dataclass
class VMFMixture:
    """vMF mixture on S^{d-1}, with exact sampling and normalisation.

    Kept sphere-specific on purpose: vMF and a Gaussian restricted to the sphere
    share an exponential-linear integrand, so the smoothed density p_sigma has a
    closed form (models/references.py) that exists on no other manifold. The
    generic, manifold-agnostic density is geometry/densities.py.
    """

    mus: torch.Tensor  # (K, d) unit vectors
    kappas: torch.Tensor  # (K,)
    weights: torch.Tensor  # (K,) simplex

    def __post_init__(self) -> None:
        """Normalise the mean directions and the mixture weights."""
        self.mus = self.mus / self.mus.norm(dim=-1, keepdim=True)
        self.weights = self.weights / self.weights.sum()

    @property
    def d(self) -> int:
        """Ambient dimension."""
        return self.mus.shape[-1]

    @property
    def n_components(self) -> int:
        """Number of mixture components."""
        return self.mus.shape[0]

    def to(self, device=None, dtype=None) -> VMFMixture:
        """Return a copy with parameters moved to a device and/or dtype."""
        return VMFMixture(
            self.mus.to(device=device, dtype=dtype),
            self.kappas.to(device=device, dtype=dtype),
            self.weights.to(device=device, dtype=dtype),
        )

    @staticmethod
    def random(
        K: int = 3,
        d: int = 4,
        kappa_range: tuple[float, float] = (5.0, 20.0),
        *,
        dtype=torch.float64,
        generator=None,
    ) -> VMFMixture:
        """Draw a random vMF mixture with uniform means and concentrations."""
        mus = torch.randn(K, d, dtype=dtype, generator=generator)
        lo, hi = kappa_range
        kappas = lo + (hi - lo) * torch.rand(K, dtype=dtype, generator=generator)
        weights = torch.rand(K, dtype=dtype, generator=generator) + 0.5
        return VMFMixture(mus, kappas, weights)

    def log_norm_const(self) -> torch.Tensor:
        """Log C_d(kappa) for each component, w.r.t. surface measure."""
        d, k = self.d, self.kappas
        nu = d / 2 - 1
        return (
            nu * k.log()
            - (d / 2)
            * torch.log(torch.tensor(2 * torch.pi, dtype=k.dtype, device=k.device))
            - _log_iv(int(nu), k)
        )

    def log_prob(self, u: torch.Tensor) -> torch.Tensor:
        """Log p_data(u) for u on the sphere. Shape (..., d) -> (...)."""
        # (..., K)
        dots = u @ self.mus.T
        terms = self.weights.log() + self.log_norm_const() + self.kappas * dots
        return torch.logsumexp(terms, dim=-1)

    def sample(self, n: int, *, device=None, generator=None) -> torch.Tensor:
        """Draw exactly from the mixture using Wood's rejection algorithm.

        Args:
            n: number of samples.
            device: destination device.
            generator: RNG for reproducibility.

        Returns:
            ``(n, d)`` points on the sphere.
        """
        dtype = self.mus.dtype
        comp = torch.multinomial(
            self.weights.to("cpu"), n, replacement=True, generator=generator
        )
        out = torch.empty(n, self.d, dtype=dtype)
        for k in range(self.n_components):
            idx = (comp == k).nonzero(as_tuple=True)[0]
            if idx.numel():
                out[idx] = _sample_vmf(
                    self.mus[k].cpu(),
                    float(self.kappas[k]),
                    idx.numel(),
                    generator=generator,
                )
        return out.to(device=device)


def _sample_vmf(
    mu: torch.Tensor, kappa: float, n: int, *, generator=None
) -> torch.Tensor:
    """Wood (1994) rejection sampler for vMF on S^{d-1}."""
    d, dtype = mu.shape[-1], mu.dtype
    dm1 = d - 1
    # numerically stable form of b = (-2k + sqrt(4k^2 + (d-1)^2)) / (d-1)
    b = dm1 / (2 * kappa + (4 * kappa**2 + dm1**2) ** 0.5)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + dm1 * torch.log(torch.tensor(1 - x0**2, dtype=dtype))

    w = torch.empty(n, dtype=dtype)
    filled = 0
    beta = torch.distributions.Beta(
        torch.tensor(dm1 / 2, dtype=dtype), torch.tensor(dm1 / 2, dtype=dtype)
    )
    while filled < n:
        m = 2 * (n - filled) + 16
        z = beta.sample((m,))
        wc = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        u = torch.rand(m, dtype=dtype, generator=generator)
        keep = kappa * wc + dm1 * torch.log1p(-x0 * wc) - c >= u.log()
        acc = wc[keep]
        take = min(acc.numel(), n - filled)
        w[filled : filled + take] = acc[:take]
        filled += take

    # uniform direction in the tangent hyperplane of mu
    g = torch.randn(n, d, dtype=dtype, generator=generator)
    v = g - (g @ mu).unsqueeze(-1) * mu
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w = w.unsqueeze(-1)
    return w * mu + (1 - w**2).clamp_min(0).sqrt() * v

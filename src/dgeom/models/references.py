"""Exact diffusion models, used as ground truth for the learned ones.

Both implement the same DiffusionModel interface as the trained model, so every
metric, sampler and experiment can be pointed at either. That is what separates
"does the method work" from "is the model good enough": run the identical
pipeline on an exact score and see which answer changes.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import ive

from ..geometry.densities import VonMisesFisherMixture
from ..geometry.sphere import bessel_ratio_i0_i1
from ..progress import track
from ..registry import MODELS
from .base import DiffusionModel, broadcast_sigma
from .schedule import NoiseSchedule


class _SphereVMF:
    """Adds the sphere-only normalisation the closed form needs.

    log C_d(kappa) exists only on a sphere, so it is bolted on here rather than
    pushed onto the generic density.
    """

    def __init__(self, mixture: VonMisesFisherMixture) -> None:
        from ..geometry.sphere import VMFMixture

        self._v = VMFMixture(
            mixture.means.clone(),
            mixture.concentrations.clone(),
            mixture.weights.clone(),
        )

    def __getattr__(self, item):
        return getattr(self._v, item)


@MODELS.register("analytic")
class AnalyticDiffusion(DiffusionModel):
    """Closed-form smoothed score of a vMF mixture on S^{d-1}.

    A vMF density and a Gaussian restricted to the sphere share an
    exponential-linear integrand, so the smoothing integral collapses onto a
    modified Bessel function and the score is exact:

        shat = -x + sum_k omega_k [ I_0(r_k)/I_1(r_k) - 2/r_k ] xi_k / r_k
        xi_k = x/sigma^2 + kappa_k mu_k,   r_k = |xi_k|

    with no sigma^-2 anywhere, which is why it stays stable at sigma_min.
    """

    def __init__(self, mixture: VonMisesFisherMixture, schedule: NoiseSchedule) -> None:
        super().__init__(dim=mixture.d, schedule=schedule)
        if mixture.d != 4:
            raise NotImplementedError(
                "specialised to d=4 (nu=1); generalise the Bessel order and the "
                "(nu+1)/r term for other ambient dimensions"
            )
        self.mix = _SphereVMF(mixture)

    def _terms(self, x: torch.Tensor, sigma):
        from scipy.special import ive

        s2 = broadcast_sigma(sigma, x) ** 2
        mu, kap = self.mix.mus, self.mix.kappas
        xi = x.unsqueeze(1) / s2.unsqueeze(1) + kap.view(1, -1, 1) * mu.unsqueeze(0)
        r = xi.norm(dim=-1).clamp_min(1e-30)
        log_ive1 = (
            torch.from_numpy(ive(1, r.detach().cpu().double().numpy()))
            .to(r.device, r.dtype)
            .log()
        )
        log_term = (
            (self.mix.weights.log() + self.mix.log_norm_const()).view(1, -1)
            + log_ive1
            + r
            - r.log()
        )
        return xi, r, log_term

    def shat(self, x: torch.Tensor, sigma) -> torch.Tensor:
        """Hat-space score sigma^2 * grad log p_sigma(x), i.e. E[x0|x] - x."""
        xi, r, log_term = self._terms(x, sigma)
        omega = torch.softmax(log_term, dim=-1)
        coef = bessel_ratio_i0_i1(r) - 2.0 / r
        return -x + (omega.unsqueeze(-1) * (coef / r).unsqueeze(-1) * xi).sum(dim=1)

    def log_prob_unnorm(self, x: torch.Tensor, sigma) -> torch.Tensor:
        """Log smoothed density, up to a constant independent of x."""
        s2 = broadcast_sigma(sigma, x) ** 2
        _, _, log_term = self._terms(x, sigma)
        return -((x * x).sum(-1, keepdim=True) + 1.0).div(2 * s2).squeeze(
            -1
        ) + torch.logsumexp(log_term, dim=-1)


@MODELS.register("quadrature")
class QuadratureDiffusion(DiffusionModel):
    """Exact score of a 2-D manifold by deterministic quadrature.

    The Klein bottle has no closed-form p_sigma, but it is two-dimensional with
    an exact chart, so the smoothed density is a 2-D integral and Tweedie gives

        shat(x) = (int Phi w) / (int w) - x,
        w = exp(-|x - Phi(u,v)|^2 / 2 sigma^2) p_data sqrt(det g).

    Gauss-Legendre rather than a uniform grid: the integrand is smooth, so GL
    converges geometrically and ~96 nodes per axis reach machine precision where
    the trapezoid rule would need thousands. Monte Carlo would be noisy and
    irreproducible; this is deterministic.

    The patch is integrated LOCALLY around P_M(x), with half-widths converted
    from arclength to chart coordinates, falling back to the full chart once the
    bump is no longer local.
    """

    def __init__(
        self,
        manifold,
        loader,
        schedule: NoiseSchedule,
        n_nodes: int = 96,
        n_sigma: float = 7.0,
        chunk: int = 256,
    ) -> None:
        super().__init__(dim=manifold.d, schedule=schedule)
        self.M = manifold
        self.density = loader.density
        self.n_nodes, self.n_sigma, self.chunk = (
            int(n_nodes),
            float(n_sigma),
            int(chunk),
        )
        xs, ws = np.polynomial.legendre.leggauss(self.n_nodes)
        self._nodes = torch.tensor(xs, dtype=torch.float64)
        self._weights = torch.tensor(ws, dtype=torch.float64)

    def _patch(self, x: torch.Tensor, sigma: float):
        uv = self.M.chart_coords(self.M.project(x))
        u0, v0 = uv[..., 0], uv[..., 1]
        Pu, Pv = self.M.frame(u0, v0)
        hu = self.n_sigma * sigma / Pu.norm(dim=-1).clamp_min(1e-12)
        hv = self.n_sigma * sigma / Pv.norm(dim=-1).clamp_min(1e-12)
        return u0, v0, hu.clamp(max=math.pi), hv.clamp(max=math.pi)

    def shat(self, x: torch.Tensor, sigma) -> torch.Tensor:
        """Hat-space score sigma^2 * grad log p_sigma(x), i.e. E[x0|x] - x."""
        s = float(broadcast_sigma(sigma, x).reshape(-1)[0])
        nodes = self._nodes.to(x.device, x.dtype)
        wts = self._weights.to(x.device, x.dtype)
        out = torch.empty_like(x)
        K = self.n_nodes

        chunks = range(0, x.shape[0], self.chunk)
        if x.shape[0] > 4 * self.chunk:
            chunks = track(
                chunks,
                desc="quadrature reference",
                total=(x.shape[0] + self.chunk - 1) // self.chunk,
            )
        for i in chunks:
            xi = x[i : i + self.chunk]
            u0, v0, hu, hv = self._patch(xi, s)
            uu = u0.unsqueeze(-1) + hu.unsqueeze(-1) * nodes
            vv = v0.unsqueeze(-1) + hv.unsqueeze(-1) * nodes
            U = uu.unsqueeze(-1).expand(-1, -1, K)
            V = vv.unsqueeze(-2).expand(-1, K, -1)
            P = self.M.from_chart(U.reshape(-1), V.reshape(-1)).reshape(
                *U.shape, self.dim
            )

            d2 = ((P - xi[:, None, None, :]) ** 2).sum(-1)
            logp = self.density.log_prob(P.reshape(-1, self.dim)).reshape(U.shape)
            logw = -d2 / (2 * s * s) + logp + self.M.volume_element(V).log()
            logw = logw - logw.amax(dim=(1, 2), keepdim=True)
            w = (
                logw.exp()
                * (hu.unsqueeze(-1) * wts).unsqueeze(-1)
                * (hv.unsqueeze(-1) * wts).unsqueeze(-2)
            )

            denom = w.sum(dim=(1, 2)).clamp_min(1e-300)
            out[i : i + self.chunk] = (w.unsqueeze(-1) * P).sum(
                dim=(1, 2)
            ) / denom.unsqueeze(-1) - xi
        return out


def reference_for(manifold, loader, schedule: NoiseSchedule, **kw) -> DiffusionModel:
    """The exact model appropriate to a manifold and its data distribution."""
    if manifold.name == "sphere":
        return AnalyticDiffusion(loader.mixture, schedule)
    return QuadratureDiffusion(manifold, loader, schedule, **kw)


class UniformSphereDiffusion(DiffusionModel):
    """Exact score of the uniform measure on a unit sphere, after smoothing.

    Uniform on the unit ``S^{k-1}`` convolved with ``N(0, sigma^2 I_k)`` has

        p_sigma(x) ∝ r^-nu I_nu(r / sigma^2) exp(-r^2 / 2 sigma^2),  nu = k/2 - 1

    whose radial derivative collapses to one Bessel ratio, the ``-nu/r`` terms
    cancelling exactly:

        d/dr log p_sigma = (1 / sigma^2) [ I_{nu+1}(z) / I_nu(z) - r ],  z = r/sigma^2

    Used as the exact reference for a hyperplane SECTION, where the constraint
    makes the conditional law uniform on a great subsphere and no learned or
    approximated component is involved anywhere.
    """

    def __init__(
        self,
        dim: int,
        schedule,
        subspace_dim: int | None = None,
        normal: torch.Tensor | None = None,
    ) -> None:
        super().__init__(dim, schedule)
        self.normal = normal
        k = subspace_dim if subspace_dim is not None else dim
        self.nu = k / 2.0 - 1.0

    def _radial(self, y: torch.Tensor, sigma) -> torch.Tensor:
        s = broadcast_sigma(sigma, y)
        r = y.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        z = (r / s**2).squeeze(-1)
        zn = z.detach().cpu().double().numpy()
        ratio = ive(self.nu + 1, zn) / np.maximum(ive(self.nu, zn), 1e-300)
        ratio = torch.from_numpy(ratio).to(y.device, y.dtype).unsqueeze(-1)
        return (y / r) * (ratio - r)

    def shat(self, x: torch.Tensor, sigma) -> torch.Tensor:
        """sigma^2 * grad log p_sigma. Splits off the constrained direction."""
        if self.normal is None:
            return self._radial(x, sigma)
        w = self.normal.to(device=x.device, dtype=x.dtype)
        par = (x @ w).unsqueeze(-1)
        # along w the law is N(0, sigma^2); within w-perp it is the sphere form
        return self._radial(x - par * w, sigma) - par * w

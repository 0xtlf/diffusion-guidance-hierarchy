"""Monte-Carlo reference for the smoothed score, independent of the closed form.

By Tweedie, shat(x) = E[x0 | x] - x, so it suffices to estimate the posterior
mean.  The posterior is p(u|x) ∝ p_data(u) exp(<x,u>/sigma^2) on the sphere.
Sampling the *proposal* vMF(x/|x|, |x|/sigma^2) reproduces the exponential
factor exactly, so the importance weights collapse to p_data(u) alone — which
keeps the estimator low-variance at every sigma, including sigma_min where
naive prior sampling would degenerate completely.

The estimator also returns its own standard error.  That matters: shat is O(sigma)
near the manifold, a difference of two nearly-equal O(1) vectors, so a fixed
relative tolerance is not a meaningful gate.  The meaningful question is whether
the closed form sits inside the MC confidence interval, which is what the
returned standard error lets the caller test.
"""

from __future__ import annotations

import torch

from .sphere import VMFMixture, _sample_vmf


def mc_shat(
    x: torch.Tensor,
    sigma: float,
    mix: VMFMixture,
    n_mc: int = 200_000,
    chunk: int = 50_000,
    return_se: bool = False,
):
    """Estimate sigma^2 * grad log p_sigma(x) for each row of x.

    Returns the estimate, and optionally the per-coordinate standard error of
    the self-normalised importance-sampling mean (delta method).
    """
    out = torch.empty_like(x)
    se = torch.empty_like(x)

    for i in range(x.shape[0]):
        xi = x[i]
        nrm = xi.norm()
        kappa = float(nrm) / sigma**2
        mu = xi / nrm.clamp_min(1e-30)

        sw = torch.zeros((), dtype=x.dtype)  # sum w
        sw2 = torch.zeros((), dtype=x.dtype)  # sum w^2
        swu = torch.zeros_like(xi)  # sum w u
        swu2 = torch.zeros_like(xi)  # sum w^2 u u
        sw2u = torch.zeros_like(xi)  # sum w^2 u

        # a single fixed shift keeps the weights comparable across chunks
        u0 = _sample_vmf(mu.cpu(), kappa, min(chunk, n_mc))
        shift = mix.log_prob(u0).max()

        for start in range(0, n_mc, chunk):
            m = min(chunk, n_mc - start)
            u = (
                u0
                if start == 0 and m == u0.shape[0]
                else _sample_vmf(mu.cpu(), kappa, m)
            )
            w = (mix.log_prob(u) - shift).exp()
            sw += w.sum()
            sw2 += (w * w).sum()
            swu += (w.unsqueeze(-1) * u).sum(0)
            sw2u += ((w * w).unsqueeze(-1) * u).sum(0)
            swu2 += ((w * w).unsqueeze(-1) * u * u).sum(0)

        mean = swu / sw
        # Var[mu_hat] ~ sum_i w_i^2 (u_i - mu_hat)^2 / (sum w)^2
        var = (swu2 - 2 * mean * sw2u + mean**2 * sw2) / sw**2
        out[i] = mean - xi
        se[i] = var.clamp_min(0).sqrt()

    return (out, se) if return_se else out

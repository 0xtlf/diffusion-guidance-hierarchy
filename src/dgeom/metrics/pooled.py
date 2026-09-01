"""Pooled check over random hyperplanes.

By rotational equivariance, if a ~ Unif(S^3) then pooling uniform-on-N samples
across random hyperplanes is *exactly* Unif(S^3), for any law on b.  So the
pooled sample can be tested directly, with no reference distribution to
estimate.  This catches systematic amortisation bias that per-hyperplane tests
smear out, and it costs nothing.

For uniform on S^3 the marginal of <u,x> has density prop to (1-t^2)^{1/2},
with CDF F(t) = 1/2 + (t sqrt(1-t^2) + arcsin t)/pi.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import kstest


def _cdf_s3_marginal(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, -1.0, 1.0)
    return 0.5 + (t * np.sqrt(1 - t**2) + np.arcsin(t)) / np.pi


def pooled_s3_ks(x: torch.Tensor, n_dirs: int = 8, generator=None) -> dict:
    """KS test of the pooled sample against Unif(S^3), along random directions."""
    xn = x.detach().cpu().double()
    xn = xn / xn.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    d = xn.shape[-1]
    stats, pvals = [], []
    for _ in range(n_dirs):
        u = torch.randn(d, dtype=xn.dtype, generator=generator)
        u = u / u.norm()
        t = (xn @ u).numpy()
        res = kstest(t, _cdf_s3_marginal)
        stats.append(float(res.statistic))
        pvals.append(float(res.pvalue))
    return {
        "ks_stat_max": max(stats),
        "ks_p_min": min(pvals),
        "ks_p_median": float(np.median(pvals)),
    }

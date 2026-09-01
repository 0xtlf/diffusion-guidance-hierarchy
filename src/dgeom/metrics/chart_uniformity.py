"""Is a sample uniform with respect to the manifold's intrinsic volume measure?

For the Klein bottle this is exact rather than estimated, because two facts line
up: the chart inverts in closed form, and sqrt(det g) depends on v alone.  Uniform
-by-area therefore FACTORISES into

    u ~ Uniform[0, 2pi)        and        v ~ proportional to sqrt(det g)(v)

so uniformity reduces to two one-dimensional KS tests against known CDFs, with no
density estimation anywhere.  A 2-D chi-square on the (u,v) grid is added because
the marginals alone cannot see joint structure -- a sample could match both
marginals while being wrong on the product.

For the sphere the existing pooled test in metrics/pooled.py already does the job
(<u,x> has density proportional to (1-t^2)^{1/2}), so this module is Klein-specific
by design rather than a general-purpose replacement.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import chi2 as _chi2
from scipy.stats import kstest

TWO_PI = 2 * np.pi


def klein_uniformity(manifold, x: torch.Tensor, nbins: int = 12) -> dict:
    """KS on u, KS on v against the volume-element CDF, plus a 2-D chi-square."""
    uv = manifold.chart_coords(manifold.project(x)).double()
    u = (uv[:, 0] % TWO_PI).cpu().numpy()
    v = (uv[:, 1] % TWO_PI).cpu().numpy()

    grid = torch.linspace(0, TWO_PI, 20001, dtype=torch.float64)
    w = manifold.volume_element(grid)
    cdf = torch.cat(
        [torch.zeros(1, dtype=torch.float64), torch.cumulative_trapezoid(w, grid)]
    )
    cdf = (cdf / cdf[-1]).numpy()
    gn = grid.numpy()

    ks_u = kstest(u / TWO_PI, "uniform")
    ks_v = kstest(v, lambda t: np.interp(t, gn, cdf))

    # joint test: cells of equal probability under the uniform-by-area measure.
    # u is split evenly; v is split at quantiles of its CDF, so every cell has
    # the same expected count and the chi-square is well conditioned.
    edges_v = np.interp(np.linspace(0, 1, nbins + 1), cdf, gn)
    iu = np.clip((u / TWO_PI * nbins).astype(int), 0, nbins - 1)
    iv = np.clip(np.searchsorted(edges_v[1:-1], v), 0, nbins - 1)
    obs = np.zeros(nbins * nbins)
    np.add.at(obs, iu * nbins + iv, 1.0)
    exp = len(u) / (nbins * nbins)
    stat = float(((obs - exp) ** 2 / exp).sum())
    dof = nbins * nbins - 1

    return {
        "ks_u_stat": float(ks_u.statistic),
        "ks_u_p": float(ks_u.pvalue),
        "ks_v_stat": float(ks_v.statistic),
        "ks_v_p": float(ks_v.pvalue),
        "chi2_joint": stat,
        "chi2_joint_dof": dof,
        "chi2_joint_p": float(_chi2.sf(stat, dof)),
        "n": len(u),
    }


def uniformity(manifold, x: torch.Tensor, **kw) -> dict:
    """Dispatch to whichever exact test the manifold admits."""
    if getattr(manifold, "name", "") == "klein":
        return klein_uniformity(manifold, x, **kw)
    from .pooled import pooled_s3_ks

    n_dirs = 12
    r = pooled_s3_ks(manifold.project(x), n_dirs=n_dirs)
    # Bonferroni-correct the minimum before returning it. Otherwise the caller
    # compares a min-of-12 p-values against a single-test threshold, and the null
    # distribution of that minimum has median ~0.06 -- so a perfectly uniform
    # sample would look borderline by construction.
    return {
        "ks_u_p": r["ks_p_median"],
        "ks_v_p": min(1.0, r["ks_p_min"] * n_dirs),
        "chi2_joint_p": r["ks_p_median"],
        "ks_stat_max": r["ks_stat_max"],
        "n": int(x.shape[0]),
    }

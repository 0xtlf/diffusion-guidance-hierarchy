"""Genuinely two-dimensional uniformity tests on a 2-sphere.

The marginal tests used elsewhere check three fixed directions. That is
necessary but not sufficient: a distribution can match every one-dimensional
marginal and still be far from uniform on the surface. These two tests look at
the joint law.

  harmonic   Expand the sample in real spherical harmonics. Under uniformity
             every coefficient of degree l >= 1 has mean zero and variance
             1/(4 pi), so the rescaled total power is chi-square distributed.
             Rotation invariant, and the per-degree spectrum localises the
             defect: power at l = 1 is a dipole (the sample is pushed to one
             side), l = 2 a quadrupole (squashed along an axis), and so on.

  chi-square Equal-area cells. Uniform on S^2 means cos(theta) is uniform on
             [-1, 1] and phi is uniform on [0, 2 pi) INDEPENDENTLY, so a regular
             grid in those coordinates already has equal-area cells and needs no
             Jacobian. Cheaper to interpret than the harmonic test but tied to
             the choice of pole.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import sph_harm_y
from scipy.stats import chi2


def _angles(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polar and azimuthal angles of unit vectors in R^3."""
    theta = np.arccos(np.clip(u[:, 2], -1.0, 1.0))
    phi = np.arctan2(u[:, 1], u[:, 0]) % (2 * np.pi)
    return theta, phi


def real_sph_harm(lmax: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Real orthonormal spherical harmonics for degrees 1..lmax.

    Returns:
        ``(N, K)`` array with ``K = lmax (lmax + 2)`` columns, orthonormal with
        respect to the surface measure.
    """
    cols = []
    for ell in range(1, lmax + 1):
        for m in range(-ell, ell + 1):
            y = sph_harm_y(ell, abs(m), theta, phi)
            if m > 0:
                v = np.sqrt(2.0) * (-1.0) ** m * y.real
            elif m < 0:
                v = np.sqrt(2.0) * (-1.0) ** m * y.imag
            else:
                v = y.real
            cols.append(v)
    return np.stack(cols, axis=-1)


def harmonic_uniformity(u: torch.Tensor | np.ndarray, lmax: int = 6) -> dict:
    """Omnibus rotation-invariant test of uniformity on S^2.

    Under the uniform measure each real harmonic of degree ``l >= 1`` satisfies
    ``E[Y] = 0`` and ``Var[Y] = 1/(4 pi)``, so with ``N`` samples

        T = 4 pi N sum_k (mean_i Y_k(x_i))^2  ~  chi^2_K,   K = lmax (lmax + 2)

    Args:
        u: ``(N, 3)`` unit vectors.
        lmax: highest degree included.

    Returns:
        ``T``, ``dof``, ``p`` and ``power`` (per-degree contribution to ``T``).
    """
    u = np.asarray(u.detach().cpu() if torch.is_tensor(u) else u, dtype=float)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    theta, phi = _angles(u)
    y = real_sph_harm(lmax, theta, phi)
    n = len(u)
    coef = y.mean(axis=0)
    contrib = 4 * np.pi * n * coef**2
    power, k = {}, 0
    for ell in range(1, lmax + 1):
        width = 2 * ell + 1
        power[ell] = float(contrib[k : k + width].sum())
        k += width
    t = float(contrib.sum())
    dof = lmax * (lmax + 2)
    return {
        "T": t,
        "dof": dof,
        "p": float(chi2.sf(t, dof)),
        "power": power,
        "n": n,
    }


def cell_uniformity(u: torch.Tensor | np.ndarray, nbins: int = 12) -> dict:
    """Chi-square on equal-area cells of S^2.

    Bins ``(cos theta, phi)`` on a regular grid, which is equal-area by
    Archimedes, so every cell has the same expected count under uniformity.

    Args:
        u: ``(N, 3)`` unit vectors.
        nbins: cells per axis; the grid is ``nbins x nbins``.

    Returns:
        ``chi2``, ``dof``, ``p`` and the largest cell excess.
    """
    u = np.asarray(u.detach().cpu() if torch.is_tensor(u) else u, dtype=float)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    theta, phi = _angles(u)
    i = np.clip(((np.cos(theta) + 1) / 2 * nbins).astype(int), 0, nbins - 1)
    j = np.clip((phi / (2 * np.pi) * nbins).astype(int), 0, nbins - 1)
    obs = np.zeros(nbins * nbins)
    np.add.at(obs, i * nbins + j, 1.0)
    exp = len(u) / (nbins * nbins)
    stat = float(((obs - exp) ** 2 / exp).sum())
    dof = nbins * nbins - 1
    return {
        "chi2": stat,
        "dof": dof,
        "p": float(chi2.sf(stat, dof)),
        "max_cell_excess": float(np.abs(obs / exp - 1).max()),
        "n": len(u),
    }


__all__ = ["cell_uniformity", "harmonic_uniformity", "real_sph_harm"]

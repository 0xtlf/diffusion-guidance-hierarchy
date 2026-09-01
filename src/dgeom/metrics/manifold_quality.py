"""Has the score model actually learned the manifold?

Three tests, none of which needs a reference score -- so they read identically on
`S^3` and on the Klein bottle, and they work on any manifold where the true
tangent space is known.

The Jacobian test is the sharp one.  Near M the hat-space score behaves like the
displacement onto the manifold, shat(x) ~ -(x - P_M x), so its Jacobian

    J = d shat / dx  ~  -P_perp

is minus the projector onto the normal space.  Its spectrum is therefore -1 with
multiplicity equal to the CODIMENSION and 0 on the tangent space.  That gives two
things for free: the model's own estimate of the intrinsic dimension (the same
observable Stanczuk et al. use), and its estimate of the normal space, which can
be compared against the truth by principal angles.

`cos_dM` is deliberately absent.  It cannot reach 1 even for a perfect score,
because shat legitimately carries a tangential Theta(1) term from the data
density -- so a value below 1 says nothing about geometric quality.
"""

from __future__ import annotations

import math

import torch

from ..geometry.base import principal_angles


def _jacobian(
    model, x: torch.Tensor, sigma: float, eps: float | None = None
) -> torch.Tensor:
    """(B, d, d) Jacobian of shat, by central differences.

    Finite differences rather than autograd so the analytic references (which
    route through scipy and numpy quadrature) can be measured with exactly the
    same code as the learned models.
    """
    d = x.shape[-1]
    eps = eps if eps is not None else max(1e-6, 1e-2 * sigma)
    cols = []
    for k in range(d):
        e = torch.zeros_like(x)
        e[:, k] = eps
        cols.append((model.shat(x + e, sigma) - model.shat(x - e, sigma)) / (2 * eps))
    return torch.stack(cols, dim=-1)  # column k = d shat / d x_k


def jacobian_spectrum(model, manifold, x: torch.Tensor, sigma: float) -> dict:
    """Codimension and normal-space accuracy, from the Jacobian's spectrum."""
    J = _jacobian(model, x, sigma)
    J = 0.5 * (J + J.transpose(-1, -2))  # J is a Hessian; symmetrise
    evals, evecs = torch.linalg.eigh(J)  # ascending
    d, n = manifold.d, manifold.n
    codim = d - n

    # infer codimension from the largest gap in the spectrum
    gaps = evals[:, 1:] - evals[:, :-1]
    inferred = gaps.argmax(dim=-1) + 1

    # learned normal space = eigenvectors of the codim most negative eigenvalues
    learned_normal = evecs[:, :, :codim].transpose(-1, -2)  # (B, codim, d)
    true_normal = manifold.normal_basis(x)
    ang = principal_angles(learned_normal, true_normal)

    return {
        "eig_mean": evals.mean(dim=0).tolist(),
        "eig_normal_mean": float(evals[:, :codim].mean()),
        "eig_tangent_mean": float(evals[:, codim:].mean()),
        "codim_correct_frac": float((inferred == codim).double().mean()),
        "normal_angle_deg_mean": float(ang.mean() * 180 / math.pi),
        "normal_angle_deg_p95": float(ang.flatten().quantile(0.95) * 180 / math.pi),
    }


def flow_to_manifold(
    model,
    manifold,
    x: torch.Tensor,
    sigma: float,
    steps: int = 200,
    step_scale: float = 0.5,
) -> dict:
    """Flow deterministically along the score, then measure distance to M.

    The most direct reading of "it learned the manifold": with no noise, the
    hat-space score is a displacement field whose fixed points should be exactly
    M.  Distances are reported relative to the manifold's own scale so the number
    means the same thing on both manifolds.
    """
    z = x.clone()
    for _ in range(steps):
        z = z + step_scale * model.shat(z, sigma)
    dist = (z - manifold.project(z)).norm(dim=-1) / manifold.scale
    moved = (z - x).norm(dim=-1).mean()
    return {
        "flow_dist_median": float(dist.median()),
        "flow_dist_p95": float(dist.quantile(0.95)),
        "flow_dist_max": float(dist.max()),
        "flow_displacement": float(moved),
        "flow_finite": bool(torch.isfinite(z).all()),
    }


def coverage(manifold, x: torch.Tensor, n_ref: int = 4096, generator=None) -> dict:
    """Do flowed samples reach all of M, or has the model collapsed onto part?

    Measured as a covering radius: draw reference points uniformly w.r.t. the
    volume measure and ask how far each is from the nearest sample. This is a
    statement about support and is metric-intrinsic.

    A chart histogram would be wrong here. Equal-width bins in chart coordinates
    are not equal-area -- uniform on S^3 has angular density proportional to
    sin^2(theta1) sin(theta2), so polar cells are legitimately sparse and get
    counted as holes. That measures the coordinate system, not the model.
    """
    ref = manifold.sample_uniform(n_ref, generator=generator).to(x.dtype)
    d = torch.cdist(ref, x).min(dim=1).values / manifold.scale
    # spacing you would expect from n samples spread over an n_dim manifold
    expected = float(x.shape[0]) ** (-1.0 / manifold.n)
    return {
        "cover_radius_p95": float(d.quantile(0.95)),
        "cover_radius_max": float(d.max()),
        "cover_ratio": float(d.quantile(0.95)) / expected,
    }


def score_error(model, reference, manifold, x: torch.Tensor, sigma: float) -> dict:
    """Hat-space error against the manifold's reference score.

    Hat space is raw x sigma^2, so "hat error = o(1)" is exactly the o(sigma^-2)
    raw score error the geometric-learning claim tolerates -- which is why this
    is reported in absolute terms, not relative.
    """
    got = model.shat(x, sigma)
    exact = reference.shat(x, sigma)
    err = (got - exact).norm(dim=-1)
    return {
        "hat_err_mean": float(err.mean()),
        "hat_err_p95": float(err.quantile(0.95)),
        "hat_err_rel": float(err.mean() / exact.norm(dim=-1).mean().clamp_min(1e-30)),
    }

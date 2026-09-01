"""Gate: the Klein bottle's geometry must be exact before anything is built on it.

Every downstream claim -- the quadrature reference score, the flow-to-manifold
quality gate, the uniformity test on the corrector's output -- rests on the
chart, the metric and the nearest-point projection being right. So they are
checked first, against independent numerics rather than against themselves.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import kstest

from dgeom.geometry import KleinVonMisesLoader
from dgeom.geometry.klein import TWO_PI, KleinBottle


def main() -> int:
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(0)
    K = KleinBottle()
    checks: list[tuple[str, float, float, bool]] = []

    def check(name, value, tol, smaller_is_better=True):
        ok = value < tol if smaller_is_better else value > tol
        checks.append((name, value, tol, ok))

    print(f"Klein bottle R={K.R} r={K.r}, RMS normalisation {K._norm:.6f}\n")

    # 1. chart round-trip
    u = torch.rand(20000, generator=g) * TWO_PI
    v = torch.rand(20000, generator=g) * TWO_PI
    x = K.from_chart(u, v)
    uv = K.chart_coords(x)
    rt = (K.from_chart(uv[:, 0], uv[:, 1]) - x).norm(dim=-1).max()
    check("chart round-trip (max)", float(rt), 1e-12)

    # 2. metric against finite differences
    eps = 1e-6
    Pu = (K.from_chart(u + eps, v) - K.from_chart(u - eps, v)) / (2 * eps)
    Pv = (K.from_chart(u, v + eps) - K.from_chart(u, v - eps)) / (2 * eps)
    E, F, G = (Pu * Pu).sum(-1), (Pu * Pv).sum(-1), (Pv * Pv).sum(-1)
    check("|F| (chart orthogonality)", float(F.abs().max()), 1e-8)
    check(
        "sqrt(det g) vs numeric",
        float(((E * G - F * F).sqrt() - K.volume_element(v)).abs().max()),
        1e-8,
    )
    fu, fv = K.frame(u, v)
    check(
        "analytic frame vs numeric",
        float(max((fu - Pu).abs().max(), (fv - Pv).abs().max())),
        1e-7,
    )

    # 3. embedding: no self-intersection
    n = 90
    gu = torch.linspace(0, TWO_PI, n + 1)[:-1]
    GU, GV = torch.meshgrid(gu, gu, indexing="ij")
    P = K.from_chart(GU.reshape(-1), GV.reshape(-1))
    D = torch.cdist(P, P)
    D.fill_diagonal_(1e9)
    j = D.argmin(dim=1)
    uu, vv = GU.reshape(-1), GV.reshape(-1)
    du = (uu - uu[j]).abs()
    du = torch.minimum(du, TWO_PI - du)
    dv = (vv - vv[j]).abs()
    dv = torch.minimum(dv, TWO_PI - dv)
    check("chart-distant near-collisions", float(((du > 0.5) & (dv > 0.5)).sum()), 0.5)

    # 4. projection: on-manifold, idempotent, and actually nearest
    x0 = K.sample_uniform(4000, generator=g)
    off = x0 + 0.05 * torch.randn(x0.shape, generator=g)
    p = K.project(off)
    check("P_M lands on M", float(K.dist(p).max()), 1e-10)
    check("P_M idempotent", float((K.project(p) - p).norm(dim=-1).max()), 1e-10)
    # nearest: no grid point beats it
    gp = K._gp
    brute = torch.cdist(off[:600], gp).min(dim=1).values
    check(
        "P_M at least as near as a 512^2 grid",
        float((K.dist(off[:600]) - brute).max()),
        1e-6,
    )
    # residual must be normal to M
    T = K.tangent_basis(off)
    resid = off - p
    resid = resid / resid.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    check(
        "residual orthogonal to T_xM",
        float(torch.einsum("bkd,bd->bk", T, resid).abs().max()),
        1e-5,
    )

    # 5. tangent/normal frames are orthonormal and complementary
    N = K.normal_basis(x0)
    T0 = K.tangent_basis(x0)
    gram_t = torch.einsum("bkd,bld->bkl", T0, T0)
    eye2 = torch.eye(2).expand_as(gram_t)
    check("tangent frame orthonormal", float((gram_t - eye2).abs().max()), 1e-8)
    check(
        "normal _|_ tangent",
        float(torch.einsum("bkd,bld->bkl", N, T0).abs().max()),
        1e-8,
    )

    # 6. uniform-by-area sampler: u uniform, v with CDF of sqrt(det g)
    xs = K.sample_uniform(50000, generator=g)
    uvs = K.chart_coords(xs)
    p_u = kstest((uvs[:, 0] / TWO_PI).numpy(), "uniform").pvalue
    grid = torch.linspace(0, TWO_PI, 20001)
    w = K.volume_element(grid)
    cdf = torch.cat([torch.zeros(1), torch.cumulative_trapezoid(w, grid)])
    cdf = cdf / cdf[-1]
    gn, cn = grid.numpy(), cdf.numpy()
    p_v = kstest(uvs[:, 1].numpy(), lambda t: np.interp(t, gn, cn)).pvalue
    check("uniform sampler: u KS p", float(p_u), 0.01, smaller_is_better=False)
    check("uniform sampler: v KS p", float(p_v), 0.01, smaller_is_better=False)

    # 7. the data density must actually be non-uniform, or the experiment is vacuous
    loader = KleinVonMisesLoader.random(K, 3, (1.0, 3.0), 1024, g)
    uvd = K.chart_coords(loader.sample(50000))
    p_ud = kstest((uvd[:, 0] / TWO_PI).numpy(), "uniform").pvalue
    check("p_data is NOT uniform (u KS p)", float(p_ud), 1e-6)

    # 8. total area, against the analytic double integral
    area_mc = float(
        K.volume_element(torch.rand(400000, generator=g) * TWO_PI).mean() * TWO_PI**2
    )
    area_qd = float(torch.trapezoid(K.volume_element(grid), grid) * TWO_PI)
    check("area: MC vs quadrature", abs(area_mc - area_qd) / area_qd, 5e-3)

    width = max(len(c[0]) for c in checks)
    print(f"{'check':<{width}} {'value':>13} {'tol':>10}   ")
    print("-" * (width + 30))
    for name, val, tol, ok in checks:
        print(f"{name:<{width}} {val:13.3e} {tol:10.1e}   {'ok' if ok else 'FAIL'}")
    failed = [c[0] for c in checks if not c[3]]
    print(f"\narea of the manifold: {area_qd:.4f} (normalised units)")
    print(
        f"\n{len(checks) - len(failed)}/{len(checks)} passed -> "
        f"{'GATE PASSED' if not failed else 'GATE FAILED: ' + ', '.join(failed)}"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

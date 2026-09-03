"""Gate for the conditional submanifold ``N = M ∩ H``.

Nothing conditional proceeds until this passes. The section is the ground truth
the whole guidance experiment is measured against, so it has to be exact rather
than approximately right, in the same way the analytic sphere score gates the
unconditional work.

Two properties carry the weight:

  on the section   samples must lie on M *and* on H, to machine precision. The
                   two residuals are checked separately because landing on the
                   manifold but off the hyperplane is a different bug from the
                   reverse, and averaging them hides which happened.
  uniform on it    the section's own marginal must be flat. On the sphere that is
                   Archimedes' theorem (<e,x> is exactly uniform on [-1,1]); on
                   the Klein bottle it is arclength on [0, L).

Everything is swept over many random hyperplanes, because the failure modes are
not uniform in w: near-tangential cuts resolve badly, and the Klein section
changes component count with w.
"""

from __future__ import annotations

import argparse

import torch
from scipy.stats import kstest

from dgeom.geometry import (
    Hyperplane,
    KleinBottle,
    Sphere,
    intersection_loader,
    section_for,
)

CFG = {
    "seed": 0,
    "data": {"density": "vmf", "n_components": 3, "kappa_range": (1.0, 3.0)},
    "train": {"score": {"batch_size": 1024}},
}


def main() -> int:
    """Run every section check and report a single verdict."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-planes", type=int, default=50)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--grid", type=int, default=400)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(0)
    checks: list[tuple[str, float, float, bool]] = []

    def check(name: str, value: float, tol: float, smaller: bool = True) -> None:
        ok = value <= tol if smaller else value >= tol
        checks.append((name, float(value), tol, bool(ok)))

    for M in (Sphere(4), KleinBottle()):
        name = M.name
        dM = ac = 0.0
        ks_min = 1.0
        comps: list[int] = []
        for _ in range(args.n_planes):
            H = Hyperplane.random(4, generator=g)
            sec = section_for(M, H, **({} if name == "sphere" else {"grid": args.grid}))
            x = sec.sample_uniform(args.n, generator=g)
            a, b = sec.residuals(x)
            dM, ac = max(dM, float(a.max())), max(ac, float(b.max()))

            # the section's own marginals must be flat under its uniform measure;
            # Bonferroni over the marginals so the minimum is comparable to a
            # single-test threshold
            ms = sec.uniform_marginals()
            for _lab, proj, _pdf, (lo, hi) in ms:
                t = (proj(x).numpy() - lo) / (hi - lo)
                ks_min = min(ks_min, kstest(t, "uniform").pvalue * len(ms))
            if name == "klein":
                comps.append(sec.n_components())

        check(f"{name}: max dist to M over planes", dM, 1e-9)
        check(f"{name}: max |<w,x>| over planes", ac, 1e-9)
        check(f"{name}: min uniformity KS p (Bonferroni)", ks_min, 0.01, smaller=False)
        check(
            f"{name}: section codim is 1",
            float(
                M.n
                - section_for(
                    M,
                    Hyperplane.random(4, generator=g),
                    **({} if name == "sphere" else {"grid": args.grid}),
                ).n
            ),
            1.0,
            smaller=False,
        )

        # p_data restricted to N must lie on N, and must NOT be uniform, or the
        # "whatever the data distribution was" claim is vacuous
        H = Hyperplane.random(4, generator=g)
        kw = {} if name == "sphere" else {"grid": args.grid}
        ld = intersection_loader(M, H, CFG, generator=g, **kw)
        lu = intersection_loader(M, H, CFG, generator=g, uniform=True, **kw)
        xd, xu = ld.sample(args.n), lu.sample(args.n)
        a, b = ld.manifold.residuals(xd)
        check(f"{name}: p_data|N stays on N", max(float(a.max()), float(b.max())), 1e-9)
        _lab, proj, _pdf, (lo, hi) = ld.manifold.uniform_marginals()[0]
        p_nonuniform = kstest((proj(xd).numpy() - lo) / (hi - lo), "uniform").pvalue
        check(f"{name}: p_data|N is NOT uniform", p_nonuniform, 1e-3)

        # importance check: reweighting uniform samples by p_data must reproduce
        # the p_data sample, which is what says the loader draws the right law
        lw = ld.density.log_prob(xu)
        w = (lw - lw.max()).exp()
        m_re = float((proj(xu) * w).sum() / w.sum())
        m_dir = float(proj(xd).mean())
        spread = float(proj(xu).std())
        check(
            f"{name}: p_data|N mean, direct vs reweighted",
            abs(m_re - m_dir) / spread,
            0.05,
        )
        if name == "klein":
            # components legitimately vary with w (2 and 4 both observed), so the
            # meaningful check is that a given w gives the same count under grid
            # refinement -- done below -- not that the count is constant
            print(f"  klein components across planes: {sorted(set(comps))}")

    # Klein length must converge under grid refinement, or the arclength measure
    # the whole ground truth rests on is not resolved
    K = KleinBottle()
    worst = 0.0
    for _ in range(8):
        H = Hyperplane.random(4, generator=g)
        s4 = section_for(K, H, grid=args.grid)
        s8 = section_for(K, H, grid=2 * args.grid)
        worst = max(worst, abs(s8.length - s4.length) / s8.length)
        if s4.n_components() != s8.n_components():
            worst = 1.0
    check("klein: length + components stable G vs 2G", worst, 1e-3)

    width = max(len(c[0]) for c in checks)
    print(f"{'check':<{width}} {'value':>13} {'tol':>10}   ")
    print("-" * (width + 30))
    for nm, val, tol, ok in checks:
        print(f"{nm:<{width}} {val:13.3e} {tol:10.1e}   {'ok' if ok else 'FAIL'}")
    failed = [c[0] for c in checks if not c[3]]
    print(
        f"\n{len(checks) - len(failed)}/{len(checks)} passed -> "
        f"{'GATE PASSED' if not failed else 'GATE FAILED: ' + ', '.join(failed)}"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

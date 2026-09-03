"""Does the guidance term of a conditional submanifold scale like GEOMETRY?

This is the measurement the conditional experiment exists to make. The paper
separates grad log p_sigma into a geometry part and a density part at different
rates in sigma; the question here is where a CONDITIONING term lands.

Everything is exact. On the sphere the ambient law and the law restricted to a
hyperplane section are both uniform-on-a-sphere after smoothing, so both scores
are closed form, and the guidance is their difference:

    grad log p_sigma(c | x) = grad log p_sigma(x | c) - grad log p_sigma(x)

No network, no Gaussian-posterior approximation, no learned classifier. A wrong
answer here would be the theory, not the implementation -- which is why the
closed forms are gated against Monte Carlo first.

Read the exponent carefully. A field with stiffness Theta(sigma^-2) has
MAGNITUDE Theta(sigma^-1) at a point noised to distance ~sigma. What decides
the regime is whether guidance matches geometry or matches density.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import t as student_t

from dgeom.config import load_config, seed_everything
from dgeom.experiment import make_loader, make_manifold, make_reference
from dgeom.models import UniformSphereDiffusion
from dgeom.models.schedule import NoiseSchedule
from dgeom.viz.rates import plot_rates

ESS_FLOOR = 200.0  # below this the MC reference is itself biased
# the rate claim is asymptotic, so the fit uses only small sigma. Above this the
# smoothing kernel is comparable to the manifold and the section stops being
# resolvable -- a regime the theory makes no claim about.
FIT_MAX_SIGMA = 0.05
N_BATCH = 16  # batches used to estimate the MC standard error
ALPHA = 0.01


def mc_score(x, sigma, sampler, gen, n_mc=200000, chunk=20000):
    """Grad log p_sigma by self-normalised importance sampling."""
    acc_w = torch.zeros(x.shape[0], dtype=x.dtype)
    acc_v = torch.zeros_like(x)
    top = torch.full((x.shape[0],), -torch.inf, dtype=x.dtype)
    for _ in range(max(1, n_mc // chunk)):
        x0 = sampler(chunk, gen)
        lw = -(torch.cdist(x, x0) ** 2) / (2 * sigma**2)
        m = lw.max(dim=1).values
        wt = (lw - m.unsqueeze(1)).exp()
        new = torch.maximum(top, m)
        so = (top - new).exp().nan_to_num(0.0)
        sn = (m - new).exp()
        acc_v = acc_v * so.unsqueeze(-1) + sn.unsqueeze(-1) * (
            wt @ x0 - wt.sum(1).unsqueeze(-1) * x
        )
        acc_w = acc_w * so + sn * wt.sum(1)
        top = new
    return acc_v / acc_w.unsqueeze(-1) / sigma**2


def terms(x, sigma, ambient, section, vmf):
    """The three magnitudes at one (ensemble, sigma), per point."""
    amb = ambient.shat(x, sigma) / sigma**2
    gui = section.shat(x, sigma) / sigma**2 - amb
    u = x / x.norm(dim=-1, keepdim=True)
    dens = vmf.shat(x, sigma) / sigma**2
    dens = dens - (dens * u).sum(-1, keepdim=True) * u
    return {
        "geometry": (amb * u).sum(-1).abs(),
        "density": dens.norm(dim=-1),
        "guidance": gui.norm(dim=-1),
    }


def fit_window(sigmas, values, max_sigma=FIT_MAX_SIGMA):
    """Slope and R^2 over the asymptotic window only."""
    s = np.asarray(sigmas)
    m = s <= max_sigma
    ls, lv = np.log(s[m]), np.log(np.asarray(values)[m])
    slope, intercept = np.polyfit(ls, lv, 1)
    resid = lv - (slope * ls + intercept)
    ss_tot = ((lv - lv.mean()) ** 2).sum()
    r2 = 1.0 - (resid**2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2)


def main() -> int:
    """Gate the closed forms, sweep sigma, write the figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/manifold_sphere.yaml")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--n-sigma", type=int, default=12)
    ap.add_argument("--n-planes", type=int, default=50)
    ap.add_argument("--out", default="runs/rates")
    ap.add_argument("--mode", choices=["light", "dark"], default="light")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    gen = torch.Generator().manual_seed(int(cfg["seed"]) + 31)

    w = torch.randn(4, generator=gen)
    w = w / w.norm()
    sch = NoiseSchedule(1e-4, 10.0)
    ambient = UniformSphereDiffusion(4, sch)
    section = UniformSphereDiffusion(4, sch, subspace_dim=3, normal=w)

    def on_sphere(n, g):
        z = torch.randn(n, 4, generator=g)
        return z / z.norm(dim=-1, keepdim=True)

    def on_section(n, g):
        z = torch.randn(n, 4, generator=g)
        z = z - (z @ w).unsqueeze(-1) * w
        return z / z.norm(dim=-1, keepdim=True)

    # ---------------------------------------------------------------- the gate
    print("closed form vs Monte Carlo (z-test against the MC standard error;")
    print("a fixed relative tolerance is meaningless -- the MC estimator's")
    print("effective sample size falls like sigma^dim, so IT is the noisy one)")
    # the reported |z| is a MAX over every point and component, and each z is a
    # t statistic on N_BATCH-1 degrees of freedom. A fixed cutoff would be
    # arbitrary, so the threshold is the Bonferroni-corrected quantile of that
    # very distribution -- what the worst case should reach under the null.
    n_cmp = 32 * 4
    z_max = float(student_t.ppf(1 - ALPHA / (2 * n_cmp), df=N_BATCH - 1))
    print(
        f"threshold |z| < {z_max:.2f}  (max of {n_cmp} t_{N_BATCH - 1} "
        f"statistics, Bonferroni at alpha={ALPHA})"
    )
    print(f"{'sigma':>7} {'ESS':>8} {'|z| ambient':>13} {'|z| section':>13}")
    ok = True
    for sig in (0.3, 0.2, 0.1):
        zs = {}
        for tag, samp, model in (
            ("ambient", on_sphere, ambient),
            ("section", on_section, section),
        ):
            x = samp(32, gen) + sig * torch.randn(32, 4, generator=gen)
            est = torch.stack([mc_score(x, sig, samp, gen) for _ in range(N_BATCH)])
            mu, se = est.mean(0), est.std(0) / N_BATCH**0.5
            cf = model.shat(x, sig) / sig**2
            zs[tag] = float(((cf - mu).abs() / se.clamp_min(1e-30)).max())
        x1 = on_sphere(1, gen)
        d2 = ((x1 - on_sphere(200000, gen)) ** 2).sum(-1)
        wt = (-d2 / (2 * sig**2)).exp()
        ess = float(wt.sum() ** 2 / (wt**2).sum())
        trusted = ess > ESS_FLOOR
        ok &= (not trusted) or max(zs.values()) < z_max
        note = "" if trusted else "  (ESS too low, not asserted)"
        print(
            f"{sig:7.3g} {ess:8.0f} {zs['ambient']:13.2f} {zs['section']:13.2f}{note}"
        )
    print(
        f"gate: {'PASSED' if ok else 'FAILED'} "
        f"(|z| < {z_max:.2f} where ESS > {ESS_FLOOR:g})\n"
    )
    if not ok:
        return 1

    # ------------------------------------------------------------ the measurement
    # density needs a NON-uniform law, so it comes from the vMF reference the
    # unconditional work already gated; geometry and guidance come from the
    # uniform forms, where the conditional score is exact
    Sm = make_manifold(cfg)
    vmf = make_reference(cfg, Sm, make_loader(cfg, Sm))
    sigmas = np.logspace(-2.5, -0.5, args.n_sigma)

    # TWO ensembles. The exponent is a property of WHERE it is measured: a point
    # drawn on N sits O(sigma) from the hyperplane, one drawn on M generally sits
    # O(1) from it. Both are reported, because the near-N regime governs the
    # stationary distribution while the far regime governs transport onto N.
    ensembles = {
        "on N (near the section)": on_section,
        "on M (away from the section)": on_sphere,
    }
    results: dict[str, dict] = {}
    for ename, sampler in ensembles.items():
        per_plane = {k: [] for k in ("geometry", "density", "guidance")}
        med = {k: [] for k in per_plane}
        q90 = {k: [] for k in per_plane}
        for _ in range(args.n_planes):
            wp = torch.randn(4, generator=gen)
            wp = wp / wp.norm()
            sec_p = UniformSphereDiffusion(4, sch, subspace_dim=3, normal=wp)

            near = sampler is on_section

            def samp(n, g, _w=wp, _near=near):
                z = torch.randn(n, 4, generator=g)
                if _near:
                    z = z - (z @ _w).unsqueeze(-1) * _w
                return z / z.norm(dim=-1, keepdim=True)

            curves = {k: [] for k in per_plane}
            for sig in sigmas:
                sig = float(sig)
                x = samp(args.n, gen) + sig * torch.randn(args.n, 4, generator=gen)
                t = terms(x, sig, ambient, sec_p, vmf)
                for k, v in t.items():
                    curves[k].append(float(v.mean()))
                    if _ == 0:
                        med[k].append(float(v.median()))
                        q90[k].append(float(v.quantile(0.9)))
            for k in per_plane:
                per_plane[k].append(curves[k])
        results[ename] = {
            "mean": {k: np.array(v) for k, v in per_plane.items()},
            "median": med,
            "q90": q90,
        }

    print(
        f"\nslopes fitted on sigma <= {FIT_MAX_SIGMA:g}, "
        f"over {args.n_planes} random hyperplanes"
    )
    summary = {}
    for ename, res in results.items():
        print(f"\n  ensemble: {ename}")
        print(
            f"    {'term':>10} {'slope mean':>11} {'sd':>7} {'min':>8} "
            f"{'max':>8} {'R^2':>7}"
        )
        summary[ename] = {}
        for k in ("geometry", "density", "guidance"):
            fits = [fit_window(sigmas, row) for row in res["mean"][k]]
            sl = np.array([f[0] for f in fits])
            r2 = np.array([f[1] for f in fits])
            summary[ename][k] = sl
            print(
                f"    {k:>10} {sl.mean():+11.3f} {sl.std():7.3f} "
                f"{sl.min():+8.3f} {sl.max():+8.3f} {r2.mean():7.4f}"
            )
        dg = np.abs(summary[ename]["guidance"] - summary[ename]["geometry"])
        dd = np.abs(summary[ename]["guidance"] - summary[ename]["density"])
        print(
            f"    guidance to geometry {dg.mean():.3f} +- {dg.std():.3f}   "
            f"to density {dd.mean():.3f} +- {dd.std():.3f}"
        )
        # "matches" needs an absolute tolerance, not merely "closer to one than
        # the other": away from the section, guidance is nearer geometry than
        # density and yet still a full power steeper than both.
        gm = summary[ename]["guidance"].mean()
        if dg.mean() < 0.15:
            note = "guidance MATCHES geometry"
        elif dd.mean() < 0.15:
            note = "guidance MATCHES density"
        else:
            note = (
                f"guidance matches NEITHER: slope {gm:+.2f}, "
                f"{dg.mean():.2f} steeper than geometry"
            )
        print(f"    -> {note}")

    print("\n  spread within a single run (first hyperplane, on-N ensemble):")
    r0 = results["on N (near the section)"]
    print(f"    {'sigma':>10} {'guidance med':>13} {'guidance p90':>13}")
    for i in (0, len(sigmas) // 2, len(sigmas) - 1):
        print(
            f"    {sigmas[i]:10.3e} {r0['median']['guidance'][i]:13.4e} "
            f"{r0['q90']['guidance'][i]:13.4e}"
        )

    series = {
        k: results["on N (near the section)"]["mean"][k].mean(0)
        for k in ("geometry", "density", "guidance")
    }
    bands = {
        k: (
            results["on N (near the section)"]["mean"][k].min(0),
            results["on N (near the section)"]["mean"][k].max(0),
        )
        for k in series
    }

    path, slopes = plot_rates(
        sigmas,
        series,
        args.out,
        bands=bands,
        slope_samples=summary,
        fit_max_sigma=FIT_MAX_SIGMA,
        title="where does the guidance term sit? (sphere, exact scores)",
        subtitle=f"{args.n_planes} random hyperplanes, N={args.n} per sigma, "
        f"slopes fitted on sigma <= {FIT_MAX_SIGMA:g}; "
        f"density from the gated vMF reference",
        mode=args.mode,
    )
    print("\nslopes: " + "  ".join(f"{k} {v:+.3f}" for k, v in slopes.items()))
    d_geo = abs(slopes["guidance"] - slopes["geometry"])
    d_den = abs(slopes["guidance"] - slopes["density"])
    print(f"guidance is {d_geo:.3f} from geometry, {d_den:.3f} from density")
    print(f"-> guidance tracks {'GEOMETRY' if d_geo < d_den else 'DENSITY'}")
    print(f"\n{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

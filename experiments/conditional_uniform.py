"""Can tempering recover the UNIFORM measure on a conditional submanifold?

The data is a von Mises-Fisher mixture on S^3. A hyperplane through the origin
is drawn at inference, cutting out N = S^3 ∩ H, and the guided tempered corrector
is asked to produce the uniform measure on N -- whatever the data distribution
was.

Three questions, in order, because each is only meaningful if the previous one
passed:

  1. is p_data restricted to N already uniform?   If it were, there would be
     nothing to demonstrate. Measured, not assumed.
  2. does the UNTEMPERED guided sampler (alpha = 0) reproduce that restriction?
     If it does not, the sampler is wrong and any tempered result is noise. This
     chain is started AT p_data|N, so it tests stationarity rather than mixing --
     alpha = 0 has dt = step_scale * sigma^2 and would need ~10^5 steps to mix.
  3. does tempering (alpha > 0) move it to uniform?  This is the result.

Every comparison is against an EXACT reference: p_data|N by rejection on the
section, uniform on N by Archimedes. Two statistics are reported -- departure
from uniform relative to the sampling noise floor, and a two-sample KS against
the restriction -- because "moved away from p_data" and "arrived at uniform" are
different claims and a run can do the first without the second.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from scipy.stats import ks_2samp, kstest

from dgeom.config import load_config, resolve_device, seed_everything
from dgeom.experiment import load_model, make_loader, make_manifold, make_reference
from dgeom.geometry import Hyperplane, intersection_loader, section_for
from dgeom.models import GuidedDiffusion
from dgeom.sampling import LangevinSampler, TemperedLangevin
from dgeom.sampling.base import Trace
from dgeom.viz.uniformity import (
    make_probe,
    plot_convergence,
    plot_uniformity,
    uniformity_summary,
)


def marginal(section, x):
    """First uniform marginal, mapped to [0, 1] so KS is comparable."""
    _lab, proj, _pdf, (lo, hi) = section.uniform_marginals()[0]
    return ((proj(x) - lo) / (hi - lo)).detach().cpu().numpy()


def report(section, name, x, ref_pdata, w):
    """One row: where this sample sits relative to both references."""
    u = uniformity_summary(section, x)
    t = marginal(section, x)
    p_unif = kstest(t, "uniform").pvalue
    p_rest = ks_2samp(t, marginal(section, ref_pdata)).pvalue
    dM = section.manifold.dist(x)
    ac = (x @ w.to(x.dtype)).abs()
    print(
        f"  {name:<34} {u['max_deviation']:8.1%} {u['ratio']:7.2f}x "
        f"{p_unif:10.2e} {p_rest:10.2e} {float(dM.mean()):8.4f} "
        f"{float(ac.mean()):8.4f}"
    )
    return {
        "name": name,
        "max_deviation": u["max_deviation"],
        "ratio_to_floor": u["ratio"],
        "ks_p_vs_uniform": float(p_unif),
        "ks_p_vs_restriction": float(p_rest),
        "dist_M": float(dM.mean()),
        "abs_constraint": float(ac.mean()),
    }


def main() -> int:
    """Run the three stages and write the figures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/manifold_sphere.yaml")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--steps-alpha0", type=int, default=5000)
    ap.add_argument("--step-scale", type=float, default=0.2)
    ap.add_argument(
        "--sim-time",
        type=float,
        default=0.0,
        help="if > 0, pick the step count per alpha to give this much SIMULATED "
        "time instead of a fixed --steps. dt = step_scale * sigma^(2-alpha) "
        "spans an order of magnitude across the alphas of interest, so equal "
        "step counts mean very unequal physical durations. Mixing on the "
        "section takes about 2.5 time units, so 15 is roughly six mixing times.",
    )
    ap.add_argument("--trace-every", type=int, default=500)
    ap.add_argument("--plot-every", type=int, default=5000)
    ap.add_argument(
        "--use-reference",
        action="store_true",
        help="exact vMF score instead of the trained network, which "
        "separates 'the method fails' from 'the model is weak'",
    )
    ap.add_argument("--load-from", default="runs/m-sphere")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/cond-uniform")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    gen = torch.Generator().manual_seed(int(cfg["seed"]) + 41)
    out = Path(args.out)
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    trace_log = out / "corrector_trace.jsonl"

    manifold = make_manifold(cfg)
    loader = make_loader(cfg, manifold)
    H = Hyperplane.random(4, generator=gen)
    section = section_for(manifold, H)

    if args.use_reference:
        base = make_reference(cfg, manifold, loader)
        src = f"REFERENCE {type(base).__name__}"
    else:
        base, step = load_model(
            args.load_from, cfg, manifold, resolve_device(args.device)
        )
        src = f"{args.load_from} @ step {step}"
    guided = GuidedDiffusion(base, H, manifold=manifold)

    print(f"manifold {manifold.name}   section dim {section.n}   score: {src}")
    print(f"{H!r}")
    print(f"sigma={args.sigma:g}  N={args.n}  step_scale={args.step_scale:g}\n")

    # ---------------------------------------------------------- exact references
    ref_pdata = intersection_loader(
        manifold, H, cfg, generator=gen, mixture=loader.density
    ).sample(args.n)
    ref_unif = section.sample_uniform(args.n, generator=gen)

    lp = loader.density.log_prob(ref_unif)
    ratio = float((lp.max() - lp.quantile(0.001)).exp())
    p0 = kstest(marginal(section, ref_pdata), "uniform").pvalue
    print("STAGE 1 -- is the target already uniform? (if so, nothing to show)")
    print(f"  p_data density ratio on N: {ratio:.1f}")
    verdict = "NOT uniform, experiment is meaningful" if p0 < 1e-3 else "TOO UNIFORM"
    print(f"  KS of p_data|N against uniform: p = {p0:.2e}  -> {verdict}\n")
    if p0 >= 1e-3:
        return 1

    hdr = (
        f"  {'sample':<34} {'max dev':>8} {'x floor':>8} {'KS vs unif':>10} "
        f"{'KS vs p_d':>10} {'dist_M':>8} {'|<w,x>|':>8}"
    )
    print("STAGE 2 -- references, then the untempered chain")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = [
        report(section, "exact uniform on N (target)", ref_unif, ref_pdata, H.w),
        report(section, "exact p_data|N (start)", ref_pdata, ref_pdata, H.w),
    ]

    samples = {"1. p_data restricted to N": ref_pdata}
    traces: dict[str, Trace] = {}

    def run(alpha, n_steps, tag):
        """One tempered chain, started from p_data|N."""
        if args.sim_time > 0 and alpha > 0:
            dt = args.step_scale * args.sigma ** (2 - alpha)
            n_steps = max(500, math.ceil(args.sim_time / dt))
            print(
                f"  alpha={alpha:g}: dt={dt:.3e}, {n_steps:,} steps "
                f"= {args.sim_time / dt / (2.5 / dt):.1f} mixing times"
            )
        recs: list[dict] = []
        inner = make_probe(section)

        def probe(x, stp):
            r = inner(x, stp)
            recs.append({"step": stp, **r})
            with trace_log.open("a") as fh:
                fh.write(json.dumps({"alpha": alpha, "step": stp, **r}) + "\n")
            if args.plot_every and stp % args.plot_every == 0:
                # checkpoint the STATE, not just a picture of it: a job killed
                # part way otherwise leaves figures that cannot be re-analysed,
                # and the 2-D uniformity tests need the points themselves
                torch.save(
                    {
                        "x": x.detach().cpu().clone(),
                        "alpha": alpha,
                        "w": H.w,
                        "step": stp,
                    },
                    out / f"latest_a{alpha:g}.pt",
                )
                plot_uniformity(
                    section,
                    {**samples, tag: x.detach().cpu().clone()},
                    figures,
                    filename=f"uniformity_alpha{alpha:g}_step{stp:07d}.png",
                    title=f"conditional uniform on N, alpha={alpha:g}, step {stp:,}",
                    subtitle=f"sigma={args.sigma:g}, N={x.shape[0]}",
                )
                print(
                    f"       [alpha={alpha:g}] step {stp:>7,}  "
                    f"max dev {r['max_deviation']:.1%}",
                    flush=True,
                )
            return r

        x0 = ref_pdata + args.sigma * torch.randn(ref_pdata.shape, generator=gen)
        # TemperedLangevin enforces 0 < alpha < 2 on purpose; alpha = 0 is
        # ordinary Langevin, whose stationary law IS the restriction p_data|N
        cls = LangevinSampler if alpha == 0.0 else TemperedLangevin
        x, _ = cls(
            sigma=args.sigma,
            alpha=alpha,
            n_steps=n_steps,
            step_scale=args.step_scale,
            trace_every=args.trace_every,
        ).sample(guided, x0, probe=probe, generator=gen)
        traces[tag] = Trace(recs)
        return x

    # alpha = 0 keeps p_data|N stationary; started AT it, so this tests the
    # sampler rather than its (very slow) mixing time
    x0chain = run(0.0, args.steps_alpha0, "2. untempered chain (alpha=0)")
    rows.append(report(section, "untempered chain (alpha=0)", x0chain, ref_pdata, H.w))
    samples["2. after alpha=0 (should match p_data|N)"] = x0chain

    print("\nSTAGE 3 -- tempered chains")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for alpha in args.alphas:
        tag = f"3. after correction, alpha={alpha:g}"
        xa = run(float(alpha), args.steps, tag)
        rows.append(report(section, f"tempered, alpha={alpha:g}", xa, ref_pdata, H.w))
        samples[tag] = xa
        torch.save({"x": xa, "alpha": alpha, "w": H.w}, out / f"samples_a{alpha:g}.pt")

    best = min(
        args.alphas,
        key=lambda a: uniformity_summary(
            section, samples[f"3. after correction, alpha={a:g}"]
        )["ratio"],
    )
    keep = {k: samples[k] for k in list(samples)[:2]}
    keep[f"3. after correction, alpha={best:g}"] = samples[
        f"3. after correction, alpha={best:g}"
    ]
    upath = plot_uniformity(
        section,
        keep,
        figures,
        filename="uniformity.png",
        title="tempering toward uniform on the conditional submanifold N = S^3 ∩ H",
        subtitle=f"vMF mixture data, sigma={args.sigma:g}, {args.steps} steps, "
        f"N={args.n}, best alpha={best:g}",
    )
    cpath = plot_convergence(
        traces,
        figures,
        title="convergence of the guided corrector on N",
        subtitle="a plateau means further steps will not help",
    )
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\n{upath}\n{cpath}\n{out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

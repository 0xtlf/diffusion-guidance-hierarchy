"""Tempered guided sampling across SEVERAL connected conditional submanifolds.

The single-hyperplane experiment answers "does it work here"; this answers "does
it work generally", which is the only version with an error bar. Each hyperplane
gets its own connected, transversal offset, so every section satisfies the
path-connectedness required by Assumption 4.1 of Li et al. -- sections through
the origin do not, and on a disconnected section the between-component mass is
frozen at initialisation at any tempering exponent.

Step counts are set per plane from that plane's own geometry: a section of length
L relaxes at (2 pi / L)^2 per unit simulated time, so a short section needs fewer
steps than a long one and a fixed budget would over- or under-run most of them.

Everything is written as it happens; a run killed part way keeps the planes it
has finished.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kstest

from dgeom.config import load_config, resolve_device, seed_everything
from dgeom.experiment import load_model, make_loader, make_manifold, make_reference
from dgeom.geometry import (
    Hyperplane,
    connected_offset,
    intersection_loader,
    section_for,
)
from dgeom.models import GuidedDiffusion
from dgeom.sampling import TemperedLangevin
from dgeom.viz.style import use_style
from dgeom.viz.uniformity import make_probe, plot_uniformity, uniformity_summary


def arclength_ks(section, x) -> float:
    """KS statistic of the section's own uniform marginal.

    On a one-dimensional section this is a COMPLETE test: arclength is a global
    coordinate, so uniform in s is equivalent to uniform on N. On the sphere the
    marginal is one of several and the test is only necessary, not sufficient.
    """
    _lab, proj, _pdf, (lo, hi) = section.uniform_marginals()[0]
    t = (proj(x).detach().cpu().numpy() - lo) / (hi - lo)
    return float(kstest(t, "uniform").statistic)


def predicted_bias(section, density, sigma, alpha, gen, n=200_000) -> float:
    """KS statistic of the predicted stationary law p_data^(sigma^alpha).

    The tempered corrector does not reach uniform at finite sigma; it reaches
    this. Comparing against it separates the theory's own bias from anything the
    score model contributes.
    """
    u = section.sample_uniform(n, generator=gen)
    lp = density.log_prob(u)
    w = ((sigma**alpha) * (lp - lp.max())).exp().numpy()
    w /= w.sum()
    _lab, proj, _pdf, (lo, hi) = section.uniform_marginals()[0]
    s = (proj(u).detach().cpu().numpy() - lo) / (hi - lo)
    idx = np.random.default_rng(0).choice(len(s), min(n, 20_000), p=w)
    return float(kstest(s[idx], "uniform").statistic)


def main() -> int:
    """Sweep hyperplanes at one tempering exponent."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/manifold_klein.yaml")
    ap.add_argument("--load-from", default="runs/m-klein")
    ap.add_argument("--n-planes", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--step-scale", type=float, default=0.2)
    ap.add_argument(
        "--steps",
        type=int,
        default=0,
        help="fixed step count per plane; 0 sizes it from --efolds instead",
    )
    ap.add_argument(
        "--efolds",
        type=float,
        default=5.0,
        help="target e-foldings of the slowest mode, per plane",
    )
    ap.add_argument("--trace-every", type=int, default=500)
    ap.add_argument("--use-reference", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/cond-sweep")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    manifold = make_manifold(cfg)
    loader = make_loader(cfg, manifold)
    kw = {} if manifold.name == "sphere" else {"grid": 400}
    if args.use_reference:
        base = make_reference(cfg, manifold, loader)
        src = f"REFERENCE {type(base).__name__}"
    else:
        base, step = load_model(
            args.load_from, cfg, manifold, resolve_device(args.device)
        )
        src = f"{args.load_from} @ step {step}"

    dt = args.step_scale * args.sigma ** (2 - args.alpha)
    print(f"manifold {manifold.name}   score: {src}")
    print(f"alpha={args.alpha:g}  sigma={args.sigma:g}  N={args.n}  dt={dt:.3e}")
    print(f"{args.n_planes} hyperplanes, each with its own connected offset\n")

    hdr = (
        f"{'#':>2} {'offset b':>9} {'L':>6} {'|PTw|':>6} {'steps':>7} "
        f"{'start D':>8} {'end D':>8} {'pred D':>8} {'excess':>8} "
        f"{'maxdev':>7} {'xfloor':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    rows: list[dict] = []
    kept: list[tuple] = []
    for i in range(args.n_planes):
        # One generator per plane, seeded from the index alone. The plane draw,
        # the initial condition and the sampler noise all consume it, so if they
        # share the run-level generator the number of steps plane i takes shifts
        # every plane after it -- changing --alpha or --efolds then silently
        # changes WHICH hyperplanes are being compared. Keyed on i, plane i is
        # the same plane in every run with the same seed.
        gen = torch.Generator().manual_seed(int(cfg["seed"]) + 71 + 1000 * i)
        w = torch.randn(4, generator=gen)
        w = w / w.norm()
        try:
            b = connected_offset(manifold, w, grid=300)
        except RuntimeError as exc:
            print(f"{i:>2}  skipped: {exc}")
            continue
        sec = section_for(manifold, Hyperplane(w, b), **kw)
        ptw = float(
            torch.einsum(
                "bnd,d->bn",
                manifold.tangent_basis(sec.sample_uniform(4000, generator=gen)),
                w,
            )
            .norm(dim=-1)
            .median()
        )
        # a section of length L relaxes at (2 pi / L)^2; a 2-D section at l(l+1)=2
        rate = (2 * math.pi / sec.length) ** 2 if sec.n == 1 else 2.0
        steps = args.steps or max(500, math.ceil(args.efolds / (rate * dt)))
        efolds = steps * rate * dt

        gm = GuidedDiffusion(base, Hyperplane(w, b), manifold=manifold)
        start = intersection_loader(
            manifold, Hyperplane(w, b), cfg, generator=gen, mixture=loader.density, **kw
        ).sample(args.n)
        x0 = start + args.sigma * torch.randn(start.shape, generator=gen)
        # The deviation dips within about one e-fold and then climbs to a
        # plateau, so the endpoint alone hides the shape of the run. A probe is
        # what puts the uniformity measurement into the trace at all: without
        # one the sampler records only the step index and dt.
        inner = make_probe(sec)
        trace_log = out / "corrector_trace.jsonl"

        def probe(xs, stp, _i=i, _inner=inner, _log=trace_log, _r=rate, _dt=dt):
            rec = _inner(xs, stp)
            with _log.open("a") as fh:
                fh.write(
                    json.dumps(
                        {"plane": _i, "step": stp, "efolds_at": stp * _r * _dt, **rec}
                    )
                    + "\n"
                )
            return rec

        x, _ = TemperedLangevin(
            sigma=args.sigma,
            alpha=args.alpha,
            n_steps=steps,
            step_scale=args.step_scale,
            trace_every=args.trace_every,
        ).sample(gm, x0, probe=probe, generator=gen)

        d0, d1 = arclength_ks(sec, start), arclength_ks(sec, x)
        pred = predicted_bias(sec, loader.density, args.sigma, args.alpha, gen)
        u = uniformity_summary(sec, x)
        excess = math.sqrt(max(d1**2 - pred**2, 0.0))
        rows.append(
            {
                "plane": i,
                "offset": b,
                "length": float(sec.length),
                "components": int(sec.n_components()) if sec.n == 1 else 1,
                "ptw_median": ptw,
                "steps": steps,
                "ks_start": d0,
                "ks_end": d1,
                "ks_predicted": pred,
                "ks_excess": excess,
                "max_deviation": u["max_deviation"],
                "ratio_to_floor": u["ratio"],
                "efolds": efolds,
            }
        )
        kept.append((i, sec, start, x, b, steps))
        print(
            f"{i:>2} {b:+9.4f} {sec.length:6.2f} {ptw:6.3f} {steps:>7,} "
            f"{d0:8.4f} {d1:8.4f} {pred:8.4f} {excess:8.4f} "
            f"{u['max_deviation']:6.1%} {u['ratio']:6.2f}x  {efolds:4.1f} e-fold",
            flush=True,
        )

        torch.save(
            {"x": x, "w": w, "b": b, "alpha": args.alpha}, out / f"samples_plane{i}.pt"
        )
        (out / "summary.json").write_text(json.dumps(rows, indent=2))
        plot_uniformity(
            sec,
            {"1. p_data|N (start)": start, f"2. tempered alpha={args.alpha:g}": x},
            out / "figures",
            filename=f"uniformity_plane{i}.png",
            title=f"plane {i}: {manifold.name} ∩ {{<w,x> = {b:+.3f}}}",
            subtitle=f"alpha={args.alpha:g}, {steps:,} steps, N={args.n}, "
            f"L={sec.length:.2f}",
        )

    if not rows:
        print("\nno planes completed")
        return 1
    e = np.array([r["ks_end"] for r in rows])
    p = np.array([r["ks_predicted"] for r in rows])
    x_ = np.array([r["ks_excess"] for r in rows])
    f = np.array([r["ratio_to_floor"] for r in rows])
    crit = 1.36 / math.sqrt(args.n)
    print(f"\nover {len(rows)} planes, KS rejects above D = {crit:.4f} at N={args.n}")
    print(
        f"  end       D  {e.mean():.4f} +- {e.std():.4f}   "
        f"[{e.min():.4f}, {e.max():.4f}]   {(e < crit).sum()}/{len(e)} pass"
    )
    print(f"  predicted D  {p.mean():.4f} +- {p.std():.4f}   (the finite-sigma bias)")
    print(
        f"  excess    D  {x_.mean():.4f} +- {x_.std():.4f}   "
        f"(what the score model adds on top)"
    )
    print(f"  max dev / noise floor  {f.mean():.2f}x +- {f.std():.2f}")
    # the three planes worth looking at: the run is only as trustworthy as its
    # worst case, and the median says what to expect
    order = sorted(range(len(rows)), key=lambda k: rows[k]["ks_end"])
    picks = {
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }
    print()
    for label, k in picks.items():
        i, sec, start, x, b, steps = kept[k]
        r = rows[k]
        path = plot_uniformity(
            sec,
            {"1. p_data|N (start)": start, f"2. tempered alpha={args.alpha:g}": x},
            out / "figures",
            filename=f"uniformity_{label}.png",
            title=f"{label.upper()} of {len(rows)} planes: "
            f"{manifold.name} ∩ {{<w,x> = {b:+.3f}}}",
            subtitle=f"KS D {r['ks_end']:.4f} "
            f"(predicted bias {r['ks_predicted']:.4f}), "
            f"{r['max_deviation']:.1%} = {r['ratio_to_floor']:.2f}x floor, "
            f"L={r['length']:.2f}, {steps:,} steps",
        )
        print(f"  {label:<7} plane {i}  D={r['ks_end']:.4f}  -> {path}")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    c = use_style("light")
    idx = np.arange(len(rows))
    ax.bar(
        idx - 0.2,
        [r["ks_predicted"] for r in rows],
        0.4,
        color=c["series"][2],
        label="predicted bias  $p_{data}^{\\sigma^\\alpha}$",
    )
    ax.bar(idx + 0.2, e, 0.4, color=c["series"][0], label="observed")
    ax.axhline(crit, ls=(0, (4, 3)), lw=1.4, color=c["critical"])
    ax.annotate(
        f"KS rejects above {crit:.4f}",
        (len(rows) - 0.5, crit),
        fontsize=8,
        va="bottom",
        ha="right",
        color=c["critical"],
    )
    for label, k in picks.items():
        ax.annotate(
            label,
            (k, e[k]),
            fontsize=8,
            ha="center",
            va="bottom",
            color=c["text_secondary"],
        )
    ax.set_xticks(idx)
    ax.set_xticklabels(
        [f"plane {r['plane']}\nL={r['length']:.1f}" for r in rows], fontsize=8
    )
    ax.set_ylabel("KS statistic $D$")
    ax.set_title(
        f"{manifold.name}: {len(rows)} connected sections, alpha={args.alpha:g}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "sweep.png", dpi=140)
    plt.close(fig)
    print(f"  spread  {out / 'figures' / 'sweep.png'}")
    print(f"\n{out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

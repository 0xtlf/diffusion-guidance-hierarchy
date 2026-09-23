"""Where does the score error live, and which part binds the tempering exponent?

Theorem 5.1 needs ||s - s*||_{L^inf(K)} = o(sigma^beta) and then admits any
alpha > max{-beta, 0}. Fitting that on the norm of the whole error vector gives
ONE number. But the error is a vector, and the three orthogonal parts of it do
different jobs:

    N_x M           confinement to the manifold
    span(P_T w)     confinement to the hyperplane, WITHIN the manifold
    T_x N           motion along the section -- where uniformity is decided

Confinement is already excellent: dist_M / sigma^(1 - alpha/2) is constant to
four significant figures across alpha. Uniformity fails on every plane. Those
two facts are hard to reconcile unless the error is concentrated tangentially,
in the one direction confinement cannot see. This script tests that by fitting
an exponent per part, and converting each to the alpha it demands.

Two things make the measurement honest.

The probe sits where the CHAIN sits, not where the data sits. A tempered chain
at exponent alpha equilibrates at distance sigma^(1 - alpha/2), which is 4 to 20
sigma -- and the theorem's norm is a sup over that region, not over the training
shell. Evaluating at data+noise (rho = 1) measures the one place the model was
never asked to work. So rho is swept explicitly.

Two guards, both learned the hard way. A sigma below the trained range does not
give a large error, it gives a CONSTANT one: the sigma-embedding extrapolates and
the network returns nearly the same vector wherever x is. Measured at sigma=0.003
against sigma_min=0.01, the error varied 1.02x across a 8x change in distance,
while a valid cell varies 13-120x. Silent, and it poisons every fit it enters. A
distance beyond the reach of M is the same kind of trap: P_M stops being unique
and "distance from M" stops meaning what the theory means by it. Both are skipped
and reported, never fitted.

The split is taken at P_M(x), not at x, so the frames are the manifold's own.
Note w is NOT orthogonal to N_x M: its tangential part is what defines the
section, so it is projected into T_x M and normalised before use, and the
residual tangent directions span T_x N.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import torch

from dgeom.config import load_config, resolve_device, seed_everything
from dgeom.experiment import load_model, make_loader, make_manifold, make_reference
from dgeom.geometry import Hyperplane
from dgeom.progress import write

BLOCKS = ("normal_M", "along_w", "tangent_N")

# Fraction of the manifold's own scale beyond which a normal displacement leaves
# the tubular neighbourhood: P_M is no longer unique and d_M is no longer the
# quantity the expansion is written in.
MAX_DIST_FRAC = 0.2
MIN_FIT_POINTS = 3


def shell(manifold, n, sigma, rho, gen):
    """Points at distance exactly ``rho * sigma`` from M, displaced normally.

    A normal displacement is the clean probe: it fixes the distance, which an
    isotropic one does not, and distance from M is the variable the theorem's
    region of validity is stated in.
    """
    x0 = manifold.sample_uniform(n, generator=gen)
    nb = manifold.normal_basis(x0)  # (n, d - dim, d)
    c = torch.randn(nb.shape[:2], generator=gen)
    c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return x0, x0 + rho * sigma * torch.einsum("bk,bkd->bd", c, nb)


def split(manifold, x0, err, w):
    """Norms of the three orthogonal parts of ``err``, in the frames at x0."""
    tb = manifold.tangent_basis(x0)  # (B, n, d)
    nb = manifold.normal_basis(x0)  # (B, d-n, d)

    e_normal = torch.einsum("bkd,bd->bk", nb, err).norm(dim=-1)

    wt = torch.einsum("bnd,d->bn", tb, w)  # w in the tangent frame
    wt = wt / wt.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    et = torch.einsum("bnd,bd->bn", tb, err)  # err in the tangent frame
    along = (et * wt).sum(-1)  # component along P_T w
    e_tangent = (et - along.unsqueeze(-1) * wt).norm(dim=-1)

    return {
        "normal_M": e_normal,
        "along_w": along.abs(),
        "tangent_N": e_tangent,
    }


def monotone(ys) -> bool:
    """Strictly increasing? A power law that is not is not a power law."""
    return all(a < b for a, b in itertools.pairwise(ys))


def fit(xs, ys):
    """Least-squares slope and intercept of log y against log x."""
    lx = [math.log(v) for v in xs]
    ly = [math.log(v) for v in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    den = sum((a - mx) ** 2 for a in lx)
    sl = sum((a - mx) * (b - my) for a, b in zip(lx, ly, strict=True)) / den
    return sl, math.exp(my - sl * mx)


def main() -> int:
    """Fit a score-error exponent per geometric block."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/manifold_klein.yaml")
    ap.add_argument("--load-from", default="runs/m-klein-180k")
    ap.add_argument("--n", type=int, default=2000, help="points per (sigma, rho) cell")
    ap.add_argument("--n-planes", type=int, default=16, help="w drawn per cell")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.01, 0.0316, 0.1, 1.0])
    ap.add_argument("--rhos", type=float, nargs="+", default=[1, 2, 5, 10, 20])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/error-anatomy")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    gen = torch.Generator().manual_seed(int(cfg["seed"]) + 907)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifold = make_manifold(cfg)
    loader = make_loader(cfg, manifold)
    reference = make_reference(cfg, manifold, loader)
    model, step = load_model(args.load_from, cfg, manifold, resolve_device(args.device))

    write(f"manifold {manifold.name}   model {args.load_from} @ step {step}")
    write(f"n={args.n} per cell, {args.n_planes} planes, dim {manifold.d}/{manifold.n}")
    write("rho = distance from M in units of sigma; the chain sits at 4-20\n")

    lo, hi = model.schedule.sigma_min, model.schedule.sigma_max
    max_dist = MAX_DIST_FRAC * manifold.scale
    write(f"trained sigma range [{lo:g}, {hi:g}]; max usable distance {max_dist:g}\n")

    rows: list[dict] = []
    skipped: list[str] = []
    hdr = f"{'sigma':>8} {'rho':>5} " + " ".join(f"{b:>11}" for b in BLOCKS)
    write(hdr)
    write("-" * len(hdr))
    for rho in args.rhos:
        for sigma in args.sigmas:
            if not model.schedule.contains(sigma):
                why = f"sigma={sigma:g} outside trained range [{lo:g}, {hi:g}]"
                skipped.append(f"  sigma={sigma:<8g} rho={rho:<4g} {why}")
                write(f"{sigma:8.4g} {rho:5g}   SKIPPED  {why}")
                continue
            if rho * sigma > max_dist:
                why = f"distance {rho * sigma:g} > {max_dist:g} (outside the reach)"
                skipped.append(f"  sigma={sigma:<8g} rho={rho:<4g} {why}")
                write(f"{sigma:8.4g} {rho:5g}   SKIPPED  {why}")
                continue
            x0, x = shell(manifold, args.n, sigma, rho, gen)
            s = torch.full((args.n,), float(sigma))
            err = model.shat(x, s).to(x.dtype) - reference.shat(x, s)
            acc = dict.fromkeys(BLOCKS, 0.0)
            for _ in range(args.n_planes):
                w = torch.randn(manifold.d, generator=gen)
                w = Hyperplane(w / w.norm()).w
                for b, v in split(manifold, x0, err, w).items():
                    acc[b] += float(v.mean()) / args.n_planes
            rows.append({"sigma": sigma, "rho": rho, **acc})
            write(
                f"{sigma:8.4g} {rho:5g} " + " ".join(f"{acc[b]:11.4e}" for b in BLOCKS)
            )
        write("")

    # ---------------------------------------------------------------- exponents
    write("\nfit over sigma at fixed rho.  hat-space ||e|| ~ C sigma^p, so the raw")
    write("score error is C sigma^(p-2), beta = p - 2, and Thm 5.1 needs")
    write("alpha > -beta = 2 - p.\n")
    hdr = f"{'rho':>5} " + " ".join(f"{b + ' p':>13}{'alpha>':>8}" for b in BLOCKS)
    write(hdr)
    write("-" * len(hdr))
    fits: list[dict] = []
    for rho in args.rhos:
        sel = sorted((r for r in rows if r["rho"] == rho), key=lambda r: r["sigma"])
        if len(sel) < MIN_FIT_POINTS:
            write(f"{rho:5g}   only {len(sel)} valid cell(s); no fit")
            continue
        xs = [r["sigma"] for r in sel]
        rec: dict = {"rho": rho, "n_points": len(sel)}
        line = f"{rho:5g} "
        for b in BLOCKS:
            ys = [r[b] for r in sel]
            p, c = fit(xs, ys)
            mono = monotone(ys)
            rec[b] = {
                "p": p,
                "C": c,
                "beta": p - 2,
                "alpha_min": 2 - p,
                "monotone": mono,
            }
            line += f"{p:13.3f}{2 - p:8.2f}{'' if mono else '*':<1}"
        fits.append(rec)
        write(line)
    write("  * not monotone in sigma: no power law, so the exponent is meaningless")

    if skipped:
        write(f"\n{len(skipped)} cell(s) skipped:")
        for line in skipped:
            write(line)

    usable = [
        (r["rho"], b, r[b]["alpha_min"])
        for r in fits
        for b in BLOCKS
        if r[b]["monotone"]
    ]
    if not usable:
        write("\nNo block admits a power law on any valid row. Nothing to fit:")
        write("widen the usable window (lower sigma_min and retrain) before reading")
        write("an exponent off this model.")
        (out / "anatomy.json").write_text(
            json.dumps(
                {"model": args.load_from, "step": step, "cells": rows, "fits": fits},
                indent=1,
            )
        )
        write(f"\n{out / 'anatomy.json'}")
        return 0
    binding = max(usable, key=lambda t: t[2])
    write(
        f"\nbinding: {binding[1]} at rho={binding[0]:g} demands "
        f"alpha > {binding[2]:.2f}"
    )
    spread = {
        b: (
            max(r[b]["alpha_min"] for r in fits if r[b]["monotone"])
            - min(r[b]["alpha_min"] for r in fits if r[b]["monotone"])
            if any(r[b]["monotone"] for r in fits)
            else None
        )
        for b in BLOCKS
    }
    write(f"spread of alpha_min across rho, per block: {spread}")
    write(
        "\nIf the three columns agree, there is no hierarchy and a single beta is "
        "the whole story.\nIf tangent_N demands more than normal_M, the error "
        "that breaks uniformity is not\nthe error that breaks confinement, and "
        "the binding column names the term to fix."
    )

    (out / "anatomy.json").write_text(
        json.dumps(
            {"model": args.load_from, "step": step, "cells": rows, "fits": fits},
            indent=1,
        )
    )
    write(f"\n{out / 'anatomy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the four figures the README cites, in both light and dark.

One figure per claim, and no figure without a claim. Each reads a run directory
that is still in the tree, so a figure can always be traced back to the numbers
that produced it and regenerated after a rerun.

Dark variants are separate renders from the same palette, not inverted images:
the README pairs them in a <picture> element so the figure follows the reader's
theme.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dgeom.viz.style import annotate_note, direct_label, use_style

OUT = Path("docs/figures")


def _save(fig, name: str, mode: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "-dark" if mode == "dark" else ""
    fig.savefig(OUT / f"{name}{suffix}.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- rates


def fig_rates(mode: str) -> None:
    """Guidance tracks geometry, not density: the headline claim."""
    p = use_style(mode)
    with open("runs/rates/rates.csv") as fh:
        rows = list(csv.DictReader(fh))
    s = [float(r["sigma"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for key, colour, label in (
        ("geometry", p["series"][0], "geometry"),
        ("guidance", p["series"][1], "guidance"),
        ("density", p["series"][2], "density"),
    ):
        y = [float(r[key]) for r in rows]
        ax.plot(s, y, color=colour, marker="o", markersize=3.5, zorder=3)
        # label at the RIGHT end: at small sigma geometry and guidance coincide
        # to 0.06%, which is the whole point, so labels there would collide.
        direct_label(ax, s[-1], y[-1], label, colour)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"noise level  $\sigma$")
    ax.set_ylabel(r"$\|\nabla \log p_\sigma\|$")
    ax.set_title("Guidance sits at the geometry rate")
    ax.set_xlim(s[0] * 0.8, s[-1] * 2.6)
    annotate_note(
        ax,
        "fitted slopes   geometry -0.970    guidance -1.024    density -0.038"
        "        guidance / geometry = 0.9994 at the smallest sigma",
        mode,
    )
    _save(fig, "rates", mode)


# ---------------------------------------------------------------- confinement


def fig_confinement(mode: str) -> None:
    """Both residuals follow sigma^(1 - alpha/2) with constant ratios."""
    p = use_style(mode)
    with open("runs/klein-p2-alpha/summary.json") as fh:
        data = json.load(fh)
    pts = []
    for r in data:
        m = re.search(r"alpha=([\d.]+)$", r["name"])
        if m and float(m.group(1)) > 0:
            pts.append((float(m.group(1)), r["dist_M"], r["abs_constraint"]))
    pts.sort()
    a = [q[0] for q in pts]
    pred = [0.01 ** (1 - x / 2) for x in a]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.2, 3.6))
    ax.plot(a, pred, color=p["muted"], ls=(0, (4, 3)), lw=1.6, zorder=2)
    direct_label(ax, a[-1], pred[-1], r"$\sigma^{1-\alpha/2}$", p["muted"], dy=-11)
    for idx, colour, label in (
        (1, p["series"][0], r"off $\mathcal{M}$"),
        (2, p["series"][1], r"off $H$"),
    ):
        y = [q[idx] for q in pts]
        ax.plot(a, y, color=colour, marker="o", markersize=4, zorder=3)
        direct_label(ax, a[-1], y[-1], label, colour)
    ax.set_yscale("log")
    ax.set_xlabel(r"tempering exponent  $\alpha$")
    ax.set_ylabel("residual")
    ax.set_title("Manifold and hyperplane confine alike")
    ax.set_xlim(0.25, 0.92)

    for idx, colour, label, ref in (
        (1, p["series"][0], r"dist$_\mathcal{M}\,/\,\sigma^{1-\alpha/2}$", 1.39),
        (2, p["series"][1], r"off $H$ / off $\mathcal{M}$", 0.603),
    ):
        y = [q[idx] / (pred[i] if idx == 1 else pts[i][1]) for i, q in enumerate(pts)]
        bx.plot(a, y, color=colour, marker="o", markersize=4, zorder=3)
        direct_label(bx, a[-1], y[-1], f"{label}\nflat at {ref}", colour, dy=0)
    bx.set_ylim(0, 1.8)
    bx.set_xlim(0.25, 1.02)
    bx.set_xlabel(r"tempering exponent  $\alpha$")
    bx.set_ylabel("ratio")
    bx.set_title("Both ratios constant in " + r"$\alpha$")
    _save(fig, "confinement", mode)


# ------------------------------------------------------------------- sweep


def fig_sweep(mode: str) -> None:
    """The negative result: every plane sits above the rejection threshold."""
    p = use_style(mode)
    with open("runs/klein-a07-long/summary.json") as fh:
        rows = json.load(fh)
    rows = sorted(rows, key=lambda r: r["ks_end"])
    y = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    thr = 0.0096
    ax.axvline(thr, color=p["critical"], lw=1.6, zorder=2)
    ax.text(
        thr * 1.12,
        -0.62,
        "KS rejects above this",
        color=p["critical"],
        fontsize=7.5,
        va="center",
    )
    for i, r in enumerate(rows):
        ax.plot(
            [r["ks_predicted"], r["ks_end"]],
            [i, i],
            color=p["grid"],
            lw=4,
            solid_capstyle="round",
            zorder=2,
        )
    ax.scatter(
        [r["ks_predicted"] for r in rows],
        y,
        s=46,
        color=p["series"][2],
        zorder=4,
        label="predicted bias (theory floor)",
    )
    ax.scatter(
        [r["ks_end"] for r in rows],
        y,
        s=46,
        color=p["series"][1],
        zorder=4,
        label="measured",
    )
    ax.set_yticks(y, [f"b = {r['offset']:+.3f}" for r in rows])
    ax.set_xscale("log")
    ax.set_xlabel("KS statistic $D$ against uniform on $N$")
    ax.set_title(r"Learned score, $\alpha=0.7$: 0 of 5 planes reach uniform")
    ax.set_ylim(-1.1, len(rows) - 0.4)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        fontsize=7.5,
        columnspacing=1.6,
    )
    ax.grid(axis="y", visible=False)
    _save(fig, "sweep", mode)


# ------------------------------------------------------------ error anatomy


def fig_anatomy(mode: str) -> None:
    """The diagnosis: the normal error stops falling and turns up at sigma_min."""
    p = use_style(mode)
    with open("runs/error-anatomy-sphere-v3/anatomy.json") as fh:
        cells = json.load(fh)["cells"]
    sel = sorted((c for c in cells if c["rho"] == 1), key=lambda c: c["sigma"])
    s = [c["sigma"] for c in sel]

    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    for key, colour, label in (
        ("normal_M", p["series"][0], r"normal to $\mathcal{M}$"),
        ("tangent_N", p["series"][1], r"tangent to $N$"),
        ("along_w", p["series"][2], r"along $w$"),
    ):
        y = [c[key] for c in sel]
        ax.plot(s, y, color=colour, marker="o", markersize=4, zorder=3)
        direct_label(ax, s[-1], y[-1], label, colour)
    nm = [c["normal_M"] for c in sel]
    i = nm.index(min(nm))
    ax.scatter(
        [s[i]],
        [nm[i]],
        s=120,
        facecolors="none",
        edgecolors=p["critical"],
        lw=1.6,
        zorder=5,
    )
    ax.annotate(
        f"minimum at $\\sigma$ = {s[i]:g}\n"
        f"{nm[0] / nm[i]:.1f}x worse at $\\sigma_{{\\min}}$",
        xy=(s[i], nm[i]),
        xytext=(14, -30),
        textcoords="offset points",
        fontsize=7.5,
        color=p["critical"],
        arrowprops={"arrowstyle": "-", "color": p["critical"], "lw": 0.9},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(s[0] * 0.85, s[-1] * 1.9)
    ax.set_xlabel(r"noise level  $\sigma$   ($\sigma_{\min}=0.01$)")
    ax.set_ylabel(r"$\|\hat{s}_\theta - \hat{s}^*\|$")
    ax.set_title("Score error does not vanish as " + r"$\sigma \to \sigma_{\min}$")
    _save(fig, "anatomy", mode)


def main() -> int:
    """Render every README figure in both modes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["light", "dark", "both"], default="both")
    args = ap.parse_args()
    modes = ["light", "dark"] if args.mode == "both" else [args.mode]
    for mode in modes:
        for fn in (fig_rates, fig_confinement, fig_sweep, fig_anatomy):
            fn(mode)
    n = len(list(OUT.glob("*.png")))
    print(f"{n} files in {OUT}")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f}  {f.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

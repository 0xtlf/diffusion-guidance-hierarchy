"""The rate figure: does guidance behave like geometry, or like density?

This is the measurement the conditional experiment exists to make, so the figure
is built to be read on its own. Two panels answer two different questions:

  left    how each term of grad log p_sigma scales. Absolute magnitudes on
          log-log, with the fitted exponent printed beside each series.
  right   the same three divided by the geometry term. "Guidance tracks
          geometry" becomes a FLAT LINE AT 1, which is far easier to judge by
          eye than two nearly-parallel lines on a log-log plot.

The exponent to expect is -1, not -2. A field whose stiffness is Theta(sigma^-2)
has magnitude Theta(sigma^-1) at a point noised to distance ~sigma, because
|grad| ~ delta / sigma^2 with delta ~ sigma. What decides the regime is not the
absolute value but whether guidance matches GEOMETRY or matches DENSITY.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .style import use_style

TERMS = ("geometry", "density", "guidance")


def fit_slopes(sigmas, series: dict) -> dict:
    """Least-squares exponent of each series against sigma, on log-log."""
    ls = np.log(np.asarray(sigmas))
    return {
        k: float(np.polyfit(ls, np.log(np.asarray(v)), 1)[0]) for k, v in series.items()
    }


def plot_rates(
    sigmas,
    series: dict,
    out_dir: str | Path,
    *,
    filename: str = "rates.png",
    bands: dict | None = None,
    slope_samples: dict | None = None,
    fit_max_sigma: float | None = None,
    title: str = "",
    subtitle: str = "",
    mode: str = "light",
) -> tuple[Path, dict]:
    """Draw the rate figure and write its companion CSV.

    Args:
        sigmas: the noise levels swept.
        series: ``{"geometry": [...], "density": [...], "guidance": [...]}``.
        bands: optional ``{term: (lo, hi)}`` spread across hyperplanes.
        slope_samples: optional ``{ensemble: {term: array of slopes}}``, drawn as
            a third panel so the error bar on the conclusion is visible.
        fit_max_sigma: if given, shade the asymptotic window the fit used.
        out_dir: directory to write into.
        filename: output file name.
        title: figure title.
        subtitle: configuration line under the title.
        mode: ``light`` or ``dark``.

    Returns:
        ``(path, slopes)``.
    """
    c = use_style(mode)
    s = np.asarray(sigmas)
    if fit_max_sigma is not None:
        m = s <= fit_max_sigma
        slopes = {
            k: float(np.polyfit(np.log(s[m]), np.log(np.asarray(v)[m]), 1)[0])
            for k, v in series.items()
        }
    else:
        slopes = fit_slopes(s, series)
    d_geo = abs(slopes["guidance"] - slopes["geometry"])
    d_den = abs(slopes["guidance"] - slopes["density"])
    regime_a = d_geo < d_den

    ncol = 3 if slope_samples else 2
    fig, axes = plt.subplots(1, ncol, figsize=(6.4 * ncol, 5.6))
    ax, bx = axes[0], axes[1]
    colour = dict(zip(TERMS, c["series"], strict=False))

    # ---- left: absolute magnitudes, log-log.
    # geometry and guidance land on top of each other -- that IS the result, so
    # guidance is drawn dashed over solid geometry to make the coincidence
    # visible rather than letting one curve hide the other.
    style = {
        "geometry": {"lw": 2.6, "ls": "-", "ms": 5, "zorder": 2},
        "density": {"lw": 2.2, "ls": "-", "ms": 4, "zorder": 2},
        "guidance": {"lw": 1.8, "ls": (0, (5, 3)), "ms": 4, "zorder": 3},
    }
    for k in TERMS:
        y = np.asarray(series[k])
        ax.plot(
            s,
            y,
            color=colour[k],
            marker="o",
            label=f"{k}   slope {slopes[k]:+.2f}",
            **style[k],
        )
        # direct-label at the RIGHT end, where the curves have separated
        ax.annotate(
            f"  {k}",
            (s[-1], y[-1]),
            color=colour[k],
            fontsize=9.5,
            va="center",
            ha="left",
            fontweight="bold",
            annotation_clip=False,
        )
    # slope guides, anchored away from the data
    g0 = np.asarray(series["geometry"])
    ref = g0[0] * 1.9 * (s / s[0]) ** -1.0
    ax.plot(s, ref, color=c["muted"], lw=1.1, ls=(0, (2, 3)), zorder=0)
    ax.annotate(
        "slope -1",
        (s[2], ref[2]),
        color=c["text_secondary"],
        fontsize=8,
        va="bottom",
        ha="left",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("noise level  sigma")
    ax.set_ylabel(r"$\|\nabla \log p_\sigma\|$   (mean over points)")
    ax.set_title("magnitude of each term of the score")
    if fit_max_sigma is not None:
        ax.axvspan(
            s[0] * 0.75, fit_max_sigma, color=c["muted"], alpha=0.09, lw=0, zorder=0
        )
        ax.annotate(
            "fit window",
            (s[0] * 0.85, ax.get_ylim()[0]),
            color=c["text_secondary"],
            fontsize=8,
            va="bottom",
            ha="left",
        )
    ax.set_xlim(s[0] * 0.75, s[-1] * 2.6)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)

    # ---- right: ratio to geometry. flat at 1 == identical rate AND size
    geo = np.asarray(series["geometry"])
    for k in ("guidance", "density"):
        y = np.asarray(series[k]) / geo
        bx.plot(
            s, y, color=colour[k], marker="o", ms=4, lw=2.0, label=f"{k} / geometry"
        )
        bx.annotate(
            f"  {k}",
            (s[-1], y[-1]),
            color=colour[k],
            fontsize=9.5,
            va="center",
            ha="left",
            fontweight="bold",
            annotation_clip=False,
        )
    bx.axhline(1.0, color=c["muted"], lw=1.4, ls=(0, (4, 3)))
    bx.annotate(
        "1.0  = same size and rate as geometry",
        (s[0], 1.0),
        color=c["text_secondary"],
        fontsize=8,
        va="bottom",
        ha="left",
    )
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlabel("noise level  sigma")
    bx.set_ylabel("term / geometry")
    bx.set_title("each term relative to geometry")
    bx.set_xlim(s[0] * 0.75, s[-1] * 2.6)
    bx.legend(fontsize=8.5, loc="lower right", framealpha=0.92)

    if slope_samples:
        cx = axes[2]
        offs = {"geometry": 0, "density": 1, "guidance": 2}
        for ename, marker in zip(slope_samples, ("o", "s"), strict=False):
            for k, sl in slope_samples[ename].items():
                y = np.asarray(sl)
                jit = (np.arange(len(y)) % 7 - 3) * 0.012
                cx.scatter(
                    y,
                    np.full_like(y, offs[k]) + jit + (0.18 if marker == "s" else -0.18),
                    s=9,
                    marker=marker,
                    color=colour[k],
                    alpha=0.55,
                    label=ename if k == "geometry" else None,
                )
        for k, o in offs.items():
            cx.annotate(
                k,
                (cx.get_xlim()[0], o),
                color=colour[k],
                fontsize=9.5,
                fontweight="bold",
                va="center",
                ha="left",
            )
        cx.set_yticks(list(offs.values()))
        cx.set_yticklabels(["", "", ""])
        cx.set_xlabel("fitted slope")
        cx.set_title("slope across hyperplanes (the error bar)")
        cx.legend(fontsize=8, loc="lower right", framealpha=0.92)
        cx.grid(axis="x", alpha=0.4)

    verdict = (
        f"guidance slope {slopes['guidance']:+.2f} is "
        f"{d_geo:.2f} from geometry and {d_den:.2f} from density  ->  "
        + (
            "GUIDANCE TRACKS GEOMETRY (conditioning adds codimension)"
            if regime_a
            else "GUIDANCE TRACKS DENSITY"
        )
    )
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.text(
        0.5,
        0.965,
        verdict,
        ha="center",
        fontsize=10,
        color=c["good"] if regime_a else c["critical"],
    )
    if subtitle:
        fig.text(
            0.5, 0.925, subtitle, ha="center", fontsize=8.5, color=c["text_secondary"]
        )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)

    with (out_dir / filename.replace(".png", ".csv")).open("w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["sigma", *TERMS])
        for i, sv in enumerate(s):
            wtr.writerow([sv, *(series[k][i] for k in TERMS)])
    return path, slopes


__all__ = ["fit_slopes", "plot_rates"]

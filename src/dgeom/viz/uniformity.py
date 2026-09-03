"""Is the sample uniform on the manifold? Answered as a picture, not a p-value.

Each manifold declares one-dimensional marginals whose density is known exactly
under the uniform measure. Plotting the empirical histogram against that curve
makes the answer immediate: on the line means uniform, off it means not.

The ratio panel is the sensitive one. A density that is 20% wrong is hard to see
against a curve but obvious as a flat line at 1.2, and the shaded band shows how
much scatter the sample size alone accounts for, so real deviation is separable
from noise by eye.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .style import use_style


def plot_uniformity(
    manifold,
    samples: dict[str, torch.Tensor],
    out_dir: str | Path,
    *,
    filename: str = "uniformity.png",
    title: str = "",
    subtitle: str = "",
    nbins: int = 48,
    mode: str = "light",
) -> Path:
    """Compare one or more samples against the exact uniform marginals.

    Args:
        manifold: supplies the marginals and their analytic densities.
        samples: label to ``(N, d)`` points; up to three are drawn.
        out_dir: directory to write into.
        filename: output file name.
        title: figure title.
        subtitle: smaller line under the title, e.g. the configuration.
        nbins: histogram resolution.
        mode: ``light`` or ``dark``.

    Returns:
        Path to the written figure.
    """
    marginals = manifold.uniform_marginals()
    if not marginals:
        raise ValueError(f"{manifold.name} declares no uniform marginals")

    colours = use_style(mode)
    names = list(samples)[:3]
    ncol = len(marginals)

    # the verdict belongs on the figure, not only in the terminal: a saved plot
    # has to be judgeable on its own months later
    stats = {n: uniformity_summary(manifold, samples[n], nbins) for n in names}
    legend = {
        n: f"{n}\n     {stats[n]['max_deviation']:.1%} off "
        f"({stats[n]['ratio']:.1f}x noise floor) - "
        f"{'UNIFORM' if stats[n]['uniform'] else 'NOT uniform'}"
        for n in names
    }
    fig, axes = plt.subplots(2, ncol, figsize=(4.3 * ncol, 6.0), squeeze=False)

    for col, (label, project, pdf, (lo, hi)) in enumerate(marginals):
        edges = np.linspace(lo, hi, nbins + 1)
        centres = 0.5 * (edges[1:] + edges[:-1])
        expected = pdf(centres)

        top, bottom = axes[0][col], axes[1][col]
        top.plot(
            centres,
            expected,
            color=colours["muted"],
            lw=2.0,
            ls=(0, (4, 3)),
            label="4. TARGET: uniform on M (exact, analytic)",
            zorder=5,
        )

        for i, n in enumerate(names):
            t = project(samples[n]).detach().cpu().double().numpy()
            dens, _ = np.histogram(t, bins=edges, density=True)
            colour = colours["series"][i % len(colours["series"])]
            top.step(centres, dens, where="mid", color=colour, lw=1.8, label=legend[n])

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(expected > 0, dens / expected, np.nan)
            bottom.step(centres, ratio, where="mid", color=colour, lw=1.8)

            if i == 0:  # noise band from this sample size, for reference
                n = len(t)
                counts = np.clip(expected * n * (edges[1] - edges[0]), 1e-9, None)
                band = 1.0 / np.sqrt(counts)
                bottom.fill_between(
                    centres,
                    1 - 2 * band,
                    1 + 2 * band,
                    color=colours["muted"],
                    alpha=0.18,
                    lw=0,
                    label="2 sigma sampling noise",
                    zorder=0,
                )

        top.set_xlabel(label)
        top.set_ylabel("density")
        top.set_title(f"marginal of {label}")
        top.set_xlim(lo, hi)
        if col == 0:
            top.legend(loc="lower center", fontsize=6.4)

        bottom.axhline(1.0, color=colours["muted"], lw=1.4, ls=(0, (4, 3)), zorder=4)
        bottom.set_xlabel(label)
        bottom.set_ylabel("empirical / uniform")
        bottom.set_title("ratio to uniform  (flat at 1 means uniform)")
        bottom.set_xlim(lo, hi)
        bottom.set_ylim(0.0, 2.0)
        if col == 0:
            bottom.legend(loc="upper center", fontsize=7.5)

    best = min(stats, key=lambda n: stats[n]["ratio"]) if stats else None
    if best is not None:
        head = (
            f"best: {best.split(':')[0]} at {stats[best]['max_deviation']:.1%} "
            f"= {stats[best]['ratio']:.1f}x the noise floor "
            f"({'UNIFORM' if stats[best]['uniform'] else 'NOT uniform'})"
        )
        fig.text(
            0.5,
            0.925,
            head,
            ha="center",
            fontsize=9.5,
            color=colours["good"]
            if stats[best]["uniform"]
            else colours["text_secondary"],
        )

    if title:
        fig.suptitle(title, fontsize=12, y=1.005)
    if subtitle:
        fig.text(
            0.5,
            0.965,
            subtitle,
            ha="center",
            fontsize=8.5,
            color=colours["text_secondary"],
        )
    fig.tight_layout(rect=(0, 0, 1, 0.915))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def uniformity_summary(manifold, x: torch.Tensor, nbins: int = 48) -> dict:
    """Departure from uniform, together with the noise floor that makes it readable.

    A raw maximum deviation is not interpretable on its own: with ``N`` points in
    ``nbins`` bins, even an exactly uniform sample deviates by roughly
    ``2 / sqrt(N / nbins)`` somewhere, purely from Poisson scatter. Reporting the
    ratio of the two is what turns the number into a verdict.

    Args:
        manifold: supplies the exact marginals.
        x: ``(N, d)`` points on the manifold.
        nbins: histogram resolution.

    Returns:
        ``max_deviation`` (largest relative departure), ``noise_floor``
        (what this sample size explains), ``ratio`` of the two, and ``uniform``
        which is True when the deviation is within ~1.5x the floor.
    """
    worst = 0.0
    n = int(x.shape[0])
    for _label, project, pdf, (lo, hi) in manifold.uniform_marginals():
        edges = np.linspace(lo, hi, nbins + 1)
        centres = 0.5 * (edges[1:] + edges[:-1])
        expected = pdf(centres)
        t = project(x).detach().cpu().double().numpy()
        dens, _ = np.histogram(t, bins=edges, density=True)
        keep = expected > 0.05 * expected.max()
        worst = max(worst, float(np.abs(dens[keep] / expected[keep] - 1).max()))

    floor = 2.0 / np.sqrt(max(n, 1) / nbins)
    return {
        "max_deviation": worst,
        "noise_floor": float(floor),
        "ratio": float(worst / floor) if floor else float("inf"),
        "uniform": bool(worst <= 1.5 * floor),
    }


def max_deviation(manifold, x: torch.Tensor, nbins: int = 48) -> float:
    """Largest relative departure from the uniform marginals."""
    return uniformity_summary(manifold, x, nbins)["max_deviation"]


def make_probe(manifold, n_probe: int | None = None, nbins: int = 48):
    """Build a ``probe`` for ``Sampler.sample``, recording progress toward uniform.

    Without this a corrector run reports only where it ended, which cannot
    distinguish a chain that converged to a biased limit from one that simply
    ran out of steps. Tracking the departure from uniform against step number
    separates the two: a plateau means converged, a descent means unfinished.

    Probe every point by default. Subsetting looks like a cheap win but sets the
    probe's own noise floor at ``2 sqrt(nbins / n_probe)``, and a floor above the
    deviation being measured makes the trace pure noise -- it will appear to
    plateau whatever the chain is doing. Reduce ``n_probe`` only for manifolds
    whose projection is an iterative search, and read the floor it reports.

    Args:
        manifold: supplies the exact marginals.
        n_probe: points used per probe; None uses all of them.
        nbins: histogram resolution.

    Returns:
        Callable mapping the current state to a dict of diagnostics.
    """

    def probe(x: torch.Tensor, step: int = 0) -> dict:
        y = x if n_probe is None or x.shape[0] <= n_probe else x[:n_probe]
        s = uniformity_summary(manifold, y, nbins)
        return {
            "max_deviation": s["max_deviation"],
            "noise_floor": s["noise_floor"],
        }

    return probe


def plot_convergence(
    traces: dict,
    out_dir: str | Path,
    *,
    filename: str = "convergence.png",
    title: str = "",
    subtitle: str = "",
    mode: str = "light",
) -> Path | None:
    """Departure from uniform against corrector step, one line per run.

    The shaded floor is what the probe's own sample size explains, so a curve
    that has settled into it has converged as far as the measurement can tell.

    Args:
        traces: label to ``Trace``; traces without probe data are skipped.
        out_dir: directory to write into.
        filename: output file name.
        title: figure title.
        subtitle: smaller line under the title.
        mode: ``light`` or ``dark``.

    Returns:
        Path to the figure, or None when nothing was recorded.
    """
    colours = use_style(mode)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    drawn, floor = False, None

    for i, (name, trace) in enumerate(traces.items()):
        steps = trace.column("step")
        dev = trace.column("max_deviation")
        if not steps or not dev:
            continue
        colour = colours["series"][i % len(colours["series"])]
        ax.plot(steps, dev, color=colour, lw=1.8, label=name)
        floors = trace.column("noise_floor")
        if floors:
            floor = floors[-1]
        drawn = True

    if not drawn:
        plt.close(fig)
        return None

    if floor is not None:
        ax.axhspan(
            0.0,
            floor,
            color=colours["muted"],
            alpha=0.18,
            lw=0,
            label="probe noise floor",
            zorder=0,
        )

    ax.set_xlabel("corrector step")
    ax.set_ylabel("max deviation from uniform")
    ax.set_yscale("log")
    ax.set_title(title or "convergence of the corrector")
    ax.legend(fontsize=8)

    if subtitle:
        fig.text(
            0.5,
            0.945,
            subtitle,
            ha="center",
            fontsize=8.5,
            color=colours["text_secondary"],
        )
    fig.tight_layout(rect=(0, 0, 1, 0.93 if subtitle else 1.0))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path

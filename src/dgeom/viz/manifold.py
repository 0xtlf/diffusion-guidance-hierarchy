"""The single most informative figure: what a trained model learned about M.

Four panels, each answering something the numbers alone do not:

  ambient projection   scatter of noisy inputs and where the score flow puts them
  distance to M        the flow collapsing onto the manifold, before against after
  Jacobian spectrum    eigenvalues of J = d shat/dx; the gap between the -1 and 0
                       clusters IS the codimension, read off the model with no
                       reference score involved

The flow uses 50 steps, not more: normal relaxation converges in about 30, while
the tangential part of shat accumulates and would sweep points into the density
modes, so a longer flow measures concentration rather than the manifold.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..metrics.manifold_quality import _jacobian
from .style import annotate_note, use_style

FLOW_STEPS = 50


def plot_learned_manifold(
    model,
    manifold,
    loader,
    out_dir: str | Path,
    *,
    n: int = 6000,
    sigma: float = 0.01,
    step: int | None = None,
    mode: str = "light",
    label: str = "",
    subtitle: str = "",
    verdict: str = "",
    generator: torch.Generator | None = None,
) -> tuple[Path, dict]:
    """Draw the four-panel summary of what a model learned.

    Args:
        model: any DiffusionModel; only ``shat`` is used.
        manifold: the true manifold, used solely to score the result.
        loader: supplies the starting points.
        out_dir: directory to write ``learned_<manifold>.png`` into.
        n: number of points to flow.
        sigma: noise level at which the score is queried.
        step: training step, for the title.
        mode: ``light`` or ``dark``.
        label: extra text for the title.
        subtitle: configuration line, so the figure is self-describing.
        verdict: headline pass/fail summary drawn beside the title.
        generator: RNG for reproducibility.

    Returns:
        The figure path and a dict of the summary statistics drawn.
    """
    g = generator or torch.Generator().manual_seed(7)
    palette = use_style(mode)

    x0 = loader.sample(n)
    noisy = x0 + (0.12 * manifold.scale) * torch.randn(x0.shape, generator=g)
    z = noisy.clone()
    for _ in range(FLOW_STEPS):
        z = z + 0.5 * model.shat(z, sigma)

    before = (noisy - manifold.project(noisy)).norm(dim=-1) / manifold.scale
    after = (z - manifold.project(z)).norm(dim=-1) / manifold.scale

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4))
    name = manifold.name

    for ax, (i, j) in zip(axes[0], ((0, 1), (2, 3)), strict=True):
        ax.scatter(
            noisy[:, i],
            noisy[:, j],
            s=3,
            alpha=0.25,
            linewidths=0,
            color=palette["muted"],
            label="noisy input",
            rasterized=True,
        )
        ax.scatter(
            z[:, i],
            z[:, j],
            s=3,
            alpha=0.55,
            linewidths=0,
            color=palette["series"][0],
            label="after flow",
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_xlabel(f"x{i}")
        ax.set_ylabel(f"x{j}")
        ax.set_title(f"ambient projection (x{i}, x{j})")
    axes[0, 0].legend(loc="upper right", markerscale=3)

    ax = axes[1, 0]
    bins = np.logspace(-6, 0.3, 60)
    ax.hist(
        before.numpy(),
        bins=bins,
        color=palette["muted"],
        alpha=0.75,
        label="before flow",
    )
    ax.hist(
        after.numpy().clip(1e-6, None),
        bins=bins,
        color=palette["series"][0],
        alpha=0.85,
        label="after flow",
    )
    ax.set_xscale("log")
    ax.set_xlabel("distance to M / scale")
    ax.set_ylabel("count")
    ax.set_title("The flow collapsing onto M")
    ax.legend(loc="upper left")
    annotate_note(
        ax, f"median {float(before.median()):.3f} -> {float(after.median()):.2e}", mode
    )

    ax = axes[1, 1]
    probe = manifold.sample_uniform(min(n, 700), generator=g)
    probe = probe + sigma * torch.randn(probe.shape, generator=g)
    jac = _jacobian(model, probe, sigma)
    evals = torch.linalg.eigvalsh(0.5 * (jac + jac.transpose(-1, -2))).numpy()
    codim = manifold.codim
    for k in range(manifold.d):
        colour = palette["series"][0] if k < codim else palette["series"][1]
        ax.hist(evals[:, k], bins=50, color=colour, alpha=0.7)
    for line in (-1.0, 0.0):
        ax.axvline(line, color=palette["muted"], ls=(0, (4, 3)), lw=1.2)
    ax.set_xlabel("eigenvalue of J = d shat/dx")
    ax.set_ylabel("count")
    ax.set_title(f"Jacobian spectrum: {codim} normal (-1), {manifold.n} tangent (0)")
    annotate_note(
        ax,
        "The gap between the clusters is the codimension, read off\n"
        "the model with no reference score involved.",
        mode,
    )

    extra = f"  [{label}]" if label else ""
    at = f"step {step}, " if step is not None else ""
    fig.suptitle(
        f"what the {name} model learned  -  {at}sigma={sigma:g}{extra}",
        fontsize=11.5,
        y=1.0,
    )
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"learned_{name}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)

    return path, {
        "dist_before_median": float(before.median()),
        "dist_after_median": float(after.median()),
        "dist_after_p95": float(after.quantile(0.95)),
        "eig_normal_mean": float(evals[:, :codim].mean()),
        "eig_tangent_mean": float(evals[:, codim:].mean()),
    }

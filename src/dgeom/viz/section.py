"""What the conditional submanifold N = M ∩ H actually looks like.

Every number elsewhere is a statistic about this object, so it is worth seeing.
The figure is also the fastest way to catch a tracing bug: a Klein section drawn
with the wrong seam handling shows visibly broken curves, which no summary
statistic makes obvious.

The two manifolds need different pictures because their sections differ in
dimension:

  sphere   N is a great S^2 inside w-perp. Drawn as its own coordinates, so a
           uniform sample looks uniform rather than foreshortened.
  klein    N is a CURVE with one to four components, whose count and total
           length change with w. Drawn in the (u, v) chart with components
           coloured, beside the ambient projection and the arclength density.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..geometry.intersection import KleinSection
from .style import use_style


def component_labels(section: KleinSection) -> np.ndarray:
    """Component index per traced segment, by union-find over shared grid edges."""
    parent: dict[int, int] = {}

    def find(i: int) -> int:
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in section._edge_ids:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    roots = [find(int(a)) for a, _ in section._edge_ids]
    order = {r: i for i, r in enumerate(sorted(set(roots)))}
    return np.array([order[r] for r in roots])


def _klein_panel(ax, section, colours, title):
    """Traced curve in the (u, v) chart, one colour per component."""
    lab = component_labels(section)
    a = section._seg_uv_a.numpy()
    b = section._seg_uv_b.numpy()
    for c in range(lab.max() + 1):
        m = lab == c
        col = colours["series"][c % len(colours["series"])]
        # draw each segment separately: the components wind through the chart
        # and joining them in index order would draw spurious chords
        for p, q in zip(a[m], b[m], strict=False):
            ax.plot(
                [p[0], q[0]], [p[1], q[1]], color=col, lw=1.6, solid_capstyle="round"
            )
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title(title, fontsize=9.5)
    ax.set_aspect("equal")


def plot_sections(
    manifold,
    hyperplanes,
    out_dir: str | Path,
    *,
    filename: str = "sections.png",
    grid: int = 400,
    mode: str = "light",
    generator=None,
) -> Path:
    """Draw the section produced by each hyperplane.

    Args:
        manifold: the ambient manifold.
        hyperplanes: the cuts to draw.
        out_dir: directory to write into.
        filename: output file name.
        grid: chart resolution for a Klein trace.
        mode: ``light`` or ``dark``.
        generator: RNG, used for the sphere scatter.

    Returns:
        Path to the written figure.
    """
    from ..geometry.intersection import section_for

    c = use_style(mode)
    n = len(hyperplanes)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.5 * cols, 3.6 * rows), squeeze=False
    )
    is_klein = manifold.name == "klein"

    for i, H in enumerate(hyperplanes):
        ax = axes[i // cols][i % cols]
        sec = section_for(manifold, H, **({"grid": grid} if is_klein else {}))
        if is_klein:
            k = sec.n_components()
            _klein_panel(
                ax,
                sec,
                c,
                f"{k} component{'s' if k > 1 else ''},  L = {sec.length:.2f}",
            )
        else:
            x = sec.sample_uniform(4000, generator=generator)
            u = (x @ sec._basis.T).numpy()
            ax.scatter(u[:, 0], u[:, 1], s=1.2, alpha=0.35, color=c["series"][0], lw=0)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_aspect("equal")
            ax.set_xlabel("<e0, x>")
            ax.set_ylabel("<e1, x>")
            ax.set_title("great S^2, always connected", fontsize=9.5)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    what = (
        "curves in the (u, v) chart; colour = connected component"
        if is_klein
        else "uniform samples in the w-perp basis"
    )
    fig.suptitle(
        f"sections N = {manifold.name} ∩ H for random hyperplanes through the origin",
        fontsize=12,
        y=1.005,
    )
    fig.text(0.5, 0.972, what, ha="center", fontsize=8.5, color=c["text_secondary"])
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


__all__ = ["component_labels", "plot_sections"]

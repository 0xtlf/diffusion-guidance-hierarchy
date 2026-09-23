"""Individual chart panels.

Each public function draws into a supplied axis so panels compose into figures.
"""

from __future__ import annotations

import numpy as np

from .style import direct_label, palette

# ------------------------------------------------------------------ line forms


def series_lines(
    ax,
    xs,
    series: dict[str, np.ndarray],
    *,
    mode="light",
    logy=False,
    xlabel="",
    ylabel="",
    title="",
    label_fmt="{}",
):
    """Up to three named series, always legended and always direct-labelled.

    Both, not either: the legend is the standing requirement for >= 2 series and
    the direct labels are the relief for the light aqua slot, which sits below
    3:1 against the surface.
    """
    p = palette(mode)
    ends = []
    for i, (name, ys) in enumerate(series.items()):
        c = p["series"][i % len(p["series"])]
        ax.plot(xs, ys, color=c, label=name, zorder=3 - i)
        finite = np.isfinite(ys)
        if finite.any():
            j = np.flatnonzero(finite)[-1]
            ends.append([xs[j], ys[j], label_fmt.format(name), c])
    if logy:
        ax.set_yscale("log")
    _place_end_labels(ax, ends)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(series) >= 2:
        ax.legend(loc="best")
    ax.margins(x=0.14)


def _place_end_labels(ax, ends, min_gap_px: float = 11.0) -> None:
    """Direct-label each series end, nudging apart labels that would overlap.

    Series that converge to the same value — which happens constantly here, since
    two candidates both being rejected puts both curves on the floor — would
    otherwise print their labels on top of each other and lose the identity that
    the labels exist to provide.
    """
    if not ends:
        return
    ax.figure.canvas.draw()
    trans = ax.transData
    order = sorted(range(len(ends)), key=lambda i: ends[i][1])
    ypx = [trans.transform((ends[i][0], ends[i][1]))[1] for i in order]
    for k in range(1, len(ypx)):
        if ypx[k] - ypx[k - 1] < min_gap_px:
            ypx[k] = ypx[k - 1] + min_gap_px
    for k, i in enumerate(order):
        x, y, text, c = ends[i]
        dy = ypx[k] - trans.transform((x, y))[1]
        direct_label(ax, x, y, text, c, dy=dy)


def threshold_line(ax, y: float, label: str, mode="light") -> None:
    """Draw a labelled dashed reference line at height ``y``."""
    p = palette(mode)
    ax.axhline(y, color=p["muted"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(
        label,
        xy=(0.995, y),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
        color=p["muted"],
    )

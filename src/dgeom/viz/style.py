"""Chart styling: one validated palette, light and dark.

Colours are assigned by the job they do, never picked per chart:

  categorical  identity  (which arm, which sigma band)  - fixed slot order
  diverging    polarity  (density above/below uniform)  - blue<->red, gray at 1
  sequential   magnitude (rarely needed here)           - one blue ramp

The three categorical slots are validated all-pairs in both modes (worst CVD
dE 9.2 light / 9.4 dark, normal-vision 24.0 / 20.9).  Three is the cap for
scatter and small multiples, and every chart here stays at or under it.  The
light aqua sits at 2.74:1 against the surface, below the 3:1 line, so the relief
rule applies: every series is direct-labelled and every figure writes a
companion CSV.  Identity is therefore never carried by colour alone.
"""

from __future__ import annotations

import matplotlib as mpl

LIGHT = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#8a8880",
    "grid": "#e6e5e1",
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    "diverging": ("#2a78d6", "#f0efec", "#e34948"),
    "good": "#008300",
    "critical": "#e34948",
}

DARK = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#8a8880",
    "grid": "#343430",
    "series": ["#3987e5", "#d95926", "#199e70"],
    "diverging": ("#3987e5", "#383835", "#e66767"),
    "good": "#3fa23f",
    "critical": "#e66767",
}


def palette(mode: str = "light") -> dict:
    """Colour tokens for a mode: surfaces, text, series, status."""
    return DARK if mode == "dark" else LIGHT


def use_style(mode: str = "light") -> dict:
    """Apply rcParams. Thin marks, recessive grid and axes, no chartjunk."""
    p = palette(mode)
    mpl.rcParams.update(
        {
            "figure.facecolor": p["surface"],
            "axes.facecolor": p["surface"],
            "savefig.facecolor": p["surface"],
            "text.color": p["text"],
            "axes.labelcolor": p["text_secondary"],
            "axes.edgecolor": p["grid"],
            "axes.titlecolor": p["text"],
            "xtick.color": p["text_secondary"],
            "ytick.color": p["text_secondary"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": p["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "medium",
            "axes.titlepad": 9,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
        }
    )
    return p


def direct_label(ax, x, y, text, color, dx: float = 0.0, dy: float = 0.0, **kw) -> None:
    """Label a series at its end.

    Required, not decorative: colour alone never carries identity, and the light
    aqua slot sits below the 3:1 contrast line.
    """
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6 + dx, dy),
        textcoords="offset points",
        color=color,
        fontsize=8,
        va="center",
        fontweight="medium",
        **kw,
    )


def annotate_note(ax, text: str, mode: str = "light") -> None:
    """Add a small muted caption beneath an axis."""
    ax.text(
        0.0,
        -0.22,
        text,
        transform=ax.transAxes,
        fontsize=7.5,
        color=palette(mode)["muted"],
        va="top",
        ha="left",
        wrap=True,
    )

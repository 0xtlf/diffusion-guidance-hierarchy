"""Figures: a validated palette, reusable panels, and per-experiment reports."""

from .manifold import plot_learned_manifold
from .panels import series_lines, status_dots, threshold_line
from .reports import build_reports, load_metrics
from .style import palette, use_style
from .uniformity import max_deviation, plot_uniformity, uniformity_summary

__all__ = [
    "build_reports",
    "load_metrics",
    "max_deviation",
    "palette",
    "plot_learned_manifold",
    "plot_uniformity",
    "series_lines",
    "status_dots",
    "threshold_line",
    "uniformity_summary",
    "use_style",
]

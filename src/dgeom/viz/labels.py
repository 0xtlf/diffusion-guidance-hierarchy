"""Turning a config into the one line a figure needs to be self-describing."""

from __future__ import annotations


def config_line(cfg: dict, extra: dict | None = None) -> str:
    """Summarise the parameters that determine what a figure shows.

    A figure without its configuration is not reproducible evidence: the same
    plot at a different sigma, alpha or network size means something different.

    Args:
        cfg: the resolved config.
        extra: anything experiment-specific to append, e.g. ``{"alpha": 0.5}``.

    Returns:
        A compact single line, for example
        ``manifold=klein | sigma in [0.01, 10] | net 384x5 | 60k steps, bs 1024``.
    """
    parts: list[str] = []
    if "manifold" in cfg:
        parts.append(f"manifold={cfg['manifold']}")

    diff = cfg.get("diffusion", {})
    if diff:
        parts.append(f"sigma in [{diff.get('sigma_min')}, {diff.get('sigma_max')}]")

    score = cfg.get("model", {}).get("score", {})
    if score:
        parts.append(f"net {score.get('width')}x{score.get('depth')}")

    train = cfg.get("train", {}).get("score", {})
    if train:
        steps = train.get("steps")
        shown = (
            f"{steps // 1000}k" if isinstance(steps, int) and steps >= 1000 else steps
        )
        parts.append(f"{shown} steps, bs {train.get('batch_size')}")
        if train.get("sigma_bias", 1.0) != 1.0:
            parts.append(f"sigma_bias {train['sigma_bias']}")

    kappa = cfg.get("data", {}).get("kappa_range")
    if kappa:
        parts.append(f"kappa {tuple(kappa)}")

    for key, value in (extra or {}).items():
        parts.append(f"{key}={value}")
    return "  |  ".join(str(p) for p in parts)


def verdict_line(passed: int, total: int) -> tuple[str, bool]:
    """Headline verdict and whether everything passed."""
    ok = passed == total
    return (f"{passed}/{total} gates passed{'' if ok else '  -- NOT YET'}"), ok

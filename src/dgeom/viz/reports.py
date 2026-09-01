"""Turn a run directory into figures.

Reads metrics.jsonl, detects which experiment produced it, and writes one PNG
per report plus a companion CSV.  The CSV is not optional: the light-mode aqua
slot sits below 3:1 against the surface, so the relief rule requires a table
view alongside every chart that uses it.

Every function degrades gracefully when a key is missing, so figures can be
produced from a run that is still in progress.
"""

from __future__ import annotations

import contextlib
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2 as _chi2

from .panels import series_lines, threshold_line
from .style import annotate_note, use_style

# a distance below this is statistically indistinguishable from the candidate
CHI2_ACCEPT = float(_chi2.ppf(0.95, 24))


def load_metrics(run_dir: str | Path) -> list[dict]:
    """Read a run's metrics.jsonl, skipping malformed lines."""
    p = Path(run_dir) / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            with contextlib.suppress(json.JSONDecodeError):
                out.append(json.loads(line))
    return out


def _stage(rows, name):
    return [r for r in rows if r.get("stage") == name]


def _write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _sigma_keys(row: dict, prefix: str) -> list[str]:
    return sorted(
        (k for k in row if k.startswith(prefix)), key=lambda k: float(k.split("_s")[-1])
    )


# ---------------------------------------------------------------------- e0


def report_manifold(rows, out: Path, mode="light") -> Path | None:
    """Quality of a trained manifold model: the gates, made visible."""
    fin = [r for r in rows if str(r.get("stage", "")).startswith("final_")]
    ev = [r for r in rows if str(r.get("stage", "")).startswith("eval_score_")]
    if not ev:
        return None
    name = ev[-1]["stage"].replace("eval_score_", "")
    use_style(mode)
    steps = np.array([r["step"] for r in ev])

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.2))

    # hat error per sigma: the tolerance the theory is stated in
    keys = _sigma_keys(ev[-1], "hat_err_mean_s")[:3]
    if keys:
        series_lines(
            axes[0, 0],
            steps,
            {
                f"sigma={k.split('_s')[-1]}": np.array([r.get(k, np.nan) for r in ev])
                for k in keys
            },
            mode=mode,
            logy=True,
            xlabel="step",
            ylabel="hat-space error",
            title="Error vs the reference score",
        )
        threshold_line(axes[0, 0], 5e-3, "gate", mode)

    series_lines(
        axes[0, 1],
        steps,
        {
            "median": np.array([r.get("flow_dist_median", np.nan) for r in ev]),
            "p95": np.array([r.get("flow_dist_p95", np.nan) for r in ev]),
        },
        mode=mode,
        logy=True,
        xlabel="step",
        ylabel="dist to M / scale",
        title="Where a deterministic flow lands",
    )
    threshold_line(axes[0, 1], 2e-3, "median gate", mode)

    # the Jacobian gates: direction (angle) and magnitude (eigenvalue)
    series_lines(
        axes[1, 0],
        steps,
        {"normal eigenvalue": np.array([r.get("eig_normal_mean", np.nan) for r in ev])},
        mode=mode,
        xlabel="step",
        ylabel="eigenvalue",
        title="Jacobian normal eigenvalue (J ~ -P_perp, want -1)",
    )
    threshold_line(axes[1, 0], -1.0, "exact", mode)
    threshold_line(axes[1, 0], -0.9, "gate", mode)
    annotate_note(
        axes[1, 0],
        "Magnitude, not direction: codimension and normal angle\n"
        "can both be perfect while this is far from -1.",
        mode,
    )

    series_lines(
        axes[1, 1],
        steps,
        {
            "normal angle (deg)": np.array(
                [r.get("normal_angle_deg_mean", np.nan) for r in ev]
            ),
            "covering ratio": np.array([r.get("cover_ratio", np.nan) for r in ev]),
        },
        mode=mode,
        logy=True,
        xlabel="step",
        ylabel="value",
        title="Normal-space angle and coverage",
    )

    sub = ""
    if fin:
        f = fin[-1]
        sub = (
            f"  |  codim correct {f.get('codim_correct_frac', float('nan')) * 100:.0f}%"
            f",  normal eig {f.get('eig_normal_mean', float('nan')):+.3f}"
        )
    fig.suptitle(f"manifold quality: {name}{sub}", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out / f"manifold_{name}.png")
    plt.close(fig)
    _write_table(out / f"manifold_{name}.csv", ev)
    return out / f"manifold_{name}.png"


def report_uniform_on_manifold(rows, out: Path, mode="light") -> Path | None:
    """E7: did the corrector reach the uniform measure, from every start?"""
    done = _stage(rows, "e7")
    if not done:
        return None
    use_style(mode)
    alphas = sorted({r["alpha"] for r in done})
    inits = list(dict.fromkeys(r["init"] for r in done))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    for phase, ax, title in [
        ("post", axes[0], "Uniformity on M (after projection)"),
        ("post", axes[1], "Distance to M"),
    ]:
        s = {}
        for init in inits:
            sub = sorted(
                [r for r in done if r["init"] == init and r["phase"] == phase],
                key=lambda r: r["alpha"],
            )
            if not sub:
                continue
            key = "chi2_joint_p" if ax is axes[0] else "dist_M"
            s[init] = np.array([max(r.get(key, np.nan), 1e-300) for r in sub])
        if s:
            series_lines(
                ax,
                np.array(alphas),
                s,
                mode=mode,
                logy=True,
                xlabel="alpha",
                ylabel=("p-value" if ax is axes[0] else "dist to M / scale"),
                title=title,
            )
    threshold_line(axes[0], 0.01, "accept above", mode)
    annotate_note(
        axes[0],
        "All initialisations must agree: that is the evidence the\n"
        "limit is the stationary distribution, not the start.",
        mode,
    )
    fig.suptitle("e7 uniform on the manifold", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "e7_uniform_on_M.png")
    plt.close(fig)
    _write_table(out / "e7_uniform_on_M.csv", done)
    return out / "e7_uniform_on_M.png"


def build_reports(run_dir: str | Path, mode: str = "light") -> list[Path]:
    """Build every figure that this run has data for. Returns their paths."""
    run_dir = Path(run_dir)
    rows = load_metrics(run_dir)
    out = run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (
        lambda: report_manifold(rows, out, mode),
        lambda: report_uniform_on_manifold(rows, out, mode),
    ):
        try:
            r = fn()
        except Exception as exc:  # a partial run should still yield its other figures
            print(f"  (skipped a report: {type(exc).__name__}: {exc})")
            r = None
        if r:
            made.append(r)
    return made

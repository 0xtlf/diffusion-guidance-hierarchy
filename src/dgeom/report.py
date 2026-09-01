"""Gather every result in runs/ into one status view.

Reads what has already been logged rather than recomputing it, so the default
path is fast and can be run against jobs that are still going. Anything
expensive (re-auditing a checkpoint, regenerating figures) is opt-in.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

GATE_ORDER = [
    "flow_dist_median",
    "flow_dist_p95",
    "codim_correct_frac",
    "normal_angle_deg_mean",
    "eig_normal_mean",
    "hat_err_mean",
    "cover_ratio",
]
GATE_SMALLER = {
    "flow_dist_median": True,
    "flow_dist_p95": True,
    "codim_correct_frac": False,
    "normal_angle_deg_mean": True,
    "eig_normal_mean": True,
    "hat_err_mean": True,
    "cover_ratio": True,
}

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, RED, YELLOW = "\033[32m", "\033[31m", "\033[33m"


def rows_of(d: Path) -> list[dict]:
    """Read a run's metrics.jsonl, skipping malformed lines."""
    f = d / "metrics.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        line = line.strip()
        if line:
            with contextlib.suppress(json.JSONDecodeError):
                out.append(json.loads(line))
    return out


def find_runs(root: Path) -> list[Path]:
    """Run directories that contain metrics, newest first."""
    if not root.exists():
        return []
    return sorted(
        (d for d in root.iterdir() if d.is_dir() and (d / "metrics.jsonl").exists()),
        key=lambda d: (d / "metrics.jsonl").stat().st_mtime,
        reverse=True,
    )


def manifold_summary(root: Path) -> dict[str, dict]:
    """Latest gate values per manifold, preferring a standalone audit if present."""
    out: dict[str, dict] = {}
    for d in find_runs(root):
        rows = rows_of(d)
        fin = [r for r in rows if str(r.get("stage", "")).startswith("final_")]
        ev = [r for r in rows if str(r.get("stage", "")).startswith("eval_score_")]
        if not (fin or ev):
            continue
        src = fin[-1] if fin else ev[-1]
        name = str(src.get("stage", "")).split("_")[-1]
        if name in out:  # runs are newest-first; keep the newest
            continue
        # audit log, if the run was audited afterwards
        audit = None
        for log in (root / f"{d.name.replace('-', '_')}.log", root / f"{d.name}.log"):
            if log.exists() and "MANIFOLD LEARNED" in log.read_text():
                audit = "LEARNED"
            elif log.exists() and "NOT YET" in log.read_text():
                audit = "NOT YET"
        steps = src.get("step") or (ev[-1].get("step") if ev else None)
        out[name] = {
            "run": d,
            "metrics": src,
            "steps": steps,
            "audit": audit,
            "n_evals": len(ev),
        }
    return out


def corrector_summary(root: Path) -> list[dict]:
    """Every logged corrector (e7) result across all runs."""
    out = []
    for d in find_runs(root):
        for r in rows_of(d):
            if r.get("stage") == "e7":
                out.append({**r, "run": d.name})
    return out


def geometry_gate(repo: Path) -> str | None:
    """Re-run the Klein geometry self-test; it is seconds and gates everything."""
    try:
        p = subprocess.run(
            [sys.executable, "experiments/validate_geometry.py"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=600,
        )
        tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        return tail
    except Exception as exc:
        return f"could not run: {exc}"


def fmt_gate(name: str, value, target) -> tuple[str, bool]:
    """Format a gate value and say whether it passes."""
    if value is None:
        return "-", False
    ok = (value <= target) if GATE_SMALLER[name] else (value >= target)
    return f"{value:.4g}", ok


def render(root: Path, gates: dict, geom: str | None, color: bool = True) -> str:
    """Render the whole status view as text."""

    def c(code: str, s: str) -> str:
        return f"{code}{s}{RESET}" if color else s

    L: list[str] = []
    L.append(c(BOLD, "DGEOM STATUS") + f"   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")

    if geom:
        ok = "PASSED" in geom
        L.append(c(BOLD, "geometry gate"))
        L.append(f"  klein self-test   {c(GREEN if ok else RED, geom)}")
        L.append("")

    man = manifold_summary(root)
    L.append(
        c(BOLD, "manifold models")
        + c(DIM, "   (unconditional score, the prerequisite)")
    )
    if not man:
        L.append("  none trained yet")
    else:
        names = [n for n in ("sphere", "klein") if n in man] + [
            n for n in man if n not in ("sphere", "klein")
        ]
        w = max(len(g) for g in GATE_ORDER) + 2
        head = f"  {'gate':<{w}}" + "".join(f"{n:>16}" for n in names)
        L.append(head)
        L.append("  " + "-" * (w + 16 * len(names)))
        allok = dict.fromkeys(names, True)
        for g in GATE_ORDER:
            cells = []
            for n in names:
                m = man[n]["metrics"]
                key = g
                if g == "hat_err_mean":
                    ks = [k for k in m if k.startswith("hat_err_mean_s")]
                    key = (
                        sorted(ks, key=lambda k: float(k.split("_s")[-1]))[0]
                        if ks
                        else g
                    )
                v = m.get(key)
                txt, ok = (
                    fmt_gate(g, v, gates.get(g)) if v is not None else ("-", False)
                )
                if v is not None and not ok:
                    allok[n] = False
                cells.append(
                    c(
                        GREEN
                        if (v is not None and ok)
                        else (RED if v is not None else DIM),
                        f"{txt:>16}",
                    )
                )
            L.append(f"  {g:<{w}}" + "".join(cells))
        L.append("  " + "-" * (w + 16 * len(names)))
        verdicts = []
        for n in names:
            a = man[n]["audit"]
            v = a if a else ("LEARNED" if allok[n] else "not yet")
            verdicts.append(c(GREEN if "LEARN" in str(v) else YELLOW, f"{v:>16}"))
        L.append(f"  {'verdict':<{w}}" + "".join(verdicts))
        L.append(
            f"  {'steps':<{w}}" + "".join(f"{man[n]['steps']!s:>16}" for n in names)
        )
        L.append(
            f"  {'run':<{w}}" + "".join(f"{man[n]['run'].name:>16}" for n in names)
        )
    L.append("")

    corr = corrector_summary(root)
    L.append(c(BOLD, "corrector -> uniform on M"))
    if not corr:
        L.append("  no e7 results yet")
    else:
        L.append(
            c(
                DIM,
                "  newest run first; older rows are kept so a stale "
                "result is never mistaken for a current one",
            )
        )
        L.append(
            f"  {'run':>22} {'manifold':>9} {'alpha':>6} {'init':>8} "
            f"{'phase':>5} {'dist_M':>10} {'p(uniform)':>11} {'verdict':>9}"
        )
        for r in corr[:24]:
            p = min(r.get("ks_u_p", 1), r.get("ks_v_p", 1), r.get("chi2_joint_p", 1))
            good = p > 0.01
            L.append(
                f"  {r.get('run', '?')[:22]:>22} {r.get('manifold', '?'):>9} "
                f"{r.get('alpha', 0):6.2f} "
                f"{r.get('init', '?'):>8} {r.get('phase', '?'):>5} "
                f"{r.get('dist_M', float('nan')):10.2e} {p:11.3e} "
                + c(GREEN if good else YELLOW, f"{'UNIFORM' if good else 'reject':>9}")
            )
    L.append("")

    figs = sorted(root.glob("*/figures/*.png"))
    L.append(c(BOLD, "figures") + c(DIM, f"   ({len(figs)})"))
    for f in figs[:14]:
        L.append(f"  {f}")
    if len(figs) > 14:
        L.append(c(DIM, f"  ... and {len(figs) - 14} more"))
    return "\n".join(L)

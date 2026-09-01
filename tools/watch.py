"""Live progress monitor for every run in runs/.

    python experiments/watch.py            # one snapshot
    python experiments/watch.py -f         # refresh until interrupted
    python experiments/watch.py -f -n 2    # every 2 seconds

Reads metrics.jsonl (which every experiment appends to as it goes) and the
resolved config, so progress is visible even though the experiments buffer their
stdout.  A run is marked LIVE if a matching process is running.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, YELLOW, BLUE = "\033[32m", "\033[33m", "\033[34m"


def _rows(p: Path) -> list[dict]:
    try:
        return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    except Exception:
        return []


def _cfg(d: Path) -> dict:
    f = d / "config.resolved.yaml"
    if not f.exists():
        return {}
    try:
        return (yaml.safe_load(f.read_text()) or {}).get("config", {})
    except Exception:
        return {}


def _live_scripts() -> set[str]:
    try:
        out = subprocess.run(
            ["ps", "-Ao", "command"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return set()
    names = set()
    for line in out.splitlines():
        for tag in (
            "train",
            "validate_reference",
        ):
            if tag in line and "watch.py" not in line:
                names.add(tag)
    return names


def _bar(frac: float, width: int = 24) -> str:
    frac = max(0.0, min(1.0, frac))
    n = round(frac * width)
    return "[" + "#" * n + "." * (width - n) + f"] {frac * 100:5.1f}%"


def describe(d: Path, live: set[str]) -> list[str]:
    """Format one run's progress as display lines."""
    rows = _rows(d / "metrics.jsonl")
    if not rows:
        return []
    cfg = _cfg(d)
    out: list[str] = []

    for tag in ("score", "classifier"):
        tr = [r for r in rows if r.get("stage") == f"train_{tag}"]
        ev = [r for r in rows if r.get("stage") == f"eval_{tag}"]
        if not tr:
            continue
        total = int(cfg.get("train", {}).get(tag, {}).get("steps", 0)) or tr[-1]["step"]
        last = tr[-1]
        frac = last["step"] / total
        rate = last.get("steps_per_s", 0) or 1
        eta = (total - last["step"]) / rate
        flag = f"{GREEN}LIVE{RESET}" if "train_models" in live and frac < 1 else "done"
        out.append(
            f"  {BOLD}{tag:<11}{RESET} {_bar(frac)} "
            f"{last['step']:>7,}/{total:,}  loss {last['loss']:.4f}  "
            f"{rate:.0f} it/s  eta {eta / 60:5.1f}m  {flag}"
        )
        if ev:
            keys = [
                k
                for k in ev[-1]
                if k.startswith(
                    ("hat_err_s", "ghat_err_s", "cos_dM_s", "cos_steer_s", "acc_s")
                )
            ]
            keys = sorted(keys)[:4]
            if keys:
                out.append(
                    "              "
                    + DIM
                    + "  ".join(f"{k}={ev[-1][k]:.4g}" for k in keys)
                    + RESET
                )

    e1 = [r for r in rows if r.get("stage") == "e1"]
    if e1:
        arms = cfg.get("grid", {}).get("arms", [])
        total = max(len(arms) * 2, len(e1))
        flag = (
            f"{GREEN}LIVE{RESET}"
            if "e1_temper_grid" in live and len(e1) < total
            else "done"
        )
        out.append(
            f"  {BOLD}e1 grid{RESET}     {_bar(len(e1) / total)} "
            f"{len(e1)}/{total} (arm,start) runs  {flag}"
        )
        for r in e1[-3:]:
            out.append(
                f"              {DIM}({r['alpha_score']:g},{r['alpha_guide']:g}) "
                f"from {r['start']:<9} -> {r['best_match']:<9} "
                f"moved={r['moved']:.4g}{RESET}"
            )
        tr = [r for r in rows if r.get("stage") == "e1_trace"]
        if tr and len(e1) < total:
            out.append(
                f"              {DIM}in-flight arm at corrector step "
                f"{tr[-1]['step']:,}{RESET}"
            )

    e5 = [r for r in rows if r.get("stage") == "e5"]
    if e5:
        total = int(cfg.get("inference", {}).get("n_hyperplanes", len(e5)))
        flag = (
            f"{GREEN}LIVE{RESET}"
            if "e5_inference" in live and len(e5) < total
            else "done"
        )
        npass = sum(1 for r in e5 if r.get("p_uniform", 0) > 0.05)
        out.append(
            f"  {BOLD}e5 inference{RESET} {_bar(len(e5) / total)} "
            f"{len(e5)}/{total} hyperplanes  {npass} uniform (p>0.05)  {flag}"
        )
        for r in e5[-2:]:
            out.append(
                f"              {DIM}b={r['b']:+.3f}  p(uniform)={r['p_uniform']:.3e}  "
                f"dist_N={r['dist_N_mean']:.2e}{RESET}"
            )

    e0 = [r for r in rows if r.get("stage") == "e0" and "sigma" in r]
    if e0:
        gate = next((r for r in rows if "worst_z" in r), None)
        v = (
            f"worst |z|={gate['worst_z']:.2f} "
            f"{GREEN + 'PASS' + RESET if gate.get('passed') else YELLOW + 'FAIL' + RESET}"  # noqa: E501
            if gate
            else "running"
        )
        out.append(f"  {BOLD}e0 gate{RESET}     {len(e0)} sigmas  {v}")

    e2 = [r for r in rows if r.get("stage") == "e2"]
    if e2:
        out.append(
            f"  {BOLD}e2 tolerance{RESET} {len(e2)} settings  "
            f"last amp={e2[-1]['amp']:g} -> {e2[-1]['best_match']}"
        )
    return out


def snapshot(root: Path) -> str:
    """Render the current state of every run."""
    live = _live_scripts()
    lines = [
        f"{BOLD}dgeom runs{RESET}  {time.strftime('%H:%M:%S')}"
        f"   {DIM}({len(live)} process(es) live){RESET}",
        "",
    ]
    dirs = sorted(
        (d for d in root.iterdir() if d.is_dir() and (d / "metrics.jsonl").exists()),
        key=lambda d: (d / "metrics.jsonl").stat().st_mtime,
        reverse=True,
    )
    shown = 0
    for d in dirs:
        body = describe(d, live)
        if not body:
            continue
        age = time.time() - (d / "metrics.jsonl").stat().st_mtime
        mark = f"{BLUE}*{RESET}" if age < 120 else " "
        lines.append(f"{mark} {d.name}  {DIM}(updated {age:.0f}s ago){RESET}")
        lines += [*body, ""]
        shown += 1
        if shown >= 6:
            break
    if shown == 0:
        lines.append("  no runs with metrics yet")
    return "\n".join(lines)


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("-f", "--follow", action="store_true")
    ap.add_argument("-n", "--interval", type=float, default=5.0)
    args = ap.parse_args()
    root = Path(args.root)

    if not args.follow:
        print(snapshot(root))
        return 0
    try:
        while True:
            sys.stdout.write("\033[H\033[J" + snapshot(root) + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

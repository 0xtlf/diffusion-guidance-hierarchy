"""Build figures for one or more run directories.

    python experiments/plot.py runs/main runs/e1-full-*   # explicit
    python experiments/plot.py --all                      # every run dir
    python experiments/plot.py runs/main --mode dark

Figures and their companion CSVs land in <run>/figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dgeom.viz import build_reports


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help="every directory under runs/")
    ap.add_argument("--mode", choices=["light", "dark"], default="light")
    ap.add_argument("--root", default="runs")
    args = ap.parse_args()

    dirs = [Path(r) for r in args.runs]
    if args.all or not dirs:
        dirs = sorted(
            d
            for d in Path(args.root).iterdir()
            if d.is_dir() and (d / "metrics.jsonl").exists()
        )
    if not dirs:
        print("no run directories with metrics.jsonl found")
        return 1

    for d in dirs:
        made = build_reports(d, args.mode)
        if made:
            print(f"{d}:")
            for m in made:
                print(f"   {m}")
        else:
            print(f"{d}: nothing to plot yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

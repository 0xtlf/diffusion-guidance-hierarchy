"""One command for the whole picture.

    python experiments/report.py                 # fast: read what is already logged
    python experiments/report.py --figures       # also (re)build every figure
    python experiments/report.py --audit         # also re-audit each checkpoint
    python experiments/report.py --all           # everything
    python experiments/report.py --md report.md  # write a shareable digest

Default is fast and safe to run against jobs that are still going: it reads
metrics.jsonl rather than recomputing. --audit reloads each checkpoint and
re-measures the gates with the CURRENT metric code, which is the only way to
know a model was judged by the current definitions.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dgeom.config import load_config
from dgeom.report import (
    find_runs,
    geometry_gate,
    render,
)

REPO = Path(__file__).resolve().parents[1]

CONFIGS = {
    "sphere": "configs/manifold_sphere.yaml",
    "klein": "configs/manifold_klein.yaml",
}


def run(cmd: list[str], quiet: bool = True) -> str:
    """Run a subprocess in the repository root and return its stdout."""
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if not quiet and p.returncode != 0:
        print(f"  (failed: {' '.join(cmd)})")
    return p.stdout


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--figures", action="store_true", help="rebuild all figures")
    ap.add_argument("--audit", action="store_true", help="re-audit checkpoints")
    ap.add_argument("--geometry", action="store_true", help="re-run the geometry gate")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--md", default=None, help="also write a markdown digest here")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.all:
        args.figures = args.audit = args.geometry = True

    root = REPO / args.root

    if args.audit:
        print("re-auditing checkpoints ...", flush=True)
        for name, cfg in CONFIGS.items():
            for d in find_runs(root):
                if not (d / "ckpt" / f"score_{name}.pt").exists():
                    continue
                out = run(
                    [
                        sys.executable,
                        "experiments/audit.py",
                        str(d),
                        "--config",
                        cfg,
                        "--n",
                        "4096",
                        "--n-flow",
                        "8192",
                    ]
                )
                verdict = [ln for ln in out.splitlines() if "gates passed" in ln]
                print(
                    f"  {d.name:<16} "
                    f"{verdict[0].strip() if verdict else '(no verdict)'}"
                )
                break  # newest run holding this manifold's checkpoint

    if args.figures:
        print("building figures ...", flush=True)
        run([sys.executable, "tools/plot.py", "--all"])
        for name, cfg in CONFIGS.items():
            for d in find_runs(root):
                if (d / "ckpt" / f"score_{name}.pt").exists():
                    subprocess.run(
                        [
                            sys.executable,
                            "tools/visualize.py",
                            str(d),
                            "--config",
                            cfg,
                        ],
                        cwd=REPO,
                        capture_output=True,
                        text=True,
                    )
                    break

    gates = load_config(REPO / CONFIGS["sphere"])["gates"]
    geom = geometry_gate(REPO) if args.geometry else None
    text = render(root, gates, geom, color=not args.no_color)
    print(text)

    if args.md:
        plain = render(root, gates, geom, color=False)
        figs = sorted(root.glob("*/figures/*.png"))
        md = ["# dgeom report", "", "```", plain, "```", "", "## Figures", ""]
        md += [f"- `{f.relative_to(REPO)}`" for f in figs]
        Path(args.md).write_text("\n".join(md))
        print(f"\nwrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

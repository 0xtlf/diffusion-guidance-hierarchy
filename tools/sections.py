"""Draw the conditional submanifolds a random hyperplane cuts out.

    python tools/sections.py klein --n-planes 8

Purely geometric: no model, no checkpoint. Useful on its own to see how the
section's topology varies with the hyperplane, and as the fastest check that
the Klein tracer is behaving.
"""

from __future__ import annotations

import argparse

import torch

from dgeom.config import load_config, seed_everything
from dgeom.experiment import make_manifold
from dgeom.geometry import Hyperplane, section_for
from dgeom.viz.section import plot_sections


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("manifold", choices=["sphere", "klein"])
    ap.add_argument("--n-planes", type=int, default=8)
    ap.add_argument("--grid", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["light", "dark"], default="light")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    cfg = load_config(f"configs/manifold_{args.manifold}.yaml")
    seed_everything(int(cfg["seed"]))
    M = make_manifold(cfg)
    gen = torch.Generator().manual_seed(int(cfg["seed"]) + 101)
    planes = [Hyperplane.random(4, generator=gen) for _ in range(args.n_planes)]

    out = args.out or f"runs/m-{args.manifold}/figures"
    path = plot_sections(
        M,
        planes,
        out,
        filename="sections.png",
        grid=args.grid,
        mode=args.mode,
        generator=gen,
    )
    print(path)
    if args.manifold == "klein":
        print(f"  {'plane':>6} {'components':>11} {'length':>9}")
        for i, H in enumerate(planes):
            s = section_for(M, H, grid=args.grid)
            print(f"  {i:>6} {s.n_components():>11} {s.length:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

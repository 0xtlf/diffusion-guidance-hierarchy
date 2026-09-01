"""Check a trained model against the quality gates, on demand.

    python experiments/audit.py runs/m2-klein \
        --config configs/manifold_klein.yaml

Deliberately separate from training: gates evaluated inside a training process
use whatever code that process imported at launch, so a metric fixed mid-run is
silently ignored. Reloading the checkpoint is the only way to know a model was
judged by the current definitions.
"""

from __future__ import annotations

import argparse

import torch

from dgeom.config import load_config, resolve_device, seed_everything
from dgeom.experiment import (
    check_gates,
    load_model,
    make_loader,
    make_manifold,
    make_reference,
    report_gates,
)
from dgeom.training.manifold_training import evaluate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--n-flow", type=int, default=4096)
    ap.add_argument("--raw", action="store_true", help="audit raw weights, not EMA")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    device = resolve_device(cfg.get("device", "auto"))
    manifold = make_manifold(cfg)
    loader = make_loader(cfg, manifold)
    model, step = load_model(args.run_dir, cfg, manifold, device, use_ema=not args.raw)
    gen = torch.Generator().manual_seed(int(cfg["seed"]) + 5)

    print(
        f"audit {args.run_dir}  manifold={manifold.name} "
        f"(d={manifold.d}, n={manifold.n}, codim={manifold.d - manifold.n})"
    )
    print(
        f"  checkpoint step {step}, {'raw' if args.raw else 'EMA'} weights, "
        f"n={args.n}/sigma, n_flow={args.n_flow}\n"
    )

    tc = cfg["train"]["score"]
    m = evaluate(
        model,
        manifold,
        loader,
        make_reference(cfg, manifold, loader),
        tc["eval_sigmas"],
        args.n,
        args.n_flow,
        gen,
    )

    print("geometry")
    print(
        f"  flow to M      median {m['flow_dist_median']:.3e}   "
        f"p95 {m['flow_dist_p95']:.3e}   max {m['flow_dist_max']:.3e}"
    )
    print(
        f"  Jacobian       normal eig {m['eig_normal_mean']:+.4f} (want -1)   "
        f"tangent eig {m['eig_tangent_mean']:+.4f} (want 0)"
    )
    print(f"  codimension    correct in {m['codim_correct_frac'] * 100:.1f}% of probes")
    print(
        f"  normal space   angle {m['normal_angle_deg_mean']:.3f} deg "
        f"(p95 {m['normal_angle_deg_p95']:.3f})"
    )
    print(
        f"  coverage       p95 radius {m['cover_radius_p95']:.4f}  "
        f"ratio {m['cover_ratio']:.2f}"
    )
    print("\nscore error vs reference (hat space; o(1) is the tolerated regime)")
    for s in tc["eval_sigmas"]:
        print(
            f"  sigma={s:<6g} mean {m[f'hat_err_mean_s{s:g}']:.3e}   "
            f"p95 {m[f'hat_err_p95_s{s:g}']:.3e}   "
            f"rel {m[f'hat_err_rel_s{s:g}']:.3f}"
        )

    failed = report_gates(
        check_gates(m, cfg["gates"], cfg["diffusion"]["sigma_min"]), manifold.name
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

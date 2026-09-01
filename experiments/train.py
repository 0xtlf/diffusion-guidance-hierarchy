"""Train an unconditional diffusion model on one manifold, then report the gates.

    python experiments/train_manifold.py --config configs/manifold_sphere.yaml
    python experiments/train_manifold.py --config configs/manifold_klein.yaml \
        --set train.score.steps=120000 train.score.sigma_bias=2.0

Exits non-zero if any gate fails, so an iteration loop can be driven from the
exit status rather than by reading numbers off a log.
"""

from __future__ import annotations

import argparse

import torch

from dgeom.config import setup
from dgeom.experiment import (
    check_gates,
    load_model,
    make_loader,
    make_manifold,
    make_reference,
    report_gates,
)
from dgeom.training.manifold_training import evaluate, train


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/manifold_sphere.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg, run, device = setup(args.config, args.overrides, default_name="manifold")
    manifold = make_manifold(cfg)
    loader = make_loader(cfg, manifold)
    print(f"run dir: {run.dir}\n")

    train(cfg, run, device, manifold, loader)

    # final evaluation on the EMA weights, which is what downstream loads
    model, _ = load_model(run.dir, cfg, manifold, device)
    tc = cfg["train"]["score"]
    gen = torch.Generator().manual_seed(int(cfg["seed"]))
    final = evaluate(
        model,
        manifold,
        loader,
        make_reference(cfg, manifold, loader),
        tc["eval_sigmas"],
        tc["eval_n"] * 2,
        tc.get("eval_n_flow", 2048) * 2,
        gen,
    )
    run.log(stage=f"final_{manifold.name}", **final)

    failed = report_gates(
        check_gates(final, cfg["gates"], cfg["diffusion"]["sigma_min"]), manifold.name
    )
    print(f"run dir: {run.dir}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

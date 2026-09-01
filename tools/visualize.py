"""Figures for an already-trained model. Nothing here trains anything.

    python tools/visualize.py runs/m-sphere --config configs/manifold_sphere.yaml

Writes into ``<run_dir>/figures/``:

  learned_<manifold>.png    what the model learned about the manifold
  uniformity.png            marginals, before and after, against the exact target

The uniformity figure shows four curves, in the order the experiment produces them:

  1. before training     p_data, the distribution the model was trained on
  2. after training      what the model learned; should sit on curve 1
  3. after correction    the tempered corrector at alpha > 0
  4. TARGET: uniform     the analytic density, drawn dashed

Curves 1 and 2 lying together says the model learned its data. Curve 3 leaving
them and approaching the dashed line is the result the project is about.

Only the checkpoint is loaded; the corrector curves are sampling, and
``--corrector-steps 0`` omits them. Sampled points are written to
``<run_dir>/samples/uniformity.pt``, so a figure can be relabelled or restyled
with ``--reuse-samples`` without paying for the corrector again.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dgeom.config import load_config, resolve_device, seed_everything
from dgeom.experiment import (
    check_gates,
    load_model,
    make_loader,
    make_manifold,
    make_reference,
)
from dgeom.sampling import LangevinSampler, TemperedLangevin
from dgeom.training.manifold_training import evaluate
from dgeom.viz.labels import config_line, verdict_line
from dgeom.viz.manifold import plot_learned_manifold
from dgeom.viz.uniformity import plot_uniformity, uniformity_summary


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument(
        "--alpha", type=float, default=0.5, help="tempering exponent of the corrector"
    )
    ap.add_argument(
        "--corrector-steps",
        type=int,
        default=4000,
        help="corrector steps; 0 omits the sampled curves",
    )
    ap.add_argument(
        "--skip-gates", action="store_true", help="skip gate evaluation (the slow part)"
    )
    ap.add_argument(
        "--reuse-samples",
        action="store_true",
        help="replot from samples/uniformity.pt instead of running the corrector",
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="cpu, mps, cuda or auto. The corrector is compute bound, so on "
        "Apple silicon mps is roughly 4x faster than cpu; it agrees with cpu "
        "to about 2e-3 relative at sigma=0.01, below the score-error gate.",
    )
    ap.add_argument("--mode", choices=["light", "dark"], default="light")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    torch.set_default_dtype(torch.float64)

    manifold = make_manifold(cfg)
    loader = make_loader(cfg, manifold)
    device = resolve_device(args.device)
    model, step = load_model(args.run_dir, cfg, manifold, device)
    print(f"device: {device}")
    figures = f"{args.run_dir}/figures"
    gen = torch.Generator().manual_seed(int(cfg["seed"]) + 7)

    verdict, ok = "", True
    if not args.skip_gates:
        tc = cfg["train"]["score"]
        metrics = evaluate(
            model,
            manifold,
            loader,
            make_reference(cfg, manifold, loader),
            tc["eval_sigmas"],
            1024,
            2048,
            gen,
        )
        rows = check_gates(metrics, cfg["gates"], cfg["diffusion"]["sigma_min"])
        verdict, ok = verdict_line(sum(r[3] for r in rows), len(rows))

    path, stats = plot_learned_manifold(
        model,
        manifold,
        loader,
        figures,
        n=min(args.n, 6000),
        sigma=args.sigma,
        step=step,
        mode=args.mode,
        subtitle=config_line(cfg),
        verdict=verdict,
        generator=gen,
    )
    print(path)
    if verdict:
        print(f"  {verdict}")
    print(
        f"  dist to M: median {stats['dist_before_median']:.4f} -> "
        f"{stats['dist_after_median']:.3e}   p95 {stats['dist_after_p95']:.3e}"
    )
    print(
        f"  eigenvalues: normal {stats['eig_normal_mean']:+.4f} (want -1)   "
        f"tangent {stats['eig_tangent_mean']:+.4f} (want 0)"
    )

    # The corrector is by far the most expensive thing here, so its output is
    # written to disk. Relabelling or restyling a figure then costs nothing:
    # rerun with --reuse-samples.
    store = Path(args.run_dir) / "samples" / "uniformity.pt"

    if args.reuse_samples:
        if not store.exists():
            raise SystemExit(
                f"no saved samples at {store}; run once without --reuse-samples"
            )
        blob = torch.load(store, weights_only=False)
        samples = blob["samples"]
        print(
            f"reusing {store}  ({blob['corrector_steps']} steps, "
            f"alpha={blob['alpha']:g}, N={blob['n']})"
        )
    else:
        samples = {"1. before training: p_data (true marginal)": loader.sample(args.n)}
        if args.corrector_steps:
            start = loader.sample(args.n)
            start = start + args.sigma * torch.randn(start.shape, generator=gen)

            learned, _ = LangevinSampler(
                sigma=args.sigma,
                alpha=0.0,
                n_steps=args.corrector_steps,
                step_scale=0.2,
            ).sample(model, start.clone(), generator=gen)
            samples["2. after training: what the model learned"] = learned

            corrected, _ = TemperedLangevin(
                sigma=args.sigma,
                alpha=args.alpha,
                n_steps=args.corrector_steps,
                step_scale=0.2,
            ).sample(model, start.clone(), generator=gen)
            samples[f"3. after correction: corrector, alpha={args.alpha:g}"] = corrected

        store.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "samples": samples,
                "manifold": manifold.name,
                "step": step,
                "sigma": args.sigma,
                "alpha": args.alpha,
                "corrector_steps": args.corrector_steps,
                "n": args.n,
            },
            store,
        )
        print(f"saved samples to {store}")

    upath = plot_uniformity(
        manifold,
        samples,
        figures,
        title=f"progress toward uniform on {manifold.name}",
        subtitle=config_line(
            cfg,
            {
                "alpha": args.alpha,
                "corrector steps": args.corrector_steps,
                "N": args.n,
            },
        ),
        mode=args.mode,
    )
    print(upath)
    print(f"  {'curve':<44} {'max dev':>8} {'noise floor':>12} {'ratio':>8}")
    for name, x in samples.items():
        u = uniformity_summary(manifold, x)
        flag = "UNIFORM" if u["uniform"] else "not uniform"
        print(
            f"  {name:<44} {u['max_deviation']:7.1%} "
            f"{u['noise_floor']:11.1%} {u['ratio']:7.1f}x  {flag}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

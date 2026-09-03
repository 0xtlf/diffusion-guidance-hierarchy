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
import json
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
from dgeom.sampling.base import Trace
from dgeom.training.manifold_training import evaluate
from dgeom.viz.labels import config_line, verdict_line
from dgeom.viz.manifold import plot_learned_manifold
from dgeom.viz.uniformity import (
    make_probe,
    plot_convergence,
    plot_uniformity,
    uniformity_summary,
)


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
        "--continue-from",
        metavar="PT",
        help="resume the corrector from a saved endpoint, e.g. "
        "runs/m-sphere/samples/uniformity.pt. Answers whether the previous run "
        "had converged: a flat trace means it had, a falling one means it had not.",
    )
    ap.add_argument(
        "--learned-steps",
        type=int,
        default=20000,
        help="steps for the alpha=0 chain that draws curve 2. It starts from "
        "p_data and targets p_data, so it needs far fewer steps than the "
        "corrector; capping it avoids doubling the cost of a long run.",
    )
    ap.add_argument(
        "--plot-every",
        type=int,
        default=0,
        help="write a uniformity figure every N corrector steps. Each is named "
        "for the step it was taken at, so snapshots accumulate rather than "
        "overwrite and the run can be watched as it goes. Must be a multiple "
        "of --trace-every.",
    )
    ap.add_argument(
        "--probe-n",
        type=int,
        default=0,
        help="points per convergence probe; 0 uses all of them. A small value "
        "raises the probe noise floor and can hide the very plateau it measures.",
    )
    ap.add_argument(
        "--trace-every",
        type=int,
        default=200,
        help="probe the departure from uniform every N corrector steps; 0 disables",
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
    model, step_trained = load_model(args.run_dir, cfg, manifold, device)
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
        step=step_trained,
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
        # adopt the settings the samples were produced under, or the caption
        # would describe this invocation's defaults rather than the experiment
        args.corrector_steps = blob["corrector_steps"]
        args.alpha, args.n, args.sigma = blob["alpha"], blob["n"], blob["sigma"]
        print(
            f"reusing {store}  ({blob['corrector_steps']} steps, "
            f"alpha={blob['alpha']:g}, N={blob['n']})"
        )
    else:
        inner = (
            make_probe(manifold, n_probe=args.probe_n or None)
            if args.trace_every
            else None
        )

        trace_log = Path(args.run_dir) / "corrector_trace.jsonl"

        def snapshot_probe(base: dict, label: str, offset: int = 0):
            """Probe that appends each measurement and writes periodic figures.

            Every probe is flushed to ``corrector_trace.jsonl`` as it happens
            rather than held until the end. A run that is interrupted after an
            hour otherwise leaves nothing behind, which is exactly how the first
            sphere continuation lost its trace.

            Snapshots are never overwritten: the absolute step is in the name,
            so a long run leaves behind the whole sequence it passed through.
            """
            if inner is None:
                return None

            last = args.corrector_steps - 1

            records: list[dict] = []

            def probe(x, step):
                rec = inner(x, step)
                k = offset + step
                records.append({"step": k, **rec})
                with trace_log.open("a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "stage": "corrector",
                                "manifold": manifold.name,
                                "alpha": args.alpha,
                                "sigma": args.sigma,
                                "n": int(x.shape[0]),
                                "step": k,
                                **rec,
                            }
                        )
                        + "\n"
                    )
                if not args.plot_every:
                    return rec
                # the probe only fires on trace steps, so the snapshot test has
                # to be aligned with those rather than with absolute multiples
                if k % args.plot_every == 0 or step == last:
                    plot_uniformity(
                        manifold,
                        {**base, label: x.detach().cpu().clone()},
                        figures,
                        filename=f"uniformity_step{k:07d}.png",
                        title=f"progress toward uniform on {manifold.name}",
                        subtitle=config_line(
                            cfg, {"alpha": args.alpha, "step": k, "N": x.shape[0]}
                        ),
                        mode=args.mode,
                    )
                    # live convergence view, overwritten so it always shows
                    # everything measured so far
                    plot_convergence(
                        {f"corrector, alpha={args.alpha:g}": Trace(list(records))},
                        figures,
                        title=f"corrector convergence on {manifold.name}",
                        subtitle="updated during the run; a plateau means "
                        "further steps will not help",
                        mode=args.mode,
                    )
                    # resumable state, so killing the job keeps the work
                    latest = Path(args.run_dir) / "samples" / "latest.pt"
                    latest.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "samples": {**base, label: x.detach().cpu().clone()},
                            "manifold": manifold.name,
                            "step": step_trained,
                            "sigma": args.sigma,
                            "alpha": args.alpha,
                            "corrector_steps": k,
                            "n": int(x.shape[0]),
                            "traces": {},
                        },
                        latest,
                    )
                    print(
                        f"  step {k:,}: max dev {rec['max_deviation']:.1%}  "
                        f"-> uniformity_step{k:07d}.png, convergence.png, "
                        f"samples/latest.pt",
                        flush=True,
                    )
                return rec

            return probe

        traces = {}
        uniformity_name, convergence_name = "uniformity.png", "convergence.png"
        corrector_label = f"3. after correction: corrector, alpha={args.alpha:g}"

        if args.continue_from:
            prev = torch.load(args.continue_from, weights_only=False)
            key = next(k for k in prev["samples"] if k.startswith("3."))
            # stays on CPU: the model moves batches to the device internally and
            # returns them here, and the CPU generator cannot seed device noise
            start = prev["samples"][key].cpu()
            done = prev["corrector_steps"]
            print(
                f"continuing {key}\n  from {args.continue_from} "
                f"({done} steps already run, N={start.shape[0]})"
            )
            samples = {
                k: v for k, v in prev["samples"].items() if not k.startswith("3.")
            }

            corrected, tr = TemperedLangevin(
                sigma=args.sigma,
                alpha=args.alpha,
                n_steps=args.corrector_steps,
                step_scale=0.2,
                trace_every=args.trace_every,
            ).sample(
                model,
                start,
                probe=snapshot_probe(samples, corrector_label, offset=done),
                generator=gen,
            )
            # keep the absolute step count so the trace joins onto the first run
            for r in tr.records:
                r["step"] += done
            traces[f"corrector, alpha={args.alpha:g}"] = tr
            total = done + args.corrector_steps
            samples[corrector_label] = corrected
            store = Path(args.run_dir) / "samples" / "continuation.pt"
            # do not clobber the figure from the original run; the two are
            # meant to be compared
            uniformity_name = "uniformity_continued.png"
            convergence_name = "convergence_continued.png"
            args.corrector_steps = total
        else:
            samples = {
                "1. before training: p_data (true marginal)": loader.sample(args.n)
            }
            if args.corrector_steps:
                start = loader.sample(args.n)
                start = start + args.sigma * torch.randn(start.shape, generator=gen)

                learned, tr0 = LangevinSampler(
                    sigma=args.sigma,
                    alpha=0.0,
                    n_steps=min(args.learned_steps, args.corrector_steps),
                    step_scale=0.2,
                    trace_every=args.trace_every,
                ).sample(model, start.clone(), probe=inner, generator=gen)
                samples["2. after training: what the model learned"] = learned
                traces["model, alpha=0"] = tr0

                corrected, tr1 = TemperedLangevin(
                    sigma=args.sigma,
                    alpha=args.alpha,
                    n_steps=args.corrector_steps,
                    step_scale=0.2,
                    trace_every=args.trace_every,
                ).sample(
                    model,
                    start.clone(),
                    probe=snapshot_probe(samples, corrector_label),
                    generator=gen,
                )
                samples[corrector_label] = corrected
                traces[f"corrector, alpha={args.alpha:g}"] = tr1

        cpath = plot_convergence(
            traces,
            figures,
            filename=convergence_name,
            title=f"corrector convergence on {manifold.name}",
            subtitle="a plateau means converged; a descent means more steps would help",
            mode=args.mode,
        )
        if cpath:
            print(cpath)
            for name, tr in traces.items():
                dev = tr.column("max_deviation")
                if len(dev) >= 6:
                    tail = sum(dev[-5:]) / 5
                    mid = sum(dev[len(dev) // 2 - 2 : len(dev) // 2 + 3]) / 5
                    verdict = (
                        "PLATEAU (converged)"
                        if tail > 0.9 * mid
                        else "still falling (more steps would help)"
                    )
                    print(
                        f"  {name:<30} half-way {mid:6.1%} -> final {tail:6.1%}  "
                        f"{verdict}"
                    )

        store.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "samples": samples,
                "manifold": manifold.name,
                "step": step_trained,
                "sigma": args.sigma,
                "alpha": args.alpha,
                "corrector_steps": args.corrector_steps,
                "n": args.n,
                "traces": {k: v.records for k, v in traces.items()},
            },
            store,
        )
        print(f"saved samples to {store}")

    upath = plot_uniformity(
        manifold,
        samples,
        figures,
        filename=locals().get("uniformity_name", "uniformity.png"),
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

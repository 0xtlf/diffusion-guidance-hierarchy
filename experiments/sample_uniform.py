"""E7 - the tempered corrector should produce the UNIFORM measure on the manifold.

No guidance: the drift is sigma^alpha * shat alone. This is Theorem 5.1/5.2 of
the paper, and the prerequisite for anything conditional.

Three things are demonstrated per manifold:

  transport          starting from p_data, the corrector must MOVE the sample to
                     uniform -- the "whatever the data distribution was" claim
  init-independence  p_data, uniform and a point mass must converge to the same
                     limit; that is the evidence we measured the stationary
                     distribution and not a memory of the start
  thickness law      the cloud should be sigma^(1-alpha/2) thick, a cheap check
                     that the implementation matches the theory

No projection anywhere. The corrector's stationary law IS the result; a
deterministic post-hoc flow is no part of the theorem, and a deterministic map
cannot preserve a measure -- measured with an exact score, projecting turned
p(uniform) = 1.5e-01 into 1.5e-04.

Uniformity is tested exactly: pooled marginals on S^3, and on the Klein bottle
two 1-D KS tests (u uniform, v against the sqrt(det g) CDF) plus a joint
chi-square, all in closed form via the exact chart inversion.
"""

from __future__ import annotations

import argparse
import json

import torch

from dgeom.config import setup
from dgeom.experiment import (
    load_model,
    make_loader,
    make_manifold,
    make_reference,
)
from dgeom.metrics import uniformity
from dgeom.sampling import AnnealedLangevin, TemperedLangevin
from dgeom.sampling.base import Trace
from dgeom.viz.uniformity import (
    make_probe,
    plot_convergence,
    plot_uniformity,
    uniformity_summary,
)

ACCEPT = 0.01  # p-value threshold for calling a sample uniform


def initial_batch(
    kind: str, manifold, loader, n: int, sigma: float, gen
) -> torch.Tensor:
    if kind == "p_data":
        x = loader.sample(n)
    elif kind == "uniform":
        x = manifold.sample_uniform(n, generator=gen)
    elif kind == "point":
        x = manifold.sample_uniform(1, generator=gen).expand(n, -1).contiguous()
    else:
        raise ValueError(f"unknown init {kind!r}")
    return x + sigma * torch.randn(x.shape, dtype=x.dtype, generator=gen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e7_uniform.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg, run, device = setup(args.config, args.overrides, default_name="e7-uniform")
    manifold = make_manifold(cfg)
    loader = make_loader(cfg, manifold)
    gen = torch.Generator().manual_seed(int(cfg["seed"]) + 17)
    sc = cfg["sampling"]
    sigma, n = float(sc["sigma"]), int(sc["n_chains"])

    if cfg.get("use_reference", False):
        # An exact score separates "does the corrector produce the uniform
        # measure" from "is the model good enough". If this fails, no amount of
        # training would help.
        model = make_reference(cfg, manifold, loader)
        source = f"REFERENCE ({type(model).__name__})"
    else:
        model, step = load_model(cfg["load_from"], cfg, manifold, device)
        source = f"{cfg['load_from']} @ step {step}"

    print(
        f"manifold {manifold.name}  (d={manifold.d}, n={manifold.n}, "
        f"codim={manifold.d - manifold.n})   score: {source}"
    )
    print(
        f"sigma={sigma:g}  chains={n}  steps={sc['n_steps']}  "
        f"step_scale={sc['step_scale']}\n"
    )

    # every probe is written as it happens, so a sweep killed part way keeps
    # the alphas it already finished
    trace_log = run.dir / "corrector_trace.jsonl"
    figures = run.dir / "figures"
    base_probe = make_probe(manifold)
    traces: dict[str, Trace] = {}
    endpoints: dict[str, torch.Tensor] = {}
    trace_every = int(sc.get("trace_every", 200))
    # republish the comparison figure during the sweep, so an alpha that is
    # clearly settled can be seen without waiting for the remaining ones
    plot_every = int(sc.get("plot_every", 0))
    live: dict[str, list] = {}

    header = (
        f"{'alpha':>6} {'init':>8} {'stage':>10} {'dist_M':>10} "
        f"{'max_dev':>9} {'x floor':>8} "
        f"{'p(u)':>10} {'p(joint)':>10} {'verdict':>8}"
    )
    print(header)
    print("-" * len(header))

    anneal = [float(s) for s in sc.get("anneal", [])]
    for alpha in cfg["sweep"]["alphas"]:
        alpha = float(alpha)
        sampler = TemperedLangevin(
            sigma=sigma,
            n_steps=int(sc["n_steps"]),
            alpha=alpha,
            step_scale=float(sc["step_scale"]),
            trace_every=trace_every,
        )

        key = f"alpha={alpha:g}"
        live.setdefault(key, [])

        def probe(x, step, _a=alpha, _k=key):
            rec = base_probe(x, step)
            rec["dist_M"] = float(
                (x - manifold.project(x)).norm(dim=-1).mean() / manifold.scale
            )
            live[_k].append({"step": step, **rec})
            if plot_every and step % plot_every == 0:
                plot_convergence(
                    {k: Trace(v) for k, v in live.items() if v},
                    figures,
                    title=f"tempered corrector on {manifold.name}: alpha sweep",
                    subtitle="updated during the sweep; a plateau means further "
                    "steps will not help",
                )
                plot_uniformity(
                    manifold,
                    {f"corrector, {_k}": x.detach().cpu().clone()},
                    figures,
                    filename=f"uniformity_{_k.replace('=', '')}_step{step:07d}.png",
                    title=f"{manifold.name}, {_k} at step {step:,}",
                    subtitle=f"sigma={sigma:g}, N={x.shape[0]}",
                )
                print(
                    f"       [{_k}] step {step:>7,}  "
                    f"max dev {rec['max_deviation']:.1%}  "
                    f"dist_M {rec['dist_M']:.4f}  -> figures/",
                    flush=True,
                )
            with trace_log.open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "stage": "sweep",
                            "manifold": manifold.name,
                            "alpha": _a,
                            "sigma": sigma,
                            "step": step,
                            **rec,
                        }
                    )
                    + "\n"
                )
            return rec

        for init in cfg["sweep"]["inits"]:
            x0 = initial_batch(init, manifold, loader, n, sigma, gen)
            x, tr = sampler.sample(model, x0, probe=probe, generator=gen)
            if init == cfg["sweep"]["inits"][0]:
                traces[f"alpha={alpha:g}"] = tr
                endpoints[f"alpha={alpha:g}"] = x.detach().cpu().clone()
            stages = [("corrector", x)]

            if anneal:
                # every rung must lie inside the trained sigma range; the sampler
                # raises if it does not
                cooler = AnnealedLangevin(
                    sigmas=tuple(anneal),
                    alpha=alpha,
                    n_steps=int(sc.get("anneal_steps", 2000)),
                    step_scale=float(sc["step_scale"]),
                )
                x, _ = cooler.sample(model, x, generator=gen)
                stages.append((f"cooled {anneal[-1]:g}", x))

            for stage, y in stages:
                dist = float(
                    (y - manifold.project(y)).norm(dim=-1).mean() / manifold.scale
                )
                u = uniformity(manifold, y)
                us = uniformity_summary(manifold, y)
                ok = min(u["ks_u_p"], u["ks_v_p"], u["chi2_joint_p"]) > ACCEPT
                print(
                    f"{alpha:6.2f} {init:>8} {stage:>10} {dist:10.3e} "
                    f"{us['max_deviation']:8.1%} {us['ratio']:7.2f}x "
                    f"{u['ks_u_p']:10.3e} "
                    f"{u['chi2_joint_p']:10.3e} "
                    f"{'UNIFORM' if ok else 'reject':>8}",
                    flush=True,
                )
                run.log(
                    stage="e7",
                    manifold=manifold.name,
                    alpha=alpha,
                    init=init,
                    phase=stage,
                    dist_M=dist,
                    max_deviation=us["max_deviation"],
                    dev_over_floor=us["ratio"],
                    uniform=ok,
                    **u,
                )
            print(
                f"{'':6} {'':>8} {'':>10} predicted thickness "
                f"sigma^(1-a/2) = {sampler.cloud_thickness:.4f}"
            )
        print()

    if traces:
        cpath = plot_convergence(
            traces,
            figures,
            title=f"tempered corrector on {manifold.name}: alpha sweep",
            subtitle=f"sigma={sigma:g}, {sc['n_steps']} steps, "
            f"N={n}, step_scale={sc['step_scale']}",
        )
        print(f"\nconvergence: {cpath}")
        print(f"  {'alpha':>7} {'first half':>11} {'second half':>12} {'verdict':>26}")
        for name, tr in traces.items():
            dev = tr.column("max_deviation")
            if len(dev) < 4:
                continue
            h = len(dev) // 2
            a, b = sum(dev[:h]) / h, sum(dev[h:]) / (len(dev) - h)
            # a flat second half means the limit is reached, so a worse plateau
            # is a property of the sampler and not of the step budget
            verdict = "converged" if b > 0.95 * a else "still falling"
            print(f"  {name.split('=')[-1]:>7} {a:10.1%} {b:11.1%} {verdict:>26}")

    if endpoints:
        upath = plot_uniformity(
            manifold,
            endpoints,
            figures,
            filename="alpha_sweep_uniformity.png",
            title=f"where each alpha lands on {manifold.name}",
            subtitle=f"sigma={sigma:g}, {sc['n_steps']} steps, N={n}",
        )
        print(f"marginals:   {upath}")

    print(f"run dir: {run.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

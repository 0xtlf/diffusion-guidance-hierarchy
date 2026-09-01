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

    header = (
        f"{'alpha':>6} {'init':>8} {'stage':>10} {'dist_M':>10} "
        f"{'p(u)':>10} {'p(v)':>10} {'p(joint)':>10} {'verdict':>8}"
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
        )
        for init in cfg["sweep"]["inits"]:
            x0 = initial_batch(init, manifold, loader, n, sigma, gen)
            x, _ = sampler.sample(model, x0, generator=gen)
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
                ok = min(u["ks_u_p"], u["ks_v_p"], u["chi2_joint_p"]) > ACCEPT
                print(
                    f"{alpha:6.2f} {init:>8} {stage:>10} {dist:10.3e} "
                    f"{u['ks_u_p']:10.3e} {u['ks_v_p']:10.3e} "
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
                    uniform=ok,
                    **u,
                )
            print(
                f"{'':6} {'':>8} {'':>10} predicted thickness "
                f"sigma^(1-a/2) = {sampler.cloud_thickness:.4f}"
            )
        print()

    print(f"run dir: {run.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

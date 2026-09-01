"""Assemble and evaluate an unconditional diffusion model on a manifold.

Nothing here knows which manifold it is on: the manifold supplies its own p_data
and its own geometry, so the same code trains S^3 and the Klein bottle.
"""

from __future__ import annotations

import torch

from ..metrics import coverage, flow_to_manifold, jacobian_spectrum, score_error
from ..models import NoiseSchedule, ScoreDiffusion, reference_for
from ..nn import ScoreNetwork
from .callbacks import Checkpointer, Evaluator, MetricLogger
from .trainer import TrainConfig, Trainer

FLOW_STEPS = 50  # normal relaxation converges in ~30; a longer flow lets the
# tangential part of shat accumulate and sweep points into the
# density modes, measuring concentration instead of coverage


def build_model(cfg: dict, manifold, device) -> ScoreDiffusion:
    """Build an untrained ScoreDiffusion sized for the manifold."""
    mc = cfg["model"]["score"]
    net = ScoreNetwork(
        dim=manifold.d, width=mc["width"], depth=mc["depth"], emb_dim=mc["emb_dim"]
    )
    return ScoreDiffusion(net, NoiseSchedule.from_cfg(cfg), device=device)


def evaluate(
    model, manifold, loader, reference, sigmas, n, n_flow, generator=None
) -> dict:
    """The quality gates."""
    out: dict = {}
    for s in sigmas:
        x0 = loader.sample(n)
        x, _ = model.add_noise(
            x0, s, torch.randn(x0.shape, dtype=x0.dtype, generator=generator)
        )
        out.update(
            {
                f"{k}_s{s:g}": v
                for k, v in score_error(model, reference, manifold, x, s).items()
            }
        )

    s = min(sigmas)
    # start broad and off-manifold so coverage is meaningful; these are NOT
    # p_data draws, which would only cover where the density already is
    base = manifold.sample_uniform(n_flow, generator=generator)
    x = base + (0.15 * manifold.scale) * torch.randn(
        base.shape, dtype=base.dtype, generator=generator
    )
    out.update(flow_to_manifold(model, manifold, x, s, steps=FLOW_STEPS))

    z = x.clone()
    for _ in range(FLOW_STEPS):
        z = z + 0.5 * model.shat(z, s)
    out.update(coverage(manifold, z, generator=generator))

    probe = manifold.sample_uniform(min(n, 512), generator=generator)
    probe = probe + s * torch.randn(probe.shape, dtype=probe.dtype, generator=generator)
    out.update(jacobian_spectrum(model, manifold, probe, s))
    out.pop("eig_mean", None)
    return out


def train(cfg: dict, run, device, manifold, loader) -> ScoreDiffusion:
    """Switch to training mode."""
    tc = cfg["train"]["score"]
    gen = torch.Generator().manual_seed(int(cfg["seed"]))
    model = build_model(cfg, manifold, device)
    reference = reference_for(
        manifold,
        loader,
        model.schedule,
        **({"n_nodes": tc["ref_nodes"]} if manifold.name != "sphere" else {}),
    )

    print(
        f"[{manifold.name}] d={manifold.d} n={manifold.n} "
        f"codim={manifold.d - manifold.n} scale={manifold.scale:.3f}"
    )
    print(f"[{manifold.name}] {model.n_params:,} params on {device}, {model.schedule}")
    print(f"[{manifold.name}] data: {loader.density!r}")

    def batch_fn(n: int) -> torch.Tensor:
        return loader.sample(n).to(torch.float32)

    trainer = Trainer(
        model,
        TrainConfig.from_cfg(tc),
        run,
        tag=f"score_{manifold.name}",
        callbacks=[
            MetricLogger(every=tc.get("log_every", 1000)),
            Evaluator(
                lambda: evaluate(
                    model,
                    manifold,
                    loader,
                    reference,
                    tc["eval_sigmas"],
                    tc["eval_n"],
                    tc.get("eval_n_flow", 2048),
                    gen,
                ),
                every=tc.get("eval_every", 20000),
            ),
            Checkpointer(every=tc.get("ckpt_every", 20000)),
        ],
    )
    return trainer.fit(batch_fn)

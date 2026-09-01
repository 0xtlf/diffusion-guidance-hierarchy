"""Building an experiment: manifold, data, model, reference, and gate checks.

Lives in the library rather than beside the scripts because auditing a checkpoint
must not depend on importing a training script, and because the seed discipline
below is a correctness property, not a convenience.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .geometry import build_manifold, loader_for
from .models import NoiseSchedule, ScoreDiffusion, reference_for
from .nn import ScoreNetwork


def make_manifold(cfg: dict):
    """The manifold. Pure geometry, no randomness."""
    return build_manifold(cfg["manifold"], cfg)


def make_loader(cfg: dict, manifold):
    """The data loader, seeded from the TRAINING seed.

    The density is drawn from that seed, so a different one would evaluate a
    model against a distribution it never saw.
    """
    return loader_for(
        manifold, cfg, generator=torch.Generator().manual_seed(int(cfg["seed"]))
    )


def load_model(
    run_dir: str | Path, cfg: dict, manifold, device, use_ema: bool = True
) -> tuple[ScoreDiffusion, int]:
    """Load a trained model from a run directory. Uses the EMA weights by default."""
    mc = cfg["model"]["score"]
    net = ScoreNetwork(
        dim=manifold.d, width=mc["width"], depth=mc["depth"], emb_dim=mc["emb_dim"]
    )
    model = ScoreDiffusion(net, NoiseSchedule.from_cfg(cfg), device=device)
    ck = torch.load(
        Path(run_dir) / "ckpt" / f"score_{manifold.name}.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(ck["ema"] if (use_ema and "ema" in ck) else ck["model"])
    return model, int(ck.get("step", -1))


def make_reference(cfg: dict, manifold, loader):
    """Build the exact reference model for this manifold and data distribution."""
    tc = cfg["train"]["score"]
    kw = {} if manifold.name == "sphere" else {"n_nodes": tc.get("ref_nodes", 96)}
    return reference_for(manifold, loader, NoiseSchedule.from_cfg(cfg), **kw)


GATE_SMALLER_IS_BETTER = {
    "flow_dist_median": True,
    "flow_dist_p95": True,
    "codim_correct_frac": False,
    "normal_angle_deg_mean": True,
    "eig_normal_mean": True,
    "hat_err_mean": True,
    "cover_ratio": True,
}


def check_gates(metrics: dict, gates: dict, sigma_min: float) -> list[tuple]:
    """Compare measured metrics against the configured gates."""
    rows = []
    for name, target in gates.items():
        key = f"hat_err_mean_s{sigma_min:g}" if name == "hat_err_mean" else name
        value = metrics.get(key)
        if value is None:
            rows.append((name, float("nan"), target, False))
            continue
        smaller = GATE_SMALLER_IS_BETTER[name]
        ok = (value <= target) if smaller else (value >= target)
        rows.append((name, float(value), float(target), bool(ok)))
    return rows


def report_gates(rows: list[tuple], title: str) -> list[str]:
    """Print the gate table and return the names of any that failed."""
    width = max(len(r[0]) for r in rows)
    print(f"\n=== quality gates: {title} ===")
    print(f"{'gate':<{width}} {'value':>12} {'target':>12}")
    print("-" * (width + 30))
    for name, value, target, ok in rows:
        print(
            f"{name:<{width}} {value:12.4g} {target:12.4g}   {'ok' if ok else 'FAIL'}"
        )
    failed = [r[0] for r in rows if not r[3]]
    print(
        f"\n{len(rows) - len(failed)}/{len(rows)} gates passed -> "
        f"{'MANIFOLD LEARNED' if not failed else 'NOT YET: ' + ', '.join(failed)}"
    )
    return failed

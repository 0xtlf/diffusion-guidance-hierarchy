"""Stage 0 validation gate: closed-form smoothed score vs Monte Carlo.

Nothing else in the project is trustworthy until this passes, because both the
theory table and every learned model are measured against this reference.

The gate is a z-test, not a relative-error threshold.  shat is O(sigma) near the
manifold — a difference of two nearly-equal O(1) vectors — so its relative error
is dominated by whichever probe point happens to have the smallest |shat|, and a
fixed tolerance would only measure Monte-Carlo variance.  Instead we ask the
statistically meaningful question: does the closed form lie inside the MC
confidence interval?
"""

from __future__ import annotations

import argparse

import torch

from dgeom.config import setup
from dgeom.geometry import Sphere, SphereVonMisesLoader
from dgeom.geometry.mc_reference import mc_shat
from dgeom.models import AnalyticDiffusion, NoiseSchedule


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e0_validate.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg, run, _ = setup(args.config, args.overrides, default_name="e0-validate")
    torch.set_default_dtype(torch.float64)  # the gate runs in double on CPU

    dc = cfg["data"]
    gen = torch.Generator().manual_seed(int(cfg["seed"]))
    sphere = Sphere(dc["dim"])
    loader = SphereVonMisesLoader.random(
        sphere, dc["n_components"], tuple(dc["kappa_range"]), 1024, gen
    )
    mix = loader.vmf
    model = AnalyticDiffusion(loader.mixture, NoiseSchedule.from_cfg(cfg))

    vc = cfg["validate"]
    worst_z, worst_norm = 0.0, 0.0
    rows = []

    for sigma in vc["sigmas"]:
        u = mix.sample(vc["n_points"], generator=gen)
        offset = torch.randn(vc["n_points"], dc["dim"], generator=gen)
        x = u + sigma * offset * float(vc["probe_offset_scale"])

        exact = model.shat(x, sigma)
        approx, se = mc_shat(x, sigma, mix, n_mc=vc["n_mc"], return_se=True)

        # z-score per coordinate against the MC standard error
        z = ((exact - approx).abs() / se.clamp_min(1e-30)).max().item()
        # scale-free magnitude check, normalised by a stable scale not a per-point one
        nerr = (
            (exact - approx).norm(dim=-1).max() / exact.norm(dim=-1).median()
        ).item()
        worst_z, worst_norm = max(worst_z, z), max(worst_norm, nerr)

        tweedie = (model.denoise(x, sigma) - x - exact).abs().max().item()
        radial = (
            torch.nn.functional.cosine_similarity(exact, sphere.project(x) - x, dim=-1)
            .mean()
            .item()
        )

        rows.append((sigma, z, nerr, tweedie, radial))
        run.log(
            stage="e0",
            sigma=sigma,
            max_z=z,
            norm_err=nerr,
            tweedie_resid=tweedie,
            radial_cos=radial,
        )

    print(
        f"{'sigma':>10} {'max_z':>9} {'err/median|s|':>14} "
        f"{'tweedie':>10} {'cos(-d_M)':>11}"
    )
    for s, z, ne, t, c in rows:
        print(f"{s:10.4g} {z:9.2f} {ne:14.3e} {t:10.1e} {c:11.4f}")

    ok = worst_z < vc["z_tol"]
    print(
        f"\nworst |z| = {worst_z:.2f}   tol {vc['z_tol']}   ->  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    print(
        "(z is vs the MC standard error: |z|<4 means the closed form is inside "
        "the Monte-Carlo confidence interval)"
    )
    run.log(stage="e0", gate="analytic_score_vs_mc", worst_z=worst_z, passed=ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Langevin samplers.

    dX = sigma^alpha * shat(X, sigma) dt + sqrt(2) dW

Discretised Euler-Maruyama with dt = step_scale * sigma^(2 - alpha), which makes
the drift step step_scale * shat, independent of sigma.

alpha = 0 is plain Langevin: the stationary law is p_data restricted to the
manifold. Any 0 < alpha < 2 tempers the score, and the stationary law becomes the
uniform (Riemannian volume) measure instead -- tempering suppresses the tangential
drift to sigma^alpha * Theta(1) while leaving the Theta(sigma^(alpha-2))
transverse walls intact, so the chain is free Brownian motion inside a thin shell
around the manifold, and free diffusion in a box equilibrates to uniform.

Song's SNR step rule must NOT be used with a tempered score: it sets dt
proportional to 1/|s|^2, so tempering inflates dt by sigma^(-2 alpha) and breaks
the integrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..models import DiffusionModel, broadcast_sigma
from ..progress import PeriodicLogger, track
from ..registry import SAMPLERS
from .base import Sampler, Trace


@SAMPLERS.register("langevin")
@dataclass(repr=False)
class LangevinSampler(Sampler):
    """Langevin at a single fixed noise level."""

    sigma: float = 0.01
    n_steps: int = 10_000
    alpha: float = 0.0
    step_scale: float = 0.1
    trace_every: int = 0
    log_every: int = 0  # 0 derives an interval giving ~20 lines per run
    progress: bool = True
    max_drift_step: float = 0.5  # guard against a blown-up early step

    @property
    def dt(self) -> float:
        """Euler-Maruyama step size, step_scale * sigma^(2 - alpha)."""
        return self.step_scale * self.sigma ** (2.0 - self.alpha)

    @property
    def cloud_thickness(self) -> float:
        """Predicted equilibrium width, sigma^(1 - alpha/2)."""
        return self.sigma ** (1.0 - self.alpha / 2.0)

    def drift(self, model: DiffusionModel, x: torch.Tensor) -> torch.Tensor:
        """Tempered drift sigma^(alpha-2) * shat at each point."""
        s = broadcast_sigma(self.sigma, x)
        return s.pow(self.alpha - 2.0) * model.shat(x, self.sigma)

    def sample(self, model, x, *, probe=None, generator=None):
        """Draw samples."""
        if not model.schedule.contains(self.sigma):
            raise ValueError(
                f"sigma={self.sigma:g} is outside the model's trained range "
                f"[{model.schedule.sigma_min:g}, {model.schedule.sigma_max:g}]. "
                "The sigma-embedding would extrapolate and the score is "
                "meaningless there."
            )
        dt, sqrt2dt = self.dt, (2.0 * self.dt) ** 0.5
        trace, x = Trace(), x.clone()

        every = self.log_every or max(1, self.n_steps // 20)
        logger = PeriodicLogger(
            f"{type(self).__name__.replace('Sampler', '').lower()} "
            f"alpha={self.alpha:g}",
            self.n_steps,
            every,
        )
        steps = range(self.n_steps)
        if self.progress:
            steps = track(
                steps,
                desc=f"corrector alpha={self.alpha:g} sigma={self.sigma:g}",
                total=self.n_steps,
            )

        latest: dict[str, Any] = {}
        for k in steps:
            step = dt * self.drift(model, x)
            norm = step.norm(dim=-1, keepdim=True)
            step = torch.where(
                norm > self.max_drift_step, step * (self.max_drift_step / norm), step
            )
            x = (
                x
                + step
                + sqrt2dt
                * torch.randn(
                    x.shape, dtype=x.dtype, device=x.device, generator=generator
                )
            )

            if self.trace_every and (
                k % self.trace_every == 0 or k == self.n_steps - 1
            ):
                rec: dict[str, Any] = {"step": k, "dt": dt}
                if probe is not None:
                    rec.update(probe(x, k))
                trace.add(**rec)
                latest = {k2: v for k2, v in rec.items() if k2 not in ("step", "dt")}
                if self.progress and hasattr(steps, "set_postfix"):
                    steps.set_postfix(
                        {k2: f"{v:.3f}" for k2, v in latest.items()},
                        refresh=False,
                    )

            if logger.due(k):
                logger.log(k, **latest)
        return x, trace


@SAMPLERS.register("tempered_langevin")
@dataclass(repr=False)
class TemperedLangevin(LangevinSampler):
    """Langevin with the score tempered by sigma^alpha. alpha must be in (0, 2)."""

    alpha: float = 1.0

    def __post_init__(self) -> None:
        """Finalise the object after dataclass construction."""
        if not 0.0 < self.alpha < 2.0:
            raise ValueError(f"alpha must lie in (0, 2), got {self.alpha}")


@SAMPLERS.register("annealed_langevin")
@dataclass(repr=False)
class AnnealedLangevin(Sampler):
    """Tempered Langevin down a ladder of noise levels.

    Cooling shrinks both finite-sigma errors at once -- the cloud thins as
    sigma^(1-alpha/2) and the residual tilt p_data^(sigma^alpha) goes to 1 -- but
    every level must lie INSIDE the model's trained range, and mixing time grows
    as sigma^(alpha-2), so a long ladder is quickly unaffordable.
    """

    sigmas: tuple[float, ...] = (0.01,)
    n_steps: int = 10_000
    alpha: float = 1.0
    step_scale: float = 0.1
    trace_every: int = 0

    def sample(self, model, x, *, probe=None, generator=None):
        """Draw samples."""
        trace = Trace()
        for sigma in self.sigmas:
            leg = TemperedLangevin(
                sigma=float(sigma),
                n_steps=self.n_steps,
                alpha=self.alpha,
                step_scale=self.step_scale,
                trace_every=self.trace_every,
            )
            x, sub = leg.sample(model, x, probe=probe, generator=generator)
            for rec in sub.records:
                trace.add(sigma=float(sigma), **rec)
        return x, trace

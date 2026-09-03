"""Diffusion models.

    DiffusionModel      the interface: a forward process plus shat(x, sigma)
    TrainableDiffusion  adds parameters and a loss
    ScoreDiffusion      backed by a score network (the trained model)
    AnalyticDiffusion   closed form on the sphere      (ground truth)
    QuadratureDiffusion exact by quadrature on a 2-D manifold (ground truth)

A conditional variant overrides `shat` alone and inherits everything else.
"""

from .base import DiffusionModel, TrainableDiffusion, broadcast_sigma
from .guidance import GuidedDiffusion
from .references import (
    AnalyticDiffusion,
    QuadratureDiffusion,
    UniformSphereDiffusion,
    reference_for,
)
from .schedule import NoiseSchedule
from .score_diffusion import ScoreDiffusion

__all__ = [
    "AnalyticDiffusion",
    "DiffusionModel",
    "GuidedDiffusion",
    "NoiseSchedule",
    "QuadratureDiffusion",
    "ScoreDiffusion",
    "TrainableDiffusion",
    "UniformSphereDiffusion",
    "broadcast_sigma",
    "reference_for",
]

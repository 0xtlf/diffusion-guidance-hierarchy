"""Sampling schemes.

Sampler            the interface
LangevinSampler    fixed sigma, alpha = 0 by default (targets p_data)
TemperedLangevin   alpha in (0, 2)      (targets the uniform measure on M)
AnnealedLangevin   a ladder of sigmas
"""

from .base import Sampler, Trace
from .langevin import AnnealedLangevin, LangevinSampler, TemperedLangevin

__all__ = [
    "AnnealedLangevin",
    "LangevinSampler",
    "Sampler",
    "TemperedLangevin",
    "Trace",
]

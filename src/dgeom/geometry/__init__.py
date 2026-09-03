"""Geometry: manifolds, densities on them, and dataloaders.

    base.py       Manifold ABC -- pure geometry, knows only the volume measure
    sphere.py     Sphere(Manifold)
    klein.py      KleinBottle(Manifold)
    densities.py  Density ABC, UniformDensity, VonMisesFisherMixture
    loaders.py    ManifoldLoader (uniform by default) and its vMF subclasses
    intersection.py  Hyperplane and the sections M ∩ H, themselves Manifolds

The split matters: which distribution the data follows is an experimental
choice, not a property of the space, so it lives in the loader rather than in
the manifold.
"""

from .base import Manifold, orthogonal_complement, principal_angles
from .densities import Density, UniformDensity, VonMisesFisherMixture
from .intersection import (
    Hyperplane,
    KleinSection,
    Section,
    SphereSection,
    section_for,
)
from .klein import KleinBottle
from .loaders import (
    KleinVonMisesLoader,
    ManifoldLoader,
    SphereVonMisesLoader,
    VonMisesFisherLoader,
    intersection_loader,
    loader_for,
)
from .sphere import Sphere, VMFMixture

__all__ = [
    "Density",
    "Hyperplane",
    "KleinBottle",
    "KleinSection",
    "KleinVonMisesLoader",
    "Manifold",
    "ManifoldLoader",
    "Section",
    "Sphere",
    "SphereSection",
    "SphereVonMisesLoader",
    "UniformDensity",
    "VMFMixture",
    "VonMisesFisherLoader",
    "VonMisesFisherMixture",
    "build_manifold",
    "intersection_loader",
    "loader_for",
    "orthogonal_complement",
    "principal_angles",
    "section_for",
]

_MANIFOLDS = {"sphere": Sphere, "klein": KleinBottle}


def build_manifold(name: str, cfg: dict):
    """Construct a manifold from config. Geometry only -- see loader_for for data."""
    if name not in _MANIFOLDS:
        raise KeyError(
            f"unknown manifold {name!r}; expected one of {sorted(_MANIFOLDS)}"
        )
    if name == "sphere":
        return Sphere(cfg["data"]["dim"])
    kc = cfg.get("klein", {})
    return KleinBottle(R=kc.get("R", 2.0), r=kc.get("r", 1.0))

"""Dataloaders over a manifold.

A loader pairs a Manifold (pure geometry) with a Density (what the data follows)
and yields batches. The default is the uniform volume measure, so
`ManifoldLoader(manifold)` is the trivial case and every non-uniform dataset is
an explicit choice.

Sampling strategy is per-loader. The generic route is rejection against the
uniform proposal, which works on any manifold because the volume element cancels
and the acceptance ratio is just the density. The sphere overrides this with
Wood's algorithm, which draws vMF exactly and far faster.
"""

from __future__ import annotations

import torch

from .base import Manifold
from .densities import Density, UniformDensity, VonMisesFisherMixture
from .intersection import Hyperplane, section_for
from .sphere import Sphere, VMFMixture


class ManifoldLoader:
    """Infinite stream of samples from a manifold. Uniform unless told otherwise."""

    def __init__(
        self,
        manifold: Manifold,
        density: Density | None = None,
        batch_size: int = 1024,
        generator: torch.Generator | None = None,
    ) -> None:
        self.manifold = manifold
        self.density = density or UniformDensity()
        self.batch_size = int(batch_size)
        self.generator = generator or torch.Generator().manual_seed(0)

    # ------------------------------------------------------------------ core

    def sample(self, n: int | None = None) -> torch.Tensor:
        """Draw a batch from the manifold.

        Args:
            n: batch size; defaults to ``self.batch_size``.

        Returns:
            ``(n, d)`` points lying exactly on the manifold.
        """
        n = self.batch_size if n is None else int(n)
        if self.density.is_uniform:
            return self.manifold.sample_uniform(n, generator=self.generator)
        return self._rejection_sample(n)

    def _rejection_sample(self, n: int, max_rounds: int = 200) -> torch.Tensor:
        """Rejection against the uniform proposal.

        Proposing uniform-by-volume means the volume element cancels, so the
        acceptance ratio is the density alone and no Jacobian bookkeeping is
        needed. Valid on any manifold whose uniform sampler is exact.
        """
        cap = self._log_cap()
        out, got, rounds = [], 0, 0
        while got < n and rounds < max_rounds:
            m = 3 * (n - got) + 64
            x = self.manifold.sample_uniform(m, generator=self.generator)
            accept = (self.density.log_prob(x) - cap).exp()
            keep = torch.rand(m, dtype=x.dtype, generator=self.generator) < accept
            out.append(x[keep])
            got += int(keep.sum())
            rounds += 1
        if got < n:
            raise RuntimeError(
                f"rejection sampling stalled at {got}/{n}; the density is far too "
                "peaked for a uniform proposal"
            )
        return torch.cat(out)[:n]

    def _log_cap(self, n_probe: int = 200_000) -> float:
        """Upper bound on log-density, estimated once and cached."""
        if getattr(self, "_cap", None) is None:
            probe = self.manifold.sample_uniform(n_probe, generator=self.generator)
            self._cap = float(self.density.log_prob(probe).max()) + 0.5
        return self._cap

    # -------------------------------------------------------------- iteration

    def __iter__(self):
        """Loaders are infinite streams; iterating never stops."""
        return self

    def __next__(self) -> torch.Tensor:
        """Next batch of ``batch_size`` samples."""
        return self.sample()

    def __call__(self, n: int) -> torch.Tensor:
        """Callable form, for Trainer.fit(batch_fn)."""
        return self.sample(n)

    def __repr__(self) -> str:
        """Manifold, density and batch size."""
        return (
            f"{type(self).__name__}({self.manifold!r}, {self.density!r}, "
            f"batch_size={self.batch_size})"
        )


class VonMisesFisherLoader(ManifoldLoader):
    """Manifold data drawn from an ambient vMF mixture restricted to it.

    Defined on the embedded point rather than in chart coordinates, so it is
    smooth and single-valued on any manifold -- including the Klein bottle, where
    a chart-space density would generically break across the identification.
    """

    def __init__(
        self,
        manifold: Manifold,
        mixture: VonMisesFisherMixture,
        batch_size: int = 1024,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(manifold, mixture, batch_size, generator)
        self.mixture = mixture

    @classmethod
    def random(
        cls,
        manifold: Manifold,
        n_components: int = 3,
        kappa_range: tuple[float, float] = (1.0, 3.0),
        batch_size: int = 1024,
        generator: torch.Generator | None = None,
    ):
        """Build a loader with a randomly drawn vMF mixture.

        Args:
            manifold: the space to sample on.
            n_components: number of vMF kernels.
            kappa_range: bounds on the concentrations.
            batch_size: default batch size.
            generator: RNG, so the density is reproducible from a seed.

        Returns:
            A loader over the manifold.
        """
        gen = generator or torch.Generator().manual_seed(0)
        mixture = VonMisesFisherMixture.random(
            n_components, manifold.d, kappa_range, generator=gen
        )
        return cls(manifold, mixture, batch_size, gen)

    def concentration_ratio(self, n_probe: int = 50_000) -> float:
        """max/min density over the manifold.

        Worth checking when choosing kappa: a density this ratio cannot be
        covered by finite training data, and score quality degrades monotonically
        with local density. A ratio around 20 is non-uniform enough to make the
        corrector work for its result while still reaching every region.
        """
        x = self.manifold.sample_uniform(n_probe, generator=self.generator)
        lp = self.density.log_prob(x)
        return float((lp.max() - lp.quantile(0.001)).exp())


class SphereVonMisesLoader(VonMisesFisherLoader):
    """vMF mixture on S^{d-1}, sampled exactly by Wood's algorithm.

    On the sphere the ambient vMF kernel IS the intrinsic vMF density, so exact
    sampling is available and rejection would be wasteful.
    """

    def __init__(
        self,
        manifold: Sphere,
        mixture: VonMisesFisherMixture,
        batch_size: int = 1024,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(manifold, mixture, batch_size, generator)
        self.vmf = VMFMixture(
            mixture.means.clone(),
            mixture.concentrations.clone(),
            mixture.weights.clone(),
        )

    def sample(self, n: int | None = None) -> torch.Tensor:
        """Draw a batch from the manifold.

        Args:
            n: batch size; defaults to ``self.batch_size``.

        Returns:
            ``(n, d)`` points lying exactly on the manifold.
        """
        n = self.batch_size if n is None else int(n)
        return self.vmf.sample(n, generator=self.generator)


class KleinVonMisesLoader(VonMisesFisherLoader):
    """vMF mixture restricted to the Klein bottle, sampled by rejection.

    No closed-form sampler exists here, but the uniform proposal is exact and the
    acceptance rate is high for the moderate concentrations used.
    """


def loader_for(manifold: Manifold, cfg: dict, generator=None) -> ManifoldLoader:
    """Build the loader named by config. Defaults to uniform."""
    dc = cfg.get("data", {})
    kind = dc.get("density", "vmf")
    gen = generator or torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    batch = int(cfg.get("train", {}).get("score", {}).get("batch_size", 1024))
    if kind == "uniform":
        return ManifoldLoader(manifold, None, batch, gen)
    mixture = VonMisesFisherMixture.random(
        dc.get("n_components", 3),
        manifold.d,
        tuple(dc.get("kappa_range", (1.0, 3.0))),
        generator=gen,
    )
    cls = SphereVonMisesLoader if manifold.name == "sphere" else KleinVonMisesLoader
    return cls(manifold, mixture, batch, gen)


def intersection_loader(
    manifold: Manifold,
    hyperplane: Hyperplane,
    cfg: dict,
    generator=None,
    mixture: VonMisesFisherMixture | None = None,
    uniform: bool = False,
    **section_kw,
) -> ManifoldLoader:
    """Data restricted to the conditional submanifold ``M ∩ H``.

    A section is itself a Manifold with an exact uniform sampler, and
    ``ManifoldLoader`` rejects against a uniform proposal -- which makes the
    volume element cancel -- so restricting the data density to the section needs
    no new sampling code at all.

    The mixture MUST be the one the unconditional model was trained on, or
    ``p_data`` restricted to the section is not the conditional of the
    distribution the model actually learned. Passing ``cfg`` alone rebuilds it
    from the same seed, which reproduces it exactly.

    Args:
        manifold: the ambient manifold.
        hyperplane: the cut, supplied at inference.
        cfg: experiment config, used to rebuild the density when none is given.
        generator: RNG for reproducibility.
        mixture: the trained density; rebuilt from ``cfg`` when omitted.
        uniform: return the uniform measure on the section instead of ``p_data``.
        **section_kw: forwarded to the section (e.g. ``grid`` for the Klein trace).

    Returns:
        A loader over ``M ∩ H``.
    """
    gen = generator or torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    section = section_for(manifold, hyperplane, **section_kw)
    batch = int(cfg.get("train", {}).get("score", {}).get("batch_size", 1024))
    if uniform or cfg.get("data", {}).get("density", "vmf") == "uniform":
        return ManifoldLoader(section, None, batch, gen)
    if mixture is None:
        dc = cfg.get("data", {})
        mixture = VonMisesFisherMixture.random(
            dc.get("n_components", 3),
            manifold.d,
            tuple(dc.get("kappa_range", (1.0, 3.0))),
            generator=torch.Generator().manual_seed(int(cfg.get("seed", 0))),
        )
    return ManifoldLoader(section, mixture, batch, gen)

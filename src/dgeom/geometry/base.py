"""The manifold interface.

A Manifold is PURE GEOMETRY. It knows its embedding, its volume measure, how to
project onto itself and what its tangent space is. It knows nothing about any
probability distribution beyond the uniform (Riemannian volume) measure, because
which distribution the data follows is an experimental choice, not a property of
the space -- that lives in `densities.py` and `loaders.py`.

Subclasses implement the geometry; everything else here is derived.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Manifold(ABC):
    """A compact embedded submanifold of R^d."""

    name: str = "manifold"

    # ------------------------------------------------------------- dimensions

    @property
    @abstractmethod
    def d(self) -> int:
        """Ambient dimension."""

    @property
    @abstractmethod
    def n(self) -> int:
        """Intrinsic dimension. Codimension is d - n."""

    @property
    def codim(self) -> int:
        """Codimension ``d - n``: how many directions point off the manifold."""
        return self.d - self.n

    @property
    def scale(self) -> float:
        """Characteristic length of the manifold.

        Distances are reported relative to this so a tolerance means the same
        thing on every manifold, however it happens to be embedded.
        """
        return 1.0

    # ---------------------------------------------------------------- geometry

    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest point on the manifold."""

    @abstractmethod
    def tangent_basis(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n, d) orthonormal basis of T_{P_M(x)} M."""

    @abstractmethod
    def chart_coords(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n) intrinsic coordinates of P_M(x)."""

    @abstractmethod
    def sample_uniform(self, n: int, *, generator=None) -> torch.Tensor:
        """Draw exactly from the intrinsic volume measure.

        This is the distribution the tempered corrector is supposed to
        reproduce, so it must be exact rather than approximate.
        """

    # ----------------------------------------------------------------- derived

    def uniform_marginals(self) -> list[tuple]:
        """One-dimensional marginals whose density is known under the uniform measure.

        Each entry is ``(label, project, pdf, (lo, hi))`` where ``project`` maps
        ambient points to a scalar and ``pdf`` is the exact density of that scalar
        when the sample is uniform on the manifold. Plotting the empirical
        histogram against ``pdf`` turns "is this uniform" into a picture rather
        than a p-value.

        Returns:
            A list of marginals; empty if the manifold declares none.
        """
        return []

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Euclidean distance from each point to its nearest point on M.

        Args:
            x: ``(B, d)`` ambient points.

        Returns:
            ``(B,)`` distances. Zero for points that lie on the manifold.
        """
        return (x - self.project(x)).norm(dim=-1)

    def normal_basis(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d-n, d) orthonormal complement of the tangent space."""
        return orthogonal_complement(self.tangent_basis(x))

    def __repr__(self) -> str:
        """Dimensions and scale, for logs."""
        return (
            f"{type(self).__name__}(d={self.d}, n={self.n}, "
            f"codim={self.codim}, scale={self.scale:g})"
        )


# ----------------------------------------------------------- linear-algebra aids


def orthogonal_complement(tangent: torch.Tensor) -> torch.Tensor:
    """Orthonormal complement of a tangent frame. (B, n, d) -> (B, d-n, d)."""
    b, n, d = tangent.shape
    eye = torch.eye(d, dtype=tangent.dtype, device=tangent.device).expand(b, d, d)
    proj = eye - torch.einsum("bni,bnj->bij", tangent, tangent)
    q, _ = torch.linalg.qr(proj.transpose(-1, -2), mode="complete")
    overlap = torch.einsum("bni,bid->bnd", tangent, q).abs().amax(dim=1)
    idx = overlap.argsort(dim=-1)[:, : d - n]
    return torch.gather(q.transpose(-1, -2), 1, idx.unsqueeze(-1).expand(-1, -1, d))


def principal_angles(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Principal angles (radians) between subspaces given by orthonormal rows.

    Both (B, k, d). Zero everywhere means the subspaces coincide, which is what
    "the model found the right normal space" looks like numerically.
    """
    m = torch.einsum("bkd,bld->bkl", a, b)
    return torch.arccos(torch.linalg.svdvals(m).clamp(-1.0, 1.0).clamp(max=1.0))

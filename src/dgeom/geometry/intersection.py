"""Conditional submanifolds: a manifold cut by a random hyperplane through 0.

Conditioning on ``<w, x> = 0`` ADDS CODIMENSION. That is the whole point: the
constraint is a measure-zero, lower-dimensional event, so its guidance term sits
at the geometry rate ``Theta(sigma^-2)`` rather than the density rate
``Theta(1)``, and the paper's separation should extend with codim + 1.

A section is itself a ``Manifold``. That is deliberate and is what keeps this
module small: every existing consumer of a manifold -- ``plot_uniformity``,
``uniformity_summary``, ``make_probe``, the samplers, the metrics -- works on a
section with no changes at all.

    S^3 ∩ H   is a great S^2 in w-perp. Exact in closed form, one component.
    K  ∩ H    is a CURVE with one to three components, depending on w, and no
              closed form. It is traced on the chart and sampled by arclength.

The Klein tracer must respect the identification ``Phi(u + 2pi, v) = Phi(u, -v)``
at the u-seam. Treating the chart as a plain torus emits points 0.82 off the
manifold and 0.53 off the hyperplane -- measured, not hypothetical.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import Manifold, orthogonal_complement
from .klein import TWO_PI, KleinBottle
from .sphere import Sphere


class Hyperplane:
    """An affine hyperplane ``{x : <w, x> = b}``.

    The offset is not cosmetic. Assumption 4.1 of Li et al. requires the set the
    conditional law concentrates on to be uniformly rectifiably path-connected,
    and on a Klein bottle the offset decides whether that holds: sections through
    the origin are disconnected for most directions (9 of 12 measured), whereas
    77% of offsets across the full range of ``<w, x>`` give a single component.
    The disconnected band straddles ``b = 0``.
    """

    def __init__(self, w: torch.Tensor, offset: float = 0.0) -> None:
        w = w / w.norm()
        # w and -w are the SAME hyperplane. Canonicalising the sign makes the
        # representation unique, which matters the moment w is fed to a network
        # or used as a dictionary key.
        nz = (w.abs() > 1e-12).nonzero()[0, 0]
        flip = w[nz] <= 0
        self.w = -w if flip else w
        # the pair (w, b) and (-w, -b) denote one hyperplane; canonicalising the
        # sign of w forces the matching sign on b
        self.b = float(-offset if flip else offset)

    @classmethod
    def random(
        cls,
        d: int = 4,
        *,
        offset: float = 0.0,
        dtype=torch.float64,
        generator=None,
    ) -> Hyperplane:
        """Draw a uniformly random hyperplane with a given offset.

        Args:
            d: ambient dimension.
            offset: the constant ``b`` in ``<w, x> = b``.
            dtype: floating point type.
            generator: RNG, so an experiment is reproducible from a seed.

        Returns:
            A hyperplane whose normal is uniform on the sphere.
        """
        return cls(torch.randn(d, dtype=dtype, generator=generator), offset)

    @property
    def d(self) -> int:
        """Ambient dimension."""
        return int(self.w.shape[0])

    def constraint(self, x: torch.Tensor) -> torch.Tensor:
        """``c(x) = <w, x> - b``; zero exactly on the hyperplane."""
        return x @ self.w.to(x.dtype).to(x.device) - self.b

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest point on the hyperplane."""
        w = self.w.to(x.dtype).to(x.device)
        return x - self.constraint(x).unsqueeze(-1) * w

    def basis(self) -> torch.Tensor:
        """``(d-1, d)`` orthonormal basis of ``w-perp``."""
        frame = orthogonal_complement(self.w.reshape(1, 1, -1))
        return frame[0]

    def __repr__(self) -> str:
        """The normal and offset, rounded, for logs."""
        vals = ", ".join(f"{v:+.3f}" for v in self.w.tolist())
        return f"Hyperplane(w=[{vals}], b={self.b:+.4f})"


class Section(Manifold):
    """Base for ``M ∩ H``. Subclasses supply the geometry."""

    def __init__(self, manifold: Manifold, hyperplane: Hyperplane) -> None:
        self.manifold, self.hyperplane = manifold, hyperplane
        self.name = f"{manifold.name}-section"

    @property
    def d(self) -> int:
        """Ambient dimension, unchanged by the cut."""
        return self.manifold.d

    @property
    def n(self) -> int:
        """Conditioning removes exactly one intrinsic dimension."""
        return self.manifold.n - 1

    @property
    def scale(self) -> float:
        """Inherited from the ambient manifold, so tolerances stay comparable."""
        return self.manifold.scale

    def constraint(self, x: torch.Tensor) -> torch.Tensor:
        """``<w, x>``, the quantity guidance drives to zero."""
        return self.hyperplane.constraint(x)

    def residuals(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Distance off ``M`` and off ``H``, kept separate.

        Landing on the manifold but off the hyperplane, and the reverse, are
        different failures with different causes. Averaging them into a single
        "distance to N" hides which one happened, so they are never combined.

        Args:
            x: ``(B, d)`` ambient points.

        Returns:
            ``(dist_M, abs_c)``, each ``(B,)``.
        """
        return self.manifold.dist(x), self.constraint(x).abs()


class SphereSection(Section):
    """``S^{d-1} ∩ H``: a sphere of radius ``r = sqrt(1 - b^2)`` in the plane.

    For ``b = 0`` this is a great ``S^{d-2}``; a nonzero offset shrinks it but
    never disconnects it, so the sphere satisfies the path-connectedness
    requirement at every offset. Exact throughout.
    """

    def __init__(self, manifold: Sphere, hyperplane: Hyperplane) -> None:
        super().__init__(manifold, hyperplane)
        self._basis = hyperplane.basis()  # (d-1, d)
        b = hyperplane.b
        if abs(b) >= 1.0:
            raise ValueError(
                f"offset b={b:g} does not meet the unit sphere (needs |b| < 1)"
            )
        self.radius = float((1.0 - b**2) ** 0.5)
        self._centre = hyperplane.b * hyperplane.w

    def project(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Project into the plane, then onto the circle of radius ``r`` in it."""
        c = self._centre.to(x.dtype).to(x.device)
        y = self.hyperplane.project(x) - c
        return c + self.radius * y / y.norm(dim=-1, keepdim=True).clamp_min(eps)

    def sample_uniform(
        self, n: int, *, device=None, dtype=torch.float64, generator=None
    ) -> torch.Tensor:
        """Uniform on the great subsphere, by normalising Gaussians in ``w-perp``.

        Args:
            n: number of samples.
            device: destination device.
            dtype: floating point type.
            generator: RNG for reproducibility.

        Returns:
            ``(n, d)`` points on ``M ∩ H``.
        """
        g = torch.randn(n, self.d, device=device, dtype=dtype, generator=generator)
        return self.project(g + self._centre.to(dtype).to(g.device))

    def tangent_basis(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, n, d)`` tangent frame: in the plane and orthogonal to the radius."""
        u = self.project(x) - self._centre.to(x.dtype).to(x.device)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        b = self._basis.to(x.dtype).to(x.device).expand(x.shape[0], -1, -1)
        # remove the radial direction from the w-perp frame, then re-orthonormalise
        coef = torch.einsum("bkd,bd->bk", b, u)
        reduced = b - coef.unsqueeze(-1) * u.unsqueeze(1)
        q, _ = torch.linalg.qr(reduced.transpose(-1, -2))
        return q.transpose(-1, -2)[:, : self.n]

    def chart_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Coordinates of ``P(x)`` in the in-plane basis, dropping the radius."""
        u = self.project(x) - self._centre.to(x.dtype).to(x.device)
        b = self._basis.to(x.dtype).to(x.device)
        return (u @ b.T)[..., : self.n]

    def uniform_marginals(self) -> list[tuple]:
        """Projections onto directions inside the hyperplane.

        For uniform on a 2-sphere of radius ``r``, ``<e, x - centre>`` is
        **exactly uniform on [-r, r]** for any unit ``e`` in the plane --
        Archimedes' hat-box theorem. That makes the check exact and readable.
        """
        b = self._basis
        r = self.radius

        def make(k: int):
            e = b[k]

            def project(x: torch.Tensor) -> torch.Tensor:
                c = self._centre.to(x.dtype).to(x.device)
                return (self.project(x) - c) @ e.to(x.dtype).to(x.device)

            def pdf(t):
                return np.full_like(t, 0.5 / r)

            return (f"<e{k}, x>", project, pdf, (-r, r))

        return [make(k) for k in range(min(3, self._basis.shape[0]))]


class KleinSection(Section):
    """``K ∩ H``: a closed curve with one to three components.

    Built by tracing the zero set of ``f(u, v) = <w, Phi(u, v)>`` on the chart and
    reducing it to a polyline with cumulative arclength, which turns "uniform on
    the section" into an inverse-CDF draw and gives an exact 1-D uniformity test.
    """

    def __init__(
        self, manifold: KleinBottle, hyperplane: Hyperplane, grid: int = 800
    ) -> None:
        super().__init__(manifold, hyperplane)
        self.grid = int(grid)
        self._trace()

    # ------------------------------------------------------------------ tracing

    def _f(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        w = self.hyperplane.w.to(self.manifold.dtype)
        return self.manifold.from_chart(u, v) @ w - self.hyperplane.b

    def _trace(self) -> None:
        """Marching squares on a CLOSED chart grid.

        The embedding already encodes the identification: evaluating Phi at
        u = 2pi reproduces Phi(0, -v) to 3e-16, and v is plainly periodic. So the
        grid simply runs over the closed square [0, 2pi]^2 and no seam bookkeeping
        is needed. What must NOT happen is wrapping index G-1 back to 0, which
        interpolates between Phi(2pi - du, v) and Phi(0, v) -- two points that are
        far apart. That mistake put traced points 0.82 off the manifold.
        """
        G, du = self.grid, TWO_PI / self.grid
        g = torch.arange(G + 1, dtype=self.manifold.dtype) * du
        U, V = torch.meshgrid(g, g, indexing="ij")
        f = self._f(U.reshape(-1), V.reshape(-1)).reshape(G + 1, G + 1).numpy()
        gu = g.numpy()

        def zero(fa, fb):
            """Fractional position of the crossing along an edge."""
            return fa / (fa - fb)

        sgn = f > 0
        # the four edges of every cell, as (mask, u, v) stacks of shape (G, G)
        cu, cv = gu[:-1, None] + 0 * gu[None, :-1], 0 * gu[:-1, None] + gu[None, :-1]
        t0 = zero(f[:-1, :-1], f[1:, :-1])
        t1 = zero(f[1:, :-1], f[1:, 1:])
        t2 = zero(f[:-1, 1:], f[1:, 1:])
        t3 = zero(f[:-1, :-1], f[:-1, 1:])
        masks = np.stack(
            [
                sgn[:-1, :-1] != sgn[1:, :-1],  # bottom, u varies
                sgn[1:, :-1] != sgn[1:, 1:],  # right,  v varies
                sgn[:-1, 1:] != sgn[1:, 1:],  # top,    u varies
                sgn[:-1, :-1] != sgn[:-1, 1:],  # left,   v varies
            ],
            axis=-1,
        )
        us = np.stack([cu + t0 * du, cu + du, cu + t2 * du, cu], axis=-1)
        vs = np.stack([cv, cv + t1 * du, cv + du, cv + t3 * du], axis=-1)
        # a unique id per grid edge, so segments sharing an edge can be linked
        ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
        eu, ev = (G + 1) * G, 0
        ids = np.stack(
            [
                ev + ii * (G + 1) + jj,
                eu + (ii + 1) * G + jj,
                ev + ii * (G + 1) + jj + 1,
                eu + ii * G + jj,
            ],
            axis=-1,
        )

        count = masks.sum(-1)
        cells = count >= 2
        if not cells.any():
            raise RuntimeError(
                f"{self.hyperplane!r} does not meet the Klein bottle on a "
                f"{G}x{G} grid; the tracer is wrong -- every hyperplane through "
                "the origin meets it"
            )
        m, uu, vv, idc = masks[cells], us[cells], vs[cells], ids[cells]
        # indices of the two crossing edges (a 4-crossing saddle is joined
        # arbitrarily; it is rare and sub-cell-sized either way)
        pick = np.argsort(~m, axis=1, kind="stable")[:, :2]
        rows = np.arange(len(m))
        a_uv = np.stack([uu[rows, pick[:, 0]], vv[rows, pick[:, 0]]], axis=-1)
        b_uv = np.stack([uu[rows, pick[:, 1]], vv[rows, pick[:, 1]]], axis=-1)
        self._edge_ids = self._canon_edges(
            np.stack([idc[rows, pick[:, 0]], idc[rows, pick[:, 1]]], -1), G
        )

        a = torch.tensor(a_uv, dtype=self.manifold.dtype)
        b = torch.tensor(b_uv, dtype=self.manifold.dtype)
        self._seg_uv_a, self._seg_uv_b = a, b
        self._seg_a = self.manifold.from_chart(a[:, 0], a[:, 1])
        self._seg_b = self.manifold.from_chart(b[:, 0], b[:, 1])

        self._lengths = (self._seg_b - self._seg_a).norm(dim=-1)
        self._cum = torch.cat(
            [torch.zeros(1, dtype=self._lengths.dtype), self._lengths.cumsum(0)]
        )
        self.length = float(self._cum[-1])

    @staticmethod
    def _canon_edges(ids: np.ndarray, G: int) -> np.ndarray:
        """Map seam-duplicated grid edges onto one canonical id.

        The chart is a CLOSED surface: column ``G`` is column ``0``, and row
        ``G`` is row ``0`` with ``v -> -v``. Edges on those far borders are the
        same edge as their partner, so leaving them distinct splits a component
        that is actually joined through the seam -- which overcounted components
        on 10 of 16 measured (w, b) pairs.

        Edge numbering (see ``_trace``): a u-edge from ``(i,j)`` to ``(i+1,j)``
        is ``i(G+1) + j``; a v-edge from ``(i,j)`` to ``(i,j+1)`` is
        ``G(G+1) + iG + j``.
        """
        out = ids.copy()
        base = G * (G + 1)
        u = out < base
        # v-seam: u-edge(i, G) is u-edge(i, 0)
        iu, ju = out[u] // (G + 1), out[u] % (G + 1)
        out[u] = np.where(ju == G, iu * (G + 1), out[u])
        # u-seam: v-edge(G, j) is v-edge(0, G-1-j), the v axis running backwards
        k = out[~u] - base
        iv, jv = k // G, k % G
        out[~u] = np.where(iv == G, base + (G - 1 - jv), out[~u])
        return out

    # ----------------------------------------------------------------- geometry

    def _refine(self, uv: torch.Tensor, iters: int = 8) -> torch.Tensor:
        """Newton in the chart, driving ``<w, Phi(u,v)>`` to zero.

        The polyline is a chord approximation, so its points sit O(du^2) off the
        true curve -- 3e-5 at G=400, far above the tolerance everything else in
        this project is held to. Newton fixes that: each iterate is Phi of some
        chart coordinate, hence *exactly* on the manifold by construction, and the
        step only has to kill the hyperplane residual.

        Args:
            uv: ``(B, 2)`` chart coordinates near the curve.
            iters: Newton steps; convergence is quadratic.

        Returns:
            ``(B, 2)`` refined chart coordinates.
        """
        w = self.hyperplane.w.to(uv.dtype)
        bb = self.hyperplane.b
        uv = uv.clone()
        for _ in range(iters):
            q = uv.detach().requires_grad_(True)
            f = self.manifold.from_chart(q[:, 0], q[:, 1]) @ w - bb
            (grad,) = torch.autograd.grad(f.sum(), q)
            uv = q.detach() - (
                f.detach().unsqueeze(-1) * grad / grad.pow(2).sum(-1, keepdim=True)
            )
        return uv.detach()

    def sample_uniform(
        self, n: int, *, device=None, dtype=torch.float64, generator=None
    ) -> torch.Tensor:
        """Uniform by ARCLENGTH along the traced curve.

        Segments are chosen with probability proportional to their length, so the
        one-to-three components are weighted by how much curve they carry.
        Sampling components equally would silently bias every downstream result.

        Args:
            n: number of samples.
            device: destination device.
            dtype: floating point type.
            generator: RNG for reproducibility.

        Returns:
            ``(n, d)`` points on ``M ∩ H``.
        """
        s = torch.rand(n, dtype=self._cum.dtype, generator=generator) * self.length
        idx = torch.searchsorted(self._cum[1:].contiguous(), s).clamp(
            max=len(self._lengths) - 1
        )
        t = ((s - self._cum[idx]) / self._lengths[idx].clamp_min(1e-30)).clamp(0, 1)
        # interpolate in the CHART (both endpoints lie in one cell, so there is no
        # wrap to worry about), then Newton back onto the exact section
        uv = self._seg_uv_a[idx] + t.unsqueeze(-1) * (
            self._seg_uv_b[idx] - self._seg_uv_a[idx]
        )
        uv = self._refine(uv)
        p = self.manifold.from_chart(uv[:, 0], uv[:, 1])
        return p.to(device=device, dtype=dtype)

    def _nearest(self, x: torch.Tensor, chunk: int = 4096):
        """Nearest segment index and its parameter, computed in chunks.

        The naive form builds a ``(B, S, d)`` tensor, which at 400k points and
        ~1100 segments is 14 GB in a single allocation. Chunking keeps it bounded
        and costs nothing in accuracy.

        Args:
            x: ``(B, d)`` ambient points.
            chunk: points per block.

        Returns:
            ``(idx, t)``: nearest segment per point and the position along it.
        """
        a = self._seg_a.to(x.dtype).to(x.device)
        b = self._seg_b.to(x.dtype).to(x.device)
        ab = b - a
        denom = (ab * ab).sum(-1).clamp_min(1e-30)
        idx_out, t_out = [], []
        for i in range(0, x.shape[0], chunk):
            xc = x[i : i + chunk]
            t = (((xc.unsqueeze(1) - a) * ab).sum(-1) / denom).clamp(0.0, 1.0)
            foot = a + t.unsqueeze(-1) * ab
            k = (xc.unsqueeze(1) - foot).norm(dim=-1).argmin(dim=1)
            idx_out.append(k)
            t_out.append(t[torch.arange(len(k), device=x.device), k])
        return torch.cat(idx_out), torch.cat(t_out)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest point on the traced polyline.

        Uses the trace rather than ``KleinBottle.project`` so the result lies on
        the SECTION, not merely on the manifold.
        """
        k, t = self._nearest(x)
        a = self._seg_a.to(x.dtype).to(x.device)[k]
        b = self._seg_b.to(x.dtype).to(x.device)[k]
        return a + t.unsqueeze(-1) * (b - a)

    def arclength(self, x: torch.Tensor) -> torch.Tensor:
        """Arclength coordinate ``s`` in ``[0, L)`` of the nearest curve point."""
        k, t = self._nearest(x)
        cum = self._cum.to(x.dtype).to(x.device)
        lens = self._lengths.to(x.dtype).to(x.device)
        return cum[k] + t * lens[k]

    def chart_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Arclength, the natural one-dimensional chart on a curve."""
        return self.arclength(x).unsqueeze(-1)

    def tangent_basis(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, d)`` unit tangent of the nearest segment."""
        k, _ = self._nearest(x)
        a = self._seg_a.to(x.dtype).to(x.device)
        b = self._seg_b.to(x.dtype).to(x.device)
        dvec = (b - a)[k]
        return (dvec / dvec.norm(dim=-1, keepdim=True).clamp_min(1e-30)).unsqueeze(1)

    def uniform_marginals(self) -> list[tuple]:
        """Arclength is uniform on ``[0, L)`` exactly when the sample is uniform."""
        L = self.length

        def project(x: torch.Tensor) -> torch.Tensor:
            return self.arclength(x)

        def pdf(t):
            return np.full_like(t, 1.0 / L)

        return [("arclength s", project, pdf, (0.0, L))]

    def n_components(self) -> int:
        """Connected components, by union-find over shared grid edges.

        Two segments belong to the same component exactly when they cross the
        same grid edge, so this is combinatorial and needs no distance tolerance.
        """
        parent: dict[int, int] = {}

        def find(i: int) -> int:
            parent.setdefault(i, i)
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b in self._edge_ids:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra
        return len({find(int(a)) for a, _ in self._edge_ids})


def section_for(manifold: Manifold, hyperplane: Hyperplane, **kw) -> Section:
    """Build the section of ``manifold`` by ``hyperplane``."""
    if isinstance(manifold, Sphere):
        return SphereSection(manifold, hyperplane)
    if isinstance(manifold, KleinBottle):
        return KleinSection(manifold, hyperplane, **kw)
    raise TypeError(f"no section implemented for {type(manifold).__name__}")


__all__ = [
    "Hyperplane",
    "KleinSection",
    "Section",
    "SphereSection",
    "connected_offset",
    "section_for",
]


def connected_offset(
    manifold: Manifold,
    w: torch.Tensor,
    *,
    n_probe: int = 41,
    margin: float = 0.04,
    grid: int = 300,
) -> float:
    """Find an offset ``b`` whose section ``M ∩ {<w,x> = b}`` is connected.

    Assumption 4.1 of Li et al. requires the set the conditional law concentrates
    on to be uniformly rectifiably path-connected. A disconnected section
    violates it, and local dynamics then cannot move mass between components at
    any tempering exponent -- the between-component allocation is frozen at
    initialisation. On the Klein bottle sections through the origin are
    disconnected for most directions, while roughly 77% of offsets are not, so
    the offset is a modelling choice rather than a detail.

    Among connected offsets this returns the most TRANSVERSAL one, maximising
    ``min |P_{T M} w|`` over the section. Maximising distance from the
    disconnected band instead drives ``b`` to the extremes of the range, where
    the section shrinks onto a critical point of ``<w, x>`` and the intersection
    becomes tangential: measured at 4% from the maximum, ``|P_T w|`` fell to
    0.38 against 0.88 mid-range, and the corrector then moved the sample away
    from uniform rather than toward it.

    Args:
        manifold: the ambient manifold.
        w: hyperplane normal, not necessarily unit.
        n_probe: offsets to scan.
        margin: fraction of the range trimmed at each end, where the section
            degenerates to a point.
        grid: chart resolution used while probing.

    Returns:
        An offset whose section is connected.

    Raises:
        RuntimeError: if no scanned offset gives a single component.
    """
    w = w / w.norm()
    probe = manifold.sample_uniform(200_000)
    vals = (probe @ w.to(probe.dtype)).numpy()
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo
    offsets = np.linspace(lo + margin * span, hi - margin * span, n_probe)

    def transversality(sec) -> float:
        """Min |P_T w| along the section: how squarely the plane cuts M.

        This is the quantity that matters for sampling. The guidance restoring
        force is proportional to |P_T w|^2, so where it is small the constraint
        confines weakly and the equilibrium cloud around N is correspondingly
        fat -- which both slows mixing and distorts the projection used to
        measure uniformity.
        """
        y = sec.sample_uniform(4000)
        t = sec.manifold.tangent_basis(y)
        return float(torch.einsum("bnd,d->bn", t, w.to(y.dtype)).norm(dim=-1).min())

    best, best_score = None, -1.0
    for b in offsets:
        try:
            sec = section_for(
                manifold,
                Hyperplane(w, float(b)),
                **({"grid": grid} if isinstance(manifold, KleinBottle) else {}),
            )
            if isinstance(sec, KleinSection) and sec.n_components() != 1:
                continue
            score = transversality(sec)
            if score > best_score:
                best, best_score = float(b), score
        except (RuntimeError, ValueError):
            continue
    if best is None:
        raise RuntimeError(
            f"no connected section found for w={w.tolist()} over {n_probe} offsets"
        )
    return best

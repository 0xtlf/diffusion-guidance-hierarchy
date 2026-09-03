"""A Klein bottle embedded in R^4.

    Phi(u,v) = ( (R + r cos v) cos u,
                 (R + r cos v) sin u,
                 r sin v cos(u/2),
                 r sin v sin(u/2) )

with fundamental domain u, v in [0, 2pi) and the identification
Phi(u + 2pi, v) = Phi(u, -v), which is exactly what makes the surface
non-orientable.  This is a genuine *embedding* into R^4 (verified: no point's
nearest neighbour is far away in the chart), not a self-intersecting immersion
like the familiar R^3 picture.

Chosen to break every convenience the sphere provided: intrinsic dimension 2 in
ambient 4 (codimension 2, not 1), non-homogeneous, non-orientable, a volume
element that varies over the surface, and no closed form for p_sigma.

Three facts make it workable anyway, all verified numerically:

  * the chart is ORTHOGONAL (F = <Phi_u, Phi_v> = 0 to 8e-10), so
        sqrt(det g) = r * sqrt((R + r cos v)^2 + r^2 sin^2 v / 4)
    which depends on v ALONE.
  * therefore uniform-with-respect-to-area FACTORISES: u ~ Unif[0,2pi) and
    v ~ proportional to sqrt(det g)(v).  Uniformity on a Klein bottle is thus two
    exact one-dimensional tests rather than a density estimate.
  * the chart inverts in closed form (round-trip error 3e-15), so a sample can be
    pushed back to (u,v) exactly.

Coordinates are RMS-normalised so the manifold occupies the same scale as the
unit sphere and a single sigma schedule serves both.
"""

from __future__ import annotations

import math

import torch

from .base import Manifold

TWO_PI = 2 * math.pi


class KleinBottle(Manifold):
    """Klein bottle in R^4, RMS-normalised to unit scale."""

    name = "klein"

    def __init__(
        self,
        R: float = 2.0,
        r: float = 1.0,
        normalise: bool = True,
        grid: int = 512,
        dtype=torch.float64,
    ) -> None:
        self.R, self.r = float(R), float(r)
        self.dtype = dtype
        self._norm = 1.0
        if normalise:
            u = torch.rand(200_000, dtype=dtype) * TWO_PI
            v = self._sample_v_by_area(200_000, dtype=dtype)
            self._norm = float((self._embed_raw(u, v).norm(dim=-1) ** 2).mean().sqrt())
        # coarse chart grid, used to seed the nearest-point Newton solve
        gu = torch.linspace(0, TWO_PI, grid + 1, dtype=dtype)[:-1]
        gv = torch.linspace(0, TWO_PI, grid + 1, dtype=dtype)[:-1]
        GU, GV = torch.meshgrid(gu, gv, indexing="ij")
        self._gu, self._gv = GU.reshape(-1), GV.reshape(-1)
        self._gp = self.from_chart(self._gu, self._gv)

    # ---------------------------------------------------------------- basics

    @property
    def d(self) -> int:
        """Ambient dimension: the Klein bottle embeds in R^4."""
        return 4

    @property
    def n(self) -> int:
        """Intrinsic dimension: it is a surface, so codimension 2."""
        return 2

    @property
    def scale(self) -> float:
        """Characteristic length; 1.0 once RMS-normalised."""
        return 1.0 if self._norm != 1.0 else float(self.R + self.r)

    def _embed_raw(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        R, r = self.R, self.r
        rho = R + r * torch.cos(v)
        return torch.stack(
            [
                rho * torch.cos(u),
                rho * torch.sin(u),
                r * torch.sin(v) * torch.cos(u / 2),
                r * torch.sin(v) * torch.sin(u / 2),
            ],
            dim=-1,
        )

    def from_chart(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Phi(u, v), normalised."""
        return self._embed_raw(u, v) / self._norm

    def chart_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Exact inverse of Phi, for points on (or projected onto) M.

        u comes straight from the angle in the first two coordinates; cos v from
        the radius; sin v from whichever of the last two coordinates has the
        larger denominator, which avoids the u/2 half-angle vanishing.
        """
        y = x * self._norm
        u = torch.atan2(y[..., 1], y[..., 0]) % TWO_PI
        rho = torch.linalg.norm(y[..., :2], dim=-1)
        cos_v = ((rho - self.R) / self.r).clamp(-1.0, 1.0)
        cu2, su2 = torch.cos(u / 2), torch.sin(u / 2)
        use_c = cu2.abs() >= su2.abs()
        den = torch.where(use_c, cu2, su2)
        num = torch.where(use_c, y[..., 2], y[..., 3])
        sin_v = num / (
            self.r * torch.where(den.abs() < 1e-12, torch.full_like(den, 1e-12), den)
        )
        v = torch.atan2(sin_v, cos_v) % TWO_PI
        return torch.stack([u, v], dim=-1)

    def uniform_marginals(self) -> list[tuple]:
        """Chart coordinates.

        Because the chart is orthogonal, ``sqrt(det g)`` depends on ``v`` alone,
        so uniform-by-area factorises: ``u`` is uniform on ``[0, 2 pi)`` and ``v``
        has density proportional to ``sqrt(det g)(v)``. Both are exact.
        """
        import numpy as np

        grid = torch.linspace(0, TWO_PI, 4001, dtype=self.dtype)
        weight = self.volume_element(grid)
        area = float(torch.trapezoid(weight, grid))
        gn, wn = grid.numpy(), (weight / area).numpy()

        # project() is a grid search plus Gauss-Newton, by far the most
        # expensive thing in a uniformity probe. Both marginals are read off the
        # same chart coordinates, so computing them once per input rather than
        # once per marginal halves the cost of every probe.
        cache: dict = {}

        def chart_of(x: torch.Tensor) -> torch.Tensor:
            key = (x.data_ptr(), tuple(x.shape), x._version)
            if cache.get("key") != key:
                cache["key"] = key
                cache["uv"] = self.chart_coords(self.project(x))
            return cache["uv"]

        def u_of(x: torch.Tensor) -> torch.Tensor:
            return chart_of(x)[..., 0]

        def v_of(x: torch.Tensor) -> torch.Tensor:
            return chart_of(x)[..., 1]

        return [
            ("u", u_of, lambda t: np.full_like(t, 1.0 / TWO_PI), (0.0, TWO_PI)),
            ("v", v_of, lambda t: np.interp(t, gn, wn), (0.0, TWO_PI)),
        ]

    # ------------------------------------------------------------- geometry

    def volume_element(self, v: torch.Tensor) -> torch.Tensor:
        """sqrt(det g) as a function of v alone (orthogonal chart)."""
        R, r = self.R, self.r
        return (
            r
            * ((R + r * torch.cos(v)) ** 2 + r**2 * torch.sin(v) ** 2 / 4).sqrt()
            / self._norm**2
        )

    def frame(self, u: torch.Tensor, v: torch.Tensor):
        """Unnormalised tangent vectors (Phi_u, Phi_v)."""
        R, r = self.R, self.r
        cu, su = torch.cos(u), torch.sin(u)
        cv, sv = torch.cos(v), torch.sin(v)
        cu2, su2 = torch.cos(u / 2), torch.sin(u / 2)
        rho = R + r * cv
        Pu = torch.stack(
            [-rho * su, rho * cu, -r * sv * su2 / 2, r * sv * cu2 / 2], dim=-1
        )
        Pv = torch.stack(
            [-r * sv * cu, -r * sv * su, r * cv * cu2, r * cv * su2], dim=-1
        )
        return Pu / self._norm, Pv / self._norm

    def tangent_basis(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2, 4) orthonormal tangent frame at P_M(x)."""
        uv = self.chart_coords(self.project(x))
        Pu, Pv = self.frame(uv[..., 0], uv[..., 1])
        Pu = Pu / Pu.norm(dim=-1, keepdim=True)
        Pv = Pv / Pv.norm(dim=-1, keepdim=True)  # chart is orthogonal already
        return torch.stack([Pu, Pv], dim=-2)

    # ----------------------------------------------------------- projection

    def project(
        self, x: torch.Tensor, newton_steps: int = 12, chunk: int = 4096
    ) -> torch.Tensor:
        """Nearest point on M: coarse chart grid, then Gauss-Newton refinement."""
        uv = []
        gp = self._gp.to(x.device, x.dtype)
        for i in range(0, x.shape[0], chunk):
            xi = x[i : i + chunk]
            j = torch.cdist(xi, gp).argmin(dim=1)
            uv.append(
                torch.stack(
                    [
                        self._gu.to(x.device, x.dtype)[j],
                        self._gv.to(x.device, x.dtype)[j],
                    ],
                    dim=-1,
                )
            )
        uv = torch.cat(uv)
        u, v = uv[..., 0].clone(), uv[..., 1].clone()

        for _ in range(newton_steps):
            p = self.from_chart(u, v)
            Pu, Pv = self.frame(u, v)
            d = p - x
            # orthogonal chart => the 2x2 normal system is diagonal
            gu = (d * Pu).sum(-1)
            gv = (d * Pv).sum(-1)
            hu = (Pu * Pu).sum(-1).clamp_min(1e-30)
            hv = (Pv * Pv).sum(-1).clamp_min(1e-30)
            u = (u - gu / hu) % TWO_PI
            v = (v - gv / hv) % TWO_PI
        return self.from_chart(u, v)

    # ------------------------------------------------------------- sampling

    def _area_cdf(self, dtype):
        """Grid and normalised CDF of sqrt(det g)(v), cached.

        Fixed for the manifold, but rebuilding it per batch dominated the
        training step (it was most of the 14 ms spent sampling 1024 points).
        """
        key = getattr(self, "_cdf_cache", None)
        if key is None or key[0] != dtype:
            grid = torch.linspace(0, TWO_PI, 20_001, dtype=dtype)
            R, r = self.R, self.r
            w = (
                r
                * (
                    (R + r * torch.cos(grid)) ** 2 + r**2 * torch.sin(grid) ** 2 / 4
                ).sqrt()
            )
            cdf = torch.cat(
                [torch.zeros(1, dtype=dtype), torch.cumulative_trapezoid(w, grid)]
            )
            self._cdf_cache = (dtype, grid, cdf / cdf[-1])
        return self._cdf_cache[1], self._cdf_cache[2]

    def _sample_v_by_area(self, n: int, *, dtype=None, generator=None) -> torch.Tensor:
        """V with density proportional to sqrt(det g)(v), by inverse CDF.

        Exact up to the quadrature grid, which is fine because the density is a
        smooth 1-D function of v.
        """
        dtype = dtype or self.dtype
        grid, cdf = self._area_cdf(dtype)
        q = torch.rand(n, dtype=dtype, generator=generator)
        idx = torch.searchsorted(cdf, q).clamp(1, len(grid) - 1)
        lo, hi = cdf[idx - 1], cdf[idx]
        t = ((q - lo) / (hi - lo).clamp_min(1e-30)).clamp(0, 1)
        return grid[idx - 1] + t * (grid[idx] - grid[idx - 1])

    def sample_uniform(self, n: int, *, generator=None) -> torch.Tensor:
        """Exactly uniform w.r.t. the intrinsic area measure."""
        u = torch.rand(n, dtype=self.dtype, generator=generator) * TWO_PI
        v = self._sample_v_by_area(n, generator=generator)
        return self.from_chart(u, v)

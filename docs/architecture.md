# Architecture

Design of `src/dgeom` and how to extend each component.

```
src/dgeom/
├── geometry/    manifolds · densities on them · dataloaders
├── models/      diffusion models: the score, and everything derived from it
├── nn/          networks: layers → backbones → networks
├── sampling/    samplers that consume a model and produce samples
├── metrics/     has it learned the manifold? is a sample uniform on it?
├── training/    the training loop and its callbacks
├── viz/         figures
└── config.py    YAML configs, run directories, metric logging
```

Four abstractions carry the design. Each has one job, and each is extended by
implementing a single method.

### `Manifold` — pure geometry

Knows its embedding, its volume measure, how to project onto itself, and its
tangent space. It knows **nothing** about any data distribution: which
distribution the data follows is an experimental choice, not a property of the
space.

```python
from dgeom.geometry import KleinBottle

M = KleinBottle()            # KleinBottle(d=4, n=2, codim=2, scale=1)
x = M.sample_uniform(1000)   # exactly uniform w.r.t. the volume measure
M.project(x)                 # nearest point on M
M.tangent_basis(x)           # (B, n, d) orthonormal tangent frame
M.chart_coords(x)            # (B, n) intrinsic coordinates
```

*To add a manifold:* implement `d`, `n`, `project`, `tangent_basis`,
`chart_coords`, `sample_uniform`. You get `codim`, `dist` and `normal_basis` free.

### `ManifoldLoader` — data, defaulting to uniform

A `Density` gives a log-density w.r.t. the volume measure, so "uniform" is
`log p = 0`. A loader pairs it with a manifold and yields batches. **The default
is uniform**, so a non-uniform dataset is always an explicit choice.

```python
from dgeom.geometry import ManifoldLoader, KleinVonMisesLoader

ManifoldLoader(M, batch_size=1024)                    # uniform
loader = KleinVonMisesLoader.random(M, n_components=3, kappa_range=(1., 3.))
loader.sample(4096)                                   # (4096, 4) points on M
loader.concentration_ratio()                          # max/min density  → ~15
```

Loaders are infinite iterators *and* callables, so `loader` drops straight into
`Trainer.fit(loader)`.

The vMF mixture is defined on the **embedded point**,
`log p(x) ∝ logsumexp_k[log wₖ + κₖ⟨μₖ, x⟩]`, not in chart coordinates. On the
sphere that is the standard vMF, which preserves the closed-form reference score.
On the Klein bottle it avoids a real trap: a density written in `(u,v)` is
generically **discontinuous** across the identification `Φ(u+2π,v) = Φ(u,−v)`.

### `DiffusionModel` — the score, and everything from it

A forward noising process plus a score, expressed in **hat space**:

```
shat(x, σ) = σ² · ∇ₓ log p_σ(x) = E[x₀ | x] − x
```

the Tweedie displacement. It is `O(1)` near the manifold where the raw score blows
up like `σ⁻²` — which keeps float32 safe at `σ_min`, and makes the theory's
tolerance readable directly: **raw error `o(σ⁻²)` is exactly hat error `o(1)`**.

> **A subclass implements `shat` and nothing else.** `score`, `denoise` and
> `add_noise` are derived. That is deliberate: a conditional model (classifier
> guidance, say) adds a term to the score, so it overrides `shat` alone and
> inherits every sampler and metric unchanged.

| class | role |
| ----- | ---- |
| `ScoreDiffusion` | backed by a `ScoreNetwork`; trainable, has `.loss()` |
| `AnalyticDiffusion` | **ground truth** on the sphere, closed form via Bessel functions |
| `QuadratureDiffusion` | **ground truth** on a 2-D manifold, Gauss-Legendre quadrature |
| `NoiseSchedule` | owns `σ`: range, sampling, and a `contains()` guard |

Because the exact models satisfy the same interface, any experiment can be pointed
at either — which is how *"does the method work"* gets separated from *"is the
model good enough"*.

### `Sampler` — how samples are drawn

Consumes a model, never inspecting how the score is computed.

```python
from dgeom.sampling import LangevinSampler, TemperedLangevin, AnnealedLangevin

sampler = TemperedLangevin(sigma=0.01, alpha=0.5, n_steps=70_000)
x, trace = sampler.sample(model, x0)
sampler.cloud_thickness        # σ^(1-α/2), the predicted equilibrium width
```

**Tempering lives in the sampler, not the model** — it is a sampling choice. The
same model with `α = 0` targets `p_data`; with `0 < α < 2` it targets the uniform
measure. Samplers refuse a `σ` outside the model's trained range.

### Networks

```
nn/layers/      FourierSigmaEmbedding · ResidualBlock
nn/backbones/   MLPBackbone     (input → features)
nn/networks/    ScoreNetwork    (backbone + task head)
```

A new network type is one file under `networks/`, reusing the layers and backbones.

### Metrics

Reference-free, so they read identically on both manifolds:

- **`flow_to_manifold`** — flow the score deterministically with no noise; where
  points land *is* the learned manifold.
- **`jacobian_spectrum`** — near `M`, `J = ∂shat/∂x ≈ −P⊥`, so the spectrum is `−1`
  with multiplicity **codim** and `0` on the tangent space. Reads the intrinsic
  dimension straight off the model.
- **`coverage`** — covering radius, catching mode collapse.
- **`uniformity`** — exact on both manifolds: pooled marginals on `S³`; on the
  Klein bottle two 1-D KS tests plus a joint χ², via exact chart inversion.

---


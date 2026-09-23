# dgeom

Uniform sampling on manifolds and on **conditional submanifolds**. Extends
[*When Scores Learn Geometry*](https://arxiv.org/abs/2509.24912)
(Li, Shen, Hsieh & He).

## Overview

### Setting

$`\mathcal{M}\subset\mathbb{R}^d`$ is a compact boundaryless $`C^4`$ submanifold,
$`\dim\mathcal{M}=n`$, carrying data $`\mu_{\mathrm{data}}`$. The Gaussian-smoothed
measure is $`p_\sigma = \mu_{\mathrm{data}} * \mathcal{N}(0,\sigma^2 I)`$, and

```math
d_{\mathcal{M}}(x) := \tfrac12\,\mathrm{dist}^2(x,\mathcal{M}),\qquad
\hat{s}(x,\sigma) := \sigma^2\nabla\log p_\sigma(x) = \mathbb{E}[x_0\mid x]-x .
```

### Rate separation (Thm 3.1)

```math
\log p_\sigma(x) = -\frac{1}{\sigma^2}d_{\mathcal{M}}(x)
+ \log p_{\mathrm{data}}\big(\Phi^{-1}(P_{\mathcal{M}}(x))\big)
- \frac{d-n}{2}\log(2\pi\sigma^2) + H(x) + o(1).
```

Geometry enters at $`\Theta(\sigma^{-2})`$, density at $`\Theta(1)`$.

### Tempered Score Langevin (eq. 8)

```math
dX_t = \sigma^\alpha s(X_t,\sigma)\,dt + \sqrt{2}\,dW_t .
```

**Thm 5.1.** If $`\lVert s-s^\ast\rVert_{L^\infty(K)} = o(\sigma^\beta)`$ for some
$`\beta>-2`$, then for any $`\max\{-\beta,0\}<\alpha<2`$ the stationary law
$`\tilde\pi_\sigma`$ converges to the uniform measure on $`\mathcal{M}`$:
$`\tilde\pi(u)\propto\sqrt{\det g(u)}`$.

Two consequences used throughout: the chain sits at transverse width
$`\sigma^{1-\alpha/2}`$, and at finite $`\sigma`$ with a gradient oracle its law is
$`\tilde\pi_\sigma\propto\exp(-\sigma^\alpha f_\sigma)`$.

### What this repo adds

Condition on a hyperplane $`H=\{x:\langle w,x\rangle = b\}`$ given **at inference
only**, giving the conditional submanifold

```math
N = \mathcal{M}\cap H,\qquad \dim N = n-1 .
```

Guidance is exact and needs no classifier. By Tweedie,

```math
m(x) = \mathbb{E}\big[\langle w,x_0\rangle \mid x_t\big]
= \langle w,\, x+\hat{s}(x,\sigma)\rangle,\qquad
\nabla\log p_\sigma(c\mid x) \approx -\frac{m(x)-b}{v_c}\,w,\quad v_c\approx\sigma^2 .
```

Transversality gives $`d_N^2 = d_{\mathcal{M}}^2 + d_H^2`$, so the constraint is
geometry, not density: one $`\alpha`$ governs both.

**Question.** Does tempering the guided score sample
$`\mathrm{Unif}(N)`$ — and at what score accuracy?

### Inference algorithm

Nothing is trained. $`w`$ and $`b`$ enter at sampling time; $`\hat{s}_\theta`$ is a
frozen unconditional model.

**Input** $`\hat{s}_\theta`$, normal $`w`$, noise level $`\sigma`$, exponent
$`\alpha`$, step scale $`\eta`$, steps $`T`$.

1. **Offset.** `connected_offset` picks $`b`$ maximising
   $`\min_{x\in N}\lvert P_{T_x\mathcal{M}}\,w\rvert`$ over offsets whose
   section is connected.
2. **Initialise.** $`X \leftarrow x_0 + \sigma\xi`$, with
   $`x_0\sim p_{\mathrm{data}}\vert_N`$ and $`\xi\sim\mathcal{N}(0,I)`$.
3. **Iterate** $`T`$ times, with $`\kappa(X)=\lvert P_{T_X\mathcal{M}}\,w\rvert`$:

```math
\hat{s} = \hat{s}_\theta(X,\sigma),\qquad
m = \langle w,\,X+\hat{s}\rangle - b,\qquad
\hat{g} = -\frac{\sigma^2}{v_c}\,\frac{m}{\kappa(X)^2}\,w,\qquad
v_c = \sigma^2
```

```math
X \leftarrow X + \eta\,\big(\hat{s} + \gamma\hat{g}\big)
+ \sqrt{2\eta\,\sigma^{2-\alpha}}\;\xi,
\qquad \xi\sim\mathcal{N}(0,I)
```

**Output** $`X \approx \mathrm{Unif}(N)`$.

Three notes.

$`m`$ is exact, not a classifier: Tweedie gives
$`\mathbb{E}[\langle w,x_0\rangle\mid X]=\langle w,X+\hat{s}\rangle`$.

The update is eq. (8) in hat space, $`dt=\eta\,\sigma^{2-\alpha}`$ and
$`\sigma^{\alpha-2}\hat{s}\,dt = \eta\hat{s}`$. **The drift is
$`\alpha`$-independent; $`\alpha`$ only sets the noise amplitude.** Tempering is a
temperature, not a different force. Drift steps are clipped at
`max_drift_step`.

$`\kappa^{-2}`$ equalises the restoring stiffness along $`N`$: without it the pull
toward $`N`$ scales as $`\lvert P_T w\rvert^2`$, constant on the sphere but varying
$`7\times`$ on the Klein bottle. Both $`v_c\approx\sigma^2`$ and the dropped
Jacobian in $`\nabla_x m = (I+\partial\hat{s}/\partial x)^\top w \approx w`$ are
switchable, so their cost is measured rather than assumed.

Unlike the CFG prescription in §6 of the paper, which tempers only the
unconditional component, **both terms are tempered here** — the constraint is
geometry, so it must scale with the geometry. Confirmed by
$`\lvert\langle w,x\rangle-b\rvert/\mathrm{dist}_{\mathcal{M}}`$ being constant
in $`\alpha`$ (Results).

### Constraint

Assumption 4.1(2) requires $`K`$ uniformly rectifiably path-connected. On the
Klein bottle, sections with $`b=0`$ are disconnected for most $`w`$; the
disconnected band is the interval between the two saddle values of
$`h(u,v)=\langle w,\Phi(u,v)\rangle`$. `connected_offset` picks $`b`$ giving a
connected, maximally transversal section.

### Manifolds

| | $`\mathcal{M}`$ | $`d`$ | $`n`$ | closed-form $`p_\sigma`$ |
| --- | --- | --- | --- | --- |
| sphere | $`S^3`$ | 4 | 3 | yes |
| klein | Klein bottle | 4 | 2 | no (quadrature) |

## Results — uniform sampling on $`N`$

Can tempering turn $`p_{\mathrm{data}}|_N`$ into $`\mathrm{Unif}(N)`$? Every run
starts from the restricted data distribution and reports departure from uniform
in units of the sampling floor, so an exactly-uniform draw is the benchmark, not
zero.

| manifold | score | $`\alpha`$ | start | result | benchmark |
| --- | --- | --- | --- | --- | --- |
| $`S^3\cap H`$ | **exact** | 0.5 | 14.97x | **1.61x** | 1.19x |
| $`S^3\cap H`$ | learned, 60k | 0.6 | 14.97x | 2.85x | 1.19x |
| Klein $`\cap\,H`$ | learned, 180k | 0.7 | — | **1.97x** (best of 5) | 1.00x |
| Klein $`\cap\,H`$ | learned, 180k | 0.7 | — | 2.41x (mean of 5) | 1.00x |

An untempered chain ($`\alpha=0`$) stays at 13.5–14.8x: **tempering, not
sampling, is what moves the distribution.**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/marginals-dark.png">
  <img alt="Marginals collapse from p_data onto uniform on both manifolds" src="docs/figures/marginals.png" width="780">
</picture>

The marginals are exact under the target — $`\langle e,x\rangle\sim\mathrm{Unif}[-1,1]`$
for a sphere section by Archimedes' hat-box theorem, arclength uniform for a
Klein section — so the flat line is truth, not a fit.

**With an exact score the method works.** On the sphere at $`\alpha=0.5`$ the
result sits at 1.61x against a 1.19x benchmark, having started at 14.97x.

**With a learned score it does not reach uniform.** All five Klein planes are
rejected by KS at $`N=20{,}000`$ (threshold $`D=0.0096`$; best plane 0.0144,
mean $`0.0331\pm0.0162`$):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/sweep-dark.png">
  <img alt="Every plane's measured D sits above the KS rejection threshold" src="docs/figures/sweep.png" width="620">
</picture>

The residual splits into the theory's own finite-$`\sigma`$ bias (mean 0.0116)
and score-model excess (mean 0.0292), and **these bind on different planes** —
$`b=-0.3417`$ is bias-limited with excess exactly 0, while $`b=+0.3983`$ has
bias 0.0057, already under threshold, and fails purely on model error. No single
knob closes the gap.

**More training does not help.** Same five planes, 60k vs 180k model: $`3\times`$
the steps, $`1.85\times`$ lower hat error, paired
$`\Delta = +0.0053\pm0.0469`$ ($`t=0.25`$). Per-plane excess correlates
$`-0.02`$ between the two checkpoints — the error is not a stable property of
the plane.

## Results — rate separation, and why there is no hierarchy

The hypothesis was a three-level hierarchy: the model learns the global manifold
first, then concentrates onto the conditional submanifold. **It does not exist,
and the reason is structural rather than empirical.**

### The guidance term is measured at the geometry rate

Fitting $`d\log\lVert\cdot\rVert/d\log\sigma`$ over a decade of noise, across
50 hyperplanes with exact scores throughout:

| term | exponent |
| --- | --- |
| geometry | $`-0.970`$ |
| **guidance** | $`-1.024`$ |
| density | $`-0.038`$ |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/rates-dark.png">
  <img alt="Guidance and geometry share an exponent and converge in magnitude" src="docs/figures/rates.png" width="620">
</picture>

Guidance and geometry do not merely share an exponent — **their magnitudes
converge**, agreeing to 0.06% by $`\sigma=0.003`$. There is no third level
between them.

### Why it cannot exist

For a transversal cut the squared distances add:

```math
d_N^2 = d_{\mathcal{M}}^2 + d_H^2 .
```

The $`\Theta(\sigma^{-2})`$ term of the expansion *is* that quadratic form, so a
hyperplane contributes one more orthogonal direction to the same sum. The score
has no way to tell "off $`\mathcal{M}`$" from "off $`H`$"; both are "off
$`N`$". **A measure-zero constraint is geometry**, raising codimension from
$`d-n`$ to $`d-n+1`$, and one tempering exponent governs both.

A direct consequence, and the sharpest confirmation: both residuals confine at
the same rate.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/confinement-dark.png">
  <img alt="Both residuals follow sigma^(1-alpha/2) with constant ratios" src="docs/figures/confinement.png" width="620">
</picture>

$`\mathrm{dist}_{\mathcal{M}}/\sigma^{1-\alpha/2}=1.39`$ constant across
$`\alpha\in[0.3,0.8]`$ (fitted slope 0.9955 against a theoretical 1), and the
off-hyperplane residual a fixed 0.603 of the off-manifold one. Had guidance sat
at a different rate, that second ratio would drift with $`\alpha`$.

### Nor is there a hierarchy in the *error*

If the signal has no third level, perhaps the model's error does — perhaps what
breaks uniformity is not what breaks confinement. Splitting
$`\hat{s}_\theta-\hat{s}^\ast`$ into components normal to $`\mathcal{M}`$,
along $`w`$, and tangent to $`N`$, on a shell at controlled distance:

| | tangent space | isotropy predicts | measured across six $`\sigma`$ |
| --- | --- | --- | --- |
| sphere | 3-D (1-D vs 2-D) | $`0.637`$ | 0.638, 0.642, 0.659, 0.641, 0.603, 0.639 |
| Klein | 2-D (1-D vs 1-D) | $`1.000`$ | 1.017, 1.033, 1.033, 1.014, 1.021, 0.988 |

The predicted ratio *changes* with the tangent dimension and the data follows it
on both manifolds. The error carries no structure relative to $`w`$ — which it
cannot, since $`w`$ is drawn after training. **No hierarchy in the signal, none
in the error.**

### What the split did find

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/anatomy-dark.png">
  <img alt="Score error has a minimum just above sigma_min and rises into it" src="docs/figures/anatomy.png" width="620">
</picture>

The error is **not a power law in $`\sigma`$**. It bottoms out just above
$`\sigma_{\min}=0.01`$ and rises into it — at $`\sigma=0.02`$ on the sphere
(2.7x worse at $`\sigma_{\min}`$) and $`\sigma=0.014`$ on Klein (2.26x). This
reproduces on a model trained $`3\times`$ longer, on a different manifold, with a
different codimension and a quadrature rather than closed-form reference, so it
is not undertraining.

Fitting $`\log\lVert e\rVert = c + P\log\sigma + Q\log\rho`$ and reading the
fixed-distance exponent $`P-Q`$ (Theorem 5.1 sups over a fixed region):

| block | Klein $`P-Q`$ | needs $`\alpha>`$ | sphere $`P-Q`$ | needs $`\alpha>`$ |
| --- | --- | --- | --- | --- |
| normal to $`\mathcal{M}`$ | $`-0.714`$ | 2.71 | $`-1.690`$ | 3.69 |
| along $`w`$ | $`-0.339`$ | 2.34 | $`-0.263`$ | 2.26 |
| tangent to $`N`$ | $`-0.346`$ | 2.35 | $`-0.239`$ | 2.24 |

All six give $`\beta<-2`$, outside Theorem 5.1's hypothesis, and demand an
$`\alpha`$ above its own upper bound of 2. **For these models the theorem does
not apply at any $`\alpha`$** — a violated premise, not a poorly chosen
exponent. Measured over $`\sigma\in[0.01,0.056]`$, less than a decade; the next
step is retraining with $`\sigma_{\min}=0.001`$ to widen the window.

> Figures regenerate with `uv run python tools/readme_figures.py`.
> **Caveat on figure verdicts:** `uniformity_summary` gates on
> max-deviation-per-bin, which has poor power against smooth error — on
> `klein-a07-long` plane 1 it prints UNIFORM while KS rejects at
> $`p=3\times10^{-7}`$. Trust the $`D`$ column, not a figure's verdict.

## Experiments

YAML in `configs/`, overridden with `--set key.path=value`. Output to
`runs/<name>/`. Each stage gates the next.

**1 — Geometry.** No model needed.

```bash
uv run python experiments/validate_geometry.py       # chart, metric, area
uv run python experiments/validate_reference.py      # analytic score vs MC
uv run python experiments/validate_intersection.py   # sections
```

**2 — Train.** Sphere ~25 min; the shipped Klein model ran 180k steps. Exits
non-zero on a failed gate.

```bash
uv run python experiments/train.py --config configs/manifold_sphere.yaml
uv run python experiments/train.py --config configs/manifold_klein.yaml
```

**3 — Unconditional sampling.** $`\alpha=0`$ is plain Langevin, targeting
$`p_{\mathrm{data}}`$.

```bash
uv run python experiments/sample_uniform.py \
    --set manifold=klein load_from=runs/m-klein-180k 'sweep.alphas=[0.5,1.0]'
```

**4 — Score rates.** Exact scores, gated against Monte Carlo.

```bash
uv run python experiments/measure_rates.py           # runs/rates/rates.png
```

**5 — One hyperplane, many $`\alpha`$.**

```bash
uv run python experiments/conditional_uniform.py \
    --config configs/manifold_klein.yaml --load-from runs/m-klein-180k \
    --offset auto --alphas 0.3 0.5 0.7 --n 20000 --sim-time 5 \
    --out runs/cond-alpha
```

`--offset auto` searches for a connected section (mandatory on klein).
`--normal PATH --offset file` pins the plane stored in a sweep's
`samples_plane<i>.pt`. `--use-reference` substitutes the exact score.

**6 — Many hyperplanes, one $`\alpha`$.** Produced the klein results above.

```bash
uv run python experiments/conditional_sweep.py \
    --config configs/manifold_klein.yaml --load-from runs/m-klein-180k \
    --n-planes 5 --n 20000 --alpha 0.7 --steps 0 --efolds 4 \
    --trace-every 200 --out runs/klein-a07
```

$`dt = \eta\,\sigma^{2-\alpha}`$ spans a decade across the $`\alpha`$ of interest, so
step counts come from the physics, never fixed: a 1-D section of length $`L`$
relaxes at $`(2\pi/L)^2`$, and `--steps 0 --efolds E` sizes each run to $`E`$
e-foldings ( `--sim-time` fixes simulated time instead). Measured: 4 e-folds
plateaus at $`\alpha=0.7`$, 2 does not.

Each plane has its own generator keyed on its index, so plane $`i`$ is fixed
across `--alpha`, `--efolds` and `--n`. With a shared generator, plane $`i`$'s
step count shifts every later plane.

## Kept runs

Superseded runs were removed; their summaries are in
`runs/ARCHIVE-superseded.md`.

| run | establishes |
| --- | --- |
| `m-sphere`, `m-klein-180k` | base score models |
| `m-klein` | 60k klein model; baseline for the training comparison |
| `rates` | guidance at the geometry rate, 50 planes |
| `cond-uniform-ref2` | sphere, exact score: uniform on $`S^3\cap H`$ |
| `cond-uniform-a06` | sphere, learned score, $`\alpha=0.6`$ |
| `klein-p2-alpha` | width $`\sim\sigma^{1-\alpha/2}`$ over $`\alpha\in[0.3,0.8]`$ |
| `klein-sweep`, `klein-sweep-180k` | 60k vs 180k on the same planes |
| `klein-a07-long` | the only converged klein sweep |

Checkpoints, samples and figures are regenerable and untracked. Metrics logs,
resolved configs and corrector traces are tracked.

## Inspecting

```bash
uv run python tools/report.py                            # all runs
uv run python tools/watch.py -f                          # live
uv run python tools/plot.py runs/klein-a07-long          # rebuild figures
uv run python tools/sections.py klein --n-planes 8       # draw sections
uv run python tools/visualize.py runs/m-klein-180k --config configs/manifold_klein.yaml
uv run python experiments/audit.py runs/m-klein-180k --config configs/manifold_klein.yaml
```

`visualize.py` writes `learned_<manifold>.png` (distance to $`\mathcal{M}`$ before
and after the deterministic flow; Jacobian spectrum, which reads $`n`$ off the
model) and `uniformity.png`. `--corrector-steps 0` and `--skip-gates` omit the
slow parts.

## Configuration

Configs inherit through `_base_`.

| key | meaning |
| --- | --- |
| `data.kappa_range` | concentration of $`p_{\mathrm{data}}`$ |
| `diffusion.sigma_min`, `sigma_max` | trained noise range; samplers may not exceed it |
| `model.score.width`, `depth` | network size |
| `train.score.steps`, `batch_size`, `lr` | optimisation |
| `train.score.sigma_bias` | $`>1`$ concentrates training near $`\sigma_{\min}`$ |
| `gates` | thresholds a run must pass |

```bash
uv run python experiments/train.py --config configs/manifold_klein.yaml \
    --set train.score.steps=120000 model.score.width=512 run.name=klein-long
```

## Layout

```
src/dgeom/
  geometry/     Manifold ABC, Sphere, KleinBottle, Hyperplane, sections,
                densities, loaders. A section IS a Manifold.
  models/       DiffusionModel ABC, trained score, closed-form and quadrature
                references, GuidedDiffusion.
  sampling/     Sampler ABC, Langevin, tempered, annealed.
  metrics/      manifold quality, chart and spherical uniformity.
  training/     Trainer, callbacks, metric tracking.
  viz/          palette, uniformity, rates, sections, manifold figures.
experiments/    one script per stage, plus audit.py
tools/          report, watch, plot, sections, visualize
report/         write-ups and figures
docs/           architecture notes
```

`src/dgeom/experiment.py` constructs manifolds, loaders, models and references.

Write-ups: [`report/sphere-experiment.tex`](report/sphere-experiment.tex) (+PDF),
[`report/conditional-submanifolds.md`](report/conditional-submanifolds.md),
[`docs/architecture.md`](docs/architecture.md).

## Development

```bash
uv run ruff check src experiments tools
uv run ruff format src experiments tools
```

# dgeom

Uniform sampling on manifolds and on **conditional submanifolds**. Extends
[*When Scores Learn Geometry*](https://arxiv.org/abs/2509.24912)
(Li, Shen, Hsieh & He).

## Overview

### Setting

$\mathcal{M}\subset\mathbb{R}^d$ is a compact boundaryless $C^4$ submanifold,
$\dim\mathcal{M}=n$, carrying data $\mu_{\mathrm{data}}$. The Gaussian-smoothed
measure is $p_\sigma = \mu_{\mathrm{data}} * \mathcal{N}(0,\sigma^2 I)$, and

$$d_{\mathcal{M}}(x) := \tfrac12\operatorname{dist}^2(x,\mathcal{M}),\qquad
\hat{s}(x,\sigma) := \sigma^2\nabla\log p_\sigma(x) = \mathbb{E}[x_0\mid x]-x .$$

### Rate separation (Thm 3.1)

$$\log p_\sigma(x) = -\frac{1}{\sigma^2}d_{\mathcal{M}}(x)
+ \log p_{\mathrm{data}}\big(\Phi^{-1}(P_{\mathcal{M}}(x))\big)
- \frac{d-n}{2}\log(2\pi\sigma^2) + H(x) + o(1).$$

Geometry enters at $\Theta(\sigma^{-2})$, density at $\Theta(1)$.

### Tempered Score Langevin (eq. 8)

$$dX_t = \sigma^\alpha s(X_t,\sigma)\,dt + \sqrt{2}\,dW_t .$$

**Thm 5.1.** If $\lVert s-s^\ast\rVert_{L^\infty(K)} = o(\sigma^\beta)$ for some
$\beta>-2$, then for any $\max\{-\beta,0\}<\alpha<2$ the stationary law
$\tilde\pi_\sigma$ converges to the uniform measure on $\mathcal{M}$:
$\tilde\pi(u)\propto\sqrt{\det g(u)}$.

Two consequences used throughout: the chain sits at transverse width
$\sigma^{1-\alpha/2}$, and at finite $\sigma$ with a gradient oracle its law is
$\tilde\pi_\sigma\propto\exp(-\sigma^\alpha f_\sigma)$.

### What this repo adds

Condition on a hyperplane $H=\{x:\langle w,x\rangle = b\}$ given **at inference
only**, giving the conditional submanifold

$$N = \mathcal{M}\cap H,\qquad \dim N = n-1 .$$

Guidance is exact and needs no classifier. By Tweedie,

$$m(x) = \mathbb{E}\big[\langle w,x_0\rangle \mid x_t\big]
= \langle w,\, x+\hat{s}(x,\sigma)\rangle,\qquad
\nabla\log p_\sigma(c\mid x) \approx -\frac{m(x)-b}{v_c}\,w,\quad v_c\approx\sigma^2 .$$

Transversality gives $d_N^2 = d_{\mathcal{M}}^2 + d_H^2$, so the constraint is
geometry, not density: one $\alpha$ governs both.

**Question.** Does tempering the guided score sample
$\mathrm{Unif}(N)$ — and at what score accuracy?

### Inference algorithm

Nothing is trained. $w$ and $b$ enter at sampling time; $\hat{s}_\theta$ is a
frozen unconditional model.

**Input** $\hat{s}_\theta$, normal $w$, noise level $\sigma$, exponent
$\alpha$, step scale $\eta$, steps $T$.

1. **Offset.** $b \leftarrow \texttt{connected\_offset}(\mathcal{M},w)$:
   maximise $\min_{x\in N}\lvert P_{T_x\mathcal{M}}\,w\rvert$ over offsets
   whose section is connected.
2. **Initialise.** $X \leftarrow x_0 + \sigma\xi$, with
   $x_0\sim p_{\mathrm{data}}\vert_N$ and $\xi\sim\mathcal{N}(0,I)$.
3. **Iterate** $T$ times, with $\kappa(X)=\lvert P_{T_X\mathcal{M}}\,w\rvert$:

$$\hat{s} = \hat{s}_\theta(X,\sigma),\qquad
m = \langle w,\,X+\hat{s}\rangle - b,\qquad
\hat{g} = -\frac{\sigma^2}{v_c}\,\frac{m}{\kappa(X)^2}\,w,\qquad
v_c = \sigma^2$$

$$X \leftarrow X + \eta\,\big(\hat{s} + \gamma\hat{g}\big)
+ \sqrt{2\eta\,\sigma^{2-\alpha}}\;\xi,
\qquad \xi\sim\mathcal{N}(0,I)$$

**Output** $X \approx \mathrm{Unif}(N)$.

Three notes.

$m$ is exact, not a classifier: Tweedie gives
$\mathbb{E}[\langle w,x_0\rangle\mid X]=\langle w,X+\hat{s}\rangle$.

The update is eq. (8) in hat space, $dt=\eta\,\sigma^{2-\alpha}$ and
$\sigma^{\alpha-2}\hat{s}\,dt = \eta\hat{s}$. **The drift is
$\alpha$-independent; $\alpha$ only sets the noise amplitude.** Tempering is a
temperature, not a different force. Drift steps are clipped at
`max_drift_step`.

$\kappa^{-2}$ equalises the restoring stiffness along $N$: without it the pull
toward $N$ scales as $\lvert P_T w\rvert^2$, constant on the sphere but varying
$7\times$ on the Klein bottle. Both $v_c\approx\sigma^2$ and the dropped
Jacobian in $\nabla_x m = (I+\partial\hat{s}/\partial x)^\top w \approx w$ are
switchable, so their cost is measured rather than assumed.

Unlike the CFG prescription in §6 of the paper, which tempers only the
unconditional component, **both terms are tempered here** — the constraint is
geometry, so it must scale with the geometry. Confirmed by
$\lvert\langle w,x\rangle-b\rvert/\mathrm{dist}_{\mathcal{M}}$ being constant
in $\alpha$ (Results).

### Constraint

Assumption 4.1(2) requires $K$ uniformly rectifiably path-connected. On the
Klein bottle, sections with $b=0$ are disconnected for most $w$; the
disconnected band is the interval between the two saddle values of
$h(u,v)=\langle w,\Phi(u,v)\rangle$. `connected_offset` picks $b$ giving a
connected, maximally transversal section.

### Manifolds

| | $\mathcal{M}$ | $d$ | $n$ | closed-form $p_\sigma$ |
| --- | --- | --- | --- | --- |
| sphere | $S^3$ | 4 | 3 | yes |
| klein | Klein bottle | 4 | 2 | no (quadrature) |

## Results

**Guidance sits at the geometry rate** (`rates`, 50 planes).
$d\log\lVert\cdot\rVert/d\log\sigma$:

| term | exponent |
| --- | --- |
| geometry | $-0.970$ |
| guidance | $-1.024$ |
| density | $-0.038$ |

Magnitudes converge: $\lVert g\rVert/\lVert\text{geom}\rVert = 0.9994$ at
$\sigma=0.003$. No intermediate rate exists.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/rates-dark.png">
  <img alt="Guidance and geometry share an exponent and converge in magnitude; density is flat" src="docs/figures/rates.png" width="620">
</picture>


**Both constraints confine alike** (`klein-p2-alpha`). Over
$\alpha\in[0.3,0.8]$: $\mathrm{dist}_{\mathcal{M}}/\sigma^{1-\alpha/2}=1.39$
constant (fitted slope $0.9955$ vs theory $1$), and
$\lvert\langle w,x\rangle-b\rvert/\mathrm{dist}_{\mathcal{M}} = 0.603$ constant.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/confinement-dark.png">
  <img alt="Both residuals follow sigma^(1-alpha/2) with constant ratios" src="docs/figures/confinement.png" width="620">
</picture>


**Exact score reaches uniform** (`cond-uniform-ref2`). Sphere, $\alpha=0.5$:
$1.61\times$ the sampling floor, against $1.19\times$ for exactly-uniform draws.

**Learned score does not** (`klein-a07-long`, the only converged sweep). 0/5
planes at $\alpha=0.7$; $D = 0.0331\pm0.0162$ against threshold $0.0096$.
Residual splits into bias $0.0116$ and excess $0.0292$, binding on *different*
planes.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/sweep-dark.png">
  <img alt="Every plane's measured D sits above the KS rejection threshold" src="docs/figures/sweep.png" width="620">
</picture>


**Training does not close it** (`klein-sweep` vs `klein-sweep-180k`, same
planes). $3\times$ steps, $1.85\times$ lower hat error, paired
$\Delta = +0.0053\pm0.0469$ ($t=0.25$). Per-plane excess correlates $-0.02$
between checkpoints.

**Open problem: the score error does not vanish as $\sigma\to\sigma_{\min}$.**
Splitting $\hat{s}_\theta-\hat{s}^\ast$ into its components normal to
$\mathcal{M}$, along $w$, and tangent to $N$ (`error-anatomy-sphere-v3`):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/anatomy-dark.png">
  <img alt="Normal error has a minimum at sigma=0.02 and rises toward sigma_min" src="docs/figures/anatomy.png" width="620">
</picture>

The tangential parts are clean power laws and agree with each other — no
hierarchy, confirmed four ways. The **normal** part is not a power law at all: it
bottoms out at $\sigma\approx0.02$ and is $2.7\times$ worse at
$\sigma_{\min}=0.01$, reproducibly across three runs.

Fitting $\log\lVert e\rVert = c + P\log\sigma + Q\log\rho$ at fixed distance
(the theorem sups over a fixed region, so its exponent is $P-Q$):

| block | $P$ | $Q$ | $P-Q$ | needs $\alpha>$ |
| --- | --- | --- | --- | --- |
| normal to $\mathcal{M}$ | $-0.098$ | $1.592$ | $-1.690$ | $3.69$ |
| along $w$ | $0.921$ | $1.184$ | $-0.263$ | $2.26$ |
| tangent to $N$ | $0.944$ | $1.182$ | $-0.239$ | $2.24$ |

All three give $\beta<-2$, outside Theorem 5.1's hypothesis, and demand an
$\alpha$ above its own upper bound of $2$. **For this model the theorem does not
apply at any $\alpha$** — not a bad choice of $\alpha$, a violated premise.
Measured over $\sigma\in[0.01,0.056]$, less than a decade; the next step is
retraining with $\sigma_{\min}=0.001$ to widen the window.

> Figures regenerate with `uv run python tools/readme_figures.py`.
> **Caveat on the figures' verdict text:** `uniformity_summary` gates on
> max-deviation-per-bin, which has poor power against smooth error — on
> `klein-a07-long` plane 1 it prints UNIFORM while KS rejects at
> $p=3\times10^{-7}$. Trust the $D$ column, not a figure's verdict.

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

**3 — Unconditional sampling.** $\alpha=0$ is plain Langevin, targeting
$p_{\mathrm{data}}$.

```bash
uv run python experiments/sample_uniform.py \
    --set manifold=klein load_from=runs/m-klein-180k 'sweep.alphas=[0.5,1.0]'
```

**4 — Score rates.** Exact scores, gated against Monte Carlo.

```bash
uv run python experiments/measure_rates.py           # runs/rates/rates.png
```

**5 — One hyperplane, many $\alpha$.**

```bash
uv run python experiments/conditional_uniform.py \
    --config configs/manifold_klein.yaml --load-from runs/m-klein-180k \
    --offset auto --alphas 0.3 0.5 0.7 --n 20000 --sim-time 5 \
    --out runs/cond-alpha
```

`--offset auto` searches for a connected section (mandatory on klein).
`--normal PATH --offset file` pins the plane stored in a sweep's
`samples_plane<i>.pt`. `--use-reference` substitutes the exact score.

**6 — Many hyperplanes, one $\alpha$.** Produced the klein results above.

```bash
uv run python experiments/conditional_sweep.py \
    --config configs/manifold_klein.yaml --load-from runs/m-klein-180k \
    --n-planes 5 --n 20000 --alpha 0.7 --steps 0 --efolds 4 \
    --trace-every 200 --out runs/klein-a07
```

$dt = \eta\,\sigma^{2-\alpha}$ spans a decade across the $\alpha$ of interest, so
step counts come from the physics, never fixed: a 1-D section of length $L$
relaxes at $(2\pi/L)^2$, and `--steps 0 --efolds E` sizes each run to $E$
e-foldings ( `--sim-time` fixes simulated time instead). Measured: 4 e-folds
plateaus at $\alpha=0.7$, 2 does not.

Each plane has its own generator keyed on its index, so plane $i$ is fixed
across `--alpha`, `--efolds` and `--n`. With a shared generator, plane $i$'s
step count shifts every later plane.

## Kept runs

Superseded runs were removed; their summaries are in
`runs/ARCHIVE-superseded.md`.

| run | establishes |
| --- | --- |
| `m-sphere`, `m-klein-180k` | base score models |
| `m-klein` | 60k klein model; baseline for the training comparison |
| `rates` | guidance at the geometry rate, 50 planes |
| `cond-uniform-ref2` | sphere, exact score: uniform on $S^3\cap H$ |
| `cond-uniform-a06` | sphere, learned score, $\alpha=0.6$ |
| `klein-p2-alpha` | width $\sim\sigma^{1-\alpha/2}$ over $\alpha\in[0.3,0.8]$ |
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

`visualize.py` writes `learned_<manifold>.png` (distance to $\mathcal{M}$ before
and after the deterministic flow; Jacobian spectrum, which reads $n$ off the
model) and `uniformity.png`. `--corrector-steps 0` and `--skip-gates` omit the
slow parts.

## Configuration

Configs inherit through `_base_`.

| key | meaning |
| --- | --- |
| `data.kappa_range` | concentration of $p_{\mathrm{data}}$ |
| `diffusion.sigma_min`, `sigma_max` | trained noise range; samplers may not exceed it |
| `model.score.width`, `depth` | network size |
| `train.score.steps`, `batch_size`, `lr` | optimisation |
| `train.score.sigma_bias` | $>1$ concentrates training near $\sigma_{\min}$ |
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

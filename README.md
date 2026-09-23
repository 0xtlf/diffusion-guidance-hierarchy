# dgeom

Sample uniformly from a curved surface, and from a slice through it.
Builds on [*When Scores Learn Geometry*](https://arxiv.org/abs/2509.24912)
(Li, Shen, Hsieh & He).

## The problem

A diffusion model learns a data distribution that lives on a curved surface
$`\mathcal{M}`$ inside $`\mathbb{R}^d`$. Sampling from it gives you the data
distribution: common things often, rare things rarely. Sometimes you want the
opposite — every part of the surface equally often. That is the **uniform
distribution** on $`\mathcal{M}`$, and the paper above shows how to get it.

We ask the same question about a **slice**. Take a flat cut
$`H=\{x:\langle w,x\rangle=b\}`$ and keep only the part of the surface that lies
on it:

```math
N = \mathcal{M}\cap H .
```

$`N`$ is one dimension smaller than $`\mathcal{M}`$. Can we sample uniformly
from $`N`$, using a model that was never trained on $`N`$ and never saw $`w`$?

## Words used here

| term | meaning |
| --- | --- |
| score | $`\nabla\log p_\sigma(x)`$, the gradient the model learns |
| hat score | $`\hat{s}=\sigma^2\nabla\log p_\sigma(x)`$, the score rescaled so it stays order 1 |
| $`\sigma`$ | how much Gaussian noise was added to the data |
| $`\alpha`$ | the tempering exponent, defined below |
| codimension | how many directions point off the surface: $`d-n`$ |
| guidance | the extra term that pulls samples onto the cut $`H`$ |

## Background

Add Gaussian noise of size $`\sigma`$ to the data. Write $`p_\sigma`$ for the
result. Let $`d_{\mathcal{M}}(x)=\tfrac12\,\mathrm{dist}^2(x,\mathcal{M})`$.

**Theorem 3.1 of the paper.** For small $`\sigma`$,

```math
\log p_\sigma(x) = -\frac{1}{\sigma^2}d_{\mathcal{M}}(x)
+ \log p_{\mathrm{data}}\big(\Phi^{-1}(P_{\mathcal{M}}(x))\big)
- \frac{d-n}{2}\log(2\pi\sigma^2) + H(x) + o(1).
```

Read the two leading terms. The shape of the surface enters at size
$`\sigma^{-2}`$. The data distribution on it enters at size $`1`$. The shape is
therefore much larger. This is the **rate separation**: as $`\sigma`$ shrinks,
shape information grows and density information does not.

**Tempering.** Multiply the score by $`\sigma^\alpha`$ and run Langevin
dynamics:

```math
dX_t = \sigma^\alpha s(X_t,\sigma)\,dt + \sqrt{2}\,dW_t .
```

This shrinks both terms, but it shrinks the small one below the noise. The
density washes out and only the shape survives.

**Theorem 5.1.** Suppose the score error obeys
$`\lVert s-s^\ast\rVert_{L^\infty(K)} = o(\sigma^\beta)`$ for some
$`\beta>-2`$. Then for any $`\alpha`$ with
$`\max\{-\beta,0\}<\alpha<2`$, the sampler settles at the uniform distribution
on $`\mathcal{M}`$.

Two facts we use later. The sampler sits a distance $`\sigma^{1-\alpha/2}`$ off
the surface. At finite $`\sigma`$ it settles not at uniform but at
$`p_\sigma^{\,\sigma^\alpha}`$, which is close to uniform but not equal to it.

## What we add

The cut $`H`$ is given at sampling time only. The model is frozen.

No classifier is needed, because the constraint is linear. Tweedie's formula
gives the expected value of $`\langle w,x_0\rangle`$ exactly:

```math
m(x) = \mathbb{E}\big[\langle w,x_0\rangle \mid x_t\big]
= \langle w,\, x+\hat{s}(x,\sigma)\rangle,\qquad
\nabla\log p_\sigma(c\mid x) \approx -\frac{m(x)-b}{v_c}\,w,\quad v_c\approx\sigma^2 .
```

### The algorithm

**Input.** A frozen model $`\hat{s}_\theta`$, a direction $`w`$, a noise level
$`\sigma`$, an exponent $`\alpha`$, a step size $`\eta`$, a step count $`T`$.

1. **Pick the offset $`b`$.** `connected_offset` chooses it so the slice is one
   connected piece. See *Why the offset matters* below.
2. **Start.** Draw $`x_0`$ from the data distribution restricted to $`N`$, then
   add noise: $`X = x_0 + \sigma\xi`$.
3. **Repeat $`T`$ times.** Write $`\kappa(X)`$ for the length of $`w`$ after
   projecting it onto the surface:

```math
\hat{s} = \hat{s}_\theta(X,\sigma),\qquad
m = \langle w,\,X+\hat{s}\rangle - b,\qquad
\hat{g} = -\frac{m}{\kappa(X)^2}\,w
```

```math
X \leftarrow X + \eta\,\big(\hat{s} + \gamma\hat{g}\big)
+ \sqrt{2\eta\,\sigma^{2-\alpha}}\;\xi,
\qquad \xi\sim\mathcal{N}(0,I)
```

**Output.** $`X`$, close to uniform on $`N`$.

Three things to note.

**$`\alpha`$ changes only the noise.** Look at the update: $`\alpha`$ appears in
the noise term and nowhere else. The pull toward the surface is the same at
every $`\alpha`$. Turning $`\alpha`$ up shakes the sampler harder, so it settles
further out, at distance $`\sigma^{1-\alpha/2}`$.

**The $`\kappa^{-2}`$ factor evens out the pull.** Without it the pull toward
$`N`$ is proportional to $`\kappa^2`$. That is constant on the sphere but varies
by a factor of 7 on the Klein bottle.

**We temper both terms.** Section 6 of the paper tempers only the model score
and leaves guidance alone. We temper both, because guidance here is part of the
shape, not part of the density. The results below confirm this.

### Why the offset matters

Assumption 4.1(2) of the paper requires the slice to be one connected piece. On
the Klein bottle a slice through the origin is usually in two pieces. A sampler
cannot move points between two pieces, so no choice of $`\alpha`$ can fix it.
`connected_offset` searches for an offset that gives one piece.

### The two surfaces

| | $`\mathcal{M}`$ | $`d`$ | $`n`$ | exact $`p_\sigma`$ |
| --- | --- | --- | --- | --- |
| sphere | $`S^3`$ | 4 | 3 | yes |
| klein | Klein bottle | 4 | 2 | no, computed numerically |

## Results: sampling uniformly from the slice

Each run starts from the data distribution on $`N`$ and tries to reach uniform.
We measure how far from uniform the result is, in units of the test's own noise.
A perfect sample does not score 0, it scores about 1. That is the number to beat.

| surface | score | $`\alpha`$ | start | result | perfect sample |
| --- | --- | --- | --- | --- | --- |
| $`S^3\cap H`$ | **exact** | 0.5 | 14.97 | **1.61** | 1.19 |
| $`S^3\cap H`$ | learned, 60k steps | 0.6 | 14.97 | 2.85 | 1.19 |
| Klein $`\cap\,H`$ | learned, 180k steps | 0.7 | — | **1.43** (best of 5) | 1.00 |
| Klein $`\cap\,H`$ | learned, 180k steps | 0.7 | — | 2.41 (mean of 5) | 1.00 |

Without tempering the sampler stays at 13.5–14.8. So tempering is what moves the
distribution, not the sampling.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/marginals-dark.png">
  <img alt="Distributions move from the data distribution onto uniform on both surfaces" src="docs/figures/marginals.png" width="780">
</picture>

The dashed line is the exact answer, not a fit. For a sphere slice
$`\langle e,x\rangle`$ is uniform on $`[-1,1]`$. For a Klein slice the distance
along the curve is uniform.

**With an exact score, it works.** On the sphere the result reaches 1.61 against
a perfect score of 1.19, from a start of 14.97.

**With a learned score, it does not reach uniform.** All five Klein slices fail
a KS test at 20,000 samples. The test rejects above $`D=0.0096`$; the best slice
scores 0.0144 and the mean is $`0.0331\pm0.0162`$.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/sweep-dark.png">
  <img alt="Every slice sits above the KS rejection threshold" src="docs/figures/sweep.png" width="620">
</picture>

The leftover error has two parts. One is built into the method: at finite
$`\sigma`$ the target is $`p_\sigma^{\,\sigma^\alpha}`$, not uniform. Call it the
**method error** (mean 0.0116). The rest comes from the model. Call it the
**model error** (mean 0.0292).

They do not fail on the same slices. On $`b=-0.3417`$ the model error is exactly
0, so all that remains is method error. On $`b=+0.3983`$ the method error is
0.0057, already below the test threshold, and the slice fails on model error
alone. Fixing one does not fix the other.

**More training does not help.** Same five slices, two models. The 180k model
ran 3 times longer and has 1.85 times lower score error. The paired difference
was $`+0.0053\pm0.0469`$ ($`t=0.25`$). Per-slice model error correlates
$`-0.02`$ between the two models, so it is not a property of the slice.

## Results: there is no hierarchy

We expected two levels. First the model learns the whole surface. Then, on top
of that, it narrows down onto the slice. **That second level does not exist.**

### Guidance is measured at the same rate as shape

We fit how each term grows as $`\sigma`$ shrinks, over a factor of 100 in
$`\sigma`$, on 50 different cuts, using exact scores:

| term | exponent |
| --- | --- |
| shape | $`-0.970`$ |
| **guidance** | $`-1.024`$ |
| density | $`-0.038`$ |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/rates-dark.png">
  <img alt="Guidance and shape share an exponent and converge in size" src="docs/figures/rates.png" width="620">
</picture>

Guidance and shape share an exponent. They also converge in size: they agree to
0.06% by $`\sigma=0.003`$. There is nothing between them.

### Why there cannot be a second level

If the cut is not tangent to the surface, the squared distances add:

```math
d_N^2 = d_{\mathcal{M}}^2 + d_H^2 .
```

The $`\sigma^{-2}`$ term in the expansion is exactly this sum. Adding a cut adds
one more direction to it. The model cannot tell "off the surface" from "off the
cut" — both are just "off the slice". So a flat cut is shape, not density. It
raises codimension by one, and a single $`\alpha`$ handles both.

This predicts something we can check: the distance off the surface and the
distance off the cut should shrink at the same rate.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/confinement-dark.png">
  <img alt="Both distances follow sigma^(1-alpha/2) with constant ratios" src="docs/figures/confinement.png" width="620">
</picture>

They do. Across $`\alpha\in[0.3,0.8]`$ the distance off the surface divided by
$`\sigma^{1-\alpha/2}`$ stays at 1.39 (fitted slope 0.9955, theory says 1). The
distance off the cut stays at 0.603 times the distance off the surface. If
guidance sat at a different rate, that second number would drift with $`\alpha`$.

### The model's error has no hierarchy either

Maybe the signal has no second level but the model's mistakes do. Maybe the
error that breaks uniformity is different from the error that breaks
confinement. We split the error into three parts: off the surface, along $`w`$,
and along the slice.

If the error points in no particular direction, the ratio between the last two
parts is fixed by their dimensions alone:

| | dimensions | if error has no preferred direction | measured, six values of $`\sigma`$ |
| --- | --- | --- | --- |
| sphere | 1 vs 2 | $`0.637`$ | 0.638, 0.642, 0.659, 0.641, 0.603, 0.639 |
| Klein | 1 vs 1 | $`1.000`$ | 1.017, 1.033, 1.033, 1.014, 1.021, 0.988 |

The predicted number changes between the two surfaces, and the measurement
follows it both times. The error points nowhere in particular. It cannot, since
$`w`$ is chosen after training.

### What the split did find

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/anatomy-dark.png">
  <img alt="Score error has a minimum just above the smallest trained noise level" src="docs/figures/anatomy.png" width="620">
</picture>

The score error does not keep shrinking as $`\sigma`$ shrinks. It reaches a
minimum and then grows again. On the sphere the minimum is at $`\sigma=0.02`$,
and the error is 2.7 times worse at $`\sigma_{\min}=0.01`$. On the Klein bottle
the minimum is at $`\sigma=0.014`$, 2.26 times worse at $`\sigma_{\min}`$.

The Klein model trained 3 times longer than the sphere model, on a different
surface, with a different codimension. The same shape appears. So this is not
undertraining.

Theorem 5.1 measures the error over a fixed region. We fit
$`\log\lVert e\rVert = c + P\log\sigma + Q\log\rho`$, where $`\rho`$ is distance
from the surface in units of $`\sigma`$, and read off $`P-Q`$:

| part of the error | Klein $`P-Q`$ | needs $`\alpha>`$ | sphere $`P-Q`$ | needs $`\alpha>`$ |
| --- | --- | --- | --- | --- |
| off the surface | $`-0.714`$ | 2.71 | $`-1.690`$ | 3.69 |
| along $`w`$ | $`-0.339`$ | 2.34 | $`-0.263`$ | 2.26 |
| along the slice | $`-0.346`$ | 2.35 | $`-0.239`$ | 2.24 |

All six give $`\beta<-2`$. Theorem 5.1 requires $`\beta>-2`$, and every
$`\alpha`$ here is above the theorem's own limit of 2. **So the theorem does not
apply to these models at any $`\alpha`$.** The assumption fails; the choice of
$`\alpha`$ is not the problem.

This is measured over $`\sigma\in[0.01,0.056]`$, a factor of 5. The next step is
to retrain with $`\sigma_{\min}=0.001`$ and measure over a wider range.

> Figures rebuild with `uv run python tools/readme_figures.py`.
>
> **Do not trust the verdict printed on a figure.** `uniformity_summary` checks
> the largest gap in any single histogram bin. That test misses errors spread
> smoothly over many bins. On `klein-a07-long` slice 1 it prints UNIFORM while a
> KS test rejects at $`p=3\times10^{-7}`$. Use the $`D`$ column in the run
> summary instead.

## Running things

Settings live in `configs/`. Override any of them with `--set key.path=value`.
Output goes to `runs/<name>/`. Run the stages in order; each one checks the last.

**1 — Check the geometry.** No model needed.

```bash
uv run python experiments/validate_geometry.py       # chart, metric, area
uv run python experiments/validate_reference.py      # exact score vs Monte Carlo
uv run python experiments/validate_intersection.py   # slices
```

**2 — Train.** The sphere takes about 25 minutes. The Klein model here ran 180k
steps. Both exit non-zero if a check fails.

```bash
uv run python experiments/train.py --config configs/manifold_sphere.yaml
uv run python experiments/train.py --config configs/manifold_klein.yaml
```

**3 — Sample the whole surface.** $`\alpha=0`$ gives plain Langevin, which
targets the data distribution.

```bash
uv run python experiments/sample_uniform.py \
    --set manifold=klein load_from=runs/m-klein-180k 'sweep.alphas=[0.5,1.0]'
```

**4 — Measure the rates.** Exact scores, checked against Monte Carlo first.

```bash
uv run python experiments/measure_rates.py           # runs/rates/rates.png
```

**5 — One cut, several $`\alpha`$.**

```bash
uv run python experiments/conditional_uniform.py \
    --config configs/manifold_klein.yaml --load-from runs/m-klein-180k \
    --offset auto --alphas 0.3 0.5 0.7 --n 20000 --sim-time 5 \
    --out runs/cond-alpha
```

`--offset auto` finds an offset whose slice is one piece. This is required on the
Klein bottle. `--normal PATH --offset file` reuses the exact cut saved in a
sweep's `samples_plane<i>.pt`. `--use-reference` swaps in the exact score.

**6 — Several cuts, one $`\alpha`$.** This produced the Klein results above.

```bash
uv run python experiments/conditional_sweep.py \
    --config configs/manifold_klein.yaml --load-from runs/m-klein-180k \
    --n-planes 5 --n 20000 --alpha 0.7 --steps 0 --efolds 4 \
    --trace-every 200 --out runs/klein-a07
```

Do not fix the step count. The step size is $`\eta\,\sigma^{2-\alpha}`$, which
changes by a factor of 10 across the $`\alpha`$ we use, so equal step counts mean
unequal simulated time. A slice of length $`L`$ needs time $`(L/2\pi)^2`$ to mix.
`--steps 0 --efolds E` sets the count so every run gets $`E`$ of those.
We measured that 4 is enough at $`\alpha=0.7`$ and 2 is not.

Each cut gets its own random seed, fixed by its index. So cut $`i`$ is the same
cut whatever you pass for `--alpha`, `--efolds` or `--n`. With one shared seed,
changing the step count of cut $`i`$ changes every later cut.

## The runs kept here

Older runs were deleted. Their summaries are in `runs/ARCHIVE-superseded.md`.

| run | what it shows |
| --- | --- |
| `m-sphere`, `m-klein-180k` | the two trained models |
| `m-klein` | the 60k Klein model, kept for the training comparison |
| `rates` | guidance sits at the shape rate, 50 cuts |
| `cond-uniform-ref2` | sphere, exact score, reaches uniform |
| `cond-uniform-a06` | sphere, learned score, $`\alpha=0.6`$ |
| `klein-p2-alpha` | distances follow $`\sigma^{1-\alpha/2}`$ |
| `klein-sweep`, `klein-sweep-180k` | 60k vs 180k on the same cuts |
| `klein-a07-long` | the only Klein sweep that ran long enough |

Checkpoints, samples and run figures are not in git; they can be rebuilt. Metrics
logs, resolved configs and traces are in git, because they record what happened.

## Looking at results

```bash
uv run python tools/report.py                            # all runs
uv run python tools/watch.py -f                          # live progress
uv run python tools/plot.py runs/klein-a07-long          # rebuild figures
uv run python tools/sections.py klein --n-planes 8       # draw slices
uv run python tools/visualize.py runs/m-klein-180k --config configs/manifold_klein.yaml
uv run python experiments/audit.py runs/m-klein-180k --config configs/manifold_klein.yaml
```

`visualize.py` writes two figures. `learned_<manifold>.png` shows the distance to
the surface before and after the sampler runs, and a plot that tells you the
dimension the model learned. `uniformity.png` shows how uniform the
samples are. `--corrector-steps 0` and `--skip-gates` skip the slow parts.

## Settings

Config files inherit through `_base_`.

| key | meaning |
| --- | --- |
| `data.kappa_range` | how peaked the data distribution is |
| `diffusion.sigma_min`, `sigma_max` | noise range the model is trained on; samplers must stay inside it |
| `model.score.width`, `depth` | network size |
| `train.score.steps`, `batch_size`, `lr` | training |
| `train.score.sigma_bias` | above 1, trains more often near $`\sigma_{\min}`$ |
| `gates` | checks a run must pass |

```bash
uv run python experiments/train.py --config configs/manifold_klein.yaml \
    --set train.score.steps=120000 model.score.width=512 run.name=klein-long
```

## Code layout

```
src/dgeom/
  geometry/     surfaces, cuts, slices, densities, data loaders.
                A slice is itself a surface, so everything else works on it.
  models/       the trained score, exact references, guided score.
  sampling/     Langevin, tempered, annealed.
  metrics/      surface quality, uniformity tests.
  training/     trainer, callbacks, metric tracking.
  viz/          colours and figures.
experiments/    one script per stage, plus audit.py
tools/          report, watch, plot, sections, visualize
report/         write-ups and their figures
docs/           architecture notes, README figures
```

`src/dgeom/experiment.py` builds surfaces, loaders, models and references.

Longer write-ups: [`report/sphere-experiment.tex`](report/sphere-experiment.tex)
(and its PDF), [`report/conditional-submanifolds.md`](report/conditional-submanifolds.md),
[`docs/architecture.md`](docs/architecture.md).

## Development

```bash
uv run ruff check src experiments tools
uv run ruff format src experiments tools
```

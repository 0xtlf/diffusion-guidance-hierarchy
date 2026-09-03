# dgeom

Score-based manifold geometry, and uniform sampling on manifolds and on
**conditional submanifolds**. Builds on
[*When Scores Learn Geometry*](https://arxiv.org/abs/2509.24912)
(Li, Shen, Hsieh & He, ICLR 2026).

Two manifolds are used throughout: `S³` in `R⁴`, where every quantity is
available in closed form, and a Klein bottle in `R⁴`, which is non-orientable,
of codimension 2, and has no closed-form smoothed score. A random hyperplane
through the origin, drawn at inference, cuts either one down to a conditional
submanifold `N = M ∩ H`.

Results and their derivation:
[`report/conditional-submanifolds.md`](report/conditional-submanifolds.md).
Library design: [`docs/architecture.md`](docs/architecture.md).

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.11–3.13

## Installation

```bash
git clone <repo> && cd diffusion-exploration
uv sync --group dev
uv run python experiments/validate_geometry.py     # 15/15 passed -> GATE PASSED
```

## Experiments

Scripts live in `experiments/`, configured by YAML in `configs/`; any value may
be overridden with `--set key.path=value`. Results go to `runs/<name>/`.

Run in order — each stage gates the next.

**1 — Geometry and reference scores.** No trained model needed.

```bash
uv run python experiments/validate_geometry.py       # Klein chart, metric, area
uv run python experiments/validate_reference.py      # analytic score vs Monte Carlo
uv run python experiments/validate_intersection.py   # sections of both manifolds
```

**2 — Train.** One model per manifold, ~25 min each. Exits non-zero on a failed gate.

```bash
uv run python experiments/train.py --config configs/manifold_sphere.yaml
uv run python experiments/train.py --config configs/manifold_klein.yaml
```

**3 — Unconditional sampling.** The tempered corrector targets the uniform
measure on `M`; `alpha=0` is plain Langevin and targets the data distribution.

```bash
uv run python experiments/sample_uniform.py \
    --set manifold=klein load_from=runs/m-klein 'sweep.alphas=[0.5,1.0]'
```

**4 — Score rates.** Measures where the guidance term of a conditional
submanifold sits relative to the geometry and density terms of
`grad log p_sigma`. Exact scores throughout, gated against Monte Carlo first.

```bash
uv run python experiments/measure_rates.py           # writes runs/rates/rates.png
```

**5 — Conditional sampling.** Starting from a von Mises–Fisher mixture, can
tempering recover the uniform measure on `N`? Reports departure from uniform,
KS against both the uniform target and the restricted data distribution, and
the residuals off `M` and off `H` separately.

```bash
uv run python experiments/conditional_uniform.py \
    --sim-time 5 --n 20000 --alphas 0.6 --out runs/cond-uniform
```

Add `--use-reference` to substitute the exact score for the trained network,
which separates a failure of the method from a failure of the model.

## Reading a run

`--sim-time T` sets the step count per alpha from the physics rather than fixing
it: `dt = step_scale * sigma^(2 - alpha)` spans orders of magnitude across the
alphas of interest, so equal step counts mean very unequal simulated time.
Mixing on the section takes about 2.5 time units, so `T = 5` is roughly twelve
e-foldings of the slowest mode.

`--plot-every N` publishes results every `N` steps instead of only at the end:

| artifact | behaviour |
| --- | --- |
| `figures/uniformity_step<step>.png` | accumulates; never overwritten |
| `figures/convergence.png` | overwritten; shows everything measured so far |
| `samples/latest_a<alpha>.pt` | overwritten; the state, not a picture of it |
| `corrector_trace.jsonl` | appended at every probe |

Nothing is held until completion, so a job killed part way keeps its results.

Uniformity is judged two ways, and they answer different questions. The
one-dimensional marginals (`viz/uniformity.py`) are exact — `<e, x>` is uniform
on `[-1, 1]` for a sphere section by Archimedes' theorem, and arclength is
uniform for a Klein section. The two-dimensional tests
(`metrics/spherical.py`) are the arbiter: a real-spherical-harmonic omnibus
statistic, whose per-degree spectrum localises the defect, plus an equal-area
chi-square. A sample can pass every marginal and still fail the joint test.

Report the sampling floor alongside any deviation. The analytic
`2 sqrt(nbins / N)` understates the scatter of a maximum over many bins by
roughly 1.5x, so an empirical null from exactly-uniform draws is the honest
comparison.

Training and sampling both show a progress bar and log a line at intervals. The
bar is drawn only when stderr is a terminal, so captured output keeps the log
without the redraws; `DGEOM_NO_PROGRESS=1` suppresses it.

## Inspecting results

```bash
uv run python tools/report.py            # summary of all runs
uv run python tools/watch.py -f          # live progress
uv run python tools/visualize.py runs/m-klein \
    --config configs/manifold_klein.yaml
```

`visualize.py` loads a checkpoint and writes `learned_<manifold>.png`
(distance to the manifold before and after the deterministic flow, and the
Jacobian spectrum, which reads the intrinsic dimension off the model) and
`uniformity.png`. It never trains; `--corrector-steps 0` and `--skip-gates`
omit the slow parts.

## Configuration

Configs inherit through `_base_`. The values most often changed, from
`configs/manifold_base.yaml`:

| key | meaning |
| --- | --- |
| `data.kappa_range` | concentration of the data distribution |
| `diffusion.sigma_min`, `sigma_max` | trained noise range; samplers may not exceed it |
| `model.score.width`, `depth` | network size |
| `train.score.steps`, `batch_size`, `lr` | optimisation |
| `train.score.sigma_bias` | `>1` concentrates training near `sigma_min` |
| `gates` | thresholds a run must pass |

```bash
uv run python experiments/train.py --config configs/manifold_klein.yaml \
    --set train.score.steps=120000 model.score.width=512 run.name=klein-long
```

## Layout

```
src/dgeom/
  geometry/     Manifold ABC, Sphere, KleinBottle, Hyperplane and sections,
                densities and loaders. A section IS a Manifold, so every
                metric, plot and sampler works on it unchanged.
  models/       DiffusionModel ABC (only shat is abstract), the trained score,
                closed-form and quadrature references, and GuidedDiffusion.
  sampling/     Sampler ABC, Langevin, tempered and annealed variants.
  metrics/      manifold quality, chart and spherical uniformity tests.
  training/     Trainer, callbacks, metric tracking.
  viz/          validated palette, uniformity, rates, manifold figures.
experiments/    one script per stage, each gating the next
tools/          report, watch, visualize
report/         the write-up and its figures
```

Manifolds, loaders, trained models and reference scores are constructed by
`src/dgeom/experiment.py`.

## Development

```bash
uv run ruff check src experiments tools
uv run ruff format src experiments tools
```

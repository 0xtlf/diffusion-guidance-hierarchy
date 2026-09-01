# dgeom

Score-based manifold learning and uniform sampling on `S³` and a Klein bottle.
Implements [*When Scores Learn Geometry*](https://arxiv.org/abs/2509.24912)
(Li, Shen, Hsieh & He, ICLR 2026).

Library design is documented in [`docs/architecture.md`](docs/architecture.md).

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.11–3.13

## Installation

```bash
git clone <repo> && cd diffusion-exploration
uv sync --group dev
```

Verify:

```bash
uv run python experiments/validate_geometry.py
# 15/15 passed -> GATE PASSED
```

## Experiments

Each experiment is a script in `experiments/` configured by a YAML file in
`configs/`. Any config value may be overridden with `--set key.path=value`.
Results are written to `runs/<name>/` as `config.resolved.yaml`,
`metrics.jsonl`, `ckpt/` and `figures/`.

Run in order; each stage gates the next.

**1 — Validate the geometry and the reference scores.** No trained model required.

```bash
uv run python experiments/validate_geometry.py
uv run python experiments/validate_reference.py
```

**2 — Train a diffusion model.** One per manifold, approximately 25 minutes each.
Exits non-zero if any quality gate fails.

```bash
uv run python experiments/train.py --config configs/manifold_sphere.yaml
uv run python experiments/train.py --config configs/manifold_klein.yaml
```

**3 — Audit a checkpoint** against the current metric definitions.

```bash
uv run python experiments/audit.py runs/m-klein \
    --config configs/manifold_klein.yaml
```

**4 — Sample.** The tempered corrector targets the uniform measure on the
manifold; `alpha=0` recovers plain Langevin, which targets the data distribution.
Add `--set use_reference=true` to substitute the exact score for the trained one.

```bash
uv run python experiments/sample_uniform.py \
    --set manifold=klein load_from=runs/m-klein 'sweep.alphas=[0.5,1.0]'
```

**5 — Inspect results.** Developer tools live in `tools/`.

```bash
uv run python tools/report.py            # summary of all runs
uv run python tools/watch.py -f          # live progress
uv run python tools/visualize.py runs/m-klein \
    --config configs/manifold_klein.yaml
```

`visualize.py` loads a checkpoint and writes two figures to
`runs/<name>/figures/`, each captioned with the run configuration and its
pass/fail verdict.

- `learned_<manifold>.png` — distance to the manifold before and after the
  deterministic flow, and the Jacobian spectrum, which reads the intrinsic
  dimension off the model.
- `uniformity.png` — the exact one-dimensional marginals, comparing four
  curves in the order the experiment produces them:

  | curve | reading |
  | --- | --- |
  | 1. before training: `p_data` | the non-uniform distribution the model was trained on |
  | 2. after training | the learned distribution; agreement with 1 confirms training |
  | 3. after correction | tempered corrector at `alpha > 0` |
  | 4. target: uniform | the analytic density, drawn as a dashed reference |

  Each curve is also reported numerically as its largest departure from
  uniform alongside the sampling noise floor `2 sqrt(nbins / N)`, so a
  deviation is immediately separable from finite-sample scatter.

The corrector curves are sampling, not training; `--corrector-steps 0` omits
them, and `--skip-gates` omits the evaluation pass. Sampled points are written
to `runs/<name>/samples/uniformity.pt`; `--reuse-samples` replots from that file
so a figure can be relabelled or restyled without rerunning the corrector.

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

Example:

```bash
uv run python experiments/train.py --config configs/manifold_klein.yaml \
    --set train.score.steps=120000 model.score.width=512 run.name=klein-long
```

## Adding an experiment

1. Add a config to `configs/` inheriting from a base via `_base_`.
2. Add a script to `experiments/` beginning with
   `cfg, run, device = setup(args.config, args.overrides)`.
3. Record results with `run.log(stage=..., **metrics)`; `report.py` and `plot.py`
   discover them automatically.

Loaders for manifolds, dataloaders, trained models and reference scores are in
`src/dgeom/experiment.py`.

## Development

```bash
uv run ruff check src experiments tools
uv run ruff format src experiments tools
```

# Classifier guidance onto a conditional submanifold — findings

**The code for this branch has been removed** (see git history if versioned).
These results are kept because they were expensive to obtain and they bound
what a future attempt should expect.

# Rate separation for classifier guidance on conditional submanifolds

Does the geometry/density rate separation of [*When Scores Learn Geometry*
(arXiv 2509.24912)](https://arxiv.org/abs/2509.24912) extend to **classifier
guidance**, and does tempering carry from the data manifold `M` to a
**conditional submanifold** `N ⊂ M` specified at inference time?

Goal: train an unconditional score on a distribution over `S³` plus a noisy
classifier, then at inference supply an arbitrary hyperplane and sample
**uniformly on `N = S³ ∩ H`** — whatever `p_data` was, with that hyperplane
unseen during training.

## The theory under test

Subtracting the paper's Theorem B.2 expansion for `p_σ(x|y)` from the one for
`p_σ(x)`:

```
log p_σ(y|x) = −(1/σ²)[d_N(x) − d_M(x)]   ← Θ(σ⁻²)   which submanifold
             + ((n_y−n)/2) log(2πσ²)       ← Θ(log σ) codimension killed
             + log(p_{data|y}/p_data)      ← Θ(1)     density reweighting
             + (H_N − H_M)(x) + o(1)       ← Θ(1)     curvature

∇log p_σ(y|x) = (1/σ²)(P_N(x) − P_M(x)) + Θ(1)
```

A noisy classifier's gradient is, at leading order, a **pure geometric steering
field** from `M` to `N`. (Verified numerically: `models/tweedie_guidance.py`
matches exact Monte Carlo near `N`, and `ĝ → P_N(x) − P_M(x)` with cosine 0.9999.)

## What to temper

With independent exponents, `drift = σ^{α₁}·s_θ + γ·σ^{α₂}·∇log p_φ`, the
surviving `Θ(1)` factor on `N` is:

| `(α₁,α₂)` | limit on `N` |
|---|---|
| `(0,0)` | `p_data\|_N` |
| `(0,α)` | `∝ p_data·e^{H_M}` |
| `(α,0)` | `∝ \|∇_M c\|^{−γ}·e^{γ(H_N−H_M)}` — purely geometric, no `p_data` |
| `(α,α)` | uniform `vol_N` |

**Answer: temper both.** Each network carries a `Θ(1)` layer inseparable from its
`Θ(σ⁻²)` layer, and `σ^α` kills the `Θ(1)` layer of whatever it multiplies.

**Caveat this repo is built around.** On `S³` cut by a *hyperplane* the co-area
factor `|∇_M c| = √(1−b²)` and both curvature terms are constant along `N`, so
the table collapses to two outcomes — `p_data|N` when `α₁=0`, uniform when
`α₁=α` — and the guidance exponent is invisible. Separating `(α,0)` from `(α,α)`
requires the **quadric** constraint, where `|∇_M c|` varies along `N`. That is
`constraint.name: quadric`.

Not tempered: the **predictor** (would break the reverse-SDE marginals) and the
**final projection** (untempered conditional score, no noise).

## Layout

```
src/dgeom/
  geometry/     S³, vMF mixture (Wood's algorithm), constraints, MC references
  nn/           architectures only: ScoreMLP, ClassifierMLP, embeddings, backbone
  models/       score/guidance models behind one protocol: analytic | tweedie | learned
  sampling/     interchangeable schemes: tempered_langevin | projection | predictor_corrector
  metrics/      harmonic uniformity (χ²), dist-to-N, pooled S³ test, limit discriminator
  training/     trainer with metric tracking, on-the-fly data
configs/        every experiment is one YAML; `_base_` inherits, `--set` overrides
experiments/    entry points
runs/           per-run: config.resolved.yaml, metrics.jsonl, ckpt/
```

Components are selected by name through `registry.py`, so a sampling scheme or a
guidance model is swapped in the YAML, not in code.

## Running

```bash
uv venv --python 3.12 && uv pip install --python .venv/bin/python -e .

# gate: closed-form score vs Monte Carlo. Nothing is trustworthy until this passes.
.venv/bin/python experiments/e0_validate_analytic.py

# Stage 0: the tempering grid, no training at all
.venv/bin/python experiments/e1_temper_grid.py

# is the o(σ⁻²) tolerance real?  inject a Θ(1) field into either network
.venv/bin/python experiments/e2_tolerance.py --set tolerance.target=guidance

# train both networks (classifier guidance = two separate models)
.venv/bin/python experiments/train_models.py --set run.name=main run.timestamp=false

# the goal: fresh hyperplane at inference, uniform on N
.venv/bin/python experiments/e5_inference.py
```

Any leaf overrides from the CLI: `--set sampling.alpha=1.5 sampling.n_chains=8192`.
Each run writes `config.resolved.yaml` (plus git rev) so a result traces back to
exactly the inputs that produced it.

## Two facts that shape every experiment

**Everything is in hat space.** Both model families return `σ²·∇log p`, which by
Tweedie is the displacement `E[x₀|x] − x`. It is `O(1)` where the raw scores blow
up like `σ⁻²`, which keeps float32 sampling safe at `σ_min`. It also makes the
tolerance directly readable: the theory permits `o(σ⁻²)` raw error, so the
tolerated regime is exactly **hat error `= o(1)`**, which is what
`eval_score`/`eval_classifier` report.

**Mixing time is set by cloud thickness, not by α or σ separately.** The tempered
equilibrium is `ε = σ^{1−α/2}` thick and tangential mixing takes `∝ 1/ε²` steps:

| σ | α | ε | steps to cover N |
|---|---|---|---|
| 0.01 | 1.0 | 0.100 | ~300 |
| 0.01 | 0.5 | 0.032 | ~3,000 |
| 0.01 | 0.0 | 0.010 | ~30,000 |
| 1e-4 | 1.0 | 0.010 | ~30,000 |

So untempered arms need ~100× the steps of `(1,1)`. E1 therefore uses a **paired
stationarity test** — each arm is launched from *both* candidate limits (both
sampled exactly) and we watch which one it stays at — rather than waiting for
global mixing from an arbitrary start.

## Gotchas found the hard way

- `torch.special.log_ndtr` saturates to 0 in the **upper** tail. When the bin sits
  below the mean (`c < 0`) both bounds are large and positive, their log-CDF
  difference is pure noise, and the guidance force silently becomes **exactly
  zero** on those chains. `_log_prob_between` reflects the interval to fix this.
  Symptom was ~31% of chains frozen at `|c| ≈ 0.2`.
- Song's SNR step-size rule must **not** be used with a tempered score: it sets
  `dt ∝ 1/‖s‖²`, so tempering inflates `dt` by `σ^{−2α}` and breaks the integrator.
- MPS has no float64. Analytic paths run CPU/float64; learned paths float32.
- A fixed *relative* tolerance is meaningless as a score gate: `ŝ` is `O(σ)` near
  the manifold, a difference of two nearly-equal `O(1)` vectors, so relative error
  is dominated by whichever probe has the smallest `‖ŝ‖`. `e0` uses a z-test
  against the Monte-Carlo standard error instead.

## Auditing a trained model

```bash
python experiments/audit.py runs/main
python experiments/audit.py runs/main --sigmas 0.01 0.02 0.05 --n 8192
```

Loads a checkpoint and measures both models against the **analytic** reference,
on demand and independent of the run that produced them.  (The reference is the
closed-form score, never the learned one — a learned-score Tweedie would fold the
score's own error into the classifier's verdict.)

Score model: `hat_err` against the closed form, `cos_dM`, and `flow_radius` —
where a short deterministic flow on the learned score settles.  `flow_radius` is
the most trustworthy single number: 0.9996 +- 3.6e-4 at sigma=0.01 means the model
has genuinely learned `S^3`.

Classifier: `ghat_ratio` (want 1), `cos_steer`, `sys_frac` (coherent-bias
fraction of the residual), and `v_ratio` — the predicted `v_c` over
`sigma^2 |P_T a|^2`.  That last one is the sharpest quality signal available: the
theory says `v_c/sigma^2` tends to the co-area factor, so `v_ratio -> 1` is direct
evidence the classifier learned the **tangent space of the manifold** from bin
labels alone, which is the actual content of "noisy classifier".

**Accuracy is not quality**, and the tool prints so on every run.  The softmax
head scored 0.81 bin accuracy at sigma=0.01 with `ghat_ratio = 0.0096` — a
gradient 100x too weak, while classifying well. Any audit that stops at accuracy
would have passed it.

## Monitoring a run

```bash
python experiments/watch.py           # snapshot of every run
python experiments/watch.py -f -n 3   # refresh every 3s until Ctrl-C
```

Reads `metrics.jsonl` rather than stdout, so progress is visible even though the
experiments buffer their output, and marks a run `LIVE` when a matching process
is actually running.

## Figures

```bash
python experiments/plot.py --all              # all runs -> <run>/figures/
python experiments/compare.py runs/<run>      # "did tempering work?" + verdict
```

`compare.py` answers the operative question directly — is the result the uniform
volume measure, or `p_data|N` (what you get without tempering)? It reports both
distances plus a plain verdict, and distinguishes *significance* from *effect
size*: with enough samples `chi^2` will resolve the finite-sigma bias (the
tempered cloud is `sigma^{1-alpha/2}` thick, and uniformity is a `sigma -> 0`
statement), so the verdict compares TV against the noise floor for that sample
size rather than reading a p-value alone.

The 1-D panel projects onto the axis where `p_data|N` varies most. Uniform on a
2-sphere has an *exactly* flat marginal there (Archimedes), so "worked" means
landing on a known straight line, not "looks smooth".

## Two more gotchas found the hard way

- **A plain softmax classifier head cannot do guidance at small sigma.** The true
  `log p(y|x)` reaches ~-1e4 away from the accepted bin; a logit vector sits at
  order +-10. Cross-entropy is satisfied once the argmax is right, so the network
  learns the ranking and never the log-probability *scale* — which is the only
  thing guidance consumes. Measured: 81% bin accuracy at sigma=0.01 with a
  gradient **100x too weak** (`ghat_ratio` 0.0096), which left the constraint
  effectively unenforced (`c_resid` 0.92 — samples uniform on S^3, not on N).
  `head: structured` predicts `(m_c, v_c)` and derives bin probabilities, so the
  scale is right by construction; same CE loss, same bin labels.
- **MPS's `erfc` is unusable in the tail** — it underflows to exactly 0.0 by
  argument 4.24 (CPU gives 2e-9) and is visibly wrong before that. Combined with
  the fact that `torch.where` evaluates *both* branches and propagates NaN
  gradients from the unselected one, this produced a finite forward pass with
  silently NaN gradients. `torch_special.log_ndtr` uses a Mills-ratio continued
  fraction below `x = -2`, needs only arithmetic, and is backend-identical.

## Results

### Stage 0 (analytic components) — theory confirmed

All 8 (arm, start) runs reproduce the predicted table. `(0,0)` and `(0,1)` are
stationary at `p_data|N` and flee uniform; `(1,0)` and `(1,1)` are stationary at
uniform and transport away from `p_data|N`. So the rate separation **does**
extend to classifier guidance, and tempering **does** carry from the data
manifold `M` to a conditional submanifold `N` given at inference.

As predicted, `(0,0)`≡`(0,1)` and `(1,0)`≡`(1,1)` on this geometry: on `S³` cut
by a hyperplane the co-area factor and both curvature terms are constant along
`N`, so the only `Theta(1)` content anywhere is `p_data`, which lives in the
score. Separating the guidance exponent needs `constraint.name: quadric`.

### Learned end-to-end — concentrates on N, not yet uniform on it

With both networks learned: `dist_N` 0.014–0.042 (vs 0.36–0.95 with the broken
softmax head, a 20–60x improvement), but `chi^2` vs uniform stays in the
thousands.

Cause is characterised, not mysterious. The learned classifier carries a
**systematic** error on `N` — `|E[err]| = 0.0088` against `mean |err| = 0.0094`,
i.e. 94% coherent bias, correlating with position on `N` at 0.635 on the l=1
harmonic. A dipole.

Why that breaks uniformity but not concentration: a hat-space error `eps` enters
the tempered drift as `sigma^{alpha-2} eps`, which at `alpha=1, sigma=0.01` is
`100 x 0.0094 ~ 0.94` — an O(1) tangential drift on `N`, competing directly with
the O(1) diffusion that produces uniformity. The transverse walls are
`Theta(sigma^{alpha-2})` and far stronger, so the constraint still holds.

In the paper's terms a *constant* hat-space error is raw score error
`Theta(sigma^-2)`, i.e. `beta = -2` — and Theorem 5.1 requires `beta > -2`
**strictly**, with `alpha > -beta`. A fixed-capacity network sits exactly on the
excluded boundary. **The practical tolerance is not "hat error bounded" but "hat
error vanishing with sigma."**

`experiments/e6_alpha_sweep.py` tests this: raising alpha from 1.0 to 1.9 cuts
the amplification `sigma^{alpha-2}` from 100x to 1.6x, so uniformity should
improve while `dist_N` degrades as `sigma^{1-alpha/2}`. Both moving together is
the signature of this mechanism; only one moving means the diagnosis is wrong.

### Use both metrics, always

`dist_N` and the uniformity `chi^2` catch different failures and each hid the
other's. The broken softmax run put samples uniform on **`S^3`**, whose
`q/|q|` projection is close to uniform on `S^2` — so its `chi^2` looked
unremarkable (318–1265) while the run was completely broken, and only `dist_N`
exposed it. With the fix the roles reverse: `dist_N` is healthy and `chi^2`
catches the density bias.


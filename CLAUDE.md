# CLAUDE.md — ranking_gam

## What is this repo?

A PyTorch library for interpretable learning-to-rank using Generalized Additive Models (GAMs). Based on Zhuang et al. "Interpretable Ranking with Generalized Additive Models" (WSDM 2021), extended with pairwise interactions (GA2M), context-weighted GAM, submodular diversity-aware reranking, and multi-objective support.

## Package structure

```
ranking_gam/
├── __init__.py              # Public API — all key symbols re-exported here
├── interactions.py           # Feature pair selection by correlation
├── models/
│   ├── towers.py            # PaperTower (MLP), ConcavePWL, MonotonePWL, LearnableMonotoneTransform
│   ├── gam.py               # GAM_Paper, GA2M_Paper (context-absent)
│   ├── context.py           # ContextWeightNetwork, ContextPresentGA2M, ContextGAM
│   ├── groupwise.py         # GroupwiseFeatureComputer (set-dependent features)
│   ├── submodular.py        # SubmodularRankingGAM (greedy diversity reranking)
│   └── multi_objective.py   # MultiObjectiveRankingGAM (Pareto-controllable)
├── losses/
│   ├── approx_ndcg.py       # ApproxNDCGLoss
│   ├── pairwise.py          # PairwiseLoss (RankNet)
│   ├── listwise.py          # ListMLELoss, ListNetLoss
│   ├── lambda_loss.py       # LambdaLoss (lambdaRank, ndcgLoss1/2/2++)
│   └── diffsort.py          # DiffSortNDCGLoss, soft_rank, PAVA isotonic
├── data/
│   └── mslr.py              # load_mslr() — MSLR-WEB10K download + parse
├── metrics/
│   └── ranking.py           # compute_ndcg, evaluate_ranking_diversity
├── training/
│   ├── trainer.py           # train_model, train_diversity_towers, train_multi_objective
│   └── boosting.py          # train_gbdt_baseline, compute_gbdt_residual_feature
├── distill/
│   └── pwl.py               # greedy_knot_selection (Algorithm 1), distill_to_pwl, pwl_predict
└── viz/
    └── plots.py             # plot_response_curves, plot_diversity_curves, plot_spider
examples/
    └── demo.py              # Full 4-demo script (GAM, Submodular, Multi-Objective, GBDT)
pyproject.toml               # Package config, deps, ruff/pytest settings
```

## Architecture hierarchy

1. **GAM_Paper** — one MLP tower per feature, no interactions. Simplest model.
2. **GA2M_Paper** — GAM + pairwise interaction towers (input dim 2).
3. **ContextGAM** — GAM with shared context network producing per-feature importance weights. `score = Σ w_j(x) * f_j(x_j)`. Tower shapes are still single-variable (distillable to PWL), context weights are query-dependent but inspectable.
4. **ContextPresentGA2M** — GA2M with separate context features (e.g., query text/category) producing weights (Eq 8-9).
5. **SubmodularRankingGAM** — base GAM + concave PWL diversity towers + greedy reranking.
6. **MultiObjectiveRankingGAM** — multiple objectives (pointwise + groupwise) with runtime weight control.

Tower types:
- `PaperTower`: unconstrained MLP `Dense -> Act -> ... -> Dense(1)`. Supports relu/silu/gelu activations.
- `ConcavePWL`: monotone non-decreasing + concave (slopes >= 0, non-increasing). Guarantees submodularity.
- `MonotonePWL`: monotone non-decreasing only (slopes >= 0, independent).
- `LearnableMonotoneTransform`: monotone piecewise-linear input normalization initialized from data percentiles.

## Key conventions

- **Padding**: relevance label `-1` marks padded/invalid positions. All losses and metrics respect this.
- **Input shape**: `[batch, list_size, num_features]` for features, `[batch, list_size]` for labels/scores.
- **log1p transform**: applied during data loading (`np.sign(x) * np.log1p(np.abs(x))`).
- **Default list size**: 40 documents per query (top-40 by label, zero-padded if fewer).
- **Default hidden dims**: `[16, 8]` for WEB30K/YAHOO, `[64, 32]` for CWS item towers and ContextGAM context net, `[128, 64]` for ContextPresentGA2M context towers.

## Training patterns

**Standard GAM/GA2M:**
```python
model = GAM_Paper(num_features=136, activation="silu", feature_transforms=True, residual=True)
model.init_transforms_from_data(train_X)
train_model(model, train_loader, val_loader, LambdaLoss.ndcg2pp(), epochs=30,
            lr_schedule="plateau", weight_decay=0.01)
```

**ContextGAM (best NDCG, still interpretable):**
```python
model = ContextGAM(num_features=136, context_hidden=[64, 32],
                   activation="silu", feature_transforms=True, residual=True)
model.init_transforms_from_data(train_X)
train_model(model, train_loader, val_loader, LambdaLoss.ndcg2pp(), epochs=50,
            lr_schedule="plateau", patience=15)
# Explain a single document:
attribution = model.explain(x_single)  # tower_outputs, context_weights, contributions
```

**SubmodularRankingGAM (2 phases):**
```python
# Phase 1: train base towers with ranking loss
train_model(submod, train_loader, val_loader, LambdaLoss.ndcg2pp(), epochs=30)
# Phase 2: freeze base, train diversity towers with step-wise ListNet
train_diversity_towers(submod, X_aug, y, epochs=10)
```

**MultiObjectiveRankingGAM:**
```python
train_multi_objective(mo_model, X_aug, y, epochs=10)
# At serving time, just change weights — no retraining:
mo_model.greedy_rerank(x, weights={"relevance": 0.5, "diversity": 0.3, ...})
```

**GBDT Residual Boosting (3 stages):**
```python
# Stage 1: Train GAM on raw features
# Stage 2: Train GBDT regressor on GAM residuals
# Stage 3: Add magic curve tower (freeze D towers, train tower D+1)
# See demo.py --demo gbdt for full pipeline
```

## Loss functions

All losses accept `(y_pred, y_true)` both `[batch, list_size]`. Padding label `-1` handled internally.

| Loss | Best for | Key param |
|------|----------|-----------|
| `LambdaLoss.ndcg2pp()` | Best NDCG (recommended) | — |
| `ApproxNDCGLoss` | Direct NDCG optimization | `alpha=10` for GAMs |
| `PairwiseLoss` | Stable pairwise training | `sigma=1.0` |
| `ListNetLoss` | Fast, stable training | `label_smoothing=0.1` |
| `ListMLELoss` | Permutation probability | — |
| `LambdaLoss` | LambdaRank variants | `weighing_scheme=` |
| `DiffSortNDCGLoss` | Differentiable soft ranks | `regularization_strength=` |

## Training options

| Option | Default | Description |
|--------|---------|-------------|
| `lr_schedule` | `"cosine"` | `"constant"`, `"cosine"`, or `"plateau"` (ReduceLROnPlateau) |
| `weight_decay` | `0.01` | AdamW weight decay (skip/bias params excluded automatically) |
| `l1_output_reg` | `0.0001` | L1 regularization on tower outputs |
| `patience` | `10` | Early stopping patience (epochs without improvement) |
| `warmup_epochs` | `3` | Linear LR warmup epochs |
| `grad_clip` | `1.0` | Gradient clipping max norm |

## Distillation

Neural towers distilled to piecewise-linear via Algorithm 1 (greedy knot selection + least squares). Paper says K=3-5 knots is usually sufficient.

```python
pwl = distill_to_pwl(model, num_knots=5, train_X=train_X)
scores = pwl_predict(pwl, X_eval)  # numpy-only inference, no torch needed
```

For ContextGAM: towers distill to PWL, context network stays as small neural net (~14K params).

## Demo usage

```bash
# Best config for pure GAM
python examples/demo.py --demo gam --loss ndcg2pp --activation silu --lr-schedule plateau --epochs 50

# ContextGAM (highest NDCG, still interpretable)
python examples/demo.py --demo gam --context --loss ndcg2pp --activation silu --lr-schedule plateau --epochs 50 --patience 15

# Bare GAM (no enhancements, for comparison)
python examples/demo.py --demo gam --no-transforms --no-residual --no-cosine --l1-reg 0

# All demos
python examples/demo.py --epochs 30

# Key flags: --context, --ga2m, --loss, --activation, --lr-schedule, --no-transforms, --no-residual
```

## Dependencies

- **Required**: `torch>=2.0`, `numpy>=1.24`, `scipy>=1.10`
- **Visualization**: `matplotlib>=3.7`
- **Data/baselines**: `lightgbm>=4.0`

## When modifying this codebase

- Every model must implement `forward(x) -> [B, L]` scores.
- Models with `get_main_effect(feature_idx, x_values)` support distillation and response curve visualization.
- Models with `greedy_rerank(x, k=)` support diversity reranking.
- Models with `explain(x_single)` support per-document attribution.
- All new losses should handle `-1` padding in `y_true`.
- Concavity/monotonicity constraints are enforced via parameterization (softplus), not projection — never break this.
- Skip/bias params are excluded from weight decay automatically in `train_model`.

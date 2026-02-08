# CLAUDE.md — ranking_gam

## What is this repo?

A PyTorch library for interpretable learning-to-rank using Generalized Additive Models (GAMs). Based on Zhuang et al. "Interpretable Ranking with Generalized Additive Models" (WSDM 2021), extended with pairwise interactions (GA2M), submodular diversity-aware reranking, and multi-objective support.

## Package structure

```
ranking_gam/
├── __init__.py              # Public API — all key symbols re-exported here
├── interactions.py           # Feature pair selection by correlation
├── models/
│   ├── towers.py            # PaperTower (MLP), ConcavePWL, MonotonePWL
│   ├── gam.py               # GAM_Paper, GA2M_Paper (context-absent)
│   ├── context.py           # ContextWeightNetwork, ContextPresentGA2M
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
│   └── trainer.py           # train_model, train_diversity_towers, train_multi_objective
├── distill/
│   └── pwl.py               # greedy_knot_selection (Algorithm 1), distill_to_pwl, pwl_predict
└── viz/
    └── plots.py             # plot_response_curves, plot_diversity_curves, plot_spider
examples/
    └── demo.py              # Full 3-demo script (GAM, Submodular, Multi-Objective)
pyproject.toml               # Package config, deps, ruff/pytest settings
```

## Architecture hierarchy

1. **GAM_Paper** — one MLP tower per feature, no interactions. Simplest model.
2. **GA2M_Paper** — GAM + pairwise interaction towers (input dim 2).
3. **ContextPresentGA2M** — GA2M with context-dependent feature weights (Eq 8-9).
4. **SubmodularRankingGAM** — base GAM + concave PWL diversity towers + greedy reranking.
5. **MultiObjectiveRankingGAM** — multiple objectives (pointwise + groupwise) with runtime weight control.

Tower types:
- `PaperTower`: unconstrained MLP `Dense -> ReLU -> ... -> Dense(1)`
- `ConcavePWL`: monotone non-decreasing + concave (slopes >= 0, non-increasing). Guarantees submodularity.
- `MonotonePWL`: monotone non-decreasing only (slopes >= 0, independent).

## Key conventions

- **Padding**: relevance label `-1` marks padded/invalid positions. All losses and metrics respect this.
- **Input shape**: `[batch, list_size, num_features]` for features, `[batch, list_size]` for labels/scores.
- **log1p transform**: applied during data loading (`np.sign(x) * np.log1p(np.abs(x))`).
- **Default list size**: 40 documents per query (top-40 by label, zero-padded if fewer).
- **Default hidden dims**: `[16, 8]` for WEB30K/YAHOO, `[64, 32]` for CWS item towers, `[128, 64]` for context towers.

## Training patterns

**Standard GAM/GA2M:**
```python
model = GAM_Paper(num_features=136)
train_model(model, train_loader, val_loader, ListNetLoss(), epochs=30)
```

**SubmodularRankingGAM (2 phases):**
```python
# Phase 1: train base towers with ranking loss
train_model(submod, train_loader, val_loader, ListNetLoss(), epochs=30)
# Phase 2: freeze base, train diversity towers with step-wise ListNet
train_diversity_towers(submod, X_aug, y, epochs=10)
```

**MultiObjectiveRankingGAM:**
```python
train_multi_objective(mo_model, X_aug, y, epochs=10)
# At serving time, just change weights — no retraining:
mo_model.greedy_rerank(x, weights={"relevance": 0.5, "diversity": 0.3, ...})
```

## Loss functions

All losses accept `(y_pred, y_true)` both `[batch, list_size]`. Padding label `-1` handled internally.

| Loss | Best for | Key param |
|------|----------|-----------|
| `ApproxNDCGLoss` | Direct NDCG optimization | `alpha=10` for GAMs |
| `ListNetLoss` | Fast, stable training | — |
| `ListMLELoss` | Permutation probability | — |
| `LambdaLoss` | LambdaRank / ndcgLoss2++ | `weighing_scheme=` |
| `PairwiseLoss` | RankNet baseline | `sigma=1.0` |
| `DiffSortNDCGLoss` | Differentiable soft ranks | `regularization_strength=` |

## Distillation

Neural towers distilled to piecewise-linear via Algorithm 1 (greedy knot selection + least squares). Paper says K=3-5 knots is usually sufficient.

```python
pwl = distill_to_pwl(model, num_knots=5, train_X=train_X)
scores = pwl_predict(pwl, X_eval)  # numpy-only inference, no torch needed
```

## Dependencies

- **Required**: `torch>=2.0`, `numpy>=1.24`, `scipy>=1.10`
- **Visualization**: `matplotlib>=3.7`
- **Data/baselines**: `lightgbm>=4.0`

## When modifying this codebase

- Every model must implement `forward(x) -> [B, L]` scores.
- Models with `get_main_effect(feature_idx, x_values)` support distillation.
- Models with `greedy_rerank(x, k=)` support diversity reranking.
- Models with `explain(x_single, order)` support per-position attribution.
- All new losses should handle `-1` padding in `y_true`.
- Concavity/monotonicity constraints are enforced via parameterization (softplus), not projection — never break this.

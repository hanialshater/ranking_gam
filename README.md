# ranking_gam

Interpretable learning-to-rank with Generalized Additive Models (GAMs) in PyTorch.

Based on Zhuang et al. ["Interpretable Ranking with Generalized Additive Models"](https://dl.acm.org/doi/10.1145/3437963.3441805) (WSDM 2021), extended with context-weighted GAM, pairwise interactions (GA2M), submodular diversity-aware reranking, multi-objective support, and GBDT residual boosting.

## Installation

```bash
pip install torch numpy scipy matplotlib
pip install -e .
```

## Quick Start

```python
import ranking_gam as rg
from torch.utils.data import DataLoader, TensorDataset
import torch

# Load data (auto-downloads MSLR-WEB10K)
train_X, train_y, eval_X, eval_y = rg.load_mslr()

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y)),
    batch_size=32, shuffle=True,
)
eval_loader = DataLoader(
    TensorDataset(torch.from_numpy(eval_X), torch.from_numpy(eval_y)),
    batch_size=32,
)

# Train a GAM with best config
model = rg.GAM_Paper(
    num_features=136, activation="silu",
    feature_transforms=True, residual=True,
)
model.init_transforms_from_data(train_X)
rg.train_model(
    model, train_loader, eval_loader, rg.LambdaLoss.ndcg2pp(),
    epochs=30, lr_schedule="plateau", weight_decay=0.01,
)

# Distill to piecewise-linear for fast serving
pwl = rg.distill_to_pwl(model, num_knots=5, train_X=train_X)
scores = rg.pwl_predict(pwl, eval_X)  # numpy-only, no torch needed
```

## Models

| Model | Description | Best NDCG |
|-------|-------------|-----------|
| `GAM_Paper` | One MLP tower per feature, no interactions | Baseline |
| `GA2M_Paper` | GAM + pairwise interaction towers (input dim 2) | +1-2 pts |
| `ContextGAM` | GAM with learned per-feature importance weights | +2-3 pts |
| `ContextPresentGA2M` | GA2M with separate context features producing weights | +2-3 pts |
| `SubmodularRankingGAM` | Base GAM + concave PWL diversity towers + greedy reranking | -- |
| `MultiObjectiveRankingGAM` | Multiple objectives with runtime weight control | -- |

### ContextGAM (recommended for best NDCG)

Context-weighted GAM: `score = sum(w_j(x) * f_j(x_j))` where weights come from a shared context network. Each tower still takes a single feature, so response curves remain interpretable. The context network learns query-dependent per-feature importance.

```python
model = rg.ContextGAM(
    num_features=136, context_hidden=[64, 32],
    activation="silu", feature_transforms=True, residual=True,
)
model.init_transforms_from_data(train_X)
rg.train_model(
    model, train_loader, eval_loader, rg.LambdaLoss.ndcg2pp(),
    epochs=50, lr_schedule="plateau", patience=15,
)

# Per-document explanation
attribution = model.explain(x_single)  # tower_outputs, context_weights, contributions
```

For distillation: towers distill to PWL, context network stays as a small neural net (~14K params).

### SubmodularRankingGAM (diversity-aware)

Two-phase training: base towers with ranking loss, then diversity towers with step-wise ListNet.

```python
submod = rg.SubmodularRankingGAM(
    num_item_features=136,
    groupwise_specs=[
        {"name": "category_novelty", "type": "category_novelty", "column": 136},
        {"name": "brand_novelty", "type": "brand_novelty", "column": 137},
    ],
    activation="silu",
)

# Phase 1: base towers
rg.train_model(submod, train_loader, eval_loader, rg.LambdaLoss.ndcg2pp(), epochs=30)

# Phase 2: diversity towers (base frozen)
rg.train_diversity_towers(submod, X_aug, y, epochs=10)

# Greedy reranking with (1-1/e) optimality guarantee
order = submod.greedy_rerank(x, k=10)
```

### MultiObjectiveRankingGAM (Pareto-controllable)

Train once, then change weights at serving time -- no retraining needed.

```python
mo = rg.MultiObjectiveRankingGAM(objectives=[
    {"name": "relevance", "type": "pointwise", "features": list(range(136)),
     "tower": "mlp", "weight": 0.5},
    {"name": "diversity", "type": "groupwise", "weight": 0.3,
     "groupwise_specs": [{"name": "cat_nov", "type": "category_novelty", "column": 136}]},
    {"name": "freshness", "type": "pointwise", "features": [20],
     "tower": "monotone", "weight": 0.2},
])

rg.train_multi_objective(mo, X_aug, y, epochs=10)

# Slide weights at serving time
mo.greedy_rerank(x, weights={"relevance": 0.3, "diversity": 0.5, "freshness": 0.2})
```

## Loss Functions

All losses accept `(y_pred, y_true)` of shape `[batch, list_size]`. Padding label `-1` is handled internally.

| Loss | Best for | Key param |
|------|----------|-----------|
| `LambdaLoss.ndcg2pp()` | Best NDCG (recommended) | -- |
| `ApproxNDCGLoss` | Direct NDCG optimization | `alpha=10` |
| `PairwiseLoss` | Stable pairwise training | `sigma=1.0` |
| `ListNetLoss` | Fast, stable training | `label_smoothing=0.1` |
| `ListMLELoss` | Permutation probability | -- |
| `LambdaLoss` | LambdaRank variants | `weighing_scheme=` |
| `DiffSortNDCGLoss` | Differentiable soft ranks | `regularization_strength=` |

## Tower Types

- **`PaperTower`**: Unconstrained MLP. Supports `residual=True` (skip connection), `input_norm=True` (BatchNorm), and activation choices (`relu`, `silu`, `gelu`).
- **`ConcavePWL`**: Monotone non-decreasing + concave. Guarantees submodularity for diversity features.
- **`MonotonePWL`**: Monotone non-decreasing only. For revenue, freshness, etc.
- **`LearnableMonotoneTransform`**: Monotone PWL mapping raw features to [0, 1], initialized from data percentiles (empirical CDF). Learned end-to-end while preserving interpretability.

Monotonicity and concavity are enforced via softplus parameterization, not projection.

## Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `lr_schedule` | `"cosine"` | `"constant"`, `"cosine"`, or `"plateau"` (ReduceLROnPlateau) |
| `weight_decay` | `0.01` | AdamW weight decay (skip/bias params excluded automatically) |
| `l1_output_reg` | `0.0001` | L1 regularization on tower outputs |
| `patience` | `10` | Early stopping patience (epochs without improvement) |
| `warmup_epochs` | `3` | Linear LR warmup epochs |
| `grad_clip` | `1.0` | Gradient clipping max norm |

## Running the Demo

```bash
# Best config for pure GAM
python examples/demo.py --demo gam --loss ndcg2pp --activation silu --lr-schedule plateau --epochs 50

# ContextGAM (highest NDCG, still interpretable)
python examples/demo.py --demo gam --context --loss ndcg2pp --activation silu --lr-schedule plateau --epochs 50 --patience 15

# GA2M with pairwise interactions
python examples/demo.py --demo gam --ga2m --ga2m-pairs 20

# Bare GAM (no enhancements, for comparison)
python examples/demo.py --demo gam --no-transforms --no-residual --no-cosine --l1-reg 0

# GBDT residual boosting (GAM -> GBDT -> magic curve)
python examples/demo.py --demo gbdt --loss ndcg2pp --activation silu

# All demos
python examples/demo.py --epochs 30

# Key flags: --context, --ga2m, --loss, --activation, --lr-schedule, --no-transforms, --no-residual
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```

All tests use synthetic data -- no downloads required.

## Package Structure

```
ranking_gam/
  models/          towers, GAM, GA2M, ContextGAM, context, submodular, multi-objective
  losses/          ApproxNDCG, pairwise, listwise, lambda, diffsort
  data/            MSLR-WEB10K loader
  metrics/         NDCG, diversity metrics (coverage, entropy, ILD, alpha-NDCG)
  training/        train loops for standard, diversity, multi-objective, GBDT boosting
  distill/         greedy PWL distillation (Algorithm 1)
  viz/             response curves, spider plots, diversity curves
  interactions.py  feature pair selection by correlation
examples/
  demo.py          4-demo script (GAM, Submodular, Multi-Objective, GBDT)
```

## References

- Zhuang et al. "Interpretable Ranking with Generalized Additive Models" (WSDM 2021)
- Lou et al. "Accurate Intelligible Models with Pairwise Interactions" (KDD 2013)
- Wang et al. "The LambdaLoss Framework for Ranking Metric Optimization" (CIKM 2018)
- Blondel et al. "Fast Differentiable Sorting and Ranking" (ICML 2020)

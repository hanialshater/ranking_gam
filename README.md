# ranking_gam

Interpretable learning-to-rank with Generalized Additive Models (GAMs) in PyTorch.

Based on Zhuang et al. ["Interpretable Ranking with Generalized Additive Models"](https://dl.acm.org/doi/10.1145/3437963.3441805) (WSDM 2021), extended with pairwise interactions (GA2M), submodular diversity-aware reranking, and multi-objective support.

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

# Train a GAM (one shape function per feature)
model = rg.GAM_Paper(num_features=136)
rg.train_model(model, train_loader, eval_loader, rg.ListNetLoss(), epochs=30)

# Distill to piecewise-linear for fast serving
pwl = rg.distill_to_pwl(model, num_knots=5, train_X=train_X)
scores = rg.pwl_predict(pwl, eval_X)  # numpy-only, no torch needed
```

## Models

| Model | Description |
|-------|-------------|
| `GAM_Paper` | One MLP tower per feature, no interactions |
| `GA2M_Paper` | GAM + pairwise interaction towers (input dim 2) |
| `ContextPresentGA2M` | GA2M with context-dependent feature weights |
| `SubmodularRankingGAM` | Base GAM + concave PWL diversity towers + greedy reranking |
| `MultiObjectiveRankingGAM` | Multiple objectives with runtime weight control |

### SubmodularRankingGAM (diversity-aware)

Two-phase training: base towers with ranking loss, then diversity towers with step-wise ListNet.

```python
submod = rg.SubmodularRankingGAM(
    num_item_features=136,
    groupwise_specs=[
        {"name": "category_novelty", "type": "category_novelty", "column": 136},
        {"name": "brand_novelty", "type": "brand_novelty", "column": 137},
    ],
)

# Phase 1: base towers
rg.train_model(submod, train_loader, eval_loader, rg.ListNetLoss(), epochs=30)

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

| Loss | Description | Key param |
|------|-------------|-----------|
| `ApproxNDCGLoss` | Direct NDCG optimization | `alpha=10` |
| `ListNetLoss` | Top-1 probability cross-entropy | -- |
| `ListMLELoss` | Permutation probability | -- |
| `LambdaLoss` | LambdaRank / ndcgLoss2++ | `weighing_scheme=` |
| `PairwiseLoss` | RankNet | `sigma=1.0` |
| `DiffSortNDCGLoss` | Differentiable soft ranks | `regularization_strength=` |

## Tower Types

- **`PaperTower`**: Unconstrained MLP. Used for item features. Supports `residual=True` (skip connection) and `input_norm=True` (BatchNorm).
- **`ConcavePWL`**: Monotone non-decreasing + concave. Guarantees submodularity for diversity features.
- **`MonotonePWL`**: Monotone non-decreasing only. For revenue, freshness, etc.
- **`LearnableMonotoneTransform`**: Monotone PWL mapping raw features to [0, 1], initialized from data percentiles (empirical CDF). Learned end-to-end while preserving interpretability.

Monotonicity and concavity are enforced via softplus parameterization, not projection.

## Performance Enhancements

All enhancements are opt-in and preserve full interpretability.

### Learnable Monotone Feature Transforms

Per-feature monotone warp initialized from data percentiles (empirical CDF). Maps raw features to [0, 1] so towers see well-normalized inputs. Since monotone-of-f is still a single-variable function, interpretability is preserved.

```python
model = rg.GAM_Paper(num_features=136, feature_transforms=True, num_transform_knots=20)
model.init_transforms_from_data(train_X)  # set knot positions from data percentiles
rg.train_model(model, train_loader, eval_loader, rg.ListNetLoss(), epochs=30)
```

### Residual Connections

Adds a linear skip from tower input to output (`out = MLP(x) + W*x`), making it easier to learn near-linear effects.

```python
model = rg.GAM_Paper(num_features=136, residual=True)
```

### Cosine Annealing LR Schedule

Decays learning rate following a cosine curve over the training epochs.

```python
rg.train_model(model, train_loader, eval_loader, loss_fn, lr_schedule="cosine")
```

### L1 Output Regularization

Penalizes large predicted scores to improve generalization.

```python
rg.train_model(model, train_loader, eval_loader, loss_fn, l1_output_reg=0.001)
```

### Combining Enhancements

```python
model = rg.GAM_Paper(
    num_features=136, hidden_dims=[16, 8],
    feature_transforms=True, residual=True,
)
model.init_transforms_from_data(train_X)
rg.train_model(
    model, train_loader, eval_loader, rg.ListNetLoss(),
    epochs=30, lr_schedule="cosine", l1_output_reg=0.001,
)
```

## Running the Demo

```bash
# All demos
python examples/demo.py

# Select specific demos
python examples/demo.py --demo gam
python examples/demo.py --demo submodular
python examples/demo.py --demo multi
python examples/demo.py --demo gam,multi

# Production quality
python examples/demo.py --epochs 30

# With performance enhancements
python examples/demo.py --demo gam --transforms                          # monotone feature transforms
python examples/demo.py --demo gam --transforms --residual --cosine      # all enhancements
python examples/demo.py --demo gam --transforms --cosine --l1-reg 0.001  # with L1 regularization
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
  models/          towers, GAM, GA2M, context, submodular, multi-objective
  losses/          ApproxNDCG, pairwise, listwise, lambda, diffsort
  data/            MSLR-WEB10K loader
  metrics/         NDCG, diversity metrics (coverage, entropy, ILD, alpha-NDCG)
  training/        train loops for standard, diversity, multi-objective
  distill/         greedy PWL distillation (Algorithm 1)
  viz/             response curves, spider plots, diversity curves
  interactions.py  feature pair selection by correlation
examples/
  demo.py          3-demo script with visualization
```

## References

- Zhuang et al. "Interpretable Ranking with Generalized Additive Models" (WSDM 2021)
- Lou et al. "Accurate Intelligible Models with Pairwise Interactions" (KDD 2013)
- Wang et al. "The LambdaLoss Framework for Ranking Metric Optimization" (CIKM 2018)
- Blondel et al. "Fast Differentiable Sorting and Ranking" (ICML 2020)

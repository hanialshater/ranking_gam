# ranking-gam v0.1.0 — Release Notes

**Interpretable Ranking with Generalized Additive Models**

Initial release of `ranking-gam`, a PyTorch library for interpretable learning-to-rank. Based on Zhuang et al. "Interpretable Ranking with Generalized Additive Models" (WSDM 2021), extended with pairwise interactions, context-weighted models, submodular diversity reranking, multi-objective support, transformer baselines, GBDT residual boosting, and optimized Java inference.

---

## Models

### GAM / GA2M (Core)
- **GAM_Paper**: One MLP tower per feature — the simplest interpretable ranker. Each feature gets its own learned shape function `f_j(x_j)`, and the final score is `sum_j f_j(x_j)`.
- **GA2M_Paper**: Extends GAM with pairwise interaction towers (input dim 2) selected via correlation-based feature pair selection. Captures feature interactions while remaining interpretable.

### Context-Weighted Models
- **ContextGAM**: Self-context model with a shared context network that produces per-feature importance weights. Scoring: `score = sum_j w_j(x) * f_j(x_j)`. Tower shapes remain single-variable (distillable to PWL), context weights are query-dependent but inspectable. Achieves the best NDCG among interpretable models.
- **ContextPresentGA2M**: GA2M with separate context features (e.g., query text/category) producing importance weights. Supports both numerical and categorical context via learned embeddings.
- **ContextWeightNetwork**: Reusable context feature subnetwork producing importance weights with softmax normalization.

### Submodular Diversity Reranking
- **SubmodularRankingGAM**: Base GAM + concave PWL diversity towers + greedy reranking. Concavity enforces submodularity, guaranteeing the greedy algorithm achieves (1-1/e) ~ 63% of the optimal diverse ranking. Two-phase training: (1) train base towers with ranking loss, (2) freeze base and train diversity towers with step-wise ListNet.
- **GroupwiseFeatureComputer**: Computes set-dependent features (e.g., category coverage, price diversity) needed for diversity-aware scoring.

### Multi-Objective Ranking
- **MultiObjectiveRankingGAM**: Multiple objectives (pointwise + groupwise) with runtime weight control. Product managers can adjust objective weights at serving time — no retraining required. Greedy reranking with Pareto-controllable tradeoffs.

### Transformer Baselines
- **TransformerRanker**: Black-box self-attention model over the document list. Cross-document interactions make it a full black-box — serves as an upper-bound comparison to quantify the NDCG cost of requiring interpretability.
- **InvertedTransformerRanker**: Inspired by iTransformer (Liu et al., ICLR 2024). Applies self-attention across *features* rather than documents. Scores each document independently (like GAM) but captures arbitrary feature interactions. Per-feature embeddings, pre-norm transformer blocks, CLS/mean pooling, and attention weight extraction for interpretability.

### Tower Types
- **PaperTower**: Unconstrained MLP with configurable activations (relu/silu/gelu), dropout, residual connections, and input normalization.
- **ConcavePWL**: Monotone non-decreasing + concave piecewise-linear. Slopes are constrained via softplus parameterization (never projection). Guarantees submodularity for diversity towers.
- **MonotonePWL**: Monotone non-decreasing piecewise-linear with independent slopes.
- **LearnableMonotoneTransform**: Monotone piecewise-linear input normalization initialized from data percentiles. Learnable feature transforms that preserve ordering.

---

## Loss Functions

A comprehensive suite of ranking loss functions, all handling padding (label = -1) internally:

| Loss | Description |
|------|-------------|
| **LambdaLoss** | Unified framework (Wang et al. 2018) with multiple weighting schemes: `lambdaRank`, `ndcgLoss1`, `ndcgLoss2`, `ndcgLoss2++` |
| **LambdaLoss.ndcg2pp()** | Convenience constructor for ndcgLoss2++ — recommended default, best NDCG |
| **ApproxNDCGLoss** | Direct NDCG optimization via smooth approximation |
| **PairwiseLoss** | RankNet-style pairwise logistic loss |
| **ListNetLoss** | Top-1 probability listwise loss with optional label smoothing |
| **ListMLELoss** | Permutation probability listwise loss |
| **DiffSortNDCGLoss** | Differentiable NDCG using soft ranking with isotonic regression (PAVA) |

---

## Training

### Training Loops
- **train_model**: Standard training loop with early stopping on NDCG. Supports cosine/plateau/constant LR schedules, linear warmup, gradient clipping, L1 output regularization, and AdamW weight decay. Automatically separates parameter groups (transforms get lower LR, skip/bias params excluded from weight decay).
- **train_diversity_towers**: Phase-2 step-wise diversity tower training for SubmodularRankingGAM.
- **train_multi_objective**: Joint multi-objective step-wise training for MultiObjectiveRankingGAM.

### GBDT Residual Boosting
- **train_gbdt_baseline**: Train a LambdaMART ranker (via LightGBM) as a performance ceiling.
- **train_gbdt_residual_boost**: Three-stage pipeline — (1) train GAM on raw features, (2) train GBDT regressor on GAM residuals, (3) add a "magic curve" tower (feature D+1) that captures what GAM missed. Combines interpretability of GAM with the residual power of GBDT.
- **compute_gbdt_residual_feature**: Extract GBDT residual scores as an additional feature for the GAM.

### Item Boosting
- **boost_items**: Apply interpretable feature overrides to selected items and re-score. The boost is fully transparent — equivalent to moving along the response curve.
- **warmup_boost_items / warmup_blend**: Time-decaying boost for cold-start items. New items get full boost, items past the warmup period use real values, items in between get a linear blend.
- **compute_boost_delta / explain_boost**: Compute and explain the exact score delta from shifting a feature on its response curve.
- **get_percentile_value**: Target boost values by percentile (e.g., "boost to the 75th percentile of popularity").

---

## Distillation

Neural towers distilled to piecewise-linear functions for fast serving and full interpretability, via Algorithm 1 from the paper (greedy knot selection + least squares refinement).

- **distill_to_pwl**: Distill GAM/GA2M/ContextGAM/SubmodularRankingGAM to PWL. Paper says K=3-5 knots is usually sufficient. Includes interaction grids (bilinear) and diversity tower export.
- **distill_context_model**: Full distillation for context-present GA2M — main effects + interactions + context weight functions.
- **pwl_predict**: NumPy-only inference from distilled models (no PyTorch needed). Supports context weights.
- **save_pwl_json / load_pwl_json**: JSON serialization for cross-language serving (Python ↔ Java).
- **evaluate_pwl**: Evaluate distilled model NDCG with optional context.

---

## Java Inference Engine

A production-ready Java inference library for distilled GAM models, with no ML framework dependencies — pure arithmetic suitable for low-latency ranking services.

### Core Components
- **DistilledGamModel**: Scores documents using `bias + sum(pwl_j(x_j)) + sum(interp_k(x_{f1}, x_{f2}))`. Supports single-document, batch, and columnar (structure-of-arrays) scoring.
- **DistilledGamLoader**: Loads models exported via Python's `save_pwl_json()`.
- **CompiledPwlFunction**: Compiles PWL lookups to specialized if/else evaluators at load time — no loops or binary search for K<=6 knots. Bulk `evaluateAndAccumulate` for column-major scoring.
- **PwlFunction**: Standard PWL evaluation with precomputed slopes and binary search.
- **BilinearGridFunction**: Bilinear interpolation on 2D grids for pairwise interaction towers. Includes bulk `evaluateAndAccumulate`.
- **ConcavePwlFunction**: Concave PWL evaluation for diversity towers with `evaluateAtMax()` for upper-bound computation.

### Submodular Reranking
- **SubmodularGamReranker**: Greedy submodular reranking with Minoux lazy greedy acceleration (1978). Uses a priority queue of upper bounds — previous marginal gains are valid upper bounds due to submodularity. Configurable max evaluation budget per position to cap worst-case cost.
- **GroupwiseFeatureComputer / DefaultGroupwiseComputer**: Interface and default implementation for computing set-dependent diversity features in Java.

### Tests
- End-to-end cross-validation tests ensuring Python-exported models produce matching scores in Java.
- Unit tests for PWL evaluation, bilinear grid interpolation, and compiled evaluator correctness.

---

## Data Loading

Unified data loading for major learning-to-rank benchmarks with auto-download support:

| Dataset | Features | Notes |
|---------|----------|-------|
| **MSLR-WEB10K** | 136 | Auto-download from GCS |
| **MSLR-WEB30K** | 136 | Manual download required |
| **Yahoo LTRC Set 1/2** | 699/700 | License required |
| **Istella-S/Full/X** | 220 | License required |
| **LETOR 4.0 MQ2007/MQ2008** | 46 | Free, auto-download |
| **FINN Slates** | 19 | HuggingFace Hub / Google Drive |

- **load_dataset**: Unified loader by name with `log1p` transform, top-k padding, and configurable list size.
- **get_num_features / DATASET_NAMES**: Registry for programmatic dataset discovery.
- SVMLight format parser shared across MSLR, Yahoo, Istella, and LETOR loaders.

---

## Metrics

- **compute_ndcg**: NDCG@k with proper padding handling (label = -1 excluded).
- **evaluate_ranking_diversity**: Diversity metrics for evaluating submodular reranking quality.

---

## Visualization

- **plot_response_curves**: Plot learned shape functions `f_j(x_j)` for top-k features by importance, with optional data histogram overlay. Reproduces paper Figures 5/7.
- **plot_diversity_curves**: Visualize concave PWL diversity tower shapes.
- **plot_spider**: Radar/spider charts for multi-objective tradeoff visualization with absolute metric bounds.

---

## Examples

| Demo | Description |
|------|-------------|
| `demo_gam.py` | GAM and ContextGAM training with distillation and response curve plots |
| `demo_submodular.py` | SubmodularRankingGAM two-phase training with diversity evaluation |
| `demo_multi.py` | Multi-objective ranking with Pareto weight sweeps |
| `demo_gbdt.py` | Three-stage GBDT residual boosting pipeline |
| `demo_boost.py` | Item boosting and cold-start warmup blend |
| `demo_transformer.py` | TransformerRanker black-box baseline |
| `demo_inverted_transformer.py` | InvertedTransformerRanker with 3-way NDCG comparison |
| `demo_finn_slates.py` | FINN slate data with GA2M interaction heatmaps |
| `demo_java_export.py` | End-to-end Python → JSON → Java inference pipeline |
| `demo.py` | Unified CLI with `--demo` selector for all demos |

---

## Dependencies

- **Required**: `torch>=2.0`, `numpy>=1.24`, `scipy>=1.10`
- **Visualization**: `matplotlib>=3.7`
- **Data/Baselines**: `lightgbm>=4.0`, `scikit-learn>=1.0`
- **FINN data**: `huggingface_hub>=0.14`, `gdown>=4.6`
- **Java inference**: No external dependencies (pure Java)

---

## Testing

178 tests covering all models, losses, training loops, distillation, data loading, metrics, visualization, boosting, and cross-language inference validation. Run with:

```bash
pytest tests/
```

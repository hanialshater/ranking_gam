# Repository Evaluation: ranking_gam

## Overview

A PyTorch library for interpretable learning-to-rank using Generalized Additive Models (GAMs).
Based on Zhuang et al. "Interpretable Ranking with Generalized Additive Models" (WSDM 2021),
extended with pairwise interactions (GA2M), context-weighted GAM, submodular diversity-aware
reranking, and multi-objective support.

**Version:** 0.1.0 | **Python:** >=3.9 | **License:** MIT
**Codebase size:** ~8,700 lines across 55 Python files | **30 commits, 1 contributor**

---

## Test Results

```
154 collected, 143 passed, 11 failed
```

| Failure Category | Count | Root Cause |
|-----------------|-------|------------|
| GroupwiseFeatureComputer IndexError | 5 | Test fixture uses column index 8 on 8-feature data (indices 0-7) |
| LightGBM requires scikit-learn | 6 | Missing `scikit-learn` dependency (not declared in pyproject.toml) |

**Ruff linting:** 36 issues (27 auto-fixable). Mostly import sorting (I001), 3 unused imports
(F401), 3 ambiguous variable names (E741), 1 unused assignment (F841).

---

## Strengths

### 1. Architecture Design

The model hierarchy is well-designed and follows a clear progression of complexity:

```
GAM_Paper → GA2M_Paper → ContextGAM → SubmodularRankingGAM → MultiObjectiveRankingGAM
```

Each model adds capability while maintaining the interpretability contract: single-variable
towers stay distillable to piecewise-linear functions. The separation between base ranking
(differentiable forward pass) and diversity reranking (greedy inference-time procedure) is
architecturally sound.

### 2. Mathematical Rigor in Tower Constraints

Tower constraints (monotonicity, concavity) are enforced via parameterization (softplus on
slopes), not projection. This is the correct approach — it guarantees constraints are never
violated during training, unlike projection methods that can oscillate. Specific highlights:

- `ConcavePWL` (`towers.py:154-217`): Slopes enforced as cumulative reversed softplus sums,
  guaranteeing both monotonicity and concavity simultaneously.
- `LearnableMonotoneTransform` (`towers.py:64-152`): Data-driven initialization from percentiles
  with proper epsilon guards throughout.

### 3. Loss Function Library

Seven loss functions covering the major approaches to learning-to-rank. All correctly handle
the `-1` padding convention. Numerical stability is handled consistently with epsilon guards,
clamp operations, and log-sum-exp stabilization. The `LambdaLoss` framework correctly unifies
multiple weighting schemes under a single implementation.

### 4. Training Infrastructure

`train_model` (`training/trainer.py`) provides a well-featured training loop with:
- Three LR schedule options (cosine, plateau, constant) with warmup
- Automatic skip/bias parameter exclusion from weight decay
- Early stopping with best-state restoration
- Gradient clipping

### 5. Distillation Pipeline

The greedy knot selection (Algorithm 1 from the paper) is faithfully implemented in
`distill/pwl.py`. The refinement loop with convergence checking and the least-squares
solution via `numpy.linalg.lstsq` are correct. For ContextGAM, the library properly
handles the hybrid case (towers → PWL, context network stays neural).

### 6. Comprehensive Data Loading

Multi-dataset support (MSLR-WEB10K/30K, Yahoo, Istella, LETOR 4.0) with a clean registry
pattern. The SVMLight parser handles edge cases well (1-indexed features, missing features,
LETOR comments). Auto-download is supported for MSLR and Istella.

### 7. Public API Design

Lazy imports via `__getattr__` in `__init__.py` allow `import ranking_gam` to succeed even
without torch installed. 57 symbols exported covering models, losses, metrics, training,
distillation, data, and utilities.

---

## Bugs Found

### BUG-1: Test Fixture Column Index Out of Bounds (5 test failures)

**Files:** `tests/test_models.py:195`

The `TestSubmodularRankingGAM._make_model()` and `TestMultiObjectiveRankingGAM._make_model()`
create `GroupwiseFeatureComputer` specs with `"column": 8`, but the synthetic test data has
only 8 features (indices 0-7). This causes `IndexError: index 8 is out of bounds for
dimension 1 with size 8` at `groupwise.py:71`.

**Impact:** 5 tests fail (greedy_rerank and explain for Submodular and MultiObjective models).
This is a test bug, not a model bug — the models themselves work correctly when given valid
column indices. However, it means greedy reranking is not tested.

### BUG-2: Missing scikit-learn Dependency (6 test failures)

**File:** `pyproject.toml`

LightGBM's sklearn interface requires scikit-learn, but it's not declared as a dependency.
All 6 boosting tests fail with `LightGBMError: scikit-learn is required`.

**Fix:** Add `scikit-learn` to the `[project.optional-dependencies] data` group.

### BUG-3: ListNetLoss All-Padded Query Handling

**File:** `ranking_gam/losses/listwise.py:84-98`

When all items in a query are padded (labels all < 0), `F.softmax` receives all `-inf` values
and produces NaN. Unlike `ListMLELoss` (which has an early return at line 24-26),
`ListNetLoss` has no guard for this case.

**Impact:** NaN loss values propagate through training if any query has zero valid documents.

### BUG-4: _ndcg2_weights Fragile Negative Indexing

**File:** `ranking_gam/losses/lambda_loss.py:143-152`

```python
delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])  # 0 on diagonal
D[0, delta_idxs - 1]  # D[0, -1] when delta_idxs=0 → wraps to last element
```

The diagonal of `delta_idxs` is 0, so `delta_idxs - 1 = -1` wraps to `D[0, L-1]` via
PyTorch negative indexing. The result is then zeroed by `deltas.diagonal().zero_()`.

**Impact:** Currently produces correct results because the diagonal is explicitly zeroed.
However, the code relies on negative indexing producing a valid (but wrong) intermediate
value that gets masked. This is fragile — any refactor that removes or reorders the
diagonal zeroing would introduce a real bug.

---

## Issues and Recommendations

### HIGH Priority

| Issue | File | Description |
|-------|------|-------------|
| No bounds checking on GroupwiseFeatureComputer column indices | `groupwise.py:68-71` | Column indices from specs are used directly without validation against feature dimension. Should raise ValueError if column >= num_features. |
| Undeclared scikit-learn dependency | `pyproject.toml` | Add `scikit-learn>=1.0` alongside `lightgbm>=4.0` in the `data` extras group. |
| Greedy reranking not tested | `tests/test_models.py` | Fix column indices in test fixtures (use indices within 0-7 range) so the 5 failing tests pass. |
| No CI/CD pipeline | Root | No GitHub Actions, no automated test runs on push/PR. |

### MEDIUM Priority

| Issue | File | Description |
|-------|------|-------------|
| O(L²) list removal in greedy loops | `submodular.py:218`, `multi_objective.py:237` | `remaining.remove(best_global)` is O(n). Use `remaining.pop(best_local)` for O(1). |
| GBDT residual normalization uses hardcoded 1e-6 | `training/boosting.py:316` | When GAM std is legitimately < 1e-6, normalizing by 1e-6 amplifies residuals by 10^5+. Use quantile-based normalization instead. |
| Ruff lint violations | Multiple files | 36 issues, 27 auto-fixable. Run `ruff check --fix` to resolve import sorting. |
| No TransformerRanker tests | `tests/` | `TransformerRanker` is the only exported model class without any test coverage. |
| ContextGAM test coverage sparse | `tests/test_models.py` | Only 2 tests (forward shape, context-absent). Missing: explain method, context weight validation, distillation roundtrip. |

### LOW Priority

| Issue | File | Description |
|-------|------|-------------|
| Unused imports | `gam.py:10`, `diffsort.py:13`, `svmlight.py:11`, `test_training.py:3` | `numpy`, `F`, `glob` imported but unused. |
| Ambiguous variable name `l` | `svmlight.py:86,94,97` | Rename to `labels` for clarity. |
| Global min/max normalization in GBDT residual feature | `training/boosting.py:236-239` | Normalizes over all queries globally. Per-query or quantile-based normalization would be more robust. |
| Interactions selection not scalable | `interactions.py:31-32` | Exhaustive O(D²×N) search over all feature pairs. Fine for D=136 but impractical for D>500. |
| No visualization tests | `viz/plots.py` | Plot functions have no test coverage (only invoked in examples). |
| `model.state_dict()` shallow copy for early stopping | `training/trainer.py:146` | Should use `copy.deepcopy(model.state_dict())` to guarantee independence from subsequent parameter updates. |

---

## Code Quality Summary

| Module | Quality | Notes |
|--------|---------|-------|
| `models/towers.py` | Excellent | Clean constraint parameterization, good numerical guards |
| `models/gam.py` | Excellent | Clean, well-documented forward passes |
| `models/context.py` | Excellent | Clever self-context design in ContextGAM |
| `models/groupwise.py` | Good | Missing input validation on column indices |
| `models/submodular.py` | Good | O(L²) greedy loop, but correct |
| `models/multi_objective.py` | Good | Flexible weight handling, same O(L²) issue |
| `losses/` (all) | Very Good | Consistent padding handling, good numerical stability |
| `training/trainer.py` | Very Good | Feature-rich training loop |
| `training/boosting.py` | Good | Fragile residual normalization |
| `distill/pwl.py` | Excellent | Faithful Algorithm 1 implementation |
| `metrics/ranking.py` | Very Good | Correct NDCG, comprehensive diversity metrics |
| `data/` (all) | Very Good | Multi-dataset support, robust SVMLight parser |
| `viz/plots.py` | Good | Functional but untested |
| Tests | Good | 143/154 pass, but 5 failures are fixable test bugs |
| Documentation | Excellent | CLAUDE.md and README.md are thorough |
| Examples | Good | 7 demos, though demo_boost.py has untested API paths |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Public API (__init__.py)                  │
│                     57 symbols, lazy imports                     │
├─────────┬──────────┬──────────┬───────────┬──────────┬─────────┤
│ Models  │  Losses  │ Training │ Distill   │  Data    │  Viz    │
├─────────┼──────────┼──────────┼───────────┼──────────┼─────────┤
│GAM_Paper│ApproxNDCG│train_    │greedy_    │load_mslr │plot_    │
│GA2M     │Pairwise  │  model   │  knot_sel │load_yahoo│ response│
│Context  │ListMLE   │train_div │distill_   │load_     │ _curves │
│  GAM    │ListNet   │train_    │  to_pwl   │  istella │plot_    │
│Submod   │Lambda    │  multi   │pwl_predict│load_mq*  │ spider  │
│MultiObj │DiffSort  │train_gbdt│           │SVMLight  │plot_    │
│Transf.  │          │          │           │  parser  │ pareto  │
└─────────┴──────────┴──────────┴───────────┴──────────┴─────────┘
```

---

## Overall Assessment

This is a **well-implemented research library** that faithfully reproduces the paper's
methodology and extends it meaningfully. The core model and loss implementations are
mathematically sound with proper numerical stability handling throughout. The codebase is
well-organized with clear module boundaries and a comprehensive public API.

The main gaps are operational: no CI/CD, 11 failing tests (5 from a test configuration
error, 6 from a missing dependency), and some lint issues. These are straightforward to fix.
The library would benefit from adding input validation in `GroupwiseFeatureComputer`,
fixing the test fixtures, declaring the scikit-learn dependency, and setting up CI.

**Maturity level:** Research prototype approaching production quality. The mathematical
foundations are solid; the operational infrastructure needs hardening.

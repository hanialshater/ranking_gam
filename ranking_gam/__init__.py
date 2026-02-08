"""
ranking_gam -- Interpretable Ranking with Generalized Additive Models.

A PyTorch library for learning-to-rank with:
  - GAM / GA2M models with per-feature shape functions
  - Context-present models with learned feature importance weights
  - Submodular diversity-aware reranking with greedy (1-1/e) guarantees
  - Multi-objective ranking with runtime weight control
  - PWL distillation for fast serving
  - Comprehensive ranking loss functions (ApproxNDCG, LambdaRank, ListMLE, etc.)

Based on Zhuang et al. "Interpretable Ranking with Generalized Additive
Models" (WSDM 2021), extended with pairwise interactions, submodular
diversity, and multi-objective support.
"""

__version__ = "0.1.0"

# Models
from .models import (
    PaperTower,
    ConcavePWL,
    MonotonePWL,
    GAM_Paper,
    GA2M_Paper,
    ContextWeightNetwork,
    ContextPresentGA2M,
    GroupwiseFeatureComputer,
    SubmodularRankingGAM,
    MultiObjectiveRankingGAM,
)

# Losses
from .losses import (
    ApproxNDCGLoss,
    PairwiseLoss,
    ListMLELoss,
    ListNetLoss,
    LambdaLoss,
    DiffSortNDCGLoss,
    soft_rank,
)

# Metrics
from .metrics import compute_ndcg, evaluate_ranking_diversity

# Training
from .training import (
    train_model,
    train_diversity_towers,
    train_multi_objective,
    get_base_ranking,
    get_greedy_ranking,
)

# Distillation
from .distill import (
    greedy_knot_selection,
    distill_to_pwl,
    distill_context_model,
    pwl_predict,
    evaluate_pwl,
)

# Data
from .data import load_mslr

# Interactions
from .interactions import select_interactions_correlation

__all__ = [
    # Models
    "PaperTower",
    "ConcavePWL",
    "MonotonePWL",
    "GAM_Paper",
    "GA2M_Paper",
    "ContextWeightNetwork",
    "ContextPresentGA2M",
    "GroupwiseFeatureComputer",
    "SubmodularRankingGAM",
    "MultiObjectiveRankingGAM",
    # Losses
    "ApproxNDCGLoss",
    "PairwiseLoss",
    "ListMLELoss",
    "ListNetLoss",
    "LambdaLoss",
    "DiffSortNDCGLoss",
    "soft_rank",
    # Metrics
    "compute_ndcg",
    "evaluate_ranking_diversity",
    # Training
    "train_model",
    "train_diversity_towers",
    "train_multi_objective",
    "get_base_ranking",
    "get_greedy_ranking",
    # Distillation
    "greedy_knot_selection",
    "distill_to_pwl",
    "distill_context_model",
    "pwl_predict",
    "evaluate_pwl",
    # Data
    "load_mslr",
    # Interactions
    "select_interactions_correlation",
]

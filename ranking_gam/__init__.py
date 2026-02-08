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


def __getattr__(name):
    """Lazy imports so that `import ranking_gam` never fails due to missing torch."""
    # Build a mapping on first access
    _import_map = {
        # Models
        "PaperTower": (".models", "PaperTower"),
        "ConcavePWL": (".models", "ConcavePWL"),
        "MonotonePWL": (".models", "MonotonePWL"),
        "GAM_Paper": (".models", "GAM_Paper"),
        "GA2M_Paper": (".models", "GA2M_Paper"),
        "ContextWeightNetwork": (".models", "ContextWeightNetwork"),
        "ContextPresentGA2M": (".models", "ContextPresentGA2M"),
        "GroupwiseFeatureComputer": (".models", "GroupwiseFeatureComputer"),
        "SubmodularRankingGAM": (".models", "SubmodularRankingGAM"),
        "MultiObjectiveRankingGAM": (".models", "MultiObjectiveRankingGAM"),
        # Losses
        "ApproxNDCGLoss": (".losses", "ApproxNDCGLoss"),
        "PairwiseLoss": (".losses", "PairwiseLoss"),
        "ListMLELoss": (".losses", "ListMLELoss"),
        "ListNetLoss": (".losses", "ListNetLoss"),
        "LambdaLoss": (".losses", "LambdaLoss"),
        "DiffSortNDCGLoss": (".losses", "DiffSortNDCGLoss"),
        "soft_rank": (".losses", "soft_rank"),
        # Metrics
        "compute_ndcg": (".metrics", "compute_ndcg"),
        "evaluate_ranking_diversity": (".metrics", "evaluate_ranking_diversity"),
        # Training
        "train_model": (".training", "train_model"),
        "train_diversity_towers": (".training", "train_diversity_towers"),
        "train_multi_objective": (".training", "train_multi_objective"),
        "get_base_ranking": (".training", "get_base_ranking"),
        "get_greedy_ranking": (".training", "get_greedy_ranking"),
        # Distillation
        "greedy_knot_selection": (".distill", "greedy_knot_selection"),
        "distill_to_pwl": (".distill", "distill_to_pwl"),
        "distill_context_model": (".distill", "distill_context_model"),
        "pwl_predict": (".distill", "pwl_predict"),
        "evaluate_pwl": (".distill", "evaluate_pwl"),
        # Data
        "load_mslr": (".data", "load_mslr"),
        # Interactions
        "select_interactions_correlation": (".interactions", "select_interactions_correlation"),
    }

    if name in _import_map:
        import importlib

        module_path, attr = _import_map[name]
        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr)
        # Cache on the module so __getattr__ isn't called again
        globals()[name] = val
        return val

    raise AttributeError(f"module 'ranking_gam' has no attribute {name!r}")


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

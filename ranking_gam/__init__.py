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
        "LearnableMonotoneTransform": (".models", "LearnableMonotoneTransform"),
        "GAM_Paper": (".models", "GAM_Paper"),
        "GA2M_Paper": (".models", "GA2M_Paper"),
        "ContextWeightNetwork": (".models", "ContextWeightNetwork"),
        "ContextPresentGA2M": (".models", "ContextPresentGA2M"),
        "ContextGAM": (".models", "ContextGAM"),
        "GroupwiseFeatureComputer": (".models", "GroupwiseFeatureComputer"),
        "SubmodularRankingGAM": (".models", "SubmodularRankingGAM"),
        "MultiObjectiveRankingGAM": (".models", "MultiObjectiveRankingGAM"),
        "TransformerRanker": (".models", "TransformerRanker"),
        "InvertedTransformerRanker": (".models", "InvertedTransformerRanker"),
        "GAMFormer": (".models", "GAMFormer"),
        "RevIN": (".models", "RevIN"),
        "TemporalEncoder": (".models", "TemporalEncoder"),
        "TemporalGAMFormer": (".models", "TemporalGAMFormer"),
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
        "train_gbdt_baseline": (".training", "train_gbdt_baseline"),
        "train_gbdt_residual_boost": (".training", "train_gbdt_residual_boost"),
        "compute_gbdt_residual_feature": (".training", "compute_gbdt_residual_feature"),
        # Distillation
        "greedy_knot_selection": (".distill", "greedy_knot_selection"),
        "distill_to_pwl": (".distill", "distill_to_pwl"),
        "distill_context_model": (".distill", "distill_context_model"),
        "pwl_predict": (".distill", "pwl_predict"),
        "evaluate_pwl": (".distill", "evaluate_pwl"),
        # Data
        "load_mslr": (".data", "load_mslr"),
        "load_mslr30k": (".data", "load_mslr30k"),
        "load_yahoo": (".data", "load_yahoo"),
        "load_istella": (".data", "load_istella"),
        "load_mq2007": (".data", "load_mq2007"),
        "load_mq2008": (".data", "load_mq2008"),
        "load_finn": (".data", "load_finn"),
        "load_dataset": (".data", "load_dataset"),
        "get_num_features": (".data", "get_num_features"),
        "DATASET_NAMES": (".data", "DATASET_NAMES"),
        "FINN_NUM_FEATURES": (".data", "FINN_NUM_FEATURES"),
        "FINN_FEATURE_NAMES": (".data", "FINN_FEATURE_NAMES"),
        "FINN_INTERACTION_PAIRS": (".data", "FINN_INTERACTION_PAIRS"),
        # Interactions
        "select_interactions_correlation": (".interactions", "select_interactions_correlation"),
        # Item boosting
        "compute_boost_delta": (".boosting", "compute_boost_delta"),
        "boost_items": (".boosting", "boost_items"),
        "get_percentile_value": (".boosting", "get_percentile_value"),
        "explain_boost": (".boosting", "explain_boost"),
        "warmup_blend": (".boosting", "warmup_blend"),
        "warmup_boost_items": (".boosting", "warmup_boost_items"),
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
    "LearnableMonotoneTransform",
    "GAM_Paper",
    "GA2M_Paper",
    "ContextWeightNetwork",
    "ContextPresentGA2M",
    "ContextGAM",
    "GroupwiseFeatureComputer",
    "SubmodularRankingGAM",
    "MultiObjectiveRankingGAM",
    "TransformerRanker",
    "InvertedTransformerRanker",
    "GAMFormer",
    "RevIN",
    "TemporalEncoder",
    "TemporalGAMFormer",
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
    "train_gbdt_baseline",
    "train_gbdt_residual_boost",
    "compute_gbdt_residual_feature",
    # Distillation
    "greedy_knot_selection",
    "distill_to_pwl",
    "distill_context_model",
    "pwl_predict",
    "evaluate_pwl",
    # Data
    "load_mslr",
    "load_mslr30k",
    "load_yahoo",
    "load_istella",
    "load_mq2007",
    "load_mq2008",
    "load_finn",
    "load_dataset",
    "get_num_features",
    "DATASET_NAMES",
    "FINN_NUM_FEATURES",
    "FINN_FEATURE_NAMES",
    "FINN_INTERACTION_PAIRS",
    # Interactions
    "select_interactions_correlation",
    # Item boosting
    "compute_boost_delta",
    "boost_items",
    "get_percentile_value",
    "explain_boost",
    "warmup_blend",
    "warmup_boost_items",
]

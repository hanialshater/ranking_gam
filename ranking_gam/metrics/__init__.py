"""Ranking and diversity metrics."""

from .ranking import compute_ndcg, evaluate_ranking_diversity

__all__ = ["compute_ndcg", "evaluate_ranking_diversity"]

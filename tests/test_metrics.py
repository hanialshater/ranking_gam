"""Tests for ranking metrics."""

import numpy as np
import torch

from ranking_gam.metrics import compute_ndcg, evaluate_ranking_diversity


class TestComputeNDCG:
    def test_perfect_ranking(self):
        y_true = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
        y_pred = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        ndcg = compute_ndcg(y_pred, y_true, k=5)
        assert abs(ndcg - 1.0) < 1e-6

    def test_worst_ranking(self):
        y_true = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
        y_pred = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])  # reversed
        ndcg = compute_ndcg(y_pred, y_true, k=5)
        assert ndcg < 1.0

    def test_handles_padding(self):
        y_true = torch.tensor([[4.0, 3.0, -1.0, -1.0]])
        y_pred = torch.tensor([[5.0, 4.0, 3.0, 2.0]])
        ndcg = compute_ndcg(y_pred, y_true, k=2)
        assert abs(ndcg - 1.0) < 1e-6

    def test_all_padded_returns_zero(self):
        y_true = torch.tensor([[-1.0, -1.0, -1.0]])
        y_pred = torch.tensor([[1.0, 2.0, 3.0]])
        ndcg = compute_ndcg(y_pred, y_true, k=3)
        assert ndcg == 0.0

    def test_numpy_input(self):
        y_true = np.array([[4.0, 3.0, 2.0]])
        y_pred = np.array([[3.0, 2.0, 1.0]])
        ndcg = compute_ndcg(y_pred, y_true, k=3)
        assert abs(ndcg - 1.0) < 1e-6


class TestEvaluateRankingDiversity:
    def test_returns_all_keys(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        B, L, D = X_aug.shape
        orders = [list(range(min(5, L))) for _ in range(B)]
        result = evaluate_ranking_diversity(
            X_aug, y, orders, k=5,
            cat_col=D - 2, brand_col=D - 1, price_col=3,
        )
        expected_keys = {
            "ndcg", "precision", "mean_rel",
            "cat_coverage", "brand_coverage",
            "cat_entropy", "brand_entropy",
            "price_std", "ild", "alpha_ndcg",
        }
        assert set(result.keys()) == expected_keys

    def test_values_are_finite(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        B, L, D = X_aug.shape
        orders = [list(range(min(5, L))) for _ in range(B)]
        result = evaluate_ranking_diversity(
            X_aug, y, orders, k=5,
            cat_col=D - 2, brand_col=D - 1, price_col=3,
        )
        for k, v in result.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

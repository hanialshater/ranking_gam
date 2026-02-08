"""Tests for GAM, GA2M, Context, Submodular, and MultiObjective models."""

import numpy as np
import torch

from ranking_gam.models import (
    GAM_Paper,
    GA2M_Paper,
    ContextPresentGA2M,
    SubmodularRankingGAM,
    MultiObjectiveRankingGAM,
)


class TestGAMPaper:
    def test_forward_shape(self, synthetic_tensors):
        X, y = synthetic_tensors
        B, L, D = X.shape
        model = GAM_Paper(num_features=D, hidden_dims=[8, 4])
        out = model(X)
        assert out.shape == (B, L)

    def test_get_main_effect(self):
        model = GAM_Paper(num_features=4, hidden_dims=[8, 4])
        x_vals = np.linspace(-1, 1, 20).astype(np.float32)
        eff = model.get_main_effect(0, x_vals)
        assert eff.shape == (20,)

    def test_gradient_flows(self, synthetic_tensors):
        X, y = synthetic_tensors
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4])
        out = model(X)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestGA2MPaper:
    def test_forward_no_interactions(self, synthetic_tensors):
        X, y = synthetic_tensors
        B, L, D = X.shape
        model = GA2M_Paper(num_features=D, hidden_dims=[8, 4])
        out = model(X)
        assert out.shape == (B, L)

    def test_forward_with_interactions(self, synthetic_tensors):
        X, y = synthetic_tensors
        B, L, D = X.shape
        pairs = [(0, 1), (2, 3)]
        model = GA2M_Paper(
            num_features=D, interaction_pairs=pairs,
            hidden_dims=[8, 4], interaction_hidden=[8, 4],
        )
        out = model(X)
        assert out.shape == (B, L)
        assert model.num_interactions == 2

    def test_get_interaction_effect(self, synthetic_tensors):
        X, y = synthetic_tensors
        D = X.shape[-1]
        model = GA2M_Paper(num_features=D, interaction_pairs=[(0, 1)])
        x1 = np.linspace(-1, 1, 10).astype(np.float32)
        x2 = np.linspace(-1, 1, 10).astype(np.float32)
        eff = model.get_interaction_effect(0, x1, x2)
        assert eff.shape == (10,)


class TestContextPresentGA2M:
    def test_forward_with_context(self):
        model = ContextPresentGA2M(
            num_item_features=4,
            context_feature_specs=[
                {"type": "numerical"},
                {"type": "categorical", "vocab_size": 10, "embed_dim": 8},
            ],
            item_hidden=[8, 4],
            context_hidden=[8, 4],
        )
        B, L = 4, 6
        x = torch.randn(B, L, 4)
        q = [torch.randn(B), torch.randint(0, 10, (B,))]
        out = model(x, q)
        assert out.shape == (B, L)

    def test_forward_no_context_fallback(self):
        model = ContextPresentGA2M(
            num_item_features=4,
            context_feature_specs=[{"type": "numerical"}],
            item_hidden=[8, 4],
            context_hidden=[8, 4],
        )
        x = torch.randn(4, 6, 4)
        out = model(x, q=None)
        assert out.shape == (4, 6)


class TestSubmodularRankingGAM:
    def _make_model(self):
        specs = [
            {"name": "cat_novelty", "type": "category_novelty", "column": 8, "x_min": 0, "x_max": 1},
            {"name": "price_spread", "type": "price_spread", "column": 3, "x_min": 0, "x_max": 3},
        ]
        return SubmodularRankingGAM(
            num_item_features=8, groupwise_specs=specs,
            item_hidden=[8, 4], num_knots=4,
        )

    def test_forward_shape(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = self._make_model()
        out = model(X)
        assert out.shape == (X.shape[0], X.shape[1])

    def test_greedy_rerank(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            order = model.greedy_rerank(X[:2], k=5)
        assert order.shape == (2, 5)
        # All indices should be valid (0..L-1) or -1
        assert (order >= -1).all()
        assert (order < X.shape[1]).all()

    def test_explain(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            order = model.greedy_rerank(X[:1], k=3)
        order_list = [o for o in order[0].tolist() if o >= 0]
        explanations = model.explain(X[0], order_list)
        assert len(explanations) == len(order_list)
        assert "base_score" in explanations[0]
        assert "diversity_breakdown" in explanations[0]

    def test_base_scores(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = self._make_model()
        base = model.base_scores(X)
        assert base.shape == (X.shape[0], X.shape[1])


class TestMultiObjectiveRankingGAM:
    def _make_model(self):
        objectives = [
            {
                "name": "relevance", "type": "pointwise",
                "features": list(range(8)), "tower": "mlp", "weight": 0.6,
            },
            {
                "name": "revenue", "type": "pointwise",
                "features": [3], "tower": "monotone",
                "x_min": 0.0, "x_max": 1.0, "weight": 0.2,
            },
            {
                "name": "diversity", "type": "groupwise",
                "groupwise_specs": [
                    {"name": "cat_nov", "type": "category_novelty", "column": 8, "x_min": 0, "x_max": 1},
                ],
                "weight": 0.2,
            },
        ]
        return MultiObjectiveRankingGAM(
            objectives=objectives, num_item_features=8,
            hidden_dims=[8, 4], num_knots=4,
        )

    def test_forward_shape(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = self._make_model()
        out = model(X)
        assert out.shape == (X.shape[0], X.shape[1])

    def test_greedy_rerank(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            order = model.greedy_rerank(X[:2], k=5)
        assert order.shape == (2, 5)

    def test_greedy_rerank_with_dict_weights(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = self._make_model()
        model.eval()
        weights = {"relevance": 0.3, "revenue": 0.5, "diversity": 0.2}
        with torch.no_grad():
            order = model.greedy_rerank(X[:1], k=3, weights=weights)
        assert order.shape[0] == 1

    def test_explain(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            order = model.greedy_rerank(X[:1], k=3)
        order_list = [o for o in order[0].tolist() if o >= 0]
        expl = model.explain(X[0], order_list)
        assert len(expl) == len(order_list)
        assert "objectives" in expl[0]
        assert "relevance" in expl[0]["objectives"]

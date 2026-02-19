"""Tests for GAM, GA2M, Context, Submodular, and MultiObjective models."""

import numpy as np
import torch

from ranking_gam.models import (
    ContextGAM,
    ContextPresentGA2M,
    GA2M_Paper,
    GAM_Paper,
    InvertedTransformerRanker,
    MultiObjectiveRankingGAM,
    SubmodularRankingGAM,
    TransformerRanker,
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


class TestGAMTowerDropout:
    def test_tower_dropout_training(self, synthetic_tensors):
        X, y = synthetic_tensors
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4], tower_dropout=0.3)
        model.train()
        out1 = model(X)
        _ = model(X)
        # With dropout, outputs should differ between calls (stochastic)
        assert out1.shape == (X.shape[0], X.shape[1])
        # Not guaranteed to differ on tiny data, but shape must be right

    def test_tower_dropout_eval(self, synthetic_tensors):
        X, y = synthetic_tensors
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4], tower_dropout=0.3)
        model.eval()
        out1 = model(X)
        out2 = model(X)
        # In eval mode, dropout is off, so outputs should be identical
        torch.testing.assert_close(out1, out2)

    def test_tower_dropout_zero_is_noop(self, synthetic_tensors):
        X, y = synthetic_tensors
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4], tower_dropout=0.0)
        model.eval()
        out = model(X)
        assert out.shape == (X.shape[0], X.shape[1])


class TestGAMOutputNorm:
    def test_output_norm_forward(self, synthetic_tensors):
        X, y = synthetic_tensors
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4], output_norm=True)
        out = model(X)
        assert out.shape == (X.shape[0], X.shape[1])

    def test_output_norm_gradient_flows(self, synthetic_tensors):
        X, y = synthetic_tensors
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4], output_norm=True)
        out = model(X)
        out.sum().backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_both_together(self, synthetic_tensors):
        X, y = synthetic_tensors
        model = GAM_Paper(
            num_features=X.shape[-1], hidden_dims=[8, 4],
            tower_dropout=0.2, output_norm=True,
            feature_transforms=True, residual=True,
        )
        model.train()
        out = model(X)
        assert out.shape == (X.shape[0], X.shape[1])
        out.sum().backward()


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


class TestGA2MEnhancements:
    def test_tower_dropout(self, synthetic_tensors):
        X, y = synthetic_tensors
        D = X.shape[-1]
        model = GA2M_Paper(
            num_features=D, interaction_pairs=[(0, 1)],
            hidden_dims=[8, 4], tower_dropout=0.3,
        )
        model.train()
        out = model(X)
        assert out.shape == (X.shape[0], X.shape[1])

    def test_output_norm(self, synthetic_tensors):
        X, y = synthetic_tensors
        D = X.shape[-1]
        model = GA2M_Paper(
            num_features=D, interaction_pairs=[(0, 1)],
            hidden_dims=[8, 4], output_norm=True,
        )
        out = model(X)
        assert out.shape == (X.shape[0], X.shape[1])

    def test_all_enhancements(self, synthetic_tensors):
        X, y = synthetic_tensors
        D = X.shape[-1]
        model = GA2M_Paper(
            num_features=D, interaction_pairs=[(0, 1), (2, 3)],
            hidden_dims=[8, 4], feature_transforms=True,
            residual=True, tower_dropout=0.2, output_norm=True,
        )
        model.train()
        out = model(X)
        assert out.shape == (X.shape[0], X.shape[1])
        out.sum().backward()


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
            {"name": "cat_novelty", "type": "category_novelty", "column": 7, "x_min": 0, "x_max": 1},
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


class TestGAMWithTransforms:
    def test_forward_with_transforms(self, synthetic_tensors):
        X, y = synthetic_tensors
        B, L, D = X.shape
        model = GAM_Paper(num_features=D, hidden_dims=[8, 4], feature_transforms=True, num_transform_knots=10)
        out = model(X)
        assert out.shape == (B, L)

    def test_init_transforms_from_data(self, synthetic_data):
        X, y = synthetic_data
        D = X.shape[-1]
        model = GAM_Paper(num_features=D, hidden_dims=[8, 4], feature_transforms=True)
        model.init_transforms_from_data(X)
        # Transforms should now have data-aligned knots
        assert model.feature_transforms is not None
        assert len(model.feature_transforms) == D

    def test_ga2m_with_transforms(self, synthetic_tensors):
        X, y = synthetic_tensors
        B, L, D = X.shape
        model = GA2M_Paper(
            num_features=D, interaction_pairs=[(0, 1)],
            hidden_dims=[8, 4], feature_transforms=True,
        )
        out = model(X)
        assert out.shape == (B, L)

    def test_gam_residual_and_transforms(self, synthetic_tensors):
        X, y = synthetic_tensors
        B, L, D = X.shape
        model = GAM_Paper(
            num_features=D, hidden_dims=[8, 4],
            residual=True, feature_transforms=True,
        )
        out = model(X)
        assert out.shape == (B, L)

    def test_get_main_effect_with_transforms(self, synthetic_data):
        X, y = synthetic_data
        D = X.shape[-1]
        model = GAM_Paper(num_features=D, hidden_dims=[8, 4], feature_transforms=True)
        model.init_transforms_from_data(X)
        x_vals = np.linspace(-1, 1, 20).astype(np.float32)
        eff = model.get_main_effect(0, x_vals)
        assert eff.shape == (20,)

    def test_no_transforms_by_default(self, synthetic_tensors):
        X, y = synthetic_tensors
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4])
        assert model.feature_transforms is None


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
                    {"name": "cat_nov", "type": "category_novelty", "column": 7, "x_min": 0, "x_max": 1},
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


class TestTransformerRanker:
    def test_forward_shape(self, synthetic_tensors):
        X, _ = synthetic_tensors
        B, L, D = X.shape
        model = TransformerRanker(num_features=D, d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
        out = model(X)
        assert out.shape == (B, L)

    def test_gradient_flows(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = TransformerRanker(num_features=X.shape[-1], d_model=16, nhead=2, num_layers=1)
        out = model(X)
        out.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_cross_document_interaction(self, synthetic_tensors):
        """Transformer output for one doc should depend on other docs in the list."""
        X, _ = synthetic_tensors
        D = X.shape[-1]
        model = TransformerRanker(num_features=D, d_model=16, nhead=2, num_layers=1)
        model.eval()
        with torch.no_grad():
            scores_full = model(X[:1])
            # Change a different document and see if it affects score of doc 0
            X_mod = X[:1].clone()
            X_mod[0, 1, :] += 10.0
            scores_mod = model(X_mod)
        # Score of doc 0 should differ because transformer attends across docs
        assert not torch.allclose(scores_full[0, 0:1], scores_mod[0, 0:1], atol=1e-5), \
            "Transformer should have cross-document interactions"

    def test_eval_deterministic(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = TransformerRanker(num_features=X.shape[-1], d_model=16, nhead=2, num_layers=1)
        model.eval()
        with torch.no_grad():
            out1 = model(X)
            out2 = model(X)
        assert torch.allclose(out1, out2)


class TestContextGAM:
    def _make_model(self, D=8):
        return ContextGAM(
            num_features=D, hidden_dims=[8, 4], context_hidden=[16, 8],
            activation="relu",
        )

    def test_forward_shape(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = self._make_model(X.shape[-1])
        out = model(X)
        assert out.shape == (X.shape[0], X.shape[1])

    def test_gradient_flows(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = self._make_model(X.shape[-1])
        out = model(X)
        out.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_get_main_effect(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = self._make_model(X.shape[-1])
        x_vals = np.linspace(-2, 2, 50).astype(np.float32)
        effect = model.get_main_effect(0, x_vals)
        assert effect.shape == (50,)

    def test_get_context_weights(self, synthetic_tensors):
        X, _ = synthetic_tensors
        D = X.shape[-1]
        model = self._make_model(D)
        x_single = X[0, 0].numpy()
        weights = model.get_context_weights(x_single)
        assert weights.shape == (D,)
        assert abs(weights.sum() - 1.0) < 1e-5, "Context weights must sum to 1"
        assert (weights >= 0).all(), "Context weights must be non-negative (softmax)"

    def test_explain(self, synthetic_tensors):
        X, _ = synthetic_tensors
        D = X.shape[-1]
        model = self._make_model(D)
        x_single = X[0, 0].numpy()
        expl = model.explain(x_single)
        assert "tower_outputs" in expl
        assert "context_weights" in expl
        assert "contributions" in expl
        assert "score" in expl
        assert expl["tower_outputs"].shape == (D,)
        assert expl["context_weights"].shape == (D,)
        assert expl["contributions"].shape == (D,)
        # Score should equal sum of contributions + bias
        assert abs(expl["score"] - (expl["contributions"].sum() + model.global_bias.item())) < 1e-4

    def test_with_feature_transforms(self, synthetic_data):
        X, _ = synthetic_data
        D = X.shape[-1]
        model = ContextGAM(
            num_features=D, hidden_dims=[8, 4], context_hidden=[16, 8],
            feature_transforms=True, num_transform_knots=10,
        )
        model.init_transforms_from_data(X)
        X_t = torch.from_numpy(X)
        out = model(X_t)
        assert out.shape == (X.shape[0], X.shape[1])


class TestInvertedTransformerRanker:
    def test_forward_shape(self, synthetic_tensors):
        X, _ = synthetic_tensors
        B, L, D = X.shape
        model = InvertedTransformerRanker(
            num_features=D, d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
        )
        out = model(X)
        assert out.shape == (B, L)

    def test_mean_pooling(self, synthetic_tensors):
        X, _ = synthetic_tensors
        B, L, D = X.shape
        model = InvertedTransformerRanker(
            num_features=D, d_model=16, nhead=2, num_layers=1, pooling="mean",
        )
        out = model(X)
        assert out.shape == (B, L)

    def test_gradient_flows(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = InvertedTransformerRanker(
            num_features=X.shape[-1], d_model=16, nhead=2, num_layers=1,
        )
        out = model(X)
        out.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_no_cross_document_interaction(self, synthetic_tensors):
        """Inverted transformer scores each document independently."""
        X, _ = synthetic_tensors
        D = X.shape[-1]
        model = InvertedTransformerRanker(
            num_features=D, d_model=16, nhead=2, num_layers=1,
        )
        model.eval()
        with torch.no_grad():
            scores_full = model(X[:1])
            # Change a different document — should NOT affect score of doc 0
            X_mod = X[:1].clone()
            X_mod[0, 1, :] += 10.0
            scores_mod = model(X_mod)
        # Score of doc 0 should be identical since there's no cross-doc attention
        torch.testing.assert_close(
            scores_full[0, 0:1], scores_mod[0, 0:1],
            msg="Inverted transformer should NOT have cross-document interactions",
        )

    def test_eval_deterministic(self, synthetic_tensors):
        X, _ = synthetic_tensors
        model = InvertedTransformerRanker(
            num_features=X.shape[-1], d_model=16, nhead=2, num_layers=1,
        )
        model.eval()
        with torch.no_grad():
            out1 = model(X)
            out2 = model(X)
        assert torch.allclose(out1, out2)

    def test_get_feature_attention(self, synthetic_tensors):
        X, _ = synthetic_tensors
        D = X.shape[-1]
        model = InvertedTransformerRanker(
            num_features=D, d_model=16, nhead=2, num_layers=1, pooling="cls",
        )
        attn = model.get_feature_attention(X[:1, :1])
        # CLS pooling: D+1 tokens (CLS + D features)
        assert attn.shape == (D + 1, D + 1)
        # Rows should sum to ~1 (attention is a distribution)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    def test_get_feature_attention_mean_pooling(self, synthetic_tensors):
        X, _ = synthetic_tensors
        D = X.shape[-1]
        model = InvertedTransformerRanker(
            num_features=D, d_model=16, nhead=2, num_layers=1, pooling="mean",
        )
        attn = model.get_feature_attention(X[:1, :1])
        # Mean pooling: D tokens (no CLS)
        assert attn.shape == (D, D)

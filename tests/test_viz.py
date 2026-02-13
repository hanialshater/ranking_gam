"""Tests for visualization utilities."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from ranking_gam.models import GAM_Paper, SubmodularRankingGAM  # noqa: E402
from ranking_gam.viz.plots import (
    plot_diversity_curves,
    plot_objective_tradeoffs,
    plot_pareto_front,
    plot_response_curves,
    plot_spider,
)


class TestPlotResponseCurves:
    def test_basic(self):
        model = GAM_Paper(num_features=4, hidden_dims=[8, 4])
        fig = plot_response_curves(model, top_k=4)
        assert fig is not None
        assert len(fig.axes) >= 4

    def test_with_data(self):
        model = GAM_Paper(num_features=4, hidden_dims=[8, 4])
        data = np.random.randn(100, 4).astype(np.float32)
        fig = plot_response_curves(model, data=data, top_k=4)
        assert fig is not None

    def test_with_feature_names(self):
        model = GAM_Paper(num_features=3, hidden_dims=[8, 4])
        fig = plot_response_curves(model, feature_names=["a", "b", "c"], top_k=3)
        assert fig is not None

    def test_top_k_limits(self):
        model = GAM_Paper(num_features=8, hidden_dims=[8, 4])
        fig = plot_response_curves(model, top_k=3)
        # Should only show 3 subplots (plus possibly empty ones from grid)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 3


class TestPlotDiversityCurves:
    def test_basic(self):
        specs = [
            {"name": "cat_novelty", "type": "category_novelty", "column": 0, "x_min": 0, "x_max": 1},
            {"name": "price_spread", "type": "price_spread", "column": 1, "x_min": 0, "x_max": 3},
        ]
        model = SubmodularRankingGAM(
            num_item_features=4, groupwise_specs=specs,
            item_hidden=[8, 4], num_knots=4,
        )
        fig = plot_diversity_curves(model)
        assert fig is not None
        assert len(fig.axes) == 2


class TestPlotSpider:
    def test_single_strategy(self):
        strategies = {
            "Test": {"ndcg": 0.5, "precision": 0.3, "cat_coverage": 0.8, "ild": 0.4},
        }
        fig = plot_spider(strategies, metrics=["ndcg", "precision", "cat_coverage", "ild"])
        assert fig is not None

    def test_multiple_strategies(self):
        strategies = {
            "A": {"ndcg": 0.5, "precision": 0.3, "cat_coverage": 0.8},
            "B": {"ndcg": 0.7, "precision": 0.6, "cat_coverage": 0.4},
        }
        fig = plot_spider(strategies, metrics=["ndcg", "precision", "cat_coverage"])
        assert fig is not None

    def test_calibration_applied(self):
        """Calibrated normalization should map values to [0, 1] based on absolute bounds."""
        strategies = {
            "X": {"ndcg": 0.5, "price_std": 1.0},
        }
        fig = plot_spider(
            strategies, metrics=["ndcg", "price_std"],
            calibration={"price_std": (0, 2)},
        )
        assert fig is not None

    def test_normalize_false(self):
        strategies = {
            "A": {"ndcg": 0.5, "precision": 0.3},
        }
        fig = plot_spider(strategies, metrics=["ndcg", "precision"], normalize=False)
        assert fig is not None

    def test_custom_calibration_overrides_defaults(self):
        strategies = {
            "A": {"ndcg": 5.0},
        }
        # Default NDCG cal is (0, 1) so 5.0 clips to 1.0
        # Custom cal (0, 10) maps 5.0 to 0.5
        fig = plot_spider(
            strategies, metrics=["ndcg"],
            calibration={"ndcg": (0.0, 10.0)},
        )
        assert fig is not None


class TestPlotParetoFront:
    def test_basic(self):
        scenarios = {
            "A": {"ndcg": 0.8, "ild": 0.2},
            "B": {"ndcg": 0.5, "ild": 0.7},
            "C": {"ndcg": 0.6, "ild": 0.5},
        }
        fig = plot_pareto_front(scenarios)
        assert fig is not None

    def test_custom_metrics(self):
        scenarios = {
            "A": {"precision": 0.8, "cat_coverage": 0.3},
            "B": {"precision": 0.5, "cat_coverage": 0.9},
        }
        fig = plot_pareto_front(scenarios, x_metric="precision", y_metric="cat_coverage")
        assert fig is not None


class TestPlotObjectiveTradeoffs:
    def test_basic(self):
        scenarios = {
            "High Rel": {"ndcg": 0.8, "ild": 0.2, "cat_coverage": 0.3},
            "High Div": {"ndcg": 0.4, "ild": 0.9, "cat_coverage": 0.8},
        }
        fig = plot_objective_tradeoffs(scenarios)
        assert fig is not None

    def test_custom_objectives(self):
        scenarios = {
            "A": {"x": 0.5, "y": 0.3},
            "B": {"x": 0.8, "y": 0.7},
        }
        fig = plot_objective_tradeoffs(scenarios, objectives=["x", "y"])
        assert fig is not None

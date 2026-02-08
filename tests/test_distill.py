"""Tests for PWL distillation."""

import numpy as np

from ranking_gam.distill import greedy_knot_selection, distill_to_pwl, pwl_predict
from ranking_gam.models import GAM_Paper


class TestGreedyKnotSelection:
    def test_linear_function(self):
        """Should approximate a line nearly perfectly."""
        x = np.linspace(0, 1, 200).astype(np.float64)
        y = 2.0 * x + 1.0
        x_knots, y_knots = greedy_knot_selection(x, y, num_knots=3)
        y_approx = np.interp(x, x_knots, y_knots)
        mse = np.mean((y - y_approx) ** 2)
        assert mse < 1e-6

    def test_num_knots_respected(self):
        x = np.linspace(-1, 1, 100).astype(np.float64)
        y = x ** 2
        x_knots, y_knots = greedy_knot_selection(x, y, num_knots=5)
        assert len(x_knots) == 5
        assert len(y_knots) == 5

    def test_knots_sorted(self):
        x = np.linspace(0, 1, 100).astype(np.float64)
        y = np.sin(x * np.pi)
        x_knots, _ = greedy_knot_selection(x, y, num_knots=5)
        assert (np.diff(x_knots) >= 0).all()


class TestDistillToPWL:
    def test_roundtrip(self):
        """Distill a small GAM and verify PWL predict works."""
        model = GAM_Paper(num_features=4, hidden_dims=[8, 4])
        train_X = np.random.randn(20, 6, 4).astype(np.float32)
        pwl = distill_to_pwl(model, num_knots=3, train_X=train_X)

        assert "main_effects" in pwl
        assert len(pwl["main_effects"]) == 4
        assert "bias" in pwl

        # Predict
        scores = pwl_predict(pwl, train_X)
        assert scores.shape == (20, 6)
        assert np.all(np.isfinite(scores))

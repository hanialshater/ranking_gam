"""Tests for tower architectures."""

import torch
from ranking_gam.models.towers import PaperTower, ConcavePWL, MonotonePWL


class TestPaperTower:
    def test_output_shape(self):
        tower = PaperTower(in_dim=1, hidden_dims=[16, 8])
        x = torch.randn(32, 1)
        out = tower(x)
        assert out.shape == (32, 1)

    def test_input_dim_2(self):
        tower = PaperTower(in_dim=2, hidden_dims=[16, 8])
        x = torch.randn(32, 2)
        out = tower(x)
        assert out.shape == (32, 1)

    def test_dropout(self):
        tower = PaperTower(in_dim=1, hidden_dims=[16, 8], dropout=0.5)
        x = torch.randn(32, 1)
        tower.train()
        out_train = tower(x)
        tower.eval()
        out_eval = tower(x)
        assert out_train.shape == out_eval.shape == (32, 1)


class TestConcavePWL:
    def test_output_shape(self):
        pwl = ConcavePWL(num_knots=5, x_min=0.0, x_max=1.0)
        x = torch.linspace(0, 1, 20)
        out = pwl(x)
        assert out.shape == (20,)

    def test_monotone_nondecreasing(self):
        pwl = ConcavePWL(num_knots=8, x_min=0.0, x_max=1.0)
        x = torch.linspace(0, 1, 100)
        with torch.no_grad():
            y = pwl(x)
        diffs = y[1:] - y[:-1]
        assert (diffs >= -1e-6).all(), "ConcavePWL must be non-decreasing"

    def test_concave(self):
        pwl = ConcavePWL(num_knots=8)
        slopes = pwl.get_slopes().detach()
        diffs = slopes[:-1] - slopes[1:]
        assert (diffs >= -1e-6).all(), "Slopes must be non-increasing (concave)"

    def test_slopes_nonnegative(self):
        pwl = ConcavePWL(num_knots=10)
        slopes = pwl.get_slopes().detach()
        assert (slopes >= -1e-6).all(), "All slopes must be >= 0"

    def test_get_curve(self):
        pwl = ConcavePWL(num_knots=5)
        x, y = pwl.get_curve(n_points=50)
        assert x.shape == (50,)
        assert y.shape == (50,)

    def test_gradient_flows(self):
        pwl = ConcavePWL(num_knots=5)
        x = torch.tensor([0.3, 0.5, 0.7], requires_grad=False)
        out = pwl(x)
        loss = out.sum()
        loss.backward()
        assert pwl.raw_params.grad is not None


class TestMonotonePWL:
    def test_monotone_nondecreasing(self):
        pwl = MonotonePWL(num_knots=8, x_min=0.0, x_max=1.0)
        x = torch.linspace(0, 1, 100)
        with torch.no_grad():
            y = pwl(x)
        diffs = y[1:] - y[:-1]
        assert (diffs >= -1e-6).all(), "MonotonePWL must be non-decreasing"

    def test_slopes_nonneg(self):
        pwl = MonotonePWL(num_knots=10)
        slopes = pwl.get_slopes().detach()
        assert (slopes >= -1e-6).all()

    def test_not_necessarily_concave(self):
        """MonotonePWL slopes are independent -- concavity is NOT enforced."""
        torch.manual_seed(99)
        pwl = MonotonePWL(num_knots=5)
        # Manually set raw_params so slopes are not non-increasing
        pwl.raw_params.data = torch.tensor([-2.0, 0.0, 2.0, 0.0, -1.0])
        slopes = pwl.get_slopes().detach()
        # Slopes should be non-negative but NOT necessarily non-increasing
        assert (slopes >= -1e-6).all()

"""Tests for tower architectures."""

import numpy as np
import torch
from ranking_gam.models.towers import PaperTower, ConcavePWL, MonotonePWL, LearnableMonotoneTransform


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


class TestPaperTowerResidual:
    def test_residual_shape(self):
        tower = PaperTower(in_dim=1, hidden_dims=[16, 8], residual=True)
        x = torch.randn(32, 1)
        out = tower(x)
        assert out.shape == (32, 1)

    def test_residual_dim2(self):
        tower = PaperTower(in_dim=2, hidden_dims=[16, 8], residual=True)
        x = torch.randn(32, 2)
        out = tower(x)
        assert out.shape == (32, 1)

    def test_residual_gradient_flows(self):
        tower = PaperTower(in_dim=1, hidden_dims=[16, 8], residual=True)
        x = torch.randn(32, 1)
        out = tower(x)
        out.sum().backward()
        assert tower.skip.weight.grad is not None


class TestPaperTowerInputNorm:
    def test_input_norm_shape(self):
        tower = PaperTower(in_dim=1, hidden_dims=[16, 8], input_norm=True)
        x = torch.randn(32, 1)
        tower.train()
        out = tower(x)
        assert out.shape == (32, 1)

    def test_input_norm_with_residual(self):
        tower = PaperTower(in_dim=2, hidden_dims=[16, 8], residual=True, input_norm=True)
        x = torch.randn(32, 2)
        tower.train()
        out = tower(x)
        assert out.shape == (32, 1)


class TestLearnableMonotoneTransform:
    def test_output_shape(self):
        t = LearnableMonotoneTransform(num_knots=10)
        x = torch.randn(50)
        out = t(x)
        assert out.shape == (50,)

    def test_output_shape_2d(self):
        t = LearnableMonotoneTransform(num_knots=10)
        x = torch.randn(32, 1)
        out = t(x)
        assert out.shape == (32, 1)

    def test_output_range_01(self):
        t = LearnableMonotoneTransform(num_knots=10)
        x = torch.linspace(0, 1, 100)
        with torch.no_grad():
            out = t(x)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_monotone_nondecreasing(self):
        t = LearnableMonotoneTransform(num_knots=15)
        x = torch.linspace(0, 1, 200)
        with torch.no_grad():
            y = t(x)
        diffs = y[1:] - y[:-1]
        assert (diffs >= -1e-6).all(), "Transform must be non-decreasing"

    def test_init_from_data(self):
        t = LearnableMonotoneTransform(num_knots=10)
        data = np.random.randn(1000).astype(np.float32)
        t.init_from_data(data)
        # Knots should span data range
        assert t.x_knots[0].item() <= data.min() + 1e-5
        assert t.x_knots[-1].item() >= data.max() - 1e-5

    def test_init_from_data_tensor(self):
        t = LearnableMonotoneTransform(num_knots=10)
        data = torch.randn(500)
        t.init_from_data(data)
        assert t.x_knots[0].item() <= data.min().item() + 1e-5

    def test_monotone_after_init(self):
        t = LearnableMonotoneTransform(num_knots=20)
        data = np.random.randn(1000).astype(np.float32)
        t.init_from_data(data)
        x = torch.linspace(data.min(), data.max(), 300)
        with torch.no_grad():
            y = t(x)
        diffs = y[1:] - y[:-1]
        assert (diffs >= -1e-6).all(), "Must stay monotone after init_from_data"

    def test_gradient_flows(self):
        t = LearnableMonotoneTransform(num_knots=10)
        x = torch.tensor([0.2, 0.5, 0.8])
        out = t(x)
        out.sum().backward()
        assert t.raw_deltas.grad is not None

    def test_get_curve(self):
        t = LearnableMonotoneTransform(num_knots=10)
        x, y = t.get_curve(n_points=100)
        assert x.shape == (100,)
        assert y.shape == (100,)

    def test_get_y_knots_sum_to_one(self):
        t = LearnableMonotoneTransform(num_knots=10)
        y_knots = t.get_y_knots().detach()
        assert abs(y_knots[0].item()) < 1e-6, "First y-knot should be ~0"
        assert abs(y_knots[-1].item() - 1.0) < 1e-5, "Last y-knot should be ~1"

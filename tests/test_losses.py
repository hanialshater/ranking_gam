"""Tests for all ranking loss functions."""

import pytest
import torch

from ranking_gam.losses import (
    ApproxNDCGLoss,
    DiffSortNDCGLoss,
    LambdaLoss,
    ListMLELoss,
    ListNetLoss,
    PairwiseLoss,
    soft_rank,
)


@pytest.fixture
def pred_and_labels():
    """Predictions and labels with padding (-1)."""
    torch.manual_seed(42)
    B, L = 8, 10
    y_pred = torch.randn(B, L)
    y_true = torch.randint(0, 5, (B, L)).float()
    y_true[:, -2:] = -1.0  # padding
    return y_pred, y_true


class TestApproxNDCG:
    def test_output_scalar(self, pred_and_labels):
        y_pred, y_true = pred_and_labels
        loss = ApproxNDCGLoss(alpha=10.0)(y_pred, y_true)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_perfect_ranking_low_loss(self):
        y_true = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
        y_pred = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        loss = ApproxNDCGLoss(alpha=10.0)(y_pred, y_true)
        assert loss.item() < 0.05

    def test_gradient_flows(self, pred_and_labels):
        y_pred, y_true = pred_and_labels
        y_pred.requires_grad_(True)
        loss = ApproxNDCGLoss()(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None


class TestPairwiseLoss:
    def test_output_scalar(self, pred_and_labels):
        y_pred, y_true = pred_and_labels
        loss = PairwiseLoss(sigma=1.0)(y_pred, y_true)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gradient_flows(self, pred_and_labels):
        y_pred, y_true = pred_and_labels
        y_pred.requires_grad_(True)
        loss = PairwiseLoss()(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None


class TestListMLELoss:
    def test_output_scalar(self, pred_and_labels):
        y_pred, y_true = pred_and_labels
        loss = ListMLELoss()(y_pred, y_true)
        assert loss.shape == ()

    def test_all_padded(self):
        y_pred = torch.randn(4, 5)
        y_true = torch.full((4, 5), -1.0)
        loss = ListMLELoss()(y_pred, y_true)
        assert loss.item() == 0.0


class TestListNetLoss:
    def test_output_scalar(self, pred_and_labels):
        y_pred, y_true = pred_and_labels
        loss = ListNetLoss()(y_pred, y_true)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gradient_flows(self, pred_and_labels):
        y_pred, y_true = pred_and_labels
        y_pred.requires_grad_(True)
        loss = ListNetLoss()(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None

    def test_label_smoothing(self, pred_and_labels):
        y_pred, y_true = pred_and_labels
        loss_no_smooth = ListNetLoss()(y_pred, y_true)
        loss_smooth = ListNetLoss(label_smoothing=0.1)(y_pred, y_true)
        assert loss_smooth.shape == ()
        assert torch.isfinite(loss_smooth)
        # Smoothing should change the loss value
        assert loss_no_smooth.item() != pytest.approx(loss_smooth.item(), abs=1e-6)

    def test_label_smoothing_gradient(self, pred_and_labels):
        y_pred, y_true = pred_and_labels
        y_pred.requires_grad_(True)
        loss = ListNetLoss(label_smoothing=0.2)(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None


class TestLambdaLoss:
    @pytest.mark.parametrize("scheme", [None, "lambdaRank", "ndcgLoss1", "ndcgLoss2", "ndcgLoss2++"])
    def test_all_schemes(self, pred_and_labels, scheme):
        y_pred, y_true = pred_and_labels
        loss = LambdaLoss(weighing_scheme=scheme)(y_pred, y_true)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_ndcg2pp_constructor(self, pred_and_labels):
        y_pred, y_true = pred_and_labels
        loss_fn = LambdaLoss.ndcg2pp(k=5, mu=10.0)
        assert loss_fn.weighing_scheme == "ndcgLoss2++"
        assert loss_fn.k == 5
        assert loss_fn.mu == 10.0
        loss = loss_fn(y_pred, y_true)
        assert loss.shape == ()
        assert torch.isfinite(loss)


class TestDiffSortNDCGLoss:
    def test_output_scalar(self, pred_and_labels):
        y_pred, y_true = pred_and_labels
        loss = DiffSortNDCGLoss(k=5, regularization_strength=1.0)(y_pred, y_true)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gradient_flows(self, pred_and_labels):
        y_pred, y_true = pred_and_labels
        y_pred.requires_grad_(True)
        loss = DiffSortNDCGLoss(k=5)(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None


class TestSoftRank:
    def test_output_shape(self):
        values = torch.randn(4, 10)
        ranks = soft_rank(values, regularization_strength=1.0)
        assert ranks.shape == (4, 10)

    def test_ranks_bounded(self):
        values = torch.randn(4, 10)
        ranks = soft_rank(values, regularization_strength=0.1)
        assert (ranks >= 0.5).all()
        assert (ranks <= 10.5).all()

    def test_ordering(self):
        """Higher scores should get lower (better) ranks."""
        values = torch.tensor([[5.0, 3.0, 1.0]])
        ranks = soft_rank(values, regularization_strength=0.01)
        assert ranks[0, 0] < ranks[0, 1] < ranks[0, 2]

"""Tests for GBDT baseline and residual boosting."""

import numpy as np
import pytest
import torch

lgb = pytest.importorskip("lightgbm", reason="lightgbm not installed")

from ranking_gam.losses import ListNetLoss  # noqa: E402
from ranking_gam.models import GAM_Paper  # noqa: E402
from ranking_gam.training.boosting import (  # noqa: E402
    compute_gbdt_residual_feature,
    train_gbdt_baseline,
    train_gbdt_residual_boost,
)


class TestGBDTBaseline:
    def test_train_and_eval(self, synthetic_data):
        X, y = synthetic_data
        model, ndcg = train_gbdt_baseline(
            X, y, X, y, k=5, n_estimators=10,
        )
        assert model is not None
        assert isinstance(ndcg, float)
        assert 0.0 <= ndcg <= 1.0

    def test_predict_shape(self, synthetic_data):
        X, y = synthetic_data
        model, _ = train_gbdt_baseline(
            X, y, X, y, k=5, n_estimators=10,
        )
        # Predict on a single query
        preds = model.predict(X[0])
        assert preds.shape == (X.shape[1],)


class TestComputeGBDTResidualFeature:
    def test_output_shape(self, synthetic_data):
        X, y = synthetic_data
        model, _ = train_gbdt_baseline(
            X, y, X, y, k=5, n_estimators=10,
        )
        X_boosted = compute_gbdt_residual_feature(model, X)
        B, L, D = X.shape
        assert X_boosted.shape == (B, L, D + 1)

    def test_gbdt_column_normalized(self, synthetic_data):
        X, y = synthetic_data
        model, _ = train_gbdt_baseline(
            X, y, X, y, k=5, n_estimators=10,
        )
        X_boosted = compute_gbdt_residual_feature(model, X)
        gbdt_col = X_boosted[:, :, -1]
        assert gbdt_col.min() >= -1e-6
        assert gbdt_col.max() <= 1.0 + 1e-6

    def test_original_features_preserved(self, synthetic_data):
        X, y = synthetic_data
        model, _ = train_gbdt_baseline(
            X, y, X, y, k=5, n_estimators=10,
        )
        X_boosted = compute_gbdt_residual_feature(model, X)
        np.testing.assert_array_equal(X_boosted[:, :, :-1], X)


class TestResidualBoost:
    def test_end_to_end(self, synthetic_data):
        X, y = synthetic_data
        D = X.shape[-1]

        gam = GAM_Paper(num_features=D, hidden_dims=[8, 4])

        def loader_fn(X_in, y_in):
            from torch.utils.data import DataLoader, TensorDataset
            ds = TensorDataset(torch.from_numpy(X_in), torch.from_numpy(y_in))
            return DataLoader(ds, batch_size=8, shuffle=True), DataLoader(ds, batch_size=8)

        result = train_gbdt_residual_boost(
            gam, X, y, X, y,
            train_loader_fn=loader_fn,
            loss_fn=ListNetLoss(),
            k=5,
            gam_epochs=2,
            boost_epochs=2,
            device=torch.device("cpu"),
        )

        assert "boosted_model" in result
        assert "gbdt_model" in result
        assert "stage1_ndcg" in result
        assert "gbdt_residual_ndcg" in result
        assert "boosted_ndcg" in result
        assert isinstance(result["boosted_ndcg"], float)
        assert isinstance(result["stage1_ndcg"], float)

        # Boosted model should have D+1 features
        assert result["boosted_model"].num_features == D + 1

        # First D towers should be frozen, only magic curve tower trainable
        boosted = result["boosted_model"]
        for j in range(D):
            for param in boosted.towers[j].parameters():
                assert not param.requires_grad, f"Tower {j} should be frozen"
        for param in boosted.towers[D].parameters():
            assert param.requires_grad, "Magic curve tower should be trainable"

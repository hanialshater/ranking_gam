"""Tests for training loops (fast, 1-2 epochs on tiny data)."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ranking_gam.losses import ListNetLoss
from ranking_gam.models import GAM_Paper, SubmodularRankingGAM, MultiObjectiveRankingGAM
from ranking_gam.training import train_model, train_diversity_towers, train_multi_objective


def _make_loaders(X, y, batch_size=8):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=True), DataLoader(ds, batch_size=batch_size)


class TestTrainModel:
    def test_basic_training(self, synthetic_data):
        X, y = synthetic_data
        train_loader, val_loader = _make_loaders(X, y)
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4])
        ndcg = train_model(
            model, train_loader, val_loader, ListNetLoss(),
            epochs=2, patience=5, device=torch.device("cpu"),
        )
        assert isinstance(ndcg, float)
        assert 0.0 <= ndcg <= 1.0


class TestTrainDiversityTowers:
    def test_phase2(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        D_item = X_aug.shape[-1] - 2
        specs = [
            {"name": "cat_nov", "type": "category_novelty", "column": D_item, "x_min": 0, "x_max": 1},
        ]
        model = SubmodularRankingGAM(
            num_item_features=D_item, groupwise_specs=specs,
            item_hidden=[8, 4], num_knots=3,
        )
        # Phase 2 only — train diversity towers
        model = train_diversity_towers(
            model, X_aug, y,
            epochs=1, lr=0.01, k=3, queries_per_epoch=8,
        )
        assert model is not None


class TestTrainMultiObjective:
    def test_basic(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        D_item = X_aug.shape[-1] - 2
        objectives = [
            {
                "name": "rel", "type": "pointwise",
                "features": list(range(D_item)), "tower": "mlp", "weight": 0.7,
            },
            {
                "name": "div", "type": "groupwise",
                "groupwise_specs": [
                    {"name": "cat_nov", "type": "category_novelty", "column": D_item, "x_min": 0, "x_max": 1},
                ],
                "weight": 0.3,
            },
        ]
        model = MultiObjectiveRankingGAM(
            objectives=objectives, num_item_features=D_item,
            hidden_dims=[8, 4], num_knots=3,
        )
        model = train_multi_objective(
            model, X_aug, y,
            epochs=1, lr=0.01, k=3, queries_per_epoch=8,
        )
        assert model is not None

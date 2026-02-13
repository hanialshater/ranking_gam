"""Tests for training loops (fast, 1-2 epochs on tiny data)."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from ranking_gam.losses import ListNetLoss
from ranking_gam.models import GAM_Paper, MultiObjectiveRankingGAM, SubmodularRankingGAM
from ranking_gam.training import train_diversity_towers, train_model, train_multi_objective


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


class TestTrainModelCosineSchedule:
    def test_cosine_schedule(self, synthetic_data):
        X, y = synthetic_data
        train_loader, val_loader = _make_loaders(X, y)
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4])
        ndcg = train_model(
            model, train_loader, val_loader, ListNetLoss(),
            epochs=2, patience=5, device=torch.device("cpu"),
            lr_schedule="cosine",
        )
        assert isinstance(ndcg, float)
        assert 0.0 <= ndcg <= 1.0

    def test_l1_output_reg(self, synthetic_data):
        X, y = synthetic_data
        train_loader, val_loader = _make_loaders(X, y)
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4])
        ndcg = train_model(
            model, train_loader, val_loader, ListNetLoss(),
            epochs=2, patience=5, device=torch.device("cpu"),
            l1_output_reg=0.01,
        )
        assert isinstance(ndcg, float)

    def test_cosine_with_l1(self, synthetic_data):
        X, y = synthetic_data
        train_loader, val_loader = _make_loaders(X, y)
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4])
        ndcg = train_model(
            model, train_loader, val_loader, ListNetLoss(),
            epochs=2, patience=5, device=torch.device("cpu"),
            lr_schedule="cosine", l1_output_reg=0.01,
        )
        assert isinstance(ndcg, float)

    def test_train_with_transforms(self, synthetic_data):
        X, y = synthetic_data
        train_loader, val_loader = _make_loaders(X, y)
        model = GAM_Paper(
            num_features=X.shape[-1], hidden_dims=[8, 4],
            feature_transforms=True,
        )
        model.init_transforms_from_data(X)
        ndcg = train_model(
            model, train_loader, val_loader, ListNetLoss(),
            epochs=2, patience=5, device=torch.device("cpu"),
        )
        assert isinstance(ndcg, float)

    def test_transform_lr_mult(self, synthetic_data):
        X, y = synthetic_data
        train_loader, val_loader = _make_loaders(X, y)
        model = GAM_Paper(
            num_features=X.shape[-1], hidden_dims=[8, 4],
            feature_transforms=True,
        )
        model.init_transforms_from_data(X)
        ndcg = train_model(
            model, train_loader, val_loader, ListNetLoss(),
            epochs=3, patience=5, device=torch.device("cpu"),
            transform_lr_mult=0.1,
        )
        assert isinstance(ndcg, float)

    def test_warmup_epochs(self, synthetic_data):
        X, y = synthetic_data
        train_loader, val_loader = _make_loaders(X, y)
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4])
        ndcg = train_model(
            model, train_loader, val_loader, ListNetLoss(),
            epochs=4, patience=5, device=torch.device("cpu"),
            warmup_epochs=2,
        )
        assert isinstance(ndcg, float)

    def test_weight_decay(self, synthetic_data):
        X, y = synthetic_data
        train_loader, val_loader = _make_loaders(X, y)
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4])
        ndcg = train_model(
            model, train_loader, val_loader, ListNetLoss(),
            epochs=2, patience=5, device=torch.device("cpu"),
            weight_decay=0.01,
        )
        assert isinstance(ndcg, float)

    def test_eval_k(self, synthetic_data):
        X, y = synthetic_data
        train_loader, val_loader = _make_loaders(X, y)
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4])
        ndcg = train_model(
            model, train_loader, val_loader, ListNetLoss(),
            epochs=2, patience=5, device=torch.device("cpu"),
            eval_k=5,
        )
        assert isinstance(ndcg, float)
        assert 0.0 <= ndcg <= 1.0

    def test_label_smoothing_loss(self, synthetic_data):
        X, y = synthetic_data
        train_loader, val_loader = _make_loaders(X, y)
        model = GAM_Paper(num_features=X.shape[-1], hidden_dims=[8, 4])
        ndcg = train_model(
            model, train_loader, val_loader, ListNetLoss(label_smoothing=0.1),
            epochs=2, patience=5, device=torch.device("cpu"),
        )
        assert isinstance(ndcg, float)

    def test_all_enhancements(self, synthetic_data):
        X, y = synthetic_data
        train_loader, val_loader = _make_loaders(X, y)
        model = GAM_Paper(
            num_features=X.shape[-1], hidden_dims=[8, 4],
            feature_transforms=True, residual=True,
        )
        model.init_transforms_from_data(X)
        ndcg = train_model(
            model, train_loader, val_loader, ListNetLoss(label_smoothing=0.1),
            epochs=4, patience=5, device=torch.device("cpu"),
            lr_schedule="cosine", l1_output_reg=0.01,
            transform_lr_mult=0.1, warmup_epochs=1,
            weight_decay=0.01, eval_k=5,
        )
        assert isinstance(ndcg, float)


def _make_submod(X_aug):
    D_item = X_aug.shape[-1] - 2
    specs = [
        {"name": "cat_nov", "type": "category_novelty", "column": D_item, "x_min": 0, "x_max": 1},
    ]
    return SubmodularRankingGAM(
        num_item_features=D_item, groupwise_specs=specs,
        item_hidden=[8, 4], num_knots=3,
    )


def _make_mo(X_aug):
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
    return MultiObjectiveRankingGAM(
        objectives=objectives, num_item_features=D_item,
        hidden_dims=[8, 4], num_knots=3,
    )


_DIV_KW = dict(epochs=1, lr=0.01, k=3, queries_per_epoch=8)


class TestTrainDiversityTowers:
    def test_phase2(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        model = _make_submod(X_aug)
        model = train_diversity_towers(model, X_aug, y, **_DIV_KW)
        assert model is not None

    def test_cosine_schedule(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        model = _make_submod(X_aug)
        model = train_diversity_towers(
            model, X_aug, y, **_DIV_KW, lr_schedule="cosine",
        )
        assert model is not None

    def test_weight_decay(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        model = _make_submod(X_aug)
        model = train_diversity_towers(
            model, X_aug, y, **_DIV_KW, weight_decay=0.01,
        )
        assert model is not None

    def test_warmup(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        model = _make_submod(X_aug)
        model = train_diversity_towers(
            model, X_aug, y, epochs=3, lr=0.01, k=3, queries_per_epoch=8,
            warmup_epochs=1,
        )
        assert model is not None

    def test_grad_clip_disabled(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        model = _make_submod(X_aug)
        model = train_diversity_towers(
            model, X_aug, y, **_DIV_KW, grad_clip=0,
        )
        assert model is not None

    def test_all_enhancements(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        model = _make_submod(X_aug)
        model = train_diversity_towers(
            model, X_aug, y, epochs=3, lr=0.01, k=3, queries_per_epoch=8,
            lr_schedule="cosine", weight_decay=0.01, warmup_epochs=1,
        )
        assert model is not None


class TestTrainMultiObjective:
    def test_basic(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        model = _make_mo(X_aug)
        model = train_multi_objective(model, X_aug, y, **_DIV_KW)
        assert model is not None

    def test_cosine_schedule(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        model = _make_mo(X_aug)
        model = train_multi_objective(
            model, X_aug, y, **_DIV_KW, lr_schedule="cosine",
        )
        assert model is not None

    def test_weight_decay(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        model = _make_mo(X_aug)
        model = train_multi_objective(
            model, X_aug, y, **_DIV_KW, weight_decay=0.01,
        )
        assert model is not None

    def test_warmup(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        model = _make_mo(X_aug)
        model = train_multi_objective(
            model, X_aug, y, epochs=3, lr=0.01, k=3, queries_per_epoch=8,
            warmup_epochs=1,
        )
        assert model is not None

    def test_all_enhancements(self, synthetic_augmented):
        X_aug, y = synthetic_augmented
        model = _make_mo(X_aug)
        model = train_multi_objective(
            model, X_aug, y, epochs=3, lr=0.01, k=3, queries_per_epoch=8,
            lr_schedule="cosine", weight_decay=0.01, warmup_epochs=1,
        )
        assert model is not None

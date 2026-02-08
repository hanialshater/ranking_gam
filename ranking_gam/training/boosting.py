"""
GBDT baseline and boosted GAM models.

Provides:
  - train_gbdt_baseline: train a LambdaMART ranker as a performance ceiling
  - train_gbdt_residual_boost: train GBDT first, then GAM with GBDT score
    as an extra interpretable feature tower
"""

import numpy as np
import torch


def train_gbdt_baseline(train_X, train_y, eval_X, eval_y, k=10, **lgbm_kwargs):
    """
    Train a LightGBM LambdaMART ranker as NDCG baseline.

    Args:
        train_X: [B, L, D] numpy features
        train_y: [B, L] numpy relevance labels (-1 = padding)
        eval_X: [B, L, D] numpy features
        eval_y: [B, L] numpy labels
        k: NDCG cutoff
        **lgbm_kwargs: passed to LGBMRanker (e.g. n_estimators, num_leaves)

    Returns:
        (model, train_ndcg, eval_ndcg) -- fitted LGBMRanker and NDCG scores
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "lightgbm is required for GBDT baseline. "
            "Install with: pip install lightgbm>=4.0"
        )

    from ..metrics import compute_ndcg

    # Flatten [B, L, D] -> [B*L, D] with group structure
    B_tr, L_tr, D = train_X.shape
    X_flat = train_X.reshape(-1, D)
    y_flat = train_y.reshape(-1).copy()

    # Build group sizes (LightGBM expects list of query sizes)
    valid_mask_tr = train_y.reshape(-1) >= 0
    groups_train = []
    offset = 0
    X_train_list = []
    y_train_list = []
    for qi in range(B_tr):
        valid = train_y[qi] >= 0
        n_valid = valid.sum()
        if n_valid > 0:
            X_train_list.append(train_X[qi][valid])
            y_train_list.append(train_y[qi][valid])
            groups_train.append(int(n_valid))

    X_tr = np.concatenate(X_train_list, axis=0)
    y_tr = np.concatenate(y_train_list, axis=0)

    B_ev, L_ev, _ = eval_X.shape
    X_eval_list = []
    y_eval_list = []
    groups_eval = []
    for qi in range(B_ev):
        valid = eval_y[qi] >= 0
        n_valid = valid.sum()
        if n_valid > 0:
            X_eval_list.append(eval_X[qi][valid])
            y_eval_list.append(eval_y[qi][valid])
            groups_eval.append(int(n_valid))

    X_ev = np.concatenate(X_eval_list, axis=0)
    y_ev = np.concatenate(y_eval_list, axis=0)

    defaults = dict(
        objective="lambdarank",
        metric="ndcg",
        eval_at=k,
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.1,
        verbose=-1,
    )
    defaults.update(lgbm_kwargs)

    ranker = lgb.LGBMRanker(**defaults)
    ranker.fit(
        X_tr, y_tr, group=groups_train,
        eval_set=[(X_ev, y_ev)], eval_group=[groups_eval],
    )

    # Compute NDCG on eval set using our metric (for comparable numbers)
    eval_preds = np.zeros_like(eval_y, dtype=np.float32)
    offset = 0
    for qi in range(B_ev):
        valid = eval_y[qi] >= 0
        n_valid = valid.sum()
        if n_valid > 0:
            pred = ranker.predict(eval_X[qi][valid])
            eval_preds[qi][valid] = pred
            eval_preds[qi][~valid] = -1e9
        else:
            eval_preds[qi] = -1e9
        offset += n_valid

    eval_ndcg = compute_ndcg(
        torch.from_numpy(eval_preds), torch.from_numpy(eval_y.astype(np.float32)), k=k
    )

    print(f"  GBDT (LambdaMART): NDCG@{k} = {eval_ndcg:.4f}")
    return ranker, float(eval_ndcg)


def compute_gbdt_residual_feature(gbdt_model, X):
    """
    Compute GBDT predicted scores as an extra feature column.

    Args:
        gbdt_model: fitted LGBMRanker
        X: [B, L, D] numpy features

    Returns:
        X_boosted: [B, L, D+1] numpy features with GBDT score appended
    """
    B, L, D = X.shape
    gbdt_scores = np.zeros((B, L), dtype=np.float32)

    for qi in range(B):
        # Predict for all items (padding handled by the model downstream)
        gbdt_scores[qi] = gbdt_model.predict(X[qi])

    # Normalize GBDT scores to [0, 1] for better tower learning
    s_min = gbdt_scores.min()
    s_max = gbdt_scores.max()
    if s_max > s_min:
        gbdt_scores = (gbdt_scores - s_min) / (s_max - s_min)

    X_boosted = np.zeros((B, L, D + 1), dtype=X.dtype)
    X_boosted[:, :, :D] = X
    X_boosted[:, :, D] = gbdt_scores

    return X_boosted


def train_gbdt_residual_boost(
    model, train_X, train_y, eval_X, eval_y,
    train_loader_fn, loss_fn, k=10,
    gam_epochs=30, boost_epochs=15,
    device=None, **train_kwargs,
):
    """
    Two-stage boosting: GBDT first, then GAM with GBDT score tower.

    Stage 1: Train GBDT (LambdaMART) on raw features
    Stage 2: Append normalized GBDT scores as feature D+1, train GAM on D+1

    The GBDT score gets its own interpretable tower -- you can plot f_gbdt(x)
    to see how much the model relies on the black-box signal.

    Args:
        model: GAM_Paper (or similar) with num_features=D (used to infer config)
        train_X: [B, L, D] numpy features
        train_y: [B, L] numpy labels
        eval_X: [B, L, D] numpy features
        eval_y: [B, L] numpy labels
        train_loader_fn: callable(X, y) -> (train_loader, val_loader)
        loss_fn: ranking loss module
        k: NDCG cutoff
        gam_epochs: epochs for GAM training (stage 2)
        boost_epochs: (unused, kept for API compat)
        device: torch device
        **train_kwargs: extra kwargs for train_model

    Returns:
        dict with:
            "boosted_model": GAM with D+1 features (original + GBDT score)
            "gbdt_model": fitted LGBMRanker
            "gbdt_ndcg": GBDT-only NDCG
            "boosted_ndcg": GAM+GBDT tower NDCG
    """
    from .trainer import train_model
    from ..models import GAM_Paper

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D = train_X.shape[-1]

    # --- Stage 1: Train GBDT on raw features ---
    print("\n  Stage 1: Training GBDT (LambdaMART)...")
    gbdt_model, gbdt_ndcg = train_gbdt_baseline(
        train_X, train_y, eval_X, eval_y, k=k,
    )

    # --- Stage 2: Train GAM with GBDT score as extra feature tower ---
    print("\n  Stage 2: Training GAM with GBDT score tower (D+1 features)...")
    train_X_boosted = compute_gbdt_residual_feature(gbdt_model, train_X)
    eval_X_boosted = compute_gbdt_residual_feature(gbdt_model, eval_X)

    # Infer model config from the original model
    tower_hidden = []
    for layer in model.towers[0].net:
        if isinstance(layer, torch.nn.Linear) and layer.out_features != 1:
            tower_hidden.append(layer.out_features)

    has_transforms = model.feature_transforms is not None
    has_residual = model.towers[0].skip is not None

    boosted_model = GAM_Paper(
        num_features=D + 1,
        hidden_dims=tower_hidden,
        residual=has_residual,
        feature_transforms=has_transforms,
    )
    if has_transforms:
        boosted_model.init_transforms_from_data(train_X_boosted)

    train_loader_b, val_loader_b = train_loader_fn(train_X_boosted, train_y)
    boosted_ndcg = train_model(
        boosted_model, train_loader_b, val_loader_b, loss_fn,
        epochs=gam_epochs, device=device, **train_kwargs,
    )
    print(f"\n  Boosted GAM (with GBDT tower): NDCG@{k} = {boosted_ndcg:.4f}")

    return {
        "boosted_model": boosted_model,
        "gbdt_model": gbdt_model,
        "gbdt_ndcg": float(gbdt_ndcg),
        "boosted_ndcg": float(boosted_ndcg),
    }

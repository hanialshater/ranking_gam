"""
GBDT baseline and residual-boosted GAM models.

Provides:
  - train_gbdt_baseline: train a LambdaMART ranker as a performance ceiling
  - train_gbdt_residual_boost: train GAM first, then GBDT on GAM residuals,
    then retrain GAM with D+1 features (original + GBDT residual score).
    The GBDT tower is the "magic curve" that captures what GAM missed.
"""

import numpy as np
import torch


def _gam_predict(model, X, device=None):
    """Get GAM scores for [B, L, D] numpy array -> [B, L] numpy."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    scores = []
    with torch.no_grad():
        for qi in range(len(X)):
            x_q = torch.from_numpy(X[qi : qi + 1]).float().to(device)
            s = model(x_q).squeeze(0).cpu().numpy()
            scores.append(s)
    return np.array(scores)


def _train_gbdt_regression(train_X, train_y, eval_X, eval_y, **lgbm_kwargs):
    """
    Train a LightGBM regressor on continuous targets (e.g. residuals).

    Unlike _train_gbdt_on_targets which uses LambdaMART ranking,
    this uses regression since residuals are continuous floats.

    Args:
        train_X: [B, L, D] numpy features
        train_y: [B, L] numpy targets (-1 = padding)
        eval_X: [B, L, D] numpy features
        eval_y: [B, L] numpy targets
        **lgbm_kwargs: passed to LGBMRegressor

    Returns:
        fitted LGBMRegressor
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "lightgbm is required for GBDT training. "
            "Install with: pip install lightgbm>=4.0"
        )

    B_tr = train_X.shape[0]

    X_train_list = []
    y_train_list = []
    for qi in range(B_tr):
        valid = train_y[qi] >= 0
        if valid.sum() > 0:
            X_train_list.append(train_X[qi][valid])
            y_train_list.append(train_y[qi][valid])

    X_tr = np.concatenate(X_train_list, axis=0)
    y_tr = np.concatenate(y_train_list, axis=0)

    B_ev = eval_X.shape[0]
    X_eval_list = []
    y_eval_list = []
    for qi in range(B_ev):
        valid = eval_y[qi] >= 0
        if valid.sum() > 0:
            X_eval_list.append(eval_X[qi][valid])
            y_eval_list.append(eval_y[qi][valid])

    X_ev = np.concatenate(X_eval_list, axis=0)
    y_ev = np.concatenate(y_eval_list, axis=0)

    defaults = dict(
        objective="regression",
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.1,
        verbose=-1,
    )
    defaults.update(lgbm_kwargs)

    regressor = lgb.LGBMRegressor(**defaults)
    regressor.fit(
        X_tr, y_tr,
        eval_set=[(X_ev, y_ev)],
    )

    return regressor


def _train_gbdt_on_targets(train_X, train_y, eval_X, eval_y, k=10, **lgbm_kwargs):
    """
    Train a LightGBM LambdaMART ranker on given targets.

    Unlike train_gbdt_baseline, this is an internal helper that trains
    on arbitrary targets (e.g. residuals), not necessarily raw relevance labels.

    Args:
        train_X: [B, L, D] numpy features
        train_y: [B, L] numpy targets (-1 = padding)
        eval_X: [B, L, D] numpy features
        eval_y: [B, L] numpy targets
        k: NDCG cutoff
        **lgbm_kwargs: passed to LGBMRanker

    Returns:
        fitted LGBMRanker
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "lightgbm is required for GBDT training. "
            "Install with: pip install lightgbm>=4.0"
        )

    B_tr, L_tr, D = train_X.shape

    X_train_list = []
    y_train_list = []
    groups_train = []
    for qi in range(B_tr):
        valid = train_y[qi] >= 0
        n_valid = valid.sum()
        if n_valid > 0:
            X_train_list.append(train_X[qi][valid])
            y_train_list.append(train_y[qi][valid])
            groups_train.append(int(n_valid))

    X_tr = np.concatenate(X_train_list, axis=0)
    y_tr = np.concatenate(y_train_list, axis=0)

    B_ev = eval_X.shape[0]
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

    return ranker


def _gbdt_eval_ndcg(ranker, eval_X, eval_y, k=10):
    """Compute NDCG for a fitted GBDT ranker on eval data."""
    from ..metrics import compute_ndcg

    B_ev = eval_X.shape[0]
    eval_preds = np.zeros_like(eval_y, dtype=np.float32)
    for qi in range(B_ev):
        valid = eval_y[qi] >= 0
        n_valid = valid.sum()
        if n_valid > 0:
            pred = ranker.predict(eval_X[qi][valid])
            eval_preds[qi][valid] = pred
            eval_preds[qi][~valid] = -1e9
        else:
            eval_preds[qi] = -1e9

    return float(compute_ndcg(
        torch.from_numpy(eval_preds), torch.from_numpy(eval_y.astype(np.float32)), k=k
    ))


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
        (model, eval_ndcg) -- fitted LGBMRanker and NDCG@k
    """
    ranker = _train_gbdt_on_targets(train_X, train_y, eval_X, eval_y, k=k, **lgbm_kwargs)
    eval_ndcg = _gbdt_eval_ndcg(ranker, eval_X, eval_y, k=k)
    print(f"  GBDT (LambdaMART): NDCG@{k} = {eval_ndcg:.4f}")
    return ranker, eval_ndcg


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
    Three-stage residual boosting: GAM -> GBDT on residuals -> GAM with magic curve.

    Stage 1: Train GAM on raw features (capture interpretable signal)
    Stage 2: Compute GAM residuals, train GBDT on what GAM missed
    Stage 3: Retrain GAM with D+1 features (original + GBDT residual score)

    Most signal stays interpretable in the D original towers. The GBDT tower
    is the "magic curve" -- one extra shape function f_{D+1}(gbdt_score)
    that captures whatever the GAM couldn't.

    Args:
        model: GAM_Paper (or similar) with num_features=D
        train_X: [B, L, D] numpy features
        train_y: [B, L] numpy labels
        eval_X: [B, L, D] numpy features
        eval_y: [B, L] numpy labels
        train_loader_fn: callable(X, y) -> (train_loader, val_loader)
        loss_fn: ranking loss module
        k: NDCG cutoff
        gam_epochs: epochs for GAM training (stages 1 and 3)
        boost_epochs: epochs for boosted GAM (stage 3)
        device: torch device
        **train_kwargs: extra kwargs for train_model

    Returns:
        dict with:
            "boosted_model": GAM with D+1 features (original + GBDT residual)
            "gbdt_model": fitted LGBMRegressor (trained on residuals)
            "stage1_ndcg": GAM-only NDCG
            "gbdt_residual_ndcg": GBDT-on-residuals NDCG (ranking by residual prediction)
            "boosted_ndcg": final GAM+magic-curve NDCG
    """
    from ..models import GAM_Paper
    from .trainer import train_model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D = train_X.shape[-1]

    # --- Stage 1: Train base GAM on raw features ---
    print("\n  Stage 1: Training base GAM...")
    train_loader, val_loader = train_loader_fn(train_X, train_y)
    stage1_ndcg = train_model(
        model, train_loader, val_loader, loss_fn,
        epochs=gam_epochs, device=device, **train_kwargs,
    )
    print(f"  Stage 1 GAM: NDCG@{k} = {stage1_ndcg:.4f}")

    # --- Stage 2: Compute residuals, train GBDT regressor on them ---
    print("\n  Stage 2: Training GBDT regressor on GAM residuals...")
    gam_scores_train = _gam_predict(model, train_X, device)
    gam_scores_eval = _gam_predict(model, eval_X, device)

    # Residual = true relevance - normalized GAM score
    # This captures what the GAM missed per document.
    # Uses IQR-based robust normalization instead of std to avoid
    # amplifying residuals when the GAM is well-calibrated (std ~ 0).
    def _make_residual_labels(y_true, gam_scores):
        valid = y_true >= 0
        residuals = np.full_like(y_true, -1, dtype=np.float32)
        if valid.any():
            gs = gam_scores.copy()
            y_valid = y_true[valid].astype(np.float32)

            g_q25, g_med, g_q75 = np.percentile(gs[valid], [25, 50, 75])
            y_q25, y_med, y_q75 = np.percentile(y_valid, [25, 50, 75])
            g_iqr = max(g_q75 - g_q25, 1e-6)
            y_iqr = max(y_q75 - y_q25, 1e-6)

            gs_norm = (gs - g_med) / g_iqr * y_iqr + y_med
            residuals[valid] = y_valid - gs_norm[valid]
        return residuals

    train_residuals = _make_residual_labels(train_y, gam_scores_train)
    eval_residuals = _make_residual_labels(eval_y, gam_scores_eval)

    # Use regression (not ranking) since residuals are continuous floats
    gbdt_model = _train_gbdt_regression(
        train_X, train_residuals, eval_X, eval_residuals,
    )

    # Diagnostics: how well does GBDT residual prediction rank on original labels?
    gbdt_res_ndcg = _gbdt_eval_ndcg(gbdt_model, eval_X, eval_y, k=k)
    print(f"  GBDT (residual regressor): NDCG@{k} = {gbdt_res_ndcg:.4f} (ranking by predicted residual)")

    # --- Stage 3: Add magic curve tower to trained GAM ---
    # Keep the D trained towers frozen, only train the new GBDT tower (D+1).
    print("\n  Stage 3: Training magic curve tower (freeze existing D towers)...")
    train_X_boosted = compute_gbdt_residual_feature(gbdt_model, train_X)
    eval_X_boosted = compute_gbdt_residual_feature(gbdt_model, eval_X)

    # Infer model config from original
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

    # Copy trained weights from Stage 1 into first D towers
    with torch.no_grad():
        for j in range(D):
            boosted_model.towers[j].load_state_dict(model.towers[j].state_dict())
        if has_transforms:
            # Copy trained transforms for original D features
            for j in range(D):
                boosted_model.feature_transforms[j].load_state_dict(
                    model.feature_transforms[j].state_dict()
                )
            # Init the D+1 transform from data
            col = train_X_boosted[:, :, D].reshape(-1)
            valid = col[col > -1e8]  # skip padding
            if len(valid) > 0:
                percentiles = np.percentile(valid, np.linspace(0, 100, 11))
                boosted_model.feature_transforms[D].init_from_percentiles(
                    torch.from_numpy(percentiles).float()
                )

    # Freeze the D original towers and their transforms
    for j in range(D):
        for param in boosted_model.towers[j].parameters():
            param.requires_grad = False
        if has_transforms:
            for param in boosted_model.feature_transforms[j].parameters():
                param.requires_grad = False

    train_loader_b, _ = train_loader_fn(train_X_boosted, train_y)
    # Build val loader from boosted eval data directly so validation uses
    # the correct D+1 features (not the un-boosted eval_X).
    from torch.utils.data import DataLoader, TensorDataset

    _val_ds = TensorDataset(
        torch.from_numpy(eval_X_boosted),
        torch.from_numpy(eval_y.astype(np.float32)),
    )
    val_loader_b = DataLoader(_val_ds, batch_size=train_loader_b.batch_size)
    boosted_ndcg = train_model(
        boosted_model, train_loader_b, val_loader_b, loss_fn,
        epochs=boost_epochs, device=device, **train_kwargs,
    )
    print(f"\n  Boosted GAM (with magic curve): NDCG@{k} = {boosted_ndcg:.4f}")

    return {
        "boosted_model": boosted_model,
        "gbdt_model": gbdt_model,
        "stage1_ndcg": float(stage1_ndcg),
        "gbdt_residual_ndcg": float(gbdt_res_ndcg),
        "boosted_ndcg": float(boosted_ndcg),
    }

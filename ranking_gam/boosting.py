"""
Item-level score boosting via GAM response curves.

Because GAM scores decompose as score = sum_j f_j(x_j), we can boost individual
items by manipulating specific feature values on the response curve. This is
uniquely enabled by the interpretable structure of GAMs.

Use cases:
  - Cold-start bootstrapping: new items have no click/popularity signal;
    override that feature to a target percentile to give them visibility.
  - Promotional boosting: temporarily boost items by shifting a feature
    (e.g., set "freshness" to max for new arrivals).
  - Fairness correction: equalize a feature's contribution across groups.

All operations are transparent and auditable -- you can see exactly how much
score delta each boost adds by reading the response curve.
"""

import numpy as np
import torch


def compute_boost_delta(model, feature_idx, from_value, to_value):
    """Compute the score delta from shifting a single feature on its response curve.

    Args:
        model: trained GAM/ContextGAM/SubmodularRankingGAM with get_main_effect()
        feature_idx: which feature to boost
        from_value: current feature value (scalar or 1D array)
        to_value: target feature value (scalar or 1D array)

    Returns:
        float or 1D array: score delta (positive = boost, negative = suppress)
    """
    from_arr = np.atleast_1d(np.asarray(from_value, dtype=np.float32))
    to_arr = np.atleast_1d(np.asarray(to_value, dtype=np.float32))

    effect_from = model.get_main_effect(feature_idx, from_arr)
    effect_to = model.get_main_effect(feature_idx, to_arr)
    delta = effect_to - effect_from

    if delta.size == 1:
        return float(delta[0])
    return delta


def boost_items(X, model, feature_idx, to_value, item_mask=None):
    """Apply a feature override to selected items and return boosted scores.

    This modifies feature values in-place (on a copy) and re-scores. The
    boost is fully interpretable: it's equivalent to moving along the
    response curve for that feature.

    Args:
        X: [B, L, D] numpy array of features
        model: trained GAM with forward()
        feature_idx: which feature to override
        to_value: target value for that feature (scalar)
        item_mask: [B, L] bool array. If None, boost all items.

    Returns:
        boosted_scores: [B, L] numpy array of new scores
        original_scores: [B, L] numpy array of original scores
        delta: [B, L] numpy array of score changes
    """
    device = next(model.parameters()).device

    # Original scores
    x_t = torch.from_numpy(X).float().to(device)
    model.eval()
    with torch.no_grad():
        original_scores = model(x_t).cpu().numpy()

    # Boosted scores
    X_boosted = X.copy()
    if item_mask is not None:
        X_boosted[item_mask, feature_idx] = to_value
    else:
        X_boosted[:, :, feature_idx] = to_value
    x_b = torch.from_numpy(X_boosted).float().to(device)
    with torch.no_grad():
        boosted_scores = model(x_b).cpu().numpy()

    delta = boosted_scores - original_scores
    return boosted_scores, original_scores, delta


def get_percentile_value(train_X, feature_idx, percentile):
    """Get the feature value at a given percentile from training data.

    Useful for setting boost targets: "boost new items to the 75th percentile
    of popularity" is more meaningful than "set popularity to 0.5".

    Args:
        train_X: [B, L, D] numpy array of training features
        feature_idx: which feature
        percentile: 0-100

    Returns:
        float: feature value at the given percentile
    """
    vals = train_X[:, :, feature_idx].flatten()
    valid = vals != 0  # skip padding zeros
    if valid.any():
        return float(np.percentile(vals[valid], percentile))
    return float(np.percentile(vals, percentile))


def explain_boost(model, feature_idx, from_value, to_value, feature_name=None):
    """Print a human-readable explanation of a boost operation.

    Args:
        model: trained GAM with get_main_effect()
        feature_idx: which feature
        from_value: current value
        to_value: target value
        feature_name: optional name for the feature
    """
    delta = compute_boost_delta(model, feature_idx, from_value, to_value)
    fname = feature_name or f"feature_{feature_idx}"

    from_effect = float(model.get_main_effect(feature_idx, np.array([from_value], dtype=np.float32))[0])
    to_effect = float(model.get_main_effect(feature_idx, np.array([to_value], dtype=np.float32))[0])

    print(f"  Boost: {fname}")
    print(f"    From: x={from_value:.4f} -> f(x)={from_effect:.4f}")
    print(f"    To:   x={to_value:.4f} -> f(x)={to_effect:.4f}")
    print(f"    Delta: {delta:+.4f}")
    return delta


def warmup_blend(real_value, boost_value, day, warmup_days=10):
    """Blend between boosted and real feature values over a warmup period.

    At day 0, uses 100% boost_value (cold-start, no real signal).
    At day >= warmup_days, uses 100% real_value (fully warmed up).
    In between, linearly interpolates.

    This gives new items initial visibility via the boost, then gradually
    transitions to the real signal as data accumulates.

    Args:
        real_value: actual observed feature value (scalar or array)
        boost_value: target boost value (scalar or array)
        day: current day since item launch (scalar or array, 0-indexed)
        warmup_days: number of days for full warmup (default: 10)

    Returns:
        blended value: same shape as inputs

    Example:
        >>> # Item launched 3 days ago, real clicks=2, boost target=50
        >>> warmup_blend(real_value=2, boost_value=50, day=3, warmup_days=10)
        35.6  # 70% boost + 30% real
    """
    real_value = np.asarray(real_value, dtype=np.float32)
    boost_value = np.asarray(boost_value, dtype=np.float32)
    day = np.asarray(day, dtype=np.float32)

    # alpha = 0 at day 0 (all boost), alpha = 1 at warmup_days (all real)
    alpha = np.clip(day / warmup_days, 0.0, 1.0)
    return alpha * real_value + (1 - alpha) * boost_value


def warmup_boost_items(X, model, feature_idx, boost_value, item_days,
                       warmup_days=10, item_mask=None):
    """Apply time-decaying boost to items based on their age.

    Combines warmup_blend with boost_items: new items get full boost,
    items past warmup_days use real values, items in between get a blend.

    Args:
        X: [B, L, D] numpy array of features
        model: trained GAM with forward()
        feature_idx: which feature to boost
        boost_value: target value for new items (scalar)
        item_days: [B, L] array of days since each item was launched
        warmup_days: days for full warmup (default: 10)
        item_mask: [B, L] bool array. If None, apply to all items.

    Returns:
        boosted_scores: [B, L] numpy array of blended scores
        original_scores: [B, L] numpy array of original scores
        delta: [B, L] score changes
    """
    device_param = next(model.parameters()).device

    # Original scores
    x_t = torch.from_numpy(X).float().to(device_param)
    model.eval()
    with torch.no_grad():
        original_scores = model(x_t).cpu().numpy()

    # Blend feature values based on item age
    X_blended = X.copy()
    real_values = X[:, :, feature_idx]
    blended = warmup_blend(real_values, boost_value, item_days, warmup_days)
    if item_mask is not None:
        X_blended[item_mask, feature_idx] = blended[item_mask]
    else:
        X_blended[:, :, feature_idx] = blended

    x_b = torch.from_numpy(X_blended).float().to(device_param)
    with torch.no_grad():
        boosted_scores = model(x_b).cpu().numpy()

    delta = boosted_scores - original_scores
    return boosted_scores, original_scores, delta

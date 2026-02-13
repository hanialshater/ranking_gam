"""Feature interaction selection for GA2M models."""

from itertools import combinations

import numpy as np


def select_interactions_correlation(X, y, top_k=50):
    """
    Select top-k feature pairs by correlation with labels.

    Args:
        X: [B, L, D] feature array
        y: [B, L] label array
        top_k: number of interaction pairs to select

    Returns:
        list of (feature_i, feature_j) tuples
    """
    print(f"Selecting top {top_k} interactions by correlation...")

    X_flat = X.reshape(-1, X.shape[-1])
    y_flat = y.flatten()

    valid = y_flat >= 0
    X_flat = X_flat[valid]
    y_flat = y_flat[valid]

    num_features = X_flat.shape[1]
    pair_scores = []

    all_pairs = list(combinations(range(num_features), 2))
    print(f"Evaluating {len(all_pairs)} candidate pairs...")

    for i, (f1, f2) in enumerate(all_pairs):
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(all_pairs)}")

        interaction = X_flat[:, f1] * X_flat[:, f2]
        corr = np.abs(np.corrcoef(interaction, y_flat)[0, 1])
        if np.isnan(corr):
            corr = 0
        pair_scores.append((f1, f2, corr))

    pair_scores.sort(key=lambda x: -x[2])
    selected = [(f1, f2) for f1, f2, _ in pair_scores[:top_k]]

    print("Top 10 interactions:")
    for f1, f2, corr in pair_scores[:10]:
        print(f"  F{f1 + 1} x F{f2 + 1}: corr={corr:.4f}")

    return selected

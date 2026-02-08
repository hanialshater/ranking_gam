"""
Ranking quality and diversity metrics.

Relevance: NDCG@k, Precision@k, Mean Relevance
Diversity: Category/Brand Coverage & Entropy, ILD, Price Spread, alpha-NDCG
"""

import numpy as np
import torch


def compute_ndcg(y_pred, y_true, k=10):
    """
    Compute NDCG@k.

    Args:
        y_pred: [B, L] predicted scores (tensor)
        y_true: [B, L] relevance labels (tensor), -1 = padding
        k: evaluation depth

    Returns:
        float mean NDCG@k
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    ndcg_list = []
    for pred, true in zip(y_pred, y_true):
        valid = true >= 0
        if not valid.any():
            continue

        pred_valid = pred[valid]
        true_valid = true[valid]

        top_idx = np.argsort(-pred_valid)[:k]
        disc = np.log2(np.arange(len(top_idx)) + 2)
        dcg = np.sum((2 ** true_valid[top_idx] - 1) / disc)

        ideal_idx = np.argsort(-true_valid)[:k]
        ideal_dcg = np.sum((2 ** true_valid[ideal_idx] - 1) / disc[: len(ideal_idx)])

        if ideal_dcg > 0:
            ndcg_list.append(dcg / ideal_dcg)

    return np.mean(ndcg_list) if ndcg_list else 0.0


def evaluate_ranking_diversity(
    X_aug, y, top_k_indices, k=10, cat_col=136, brand_col=137, price_col=10
):
    """
    Evaluate both relevance and diversity of a ranked list.

    Diversity metrics:
        - Category/Brand Coverage: unique in top-k / unique in query
        - Category/Brand Entropy: Shannon entropy over distribution
        - Price Spread: std(price) of top-k items
        - ILD (Intra-List Diversity): 1 - avg pairwise category match rate
        - alpha-NDCG: rewards relevance, discounts items from seen categories

    Relevance metrics:
        - NDCG@k, Precision@k, Mean Relevance

    Args:
        X_aug: [B, L, D_aug] numpy feature matrix (with cat/brand columns)
        y: [B, L] numpy relevance labels
        top_k_indices: list of lists of item indices per query
        k: evaluation depth
        cat_col: column index for category feature
        brand_col: column index for brand feature
        price_col: column index for price feature

    Returns:
        dict of {metric_name: mean_value}
    """
    metrics = {
        "ndcg": [],
        "precision": [],
        "mean_rel": [],
        "cat_coverage": [],
        "brand_coverage": [],
        "cat_entropy": [],
        "brand_entropy": [],
        "price_std": [],
        "ild": [],
        "alpha_ndcg": [],
    }

    for qi in range(len(X_aug)):
        y_q = y[qi]
        valid = y_q >= 0
        if valid.sum() < 2:
            continue

        order = top_k_indices[qi]
        if hasattr(order, "tolist"):
            order = order.tolist()
        order = [o for o in order if 0 <= o < len(y_q) and y_q[o] >= 0][:k]
        if len(order) < 2:
            continue

        labels = np.array([y_q[o] for o in order])
        feats = X_aug[qi]

        # Relevance
        disc = np.log2(np.arange(len(labels)) + 2)
        dcg = np.sum((2**labels - 1) / disc)
        ideal = np.sort(y_q[valid])[::-1][:k]
        idcg = np.sum((2**ideal - 1) / np.log2(np.arange(len(ideal)) + 2))
        if idcg > 0:
            metrics["ndcg"].append(dcg / idcg)
        metrics["precision"].append(np.mean(labels > 0))
        metrics["mean_rel"].append(np.mean(labels))

        # Category diversity
        cats = np.array([feats[o, cat_col] for o in order])
        all_cats = feats[valid, cat_col]
        unique_cats_topk = len(np.unique(cats))
        unique_cats_all = max(len(np.unique(all_cats)), 1)
        metrics["cat_coverage"].append(unique_cats_topk / unique_cats_all)

        _, counts = np.unique(cats, return_counts=True)
        probs = counts / counts.sum()
        metrics["cat_entropy"].append(-np.sum(probs * np.log2(probs + 1e-10)))

        # Brand diversity
        brands = np.array([feats[o, brand_col] for o in order])
        all_brands = feats[valid, brand_col]
        unique_brands_topk = len(np.unique(brands))
        unique_brands_all = max(len(np.unique(all_brands)), 1)
        metrics["brand_coverage"].append(unique_brands_topk / unique_brands_all)

        _, counts = np.unique(brands, return_counts=True)
        probs = counts / counts.sum()
        metrics["brand_entropy"].append(-np.sum(probs * np.log2(probs + 1e-10)))

        # Price diversity
        prices = np.array([feats[o, price_col] for o in order])
        metrics["price_std"].append(np.std(prices))

        # ILD
        n = len(order)
        if n >= 2:
            pair_matches = 0
            pair_total = 0
            for i in range(n):
                for j in range(i + 1, n):
                    pair_total += 1
                    if cats[i] == cats[j]:
                        pair_matches += 1
            metrics["ild"].append(1.0 - pair_matches / pair_total)

        # alpha-NDCG (alpha=0.5)
        alpha = 0.5
        seen_cats = {}
        alpha_dcg = 0.0
        for pos, o in enumerate(order):
            cat = feats[o, cat_col]
            count = seen_cats.get(cat, 0)
            gain = (2 ** labels[pos] - 1) * (1 - alpha) ** count
            alpha_dcg += gain / np.log2(pos + 2)
            seen_cats[cat] = count + 1

        alpha_idcg_labels = np.sort(y_q[valid])[::-1][:k]
        alpha_idcg = np.sum(
            (2**alpha_idcg_labels - 1)
            / np.log2(np.arange(len(alpha_idcg_labels)) + 2)
        )
        if alpha_idcg > 0:
            metrics["alpha_ndcg"].append(alpha_dcg / alpha_idcg)

    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}

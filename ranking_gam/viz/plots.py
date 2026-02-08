"""
Visualization utilities for ranking GAM models.

Includes response curve plots, diversity curve plots, spider/radar charts,
and diversity comparison tables.
"""

import numpy as np
import torch


def plot_response_curves(
    model, feature_names=None, data=None, n_points=200, top_k=20,
    figsize_per=(3.2, 2.5),
):
    """
    Plot learned response curves f_j(x_j) for top-k most important features.

    Reproduces paper Fig 5/7: each subplot shows one feature's shape function
    with optional data histogram overlay.

    Works with GAM_Paper, GA2M_Paper, or any model that has a `.towers`
    or `.item_towers` ModuleList and `.num_features` / `.num_item_features`.

    Args:
        model: trained GAM/GA2M/SubmodularRankingGAM
        feature_names: list of str names
        data: [N, D] array for x-range and histograms
        n_points: curve resolution
        top_k: show top-k features by importance (output variance)
        figsize_per: (width, height) per subplot

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    model.eval()
    device = next(model.parameters()).device

    # Resolve the tower list
    if hasattr(model, "towers"):
        towers = model.towers
    elif hasattr(model, "main_towers"):
        towers = model.main_towers
    elif hasattr(model, "item_towers"):
        towers = model.item_towers
    else:
        raise AttributeError("Model must have .towers, .main_towers, or .item_towers")

    num_features = len(towers)

    importances = []
    for j in range(num_features):
        if data is not None:
            col = data[:, j] if isinstance(data, np.ndarray) else data[:, j].cpu().numpy()
            # Use percentiles to cover most of the density without zooming
            # out too far into sparse tails where curves can be wild
            x_lo = float(np.percentile(col, 1))
            x_hi = float(np.percentile(col, 99))
            # Fallback if degenerate (constant feature)
            if x_hi - x_lo < 1e-8:
                x_lo, x_hi = float(np.min(col)), float(np.max(col))
            if x_hi - x_lo < 1e-8:
                x_lo, x_hi = x_lo - 1.0, x_hi + 1.0
        else:
            x_lo, x_hi = -3.0, 3.0

        x_sweep = torch.linspace(x_lo, x_hi, n_points, device=device).unsqueeze(1)
        with torch.no_grad():
            y_sweep = towers[j](x_sweep).squeeze()
        importances.append((j, y_sweep.std().item(), x_lo, x_hi))

    importances.sort(key=lambda t: -t[1])
    top_feats = importances[:top_k]

    n_cols = min(5, len(top_feats))
    n_rows = (len(top_feats) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(figsize_per[0] * n_cols, figsize_per[1] * n_rows)
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (j, imp, x_lo, x_hi) in enumerate(top_feats):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        x_sweep = torch.linspace(x_lo, x_hi, n_points, device=device).unsqueeze(1)
        with torch.no_grad():
            y_sweep = towers[j](x_sweep).squeeze().cpu().numpy()
        x_np = x_sweep.squeeze().cpu().numpy()

        if data is not None:
            col_data = data[:, j] if isinstance(data, np.ndarray) else data[:, j].cpu().numpy()
            # Clip histogram to the same percentile range as the curve
            col_clipped = col_data[(col_data >= x_lo) & (col_data <= x_hi)]
            if len(col_clipped) > 0:
                ax.hist(col_clipped, bins=50, alpha=0.15, color="steelblue", density=True)

        ax.plot(x_np, y_sweep, color="#e74c3c", linewidth=1.8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

        fname = feature_names[j] if feature_names and j < len(feature_names) else f"feature_{j}"
        ax.set_title(f"{fname}\n(imp={imp:.3f})", fontsize=8)
        ax.set_xlabel(fname, fontsize=7)
        ax.set_ylabel("f(x)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    for idx in range(len(top_feats), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"GAM Response Curves (top {len(top_feats)} by importance)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def plot_diversity_curves(model):
    """
    Plot all concave monotone PWL diversity curves.

    Works with SubmodularRankingGAM or any model with
    .diversity_towers and .groupwise_specs.

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    n = len(model.diversity_towers)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]

    for k, (spec, tower) in enumerate(
        zip(model.groupwise_specs, model.diversity_towers)
    ):
        x, y = tower.get_curve()
        slopes = tower.get_slopes().detach().cpu().numpy()

        axes[k].plot(x, y, "b-", linewidth=2)
        axes[k].set_title(
            f"{spec['name']}\n(slopes: {', '.join(f'{s:.2f}' for s in slopes[:5])}...)"
        )
        axes[k].set_xlabel(spec["name"])
        axes[k].set_ylabel("Score contribution")
        axes[k].grid(True, alpha=0.3)

        is_concave = all(
            slopes[i] >= slopes[i + 1] - 1e-6 for i in range(len(slopes) - 1)
        )
        is_monotone = all(s >= -1e-6 for s in slopes)

        if is_concave and is_monotone:
            label, color = "monotone concave", "green"
        elif is_concave:
            label, color = "concave (not monotone)", "orange"
        else:
            label, color = "constraint violated", "red"
        axes[k].text(
            0.05, 0.95, label, transform=axes[k].transAxes,
            color=color, fontsize=9, va="top", fontweight="bold",
        )

    plt.tight_layout()
    return fig


def plot_spider(
    strategies, metrics=None, title="Multi-Objective Ranking Comparison",
    normalize=True, figsize=(8, 8), colors=None,
):
    """
    Spider/radar plot comparing multiple ranking strategies across objectives.

    Args:
        strategies: dict {strategy_name: metrics_dict}
        metrics: list of metric keys to show (or None for defaults)
        title: plot title
        normalize: if True, scale each axis to [0, 1]
        figsize: figure size
        colors: list of colors per strategy

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = [
            "ndcg", "alpha_ndcg", "precision", "cat_coverage",
            "brand_coverage", "cat_entropy", "ild", "price_std",
        ]

    display_names = {
        "ndcg": "NDCG@10",
        "alpha_ndcg": "alpha-NDCG",
        "precision": "Precision",
        "mean_rel": "Avg Relevance",
        "cat_coverage": "Category\nCoverage",
        "brand_coverage": "Brand\nCoverage",
        "cat_entropy": "Category\nEntropy",
        "brand_entropy": "Brand\nEntropy",
        "price_std": "Price\nSpread",
        "ild": "Intra-List\nDiversity",
    }

    if colors is None:
        colors = [
            "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
            "#1abc9c", "#e67e22", "#34495e",
        ]

    strategy_names = list(strategies.keys())
    n_metrics = len(metrics)
    n_strategies = len(strategy_names)

    raw = np.zeros((n_strategies, n_metrics))
    for i, name in enumerate(strategy_names):
        for j, m in enumerate(metrics):
            raw[i, j] = strategies[name].get(m, 0.0)

    if normalize:
        mins = raw.min(axis=0)
        maxs = raw.max(axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-10] = 1.0
        values = (raw - mins) / ranges * 0.8 + 0.1
    else:
        values = raw

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.spines["polar"].set_visible(False)

    if normalize:
        for gl in [0.2, 0.4, 0.6, 0.8, 1.0]:
            ax.plot(angles, [gl] * (n_metrics + 1), "--", color="gray", alpha=0.15, linewidth=0.5)

    labels = [display_names.get(m, m) for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")

    for i, name in enumerate(strategy_names):
        vals = values[i].tolist()
        vals += vals[:1]

        color = colors[i % len(colors)]
        ax.plot(angles, vals, "o-", linewidth=2.2, label=name, color=color, markersize=6, zorder=5)
        ax.fill(angles, vals, alpha=0.08, color=color)

    for j, m in enumerate(metrics):
        angle = angles[j]
        for i, sname in enumerate(strategy_names):
            r = values[i, j]
            raw_val = raw[i, j]
            offset = 0.08 * (i - (n_strategies - 1) / 2)
            ax.annotate(
                f"{raw_val:.3f}", xy=(angle, r + offset),
                fontsize=6, ha="center", va="center",
                color=colors[i % len(colors)], alpha=0.8,
            )

    if normalize:
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["", "", "", ""], fontsize=6)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.9, edgecolor="gray")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=25)

    plt.tight_layout()
    return fig


def print_diversity_comparison(base_metrics, greedy_metrics, title="Relevance vs Diversity"):
    """Pretty-print side-by-side comparison of base vs greedy rankings."""
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<25} {'Base':>12} {'Greedy':>12} {'Delta':>10}")
    print(f"  {'-' * 55}")

    groups = [
        ("RELEVANCE", ["ndcg", "alpha_ndcg", "precision", "mean_rel"]),
        (
            "DIVERSITY",
            [
                "cat_coverage", "brand_coverage", "cat_entropy",
                "brand_entropy", "price_std", "ild",
            ],
        ),
    ]

    labels = {
        "ndcg": "NDCG@10",
        "alpha_ndcg": "alpha-NDCG@10 (a=0.5)",
        "precision": "Precision@10",
        "mean_rel": "Mean Relevance",
        "cat_coverage": "Category Coverage",
        "brand_coverage": "Brand Coverage",
        "cat_entropy": "Category Entropy",
        "brand_entropy": "Brand Entropy",
        "price_std": "Price Spread (std)",
        "ild": "Intra-List Diversity",
    }

    for group_name, keys in groups:
        print(f"\n  {group_name}:")
        for key in keys:
            b = base_metrics.get(key, 0)
            g = greedy_metrics.get(key, 0)
            delta = g - b
            arrow = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
            print(
                f"    {labels.get(key, key):<23} {b:>10.4f}   {g:>10.4f}   {delta:>+8.4f} {arrow}"
            )

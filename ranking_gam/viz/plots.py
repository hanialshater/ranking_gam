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
    normalize=True, figsize=(8, 8), colors=None, calibration=None,
):
    """
    Spider/radar plot comparing multiple ranking strategies across objectives.

    Args:
        strategies: dict {strategy_name: metrics_dict}
        metrics: list of metric keys to show (or None for defaults)
        title: plot title
        normalize: if True, scale each axis to [0, 1] using calibration bounds
        figsize: figure size
        colors: list of colors per strategy
        calibration: dict {metric_key: (min, max)} defining the absolute scale
            for each axis.  When normalize=True (default), each axis is mapped
            from [cal_min, cal_max] -> [0, 1] so that the radar shape reflects
            how close each metric is to its theoretical best, not just how
            strategies compare to each other.
            Defaults are provided for common ranking metrics (NDCG, coverage,
            entropy, etc.).  Pass explicit bounds to override.

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

    # Sensible absolute bounds for common ranking metrics.
    # (min_meaningful, max_meaningful) -- values are clipped to this range.
    default_calibration = {
        "ndcg": (0.0, 1.0),
        "alpha_ndcg": (0.0, 1.0),
        "precision": (0.0, 1.0),
        "mean_rel": (0.0, 4.0),
        "cat_coverage": (0.0, 1.0),
        "brand_coverage": (0.0, 1.0),
        "cat_entropy": (0.0, 3.0),
        "brand_entropy": (0.0, 3.0),
        "price_std": (0.0, 2.0),
        "ild": (0.0, 1.0),
    }
    cal = {**default_calibration, **(calibration or {})}

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
        values = np.zeros_like(raw)
        for j, m in enumerate(metrics):
            lo, hi = cal.get(m, (0.0, float(raw[:, j].max()) or 1.0))
            span = hi - lo
            if span < 1e-10:
                span = 1.0
            values[:, j] = np.clip((raw[:, j] - lo) / span, 0.0, 1.0)
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
        ax.set_yticklabels(["20%", "40%", "60%", "80%"], fontsize=6, color="gray")

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


def plot_pareto_front(scenario_metrics, x_metric="ndcg", y_metric="ild",
                      x_label=None, y_label=None, title=None,
                      colors=None, figsize=(8, 6)):
    """Plot Pareto front of multi-objective ranking scenarios.

    Each scenario is a point in (x_metric, y_metric) space. The Pareto front
    connects non-dominated points.

    Args:
        scenario_metrics: dict of {scenario_name: metrics_dict}
        x_metric: key for x-axis metric (default: "ndcg")
        y_metric: key for y-axis metric (default: "ild")
        x_label: axis label (default: auto from metric name)
        y_label: axis label (default: auto from metric name)
        title: plot title
        colors: list of colors for scenarios
        figsize: figure size

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if colors is None:
        colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]
    if x_label is None:
        x_label = x_metric.upper().replace("_", " ")
    if y_label is None:
        y_label = y_metric.upper().replace("_", " ")
    if title is None:
        title = f"Pareto Front: {x_label} vs {y_label}"

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    names = list(scenario_metrics.keys())
    xs = [scenario_metrics[n][x_metric] for n in names]
    ys = [scenario_metrics[n][y_metric] for n in names]

    for i, (name, x, y) in enumerate(zip(names, xs, ys)):
        c = colors[i % len(colors)]
        ax.scatter(x, y, s=120, color=c, zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 8),
                    fontsize=10, color=c, fontweight="bold")

    # Compute and draw Pareto front (maximize both metrics)
    points = sorted(zip(xs, ys, names), key=lambda p: -p[0])
    pareto_x, pareto_y = [], []
    max_y = -float("inf")
    for x, y, _ in points:
        if y > max_y:
            pareto_x.append(x)
            pareto_y.append(y)
            max_y = y
    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, "--", color="#7f8c8d", linewidth=1.5,
                alpha=0.7, label="Pareto front")

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if len(pareto_x) > 1:
        ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def plot_interaction_heatmap(
    model, pair_idx, x1_range=None, x2_range=None, n_points=50,
    feature_names=None, figsize=(6, 5), cmap="RdBu_r",
):
    """
    Plot 2D heatmap of a GA2M interaction effect f_{jk}(x_j, x_k).

    Evaluates the interaction tower on a grid and renders as a heatmap,
    showing how the pairwise effect varies across both features.

    Args:
        model: trained GA2M_Paper or similar with get_interaction_effect()
        pair_idx: index into model.interaction_pairs
        x1_range: (min, max) for feature j, or None for (0, 1)
        x2_range: (min, max) for feature k, or None for (0, 1)
        n_points: grid resolution per axis
        feature_names: list of str names (indexed by feature index)
        figsize: figure size
        cmap: matplotlib colormap name

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    if x1_range is None:
        x1_range = (0.0, 1.0)
    if x2_range is None:
        x2_range = (0.0, 1.0)

    f1_idx, f2_idx = model.interaction_pairs[pair_idx]

    x1 = np.linspace(x1_range[0], x1_range[1], n_points)
    x2 = np.linspace(x2_range[0], x2_range[1], n_points)
    x1_grid, x2_grid = np.meshgrid(x1, x2)

    x1_flat = x1_grid.flatten().astype(np.float32)
    x2_flat = x2_grid.flatten().astype(np.float32)

    z_flat = model.get_interaction_effect(pair_idx, x1_flat, x2_flat)
    z_grid = z_flat.reshape(n_points, n_points)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        z_grid, origin="lower", aspect="auto",
        extent=[x1_range[0], x1_range[1], x2_range[0], x2_range[1]],
        cmap=cmap, interpolation="bilinear",
    )
    fig.colorbar(im, ax=ax, label="Interaction effect")

    f1_name = feature_names[f1_idx] if feature_names and f1_idx < len(feature_names) else f"feature_{f1_idx}"
    f2_name = feature_names[f2_idx] if feature_names and f2_idx < len(feature_names) else f"feature_{f2_idx}"

    ax.set_xlabel(f1_name, fontsize=11)
    ax.set_ylabel(f2_name, fontsize=11)
    ax.set_title(f"Interaction: {f1_name} x {f2_name}", fontsize=13, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_interaction_grid(
    model, feature_names=None, x_ranges=None, n_points=40,
    figsize_per=(4, 3.5), cmap="RdBu_r",
):
    """
    Plot heatmaps for all interaction pairs in a GA2M model.

    Args:
        model: trained GA2M_Paper with interaction_pairs
        feature_names: list of str names
        x_ranges: dict {feature_idx: (min, max)} or None for (0, 1)
        n_points: grid resolution per axis
        figsize_per: (width, height) per subplot
        cmap: matplotlib colormap

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    n_pairs = len(model.interaction_pairs)
    if n_pairs == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No interaction pairs", ha="center", va="center")
        return fig

    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per[0] * n_cols, figsize_per[1] * n_rows),
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    if x_ranges is None:
        x_ranges = {}

    for idx in range(n_pairs):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        f1_idx, f2_idx = model.interaction_pairs[idx]
        x1_range = x_ranges.get(f1_idx, (0.0, 1.0))
        x2_range = x_ranges.get(f2_idx, (0.0, 1.0))

        x1 = np.linspace(x1_range[0], x1_range[1], n_points)
        x2 = np.linspace(x2_range[0], x2_range[1], n_points)
        x1_grid, x2_grid = np.meshgrid(x1, x2)

        x1_flat = x1_grid.flatten().astype(np.float32)
        x2_flat = x2_grid.flatten().astype(np.float32)

        z_flat = model.get_interaction_effect(idx, x1_flat, x2_flat)
        z_grid = z_flat.reshape(n_points, n_points)

        im = ax.imshow(
            z_grid, origin="lower", aspect="auto",
            extent=[x1_range[0], x1_range[1], x2_range[0], x2_range[1]],
            cmap=cmap, interpolation="bilinear",
        )
        fig.colorbar(im, ax=ax, shrink=0.8)

        f1_name = feature_names[f1_idx] if feature_names and f1_idx < len(feature_names) else f"f{f1_idx}"
        f2_name = feature_names[f2_idx] if feature_names and f2_idx < len(feature_names) else f"f{f2_idx}"
        ax.set_xlabel(f1_name, fontsize=9)
        ax.set_ylabel(f2_name, fontsize=9)
        ax.set_title(f"{f1_name} x {f2_name}", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)

    for idx in range(n_pairs, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle("GA2M Interaction Effects", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_objective_tradeoffs(scenario_metrics, objectives=None, figsize=(10, 5)):
    """Plot grouped bar chart of metrics across scenarios.

    Shows how each scenario scores on different objectives,
    making the tradeoffs explicit.

    Args:
        scenario_metrics: dict of {scenario_name: metrics_dict}
        objectives: list of metric keys to show (default: ndcg, ild, cat_coverage)
        figsize: figure size

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    names = list(scenario_metrics.keys())
    if objectives is None:
        objectives = ["ndcg", "ild", "cat_coverage", "brand_coverage"]
        objectives = [o for o in objectives if o in scenario_metrics[names[0]]]

    n_scenarios = len(names)
    n_objectives = len(objectives)

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#34495e"]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = np.arange(n_scenarios)
    width = 0.8 / n_objectives

    for i, obj in enumerate(objectives):
        values = [scenario_metrics[n].get(obj, 0) for n in names]
        offset = (i - n_objectives / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=obj.replace("_", " "),
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=9, ncol=min(n_objectives, 4))
    ax.set_ylabel("Metric value")
    ax.set_title("Objective Tradeoffs Across Scenarios", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig

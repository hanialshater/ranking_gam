#!/usr/bin/env python3
"""
Item boosting demo -- cold-start bootstrapping via response curve manipulation.

Because GAM scores decompose as score = sum_j f_j(x_j), we can boost individual
items by overriding a specific feature value on its response curve. This is a
GAM-native advantage: the boost is fully transparent and auditable.

Use case: new items have no click/popularity signal (feature = 0). We override
that feature to the 75th percentile of the training distribution, which moves
the item up the response curve and gives it visibility.

Usage:
    python examples/demo_boost.py
    python examples/demo_boost.py --loss ndcg2pp --activation silu --boost-feature 0
    python examples/demo_boost.py --boost-percentile 90 --boost-feature 5
"""

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ranking_gam as rg
from ranking_gam.viz import plot_response_curves

from _common import (
    add_common_args, device, load_data, make_loss, print_config, resolve_args,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Item boosting via response curves")
    add_common_args(parser)
    parser.add_argument("--boost-feature", type=int, default=0,
                        help="feature index to boost (default: 0, typically a popularity/click feature)")
    parser.add_argument("--boost-percentile", type=float, default=75,
                        help="target percentile for boosted items (default: 75)")
    parser.add_argument("--cold-start-fraction", type=float, default=0.1,
                        help="fraction of items to treat as cold-start (default: 0.1)")
    args = resolve_args(parser.parse_args())
    print_config(args, [f"boost_feature={args.boost_feature}",
                         f"boost_pctile={args.boost_percentile}"])

    data = load_data(dataset=args.dataset, data_dir=args.data_dir)
    num_features = data["num_features"]

    # --- Train GAM ---
    print("\n" + "=" * 70)
    print("Training GAM for item boosting demo")
    print("=" * 70)

    model = rg.GAM_Paper(
        num_features=num_features, hidden_dims=[16, 8],
        feature_transforms=args.transforms, residual=args.residual,
        activation=args.activation,
    )
    if args.transforms:
        model.init_transforms_from_data(data["train_X"])

    loss_fn = make_loss(args.loss, args.label_smoothing)
    ndcg = rg.train_model(
        model, data["train_loader"], data["eval_loader"], loss_fn,
        epochs=args.epochs, patience=args.patience, device=device,
        lr_schedule=args.lr_schedule,
        l1_output_reg=args.l1_reg, weight_decay=args.weight_decay,
    )
    print(f"\nTrained GAM: NDCG@{args.k} = {ndcg:.4f}")

    # --- Boosting analysis ---
    feat_idx = args.boost_feature
    feat_name = f"feature_{feat_idx}"
    try:
        from ranking_gam.data.mslr import MSLR_FEATURE_NAMES
        if num_features == 136 and feat_idx < len(MSLR_FEATURE_NAMES):
            feat_name = MSLR_FEATURE_NAMES[feat_idx]
    except ImportError:
        pass

    print("\n" + "=" * 70)
    print(f"Item Boosting: {feat_name} (feature {feat_idx})")
    print("=" * 70)

    # Get percentile value from training data
    target_value = rg.get_percentile_value(data["train_X"], feat_idx, args.boost_percentile)
    low_value = rg.get_percentile_value(data["train_X"], feat_idx, 10)
    print(f"\n  Feature: {feat_name}")
    print(f"  10th percentile (cold-start proxy): {low_value:.4f}")
    print(f"  Target ({args.boost_percentile}th percentile):     {target_value:.4f}")

    # Explain the boost
    print()
    rg.explain_boost(model, feat_idx, low_value, target_value, feature_name=feat_name)

    # --- Simulate cold-start scenario ---
    print(f"\n  Simulating cold-start: zeroing {feat_name} for {args.cold_start_fraction*100:.0f}% of eval items")
    eval_X = data["eval_X"].copy()
    eval_y = data["eval_y"]
    B, L, D = eval_X.shape

    # Create cold-start mask: randomly zero out the feature for some items
    rng = np.random.RandomState(42)
    cold_mask = rng.random((B, L)) < args.cold_start_fraction
    # Don't touch padded items
    valid = eval_y >= 0
    cold_mask = cold_mask & valid

    eval_X_cold = eval_X.copy()
    eval_X_cold[cold_mask, feat_idx] = 0.0

    # Score without boost
    import torch
    model.eval()
    with torch.no_grad():
        scores_cold = model(torch.from_numpy(eval_X_cold).float().to(device)).cpu().numpy()
        scores_orig = model(torch.from_numpy(eval_X).float().to(device)).cpu().numpy()

    # Score with boost (override cold-start items to target percentile)
    eval_X_boosted = eval_X_cold.copy()
    eval_X_boosted[cold_mask, feat_idx] = target_value
    with torch.no_grad():
        scores_boosted = model(torch.from_numpy(eval_X_boosted).float().to(device)).cpu().numpy()

    # Compute NDCG for each scenario
    ndcg_orig = rg.compute_ndcg(scores_orig, eval_y, k=args.k)
    ndcg_cold = rg.compute_ndcg(scores_cold, eval_y, k=args.k)
    ndcg_boosted = rg.compute_ndcg(scores_boosted, eval_y, k=args.k)

    print(f"\n  NDCG@{args.k} comparison:")
    print(f"    Original (no cold-start):  {ndcg_orig:.4f}")
    print(f"    Cold-start (zeroed {feat_name}): {ndcg_cold:.4f}  (drop: {ndcg_cold - ndcg_orig:+.4f})")
    print(f"    Boosted (-> p{args.boost_percentile:.0f}):         {ndcg_boosted:.4f}  (recovery: {ndcg_boosted - ndcg_cold:+.4f})")

    # --- Rank position analysis ---
    n_cold = cold_mask.sum()
    cold_rank_before = []
    cold_rank_after = []
    for b in range(B):
        cold_items = np.where(cold_mask[b])[0]
        if len(cold_items) == 0:
            continue
        rank_cold = np.argsort(-scores_cold[b])
        rank_boosted = np.argsort(-scores_boosted[b])
        for item in cold_items:
            pos_cold = np.where(rank_cold == item)[0][0]
            pos_boosted = np.where(rank_boosted == item)[0][0]
            cold_rank_before.append(pos_cold)
            cold_rank_after.append(pos_boosted)

    if cold_rank_before:
        print(f"\n  Cold-start items rank position (lower = better):")
        print(f"    Before boost: mean={np.mean(cold_rank_before):.1f}, median={np.median(cold_rank_before):.1f}")
        print(f"    After boost:  mean={np.mean(cold_rank_after):.1f}, median={np.median(cold_rank_after):.1f}")
        print(f"    Avg improvement: {np.mean(cold_rank_before) - np.mean(cold_rank_after):.1f} positions")

    # --- Plot the response curve with boost annotation ---
    x_range = np.linspace(
        rg.get_percentile_value(data["train_X"], feat_idx, 1),
        rg.get_percentile_value(data["train_X"], feat_idx, 99),
        200,
    ).astype(np.float32)
    effect = model.get_main_effect(feat_idx, x_range)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(x_range, effect, linewidth=2, color="#2c3e50")

    # Annotate boost
    from_effect = float(model.get_main_effect(feat_idx, np.array([low_value], dtype=np.float32))[0])
    to_effect = float(model.get_main_effect(feat_idx, np.array([target_value], dtype=np.float32))[0])

    ax.annotate("", xy=(target_value, to_effect), xytext=(low_value, from_effect),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2.5))
    ax.plot(low_value, from_effect, "o", color="#e74c3c", markersize=10, label=f"Cold-start (p10)")
    ax.plot(target_value, to_effect, "o", color="#2ecc71", markersize=10, label=f"Boosted (p{args.boost_percentile:.0f})")

    delta = to_effect - from_effect
    ax.text((low_value + target_value) / 2, (from_effect + to_effect) / 2 + abs(delta) * 0.15,
            f"delta = {delta:+.3f}", ha="center", fontsize=11, color="#e74c3c", fontweight="bold")

    ax.set_xlabel(f"{feat_name} value")
    ax.set_ylabel("Tower output (score contribution)")
    ax.set_title(f"Item Boosting: {feat_name} response curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("boost_response_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: boost_response_curve.png")

    # --- Warmup blend visualization ---
    print(f"\n  Warmup blend: linear interpolation over 10 days")
    warmup_days = 10
    days = np.arange(0, warmup_days + 1)
    blended_values = rg.warmup_blend(
        real_value=low_value, boost_value=target_value,
        day=days, warmup_days=warmup_days,
    )
    blended_effects = model.get_main_effect(feat_idx, blended_values)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: feature value over time
    ax1.plot(days, blended_values, "o-", color="#3498db", linewidth=2, markersize=6)
    ax1.axhline(y=target_value, color="#2ecc71", linestyle="--", alpha=0.7, label=f"Boost target (p{args.boost_percentile:.0f})")
    ax1.axhline(y=low_value, color="#e74c3c", linestyle="--", alpha=0.7, label="Real value (p10)")
    ax1.set_xlabel("Days since launch")
    ax1.set_ylabel(f"{feat_name} value")
    ax1.set_title("Feature value during warmup")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: score contribution over time
    ax2.plot(days, blended_effects, "o-", color="#e67e22", linewidth=2, markersize=6)
    ax2.axhline(y=to_effect, color="#2ecc71", linestyle="--", alpha=0.7, label="Boosted score")
    ax2.axhline(y=from_effect, color="#e74c3c", linestyle="--", alpha=0.7, label="Real score")
    ax2.set_xlabel("Days since launch")
    ax2.set_ylabel("Score contribution f(x)")
    ax2.set_title("Score contribution during warmup")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig2.suptitle(f"Cold-start warmup: {feat_name} ({warmup_days}-day blend)", fontsize=13)
    fig2.tight_layout()
    fig2.savefig("boost_warmup_blend.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: boost_warmup_blend.png")

    for d in [0, 3, 5, 7, 10]:
        v = rg.warmup_blend(low_value, target_value, d, warmup_days)
        e = float(model.get_main_effect(feat_idx, np.array([v], dtype=np.float32))[0])
        pct = d / warmup_days * 100
        print(f"    Day {d:2d}: x={v:.4f}, f(x)={e:.4f}  ({pct:.0f}% real)")


if __name__ == "__main__":
    main()

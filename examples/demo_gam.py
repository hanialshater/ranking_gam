#!/usr/bin/env python3
"""
GAM / GA2M / ContextGAM demo with response curves and PWL distillation.

Usage:
    python examples/demo_gam.py                                          # default GAM
    python examples/demo_gam.py --context --loss ndcg2pp --activation silu  # best config
    python examples/demo_gam.py --ga2m --ga2m-pairs 20                   # GA2M with interactions
    python examples/demo_gam.py --no-transforms --no-residual --l1-reg 0 # bare GAM
"""

import warnings

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
    parser = argparse.ArgumentParser(description="GAM / GA2M / ContextGAM demo")
    add_common_args(parser)
    parser.add_argument("--context", action="store_true",
                        help="use ContextGAM with learned per-feature importance weights")
    parser.add_argument("--ga2m", action="store_true",
                        help="use GA2M with pairwise interactions")
    parser.add_argument("--ga2m-pairs", type=int, default=20,
                        help="number of interaction pairs for GA2M (default: 20)")
    parser.add_argument("--tower-dropout", type=float, default=0.0,
                        help="tower output dropout rate (default: 0)")
    parser.add_argument("--output-norm", action="store_true",
                        help="enable tower output normalization (BatchNorm)")
    args = resolve_args(parser.parse_args())

    extra = []
    if args.context:
        extra.append("context_weights")
    if args.ga2m:
        extra.append(f"ga2m(pairs={args.ga2m_pairs})")
    if args.tower_dropout > 0:
        extra.append(f"tower_drop={args.tower_dropout}")
    if args.output_norm:
        extra.append("output_norm")
    print_config(args, extra)

    data = load_data(dataset=args.dataset, data_dir=args.data_dir)
    results = run(data, args)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for name, ndcg in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<28} NDCG@{args.k} = {ndcg:.4f}")


def run(data, args):
    """Run GAM/GA2M/ContextGAM training + distillation. Returns results dict."""
    if args.context:
        model_name = "ContextGAM"
    elif args.ga2m:
        model_name = "GA2M"
    else:
        model_name = "GAM"

    K = args.k
    num_features = data["num_features"]
    print("\n" + "=" * 70)
    print(f"{model_name} -- Interpretable Ranking")
    print("=" * 70)

    # --- Build model ---
    interaction_pairs = None
    if args.ga2m:
        from ranking_gam.interactions import select_interactions_correlation
        interaction_pairs = select_interactions_correlation(
            data["train_X"], data["train_y"], top_k=args.ga2m_pairs,
        )

    tower_dropout = getattr(args, "tower_dropout", 0.0)
    output_norm = getattr(args, "output_norm", False)

    if args.context:
        model = rg.ContextGAM(
            num_features=num_features, hidden_dims=[16, 8],
            context_hidden=[64, 32],
            feature_transforms=args.transforms, residual=args.residual,
            tower_dropout=tower_dropout, output_norm=output_norm,
            activation=args.activation,
        )
    elif args.ga2m:
        model = rg.GA2M_Paper(
            num_features=num_features, hidden_dims=[16, 8],
            interaction_pairs=interaction_pairs,
            feature_transforms=args.transforms, residual=args.residual,
            tower_dropout=tower_dropout, output_norm=output_norm,
            activation=args.activation,
        )
    else:
        model = rg.GAM_Paper(
            num_features=num_features, hidden_dims=[16, 8],
            feature_transforms=args.transforms, residual=args.residual,
            tower_dropout=tower_dropout, output_norm=output_norm,
            activation=args.activation,
        )

    if args.transforms:
        model.init_transforms_from_data(data["train_X"])
        print("  Initialized feature transforms from training data percentiles")

    # --- Train ---
    loss_fn = make_loss(args.loss, args.label_smoothing)
    ndcg = rg.train_model(
        model, data["train_loader"], data["eval_loader"], loss_fn,
        epochs=args.epochs, patience=args.patience, device=device,
        lr_schedule=args.lr_schedule,
        l1_output_reg=args.l1_reg, weight_decay=args.weight_decay,
    )
    print(f"\n{model_name} ({args.loss}): NDCG@{K} = {ndcg:.4f}")

    # --- Response curves ---
    feature_names = [f"f_{i}" for i in range(num_features)]
    try:
        from ranking_gam.data.mslr import MSLR_FEATURE_NAMES
        if num_features == 136:
            feature_names = MSLR_FEATURE_NAMES
    except ImportError:
        pass

    fig = plot_response_curves(
        model, feature_names=feature_names, data=data["X_flat"], top_k=15,
    )
    fig.savefig(f"response_curves_{model_name.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: response_curves_{model_name.lower()}.png")

    # --- Distillation: neural towers -> piecewise-linear ---
    if args.context:
        print(f"\n  Distilling {model_name} towers to PWL (K=5 knots)...")
        print("  Note: context weights (w_j) are NOT distilled -- keep context_net for serving")
    else:
        print(f"\n  Distilling {model_name} to piecewise-linear (K=5 knots)...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pwl = rg.distill_to_pwl(model, num_knots=5, train_X=data["train_X"])
    pwl_ndcg = rg.evaluate_pwl(pwl, data["eval_X"], data["eval_y"], k=K)
    gap = ndcg - pwl_ndcg
    print(f"  Neural {model_name}: NDCG@{K} = {ndcg:.4f}")
    print(f"  PWL distilled:      NDCG@{K} = {pwl_ndcg:.4f}  (gap: {gap:+.4f})")
    if args.context:
        print("  (Larger gap expected -- context weights dropped in PWL-only eval)")

    # --- PWL response curves ---
    fig_pwl, axes = plt.subplots(3, 5, figsize=(20, 10))
    axes = axes.flatten()
    importances = []
    for p in pwl["main_effects"]:
        y_range = max(p["y"]) - min(p["y"]) if p["y"] else 0
        importances.append((p["feature"], y_range))
    importances.sort(key=lambda x: -x[1])
    for ax_idx, (feat_idx, _) in enumerate(importances[:15]):
        p = pwl["main_effects"][feat_idx]
        ax = axes[ax_idx]
        ax.plot(p["x"], p["y"], "o-", color="#e74c3c", linewidth=2, markersize=4)
        fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f"f{feat_idx}"
        ax.set_title(f"{fname} (K={len(p['x'])})", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig_pwl.suptitle(f"Distilled PWL Response Curves (NDCG@{K}={pwl_ndcg:.4f})", fontsize=14)
    fig_pwl.tight_layout()
    fig_pwl.savefig(f"pwl_curves_{model_name.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig_pwl)
    print(f"  Saved: pwl_curves_{model_name.lower()}.png")

    return {f"{model_name}_{args.loss}": ndcg, f"{model_name}_PWL": pwl_ndcg}


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SubmodularRankingGAM demo -- diversity-aware reranking with greedy guarantees.

Two-phase training:
  Phase 1: Train base GAM towers with ranking loss
  Phase 2: Freeze base, train concave diversity towers with step-wise ListNet

Usage:
    python examples/demo_submodular.py
    python examples/demo_submodular.py --loss ndcg2pp --activation silu --epochs 30
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ranking_gam as rg
from ranking_gam.viz import plot_diversity_curves, plot_spider, print_diversity_comparison

from _common import (
    add_common_args, device, groupwise_specs, load_data, make_loss,
    print_config, resolve_args,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SubmodularRankingGAM demo")
    add_common_args(parser)
    parser.add_argument("--queries-per-epoch", type=int, default=400,
                        help="queries per epoch for diversity training (default: 400)")
    args = resolve_args(parser.parse_args())
    print_config(args)

    data = load_data(dataset=args.dataset, data_dir=args.data_dir)
    results = run(data, args)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for name, ndcg in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<28} NDCG@{args.k} = {ndcg:.4f}")


def run(data, args):
    """Run SubmodularRankingGAM training + diversity evaluation. Returns results dict."""
    K = args.k
    num_base = data["num_features"]

    print("\n" + "=" * 70)
    print("SubmodularRankingGAM -- Diversity with Greedy Guarantees")
    print("=" * 70)

    submod = rg.SubmodularRankingGAM(
        num_item_features=num_base,
        groupwise_specs=groupwise_specs(num_base),
        item_hidden=[16, 8],
        num_knots=8,
        train_mode="pointwise",
        feature_transforms=args.transforms,
        residual=args.residual,
        activation=args.activation,
    )
    if args.transforms:
        submod.init_transforms_from_data(data["train_X"])
        print("  Initialized feature transforms from training data percentiles")

    # Phase 1: base towers
    loss_fn = make_loss(args.loss, args.label_smoothing, k=args.k)
    submod_ndcg = rg.train_model(
        submod, data["train_loader"], data["eval_loader"],
        loss_fn,
        epochs=args.epochs, patience=args.patience, device=device,
        lr_schedule=args.lr_schedule,
        l1_output_reg=args.l1_reg, weight_decay=args.weight_decay,
    )

    # Phase 2: diversity towers
    qpe = getattr(args, "queries_per_epoch", 400)
    submod = rg.train_diversity_towers(
        submod, data["train_X_aug"], data["train_y"],
        epochs=args.epochs, lr=0.003, k=K, queries_per_epoch=qpe,
        lr_schedule=args.lr_schedule,
        weight_decay=args.weight_decay,
    )

    # Evaluate: base ranking vs greedy reranking
    base_orders = rg.get_base_ranking(
        submod, data["eval_X_aug"], data["eval_y"], k=K, device=device,
    )
    greedy_orders = rg.get_greedy_ranking(
        submod, data["eval_X_aug"], k=K, device=device,
    )

    base_metrics = rg.evaluate_ranking_diversity(
        data["eval_X_aug"], data["eval_y"], base_orders, k=K,
        cat_col=num_base, brand_col=num_base + 1, price_col=min(10, num_base - 1),
    )
    greedy_metrics = rg.evaluate_ranking_diversity(
        data["eval_X_aug"], data["eval_y"], greedy_orders, k=K,
        cat_col=num_base, brand_col=num_base + 1, price_col=min(10, num_base - 1),
    )

    print_diversity_comparison(base_metrics, greedy_metrics)

    # Spider plot
    fig = plot_spider(
        {"Base": base_metrics, "Greedy": greedy_metrics},
        title="SubmodularRankingGAM: Base vs Greedy",
    )
    fig.savefig("spider_base_vs_greedy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: spider_base_vs_greedy.png")

    # Diversity curves
    fig_div = plot_diversity_curves(submod)
    fig_div.savefig("diversity_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig_div)
    print("  Saved: diversity_curves.png")

    return {
        "SubmodularGAM_base": submod_ndcg,
        "SubmodularGAM_greedy": greedy_metrics["ndcg"],
    }


if __name__ == "__main__":
    main()

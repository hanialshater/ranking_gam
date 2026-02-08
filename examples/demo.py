#!/usr/bin/env python3
"""
Full demo: GAM, SubmodularRankingGAM, and MultiObjectiveRankingGAM
on MSLR-WEB10K with diversity evaluation and visualization.

Usage:
    python examples/demo.py                    # run all 3 demos
    python examples/demo.py --demo gam         # only GAM
    python examples/demo.py --demo submodular  # only SubmodularRankingGAM
    python examples/demo.py --demo multi       # only MultiObjectiveRankingGAM
    python examples/demo.py --demo gam,multi   # GAM + Multi-Objective
    python examples/demo.py --epochs 30        # production quality

Performance enhancements:
    python examples/demo.py --demo gam --transforms             # learnable monotone feature transforms
    python examples/demo.py --demo gam --residual               # residual skip connections
    python examples/demo.py --demo gam --cosine                 # cosine annealing LR
    python examples/demo.py --demo gam --l1-reg 0.001           # L1 output regularization
    python examples/demo.py --demo gam --transforms --residual --cosine  # all enhancements
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import ranking_gam as rg
from ranking_gam.data.mslr import MSLR_FEATURE_NAMES
from ranking_gam.viz import (
    plot_response_curves,
    plot_diversity_curves,
    plot_spider,
    print_diversity_comparison,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEMO_CHOICES = {"gam", "submodular", "multi"}


def load_data():
    """Load MSLR-WEB10K and create augmented features with synthetic groupwise columns."""
    print("=" * 70)
    print("Loading MSLR-WEB10K + Creating Synthetic Groupwise Features")
    print("=" * 70)

    train_X, train_y, eval_X, eval_y = rg.load_mslr()

    train_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    eval_ds = TensorDataset(torch.from_numpy(eval_X), torch.from_numpy(eval_y))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False)

    # Synthetic groupwise features (MSLR has no category/brand)
    f0_flat = train_X[:, :, 0].flatten()
    f5_flat = train_X[:, :, 5].flatten()
    f0_bins = np.digitize(
        f0_flat, np.percentile(f0_flat[f0_flat > 0], [20, 40, 60, 80])
    )
    f5_bins = np.digitize(
        f5_flat,
        np.percentile(f5_flat[f5_flat > 0], [12.5, 25, 37.5, 50, 62.5, 75, 87.5]),
    )

    def augment(X, col0_bins, col5_bins):
        B, L, D = X.shape
        X_aug = np.zeros((B, L, D + 2), dtype=X.dtype)
        X_aug[:, :, :D] = X
        X_aug[:, :, D] = col0_bins.reshape(B, L)
        X_aug[:, :, D + 1] = col5_bins.reshape(B, L)
        return X_aug

    train_X_aug = augment(train_X, f0_bins, f5_bins)

    ef0 = eval_X[:, :, 0].flatten()
    ef5 = eval_X[:, :, 5].flatten()
    ef0_bins = np.digitize(
        ef0, np.percentile(f0_flat[f0_flat > 0], [20, 40, 60, 80])
    )
    ef5_bins = np.digitize(
        ef5,
        np.percentile(f5_flat[f5_flat > 0], [12.5, 25, 37.5, 50, 62.5, 75, 87.5]),
    )
    eval_X_aug = augment(eval_X, ef0_bins, ef5_bins)

    X_flat = train_X.reshape(-1, 136)

    return {
        "train_X": train_X,
        "train_y": train_y,
        "eval_X": eval_X,
        "eval_y": eval_y,
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "train_X_aug": train_X_aug,
        "eval_X_aug": eval_X_aug,
        "X_flat": X_flat,
    }


def groupwise_specs(num_base=136):
    return [
        {
            "name": "category_novelty",
            "type": "category_novelty",
            "column": num_base,
            "x_min": 0.0,
            "x_max": 1.0,
        },
        {
            "name": "brand_novelty",
            "type": "brand_novelty",
            "column": num_base + 1,
            "x_min": 0.0,
            "x_max": 1.0,
        },
        {
            "name": "price_spread",
            "type": "price_spread",
            "column": 10,
            "x_min": 0.0,
            "x_max": 3.0,
        },
    ]


# =========================================================================
# Individual demos
# =========================================================================

def demo_gam(data, epochs, K, transforms=False, residual=False, cosine=False, l1_reg=0.0):
    """Demo 1: GAM -- Interpretable Ranking."""
    print("\n" + "=" * 70)
    print("Demo 1: GAM -- Interpretable Ranking")
    print("=" * 70)

    extras = []
    if transforms:
        extras.append("transforms")
    if residual:
        extras.append("residual")
    if cosine:
        extras.append("cosine_lr")
    if l1_reg > 0:
        extras.append(f"l1={l1_reg}")
    if extras:
        print(f"  Enhancements: {', '.join(extras)}")

    gam = rg.GAM_Paper(
        num_features=136, hidden_dims=[16, 8],
        feature_transforms=transforms, residual=residual,
    )
    if transforms:
        gam.init_transforms_from_data(data["train_X"])
        print("  Initialized feature transforms from training data percentiles")

    gam_ndcg = rg.train_model(
        gam, data["train_loader"], data["eval_loader"], rg.ListNetLoss(),
        epochs=epochs, patience=7, device=device,
        lr_schedule="cosine" if cosine else "constant",
        l1_output_reg=l1_reg,
    )
    print(f"\nGAM (ListNet): NDCG@{K} = {gam_ndcg:.4f}")

    fig = plot_response_curves(
        gam, feature_names=MSLR_FEATURE_NAMES, data=data["X_flat"], top_k=15,
    )
    fig.savefig("response_curves_gam.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: response_curves_gam.png")

    return {"GAM_ListNet": gam_ndcg}


def demo_submodular(data, epochs, K, queries_per_epoch,
                    transforms=False, residual=False, cosine=False, l1_reg=0.0):
    """Demo 2: SubmodularRankingGAM -- Diversity with Greedy Guarantees."""
    print("\n" + "=" * 70)
    print("Demo 2: SubmodularRankingGAM -- Diversity with Greedy Guarantees")
    print("=" * 70)

    num_base = 136
    submod = rg.SubmodularRankingGAM(
        num_item_features=num_base,
        groupwise_specs=groupwise_specs(num_base),
        item_hidden=[16, 8],
        num_knots=8,
        train_mode="pointwise",
        feature_transforms=transforms,
        residual=residual,
    )
    if transforms:
        submod.init_transforms_from_data(data["train_X"])
        print("  Initialized feature transforms from training data percentiles")

    # Phase 1: base towers
    submod_ndcg = rg.train_model(
        submod, data["train_loader"], data["eval_loader"], rg.ListNetLoss(),
        epochs=epochs, patience=7, device=device,
        lr_schedule="cosine" if cosine else "constant",
        l1_output_reg=l1_reg,
    )

    # Phase 2: diversity towers
    submod = rg.train_diversity_towers(
        submod, data["train_X_aug"], data["train_y"],
        epochs=epochs, lr=0.003, k=K, queries_per_epoch=queries_per_epoch,
    )

    base_orders = rg.get_base_ranking(
        submod, data["eval_X_aug"], data["eval_y"], k=K, device=device,
    )
    greedy_orders = rg.get_greedy_ranking(
        submod, data["eval_X_aug"], k=K, device=device,
    )

    base_metrics = rg.evaluate_ranking_diversity(
        data["eval_X_aug"], data["eval_y"], base_orders, k=K,
        cat_col=num_base, brand_col=num_base + 1, price_col=10,
    )
    greedy_metrics = rg.evaluate_ranking_diversity(
        data["eval_X_aug"], data["eval_y"], greedy_orders, k=K,
        cat_col=num_base, brand_col=num_base + 1, price_col=10,
    )

    print_diversity_comparison(base_metrics, greedy_metrics)

    fig = plot_spider(
        {"Base": base_metrics, "Greedy": greedy_metrics},
        title="SubmodularRankingGAM: Base vs Greedy",
    )
    fig.savefig("spider_base_vs_greedy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: spider_base_vs_greedy.png")

    fig_div = plot_diversity_curves(submod)
    fig_div.savefig("diversity_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig_div)
    print("  Saved: diversity_curves.png")

    return {
        "SubmodularGAM_base": submod_ndcg,
        "SubmodularGAM_greedy": greedy_metrics["ndcg"],
    }


def demo_multi_objective(data, epochs, K, queries_per_epoch):
    """Demo 3: Multi-Objective Ranking GAM."""
    print("\n" + "=" * 70)
    print("Demo 3: Multi-Objective Ranking GAM")
    print("=" * 70)

    num_base = 136
    mo_objectives = [
        {
            "name": "relevance",
            "type": "pointwise",
            "features": list(range(num_base)),
            "tower": "mlp",
            "weight": 0.35,
        },
        {
            "name": "revenue",
            "type": "pointwise",
            "features": [10],
            "tower": "monotone",
            "x_min": 0.0,
            "x_max": 1.0,
            "weight": 0.35,
        },
        {
            "name": "diversity",
            "type": "groupwise",
            "groupwise_specs": [
                {
                    "name": "category_novelty",
                    "type": "category_novelty",
                    "column": num_base,
                    "x_min": 0.0,
                    "x_max": 1.0,
                },
                {
                    "name": "brand_novelty",
                    "type": "brand_novelty",
                    "column": num_base + 1,
                    "x_min": 0.0,
                    "x_max": 1.0,
                },
            ],
            "weight": 0.20,
        },
        {
            "name": "freshness",
            "type": "pointwise",
            "features": [20],
            "tower": "monotone",
            "x_min": 0.0,
            "x_max": 1.0,
            "weight": 0.10,
        },
    ]

    mo_model = rg.MultiObjectiveRankingGAM(
        objectives=mo_objectives,
        num_item_features=num_base,
        hidden_dims=[16, 8],
        num_knots=8,
    ).to(device)

    rg.train_multi_objective(
        mo_model, data["train_X_aug"], data["train_y"],
        epochs=epochs, lr=0.003, k=K, queries_per_epoch=queries_per_epoch,
    )

    scenarios = {
        "Max Revenue": {"relevance": 0.20, "revenue": 0.70, "diversity": 0.05, "freshness": 0.05},
        "Balanced": {"relevance": 0.35, "revenue": 0.35, "diversity": 0.20, "freshness": 0.10},
        "Max Diversity": {"relevance": 0.20, "revenue": 0.10, "diversity": 0.60, "freshness": 0.10},
        "New Arrivals": {"relevance": 0.30, "revenue": 0.15, "diversity": 0.10, "freshness": 0.45},
    }

    scenario_metrics = {}
    for name, w in scenarios.items():
        orders = []
        with torch.no_grad():
            for qi in range(min(500, len(data["eval_X_aug"]))):
                x_q = torch.from_numpy(data["eval_X_aug"][qi : qi + 1]).float().to(device)
                order = mo_model.greedy_rerank(x_q, k=K, weights=w)
                orders.append([o for o in order[0].tolist() if o >= 0])

        m = rg.evaluate_ranking_diversity(
            data["eval_X_aug"][: len(orders)], data["eval_y"][: len(orders)],
            orders, k=K,
            cat_col=num_base, brand_col=num_base + 1, price_col=10,
        )
        scenario_metrics[name] = m
        print(f"  {name:<18} NDCG={m['ndcg']:.4f}  ILD={m['ild']:.4f}")

    fig = plot_spider(
        scenario_metrics,
        title="Multi-Objective: Weight Scenarios",
        colors=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"],
    )
    fig.savefig("spider_weight_scenarios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: spider_weight_scenarios.png")

    return {}


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ranking_gam demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Demo choices:
  gam          GAM with response curve visualization
  submodular   SubmodularRankingGAM with diversity evaluation
  multi        Multi-Objective Ranking GAM with weight scenarios

Examples:
  python examples/demo.py --demo gam
  python examples/demo.py --demo submodular,multi
  python examples/demo.py --demo all --epochs 30
  python examples/demo.py --demo gam --transforms --cosine --residual
        """,
    )
    parser.add_argument(
        "--demo",
        type=str,
        default="all",
        help="comma-separated demos to run: gam,submodular,multi,all (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries-per-epoch", type=int, default=400)
    parser.add_argument("--transforms", action="store_true",
                        help="enable learnable monotone feature transforms (percentile-initialized)")
    parser.add_argument("--residual", action="store_true",
                        help="add residual skip connections to MLP towers")
    parser.add_argument("--cosine", action="store_true",
                        help="use cosine annealing LR schedule")
    parser.add_argument("--l1-reg", type=float, default=0.0,
                        help="L1 output regularization weight (default: 0)")
    args = parser.parse_args()

    # Parse demo selection
    if args.demo == "all":
        demos = DEMO_CHOICES
    else:
        demos = {d.strip() for d in args.demo.split(",")}
        unknown = demos - DEMO_CHOICES
        if unknown:
            parser.error(f"Unknown demo(s): {unknown}. Choose from: {DEMO_CHOICES}")

    EPOCHS = args.epochs
    K = args.k
    QPE = args.queries_per_epoch

    print(f"Running demos: {', '.join(sorted(demos))}")
    config_parts = [f"epochs={EPOCHS}", f"k={K}", f"queries_per_epoch={QPE}"]
    if args.transforms:
        config_parts.append("transforms=ON")
    if args.residual:
        config_parts.append("residual=ON")
    if args.cosine:
        config_parts.append("cosine_lr=ON")
    if args.l1_reg > 0:
        config_parts.append(f"l1_reg={args.l1_reg}")
    print(f"Config: {', '.join(config_parts)}\n")

    data = load_data()
    results = {}

    enhance_kw = dict(
        transforms=args.transforms, residual=args.residual,
        cosine=args.cosine, l1_reg=args.l1_reg,
    )

    if "gam" in demos:
        results.update(demo_gam(data, EPOCHS, K, **enhance_kw))

    if "submodular" in demos:
        results.update(demo_submodular(data, EPOCHS, K, QPE, **enhance_kw))

    if "multi" in demos:
        results.update(demo_multi_objective(data, EPOCHS, K, QPE))

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        for name, ndcg in sorted(results.items(), key=lambda x: -x[1]):
            print(f"  {name:<28} NDCG@{K} = {ndcg:.4f}")
        print(f"\n  (Set --epochs 30 for production-quality results)")


if __name__ == "__main__":
    main()

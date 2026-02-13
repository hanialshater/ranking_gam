#!/usr/bin/env python3
"""
Multi-Objective Ranking GAM demo -- Pareto-controllable reranking.

Train once, then slide objective weights at serving time (no retraining).
Evaluates 4 weight scenarios: Max Revenue, Balanced, Max Diversity, New Arrivals.

Usage:
    python examples/demo_multi.py
    python examples/demo_multi.py --epochs 20 --k 10
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ranking_gam as rg
from ranking_gam.viz import plot_spider

from _common import (
    add_common_args, device, groupwise_specs, load_data, print_config,
    resolve_args,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Objective Ranking GAM demo")
    add_common_args(parser)
    parser.add_argument("--queries-per-epoch", type=int, default=400,
                        help="queries per epoch (default: 400)")
    args = resolve_args(parser.parse_args())
    print_config(args)

    data = load_data()
    run(data, args)


def run(data, args):
    """Run MultiObjectiveRankingGAM training + scenario evaluation."""
    K = args.k
    num_base = 136
    qpe = getattr(args, "queries_per_epoch", 400)

    print("\n" + "=" * 70)
    print("Multi-Objective Ranking GAM")
    print("=" * 70)

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
            "groupwise_specs": groupwise_specs(num_base)[:2],  # category + brand
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
        epochs=args.epochs, lr=0.003, k=K, queries_per_epoch=qpe,
        lr_schedule=args.lr_schedule,
        weight_decay=args.weight_decay,
    )

    # Evaluate under different weight scenarios
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

    return scenario_metrics


if __name__ == "__main__":
    main()

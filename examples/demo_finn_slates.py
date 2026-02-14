#!/usr/bin/env python3
"""
FINN Slate Optimization with GA2M -- Interpretable position-item interactions.

Trains a GA2M on the FINN.no RecSys Slates dataset to learn:
  - Per-feature shape functions f_j(x_j) (position bias, popularity effect, ...)
  - Pairwise interaction surfaces f_{jk}(x_j, x_k) that reveal:
      * Which categories need top slate positions
      * How popularity interacts with position (position bias)
      * Search vs recommendation position sensitivity
      * Category redundancy effects (diminishing returns = submodularity)

Usage:
    python examples/demo_finn_slates.py
    python examples/demo_finn_slates.py --max-queries 50000 --epochs 30
    python examples/demo_finn_slates.py --loss ndcg2pp --activation silu

Requires: pip install gdown  (for automatic FINN data download)
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import ranking_gam as rg  # noqa: E402
from ranking_gam.data.finn import (  # noqa: E402
    load_finn,
)
from ranking_gam.viz.plots import (  # noqa: E402
    plot_interaction_grid,
    plot_interaction_heatmap,
    plot_response_curves,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loss(name):
    if name == "listnet":
        return rg.ListNetLoss(label_smoothing=0.1)
    elif name == "ndcg2pp":
        return rg.LambdaLoss.ndcg2pp()
    elif name == "approxndcg":
        return rg.ApproxNDCGLoss(alpha=10)
    elif name == "pairwise":
        return rg.PairwiseLoss(sigma=1.0)
    else:
        raise ValueError(f"Unknown loss: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="FINN Slate Optimization with GA2M"
    )
    parser.add_argument("--data-dir", default="data/finn",
                        help="FINN data directory (auto-downloads)")
    parser.add_argument("--max-queries", type=int, default=20000,
                        help="max interactions to use (default: 20000)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="training epochs (default: 20)")
    parser.add_argument("--patience", type=int, default=8,
                        help="early stopping patience")
    parser.add_argument("--loss", default="listnet",
                        choices=["listnet", "ndcg2pp", "approxndcg", "pairwise"],
                        help="ranking loss (default: listnet)")
    parser.add_argument("--activation", default="silu",
                        choices=["relu", "silu", "gelu"],
                        help="tower activation (default: silu)")
    parser.add_argument("--no-interactions", action="store_true",
                        help="train GAM only (no interactions) for comparison")
    parser.add_argument("--output-dir", default=".",
                        help="directory for output plots")
    args = parser.parse_args()

    print("=" * 70)
    print("FINN Slate Optimization with GA2M")
    print("=" * 70)
    print(f"Config: max_queries={args.max_queries}, epochs={args.epochs}, "
          f"loss={args.loss}, activation={args.activation}")
    print()

    # --- Load data ---
    data = load_finn(
        data_dir=args.data_dir,
        max_queries=args.max_queries,
    )

    train_X = data["train_X"]
    train_y = data["train_y"]
    eval_X = data["eval_X"]
    eval_y = data["eval_y"]
    num_features = data["num_features"]
    feature_names = data["feature_names"]
    interaction_pairs = data["interaction_pairs"]

    train_ds = TensorDataset(
        torch.from_numpy(train_X),
        torch.from_numpy(train_y),
    )
    eval_ds = TensorDataset(
        torch.from_numpy(eval_X),
        torch.from_numpy(eval_y),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=64)

    loss_fn = make_loss(args.loss)

    # --- Train GAM baseline ---
    print("\n" + "=" * 70)
    print("[1/2] Training GAM baseline (no interactions)...")
    print("=" * 70)

    gam = rg.GAM_Paper(
        num_features=num_features,
        hidden_dims=[32, 16],
        activation=args.activation,
        feature_transforms=True,
        residual=True,
    )
    gam.init_transforms_from_data(train_X)

    gam_ndcg = rg.train_model(
        gam, train_loader, eval_loader, loss_fn,
        epochs=args.epochs, patience=args.patience, device=device,
        lr_schedule="plateau", weight_decay=0.01,
    )
    print(f"\n  GAM baseline NDCG@10 = {gam_ndcg:.4f}")

    if args.no_interactions:
        # Plot response curves and exit
        fig = plot_response_curves(
            gam, feature_names=feature_names,
            data=train_X.reshape(-1, num_features),
            top_k=num_features,
        )
        outpath = f"{args.output_dir}/finn_gam_responses.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outpath}")
        return

    # --- Train GA2M with position-item interactions ---
    print("\n" + "=" * 70)
    print("[2/2] Training GA2M with position-item interactions...")
    print(f"  Interaction pairs: {interaction_pairs}")
    for f1, f2 in interaction_pairs:
        print(f"    ({feature_names[f1]}, {feature_names[f2]})")
    print("=" * 70)

    ga2m = rg.GA2M_Paper(
        num_features=num_features,
        interaction_pairs=interaction_pairs,
        hidden_dims=[32, 16],
        interaction_hidden=[16, 8],
        activation=args.activation,
        feature_transforms=True,
        residual=True,
    )
    ga2m.init_transforms_from_data(train_X)

    ga2m_ndcg = rg.train_model(
        ga2m, train_loader, eval_loader, loss_fn,
        epochs=args.epochs, patience=args.patience, device=device,
        lr_schedule="plateau", weight_decay=0.01,
    )
    print(f"\n  GA2M NDCG@10 = {ga2m_ndcg:.4f}")

    # --- Results ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    delta = ga2m_ndcg - gam_ndcg
    print(f"  GAM (no interactions):   NDCG@10 = {gam_ndcg:.4f}")
    print(f"  GA2M (with interactions): NDCG@10 = {ga2m_ndcg:.4f}")
    print(f"  Interaction gain:         {delta:+.4f}")

    # --- Visualize ---
    print("\n  Generating plots...")

    # 1. Response curves (main effects)
    fig = plot_response_curves(
        ga2m, feature_names=feature_names,
        data=train_X.reshape(-1, num_features),
        top_k=num_features,
    )
    outpath = f"{args.output_dir}/finn_ga2m_responses.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")

    # 2. All interaction heatmaps in a grid
    fig = plot_interaction_grid(
        ga2m, feature_names=feature_names, n_points=50,
    )
    outpath = f"{args.output_dir}/finn_ga2m_interactions.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")

    # 3. Individual high-res heatmap for position x category
    fig = plot_interaction_heatmap(
        ga2m, pair_idx=0, n_points=80,
        feature_names=feature_names,
    )
    outpath = f"{args.output_dir}/finn_position_x_category.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")

    print("\n  Done! Inspect the heatmaps to see which categories need")
    print("  top positions, how popularity interacts with slot, and")
    print("  where category redundancy causes diminishing returns.")


if __name__ == "__main__":
    main()

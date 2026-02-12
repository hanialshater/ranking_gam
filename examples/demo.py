#!/usr/bin/env python3
"""
Full demo: GAM, SubmodularRankingGAM, and MultiObjectiveRankingGAM
on MSLR-WEB10K with diversity evaluation and visualization.

Usage:
    python examples/demo.py                    # run all demos (best config by default)
    python examples/demo.py --demo gam         # only GAM
    python examples/demo.py --demo submodular  # only SubmodularRankingGAM
    python examples/demo.py --demo multi       # only MultiObjectiveRankingGAM
    python examples/demo.py --demo gbdt        # GBDT baseline + boosted GAM

All enhancements are ON by default. Use --no-* flags to disable:
    python examples/demo.py --demo gam --no-transforms          # disable feature transforms
    python examples/demo.py --demo gam --no-residual            # disable residual skip
    python examples/demo.py --demo gam --no-cosine              # disable cosine LR
    python examples/demo.py --demo gam --l1-reg 0               # disable L1 reg
    python examples/demo.py --demo gam --loss pairwise          # pairwise loss (stable gradients)
    python examples/demo.py --demo gam --lr-schedule plateau    # adaptive LR reduction
    python examples/demo.py --demo gam --context                # context-weighted GAM (+2-3 NDCG pts)
    python examples/demo.py --demo gam --no-transforms --no-residual --no-cosine --l1-reg 0  # bare GAM
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

DEMO_CHOICES = {"gam", "submodular", "multi", "gbdt"}


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

def _make_loss(loss_name, label_smoothing=0.0):
    """Create loss function by name."""
    if loss_name == "listnet":
        return rg.ListNetLoss(label_smoothing=label_smoothing)
    elif loss_name == "pairwise":
        return rg.PairwiseLoss(sigma=1.0)
    elif loss_name == "approxndcg":
        return rg.ApproxNDCGLoss(alpha=10)
    elif loss_name == "ndcg2pp":
        return rg.LambdaLoss.ndcg2pp()
    elif loss_name == "listmle":
        return rg.ListMLELoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}. Choose from: listnet, pairwise, approxndcg, ndcg2pp, listmle")


def demo_gam(data, epochs, K, transforms=True, residual=True, lr_schedule="cosine",
             l1_reg=0.0001, label_smoothing=0.1, weight_decay=0.01,
             loss="listnet", tower_dropout=0.0, output_norm=False,
             ga2m=False, ga2m_pairs=20, activation="relu", context=False,
             patience=10):
    """Demo 1: GAM / GA2M / ContextGAM -- Interpretable Ranking."""
    if context:
        model_name = "ContextGAM"
    elif ga2m:
        model_name = "GA2M"
    else:
        model_name = "GAM"
    print("\n" + "=" * 70)
    print(f"Demo 1: {model_name} -- Interpretable Ranking")
    print("=" * 70)

    extras = []
    if transforms:
        extras.append("transforms")
    if residual:
        extras.append("residual")
    if lr_schedule != "constant":
        extras.append(f"lr={lr_schedule}")
    if l1_reg > 0:
        extras.append(f"l1={l1_reg}")
    if label_smoothing > 0:
        extras.append(f"label_smooth={label_smoothing}")
    if weight_decay > 0:
        extras.append(f"wd={weight_decay}")
    if tower_dropout > 0:
        extras.append(f"tower_drop={tower_dropout}")
    if output_norm:
        extras.append("output_norm")
    if context:
        extras.append("context_weights")
    if ga2m:
        extras.append(f"ga2m(top_{ga2m_pairs}_pairs)")
    if activation != "relu":
        extras.append(f"activation={activation}")
    extras.append(f"loss={loss}")
    print(f"  Enhancements: {', '.join(extras)}")

    interaction_pairs = None
    if ga2m:
        from ranking_gam.interactions import select_interactions_correlation
        interaction_pairs = select_interactions_correlation(
            data["train_X"], data["train_y"], top_k=ga2m_pairs,
        )

    if context:
        model = rg.ContextGAM(
            num_features=136, hidden_dims=[16, 8],
            context_hidden=[64, 32],
            feature_transforms=transforms, residual=residual,
            tower_dropout=tower_dropout, output_norm=output_norm,
            activation=activation,
        )
    elif ga2m:
        model = rg.GA2M_Paper(
            num_features=136, hidden_dims=[16, 8],
            interaction_pairs=interaction_pairs,
            feature_transforms=transforms, residual=residual,
            tower_dropout=tower_dropout, output_norm=output_norm,
            activation=activation,
        )
    else:
        model = rg.GAM_Paper(
            num_features=136, hidden_dims=[16, 8],
            feature_transforms=transforms, residual=residual,
            tower_dropout=tower_dropout, output_norm=output_norm,
            activation=activation,
        )
    if transforms:
        model.init_transforms_from_data(data["train_X"])
        print("  Initialized feature transforms from training data percentiles")

    loss_fn = _make_loss(loss, label_smoothing)
    ndcg = rg.train_model(
        model, data["train_loader"], data["eval_loader"], loss_fn,
        epochs=epochs, patience=patience, device=device,
        lr_schedule=lr_schedule,
        l1_output_reg=l1_reg, weight_decay=weight_decay,
    )
    print(f"\n{model_name} ({loss}): NDCG@{K} = {ndcg:.4f}")

    fig = plot_response_curves(
        model, feature_names=MSLR_FEATURE_NAMES, data=data["X_flat"], top_k=15,
    )
    fig.savefig(f"response_curves_{model_name.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: response_curves_{model_name.lower()}.png")

    return {f"{model_name}_{loss}": ndcg}


def demo_submodular(data, epochs, K, queries_per_epoch,
                    transforms=True, residual=True, lr_schedule="cosine",
                    l1_reg=0.0001, label_smoothing=0.1, weight_decay=0.01,
                    patience=10, activation="relu", loss="listnet"):
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
        activation=activation,
    )
    if transforms:
        submod.init_transforms_from_data(data["train_X"])
        print("  Initialized feature transforms from training data percentiles")

    # Phase 1: base towers
    loss_fn = _make_loss(loss, label_smoothing)
    submod_ndcg = rg.train_model(
        submod, data["train_loader"], data["eval_loader"],
        loss_fn,
        epochs=epochs, patience=patience, device=device,
        lr_schedule=lr_schedule,
        l1_output_reg=l1_reg, weight_decay=weight_decay,
    )

    # Phase 2: diversity towers
    submod = rg.train_diversity_towers(
        submod, data["train_X_aug"], data["train_y"],
        epochs=epochs, lr=0.003, k=K, queries_per_epoch=queries_per_epoch,
        lr_schedule=lr_schedule,
        weight_decay=weight_decay,
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


def demo_multi_objective(data, epochs, K, queries_per_epoch,
                         transforms=True, residual=True, lr_schedule="cosine",
                         l1_reg=0.0001, label_smoothing=0.1, weight_decay=0.01,
                         patience=10):
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
        lr_schedule=lr_schedule,
        weight_decay=weight_decay,
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


def demo_gbdt(data, epochs, K, transforms=True, residual=True, lr_schedule="cosine",
              l1_reg=0.0001, label_smoothing=0.1, weight_decay=0.01,
              patience=10, activation="relu", loss="listnet"):
    """Demo 4: GAM -> GBDT on residuals -> magic curve tower."""
    print("\n" + "=" * 70)
    print("Demo 4: GAM + GBDT Magic Curve (Residual Boosting)")
    print("=" * 70)
    print("  Pipeline: GAM first -> GBDT on residuals -> add magic curve tower")

    from ranking_gam.training.boosting import (
        train_gbdt_baseline,
        compute_gbdt_residual_feature,
    )

    def make_loaders(X, y):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        train_l = DataLoader(ds, batch_size=32, shuffle=True)
        eval_ds = TensorDataset(torch.from_numpy(data["eval_X"]), torch.from_numpy(data["eval_y"]))
        eval_l = DataLoader(eval_ds, batch_size=32)
        return train_l, eval_l

    # --- Stage 1: Train GAM on raw features (interpretable signal) ---
    print("\n  [Stage 1] Training GAM on raw features...")
    gam = rg.GAM_Paper(
        num_features=136, hidden_dims=[16, 8],
        feature_transforms=transforms, residual=residual,
        activation=activation,
    )
    if transforms:
        gam.init_transforms_from_data(data["train_X"])

    loss_fn = _make_loss(loss, label_smoothing)
    train_loader, eval_loader = make_loaders(data["train_X"], data["train_y"])
    gam_ndcg = rg.train_model(
        gam, train_loader, eval_loader, loss_fn,
        epochs=epochs, patience=patience, device=device,
        lr_schedule=lr_schedule,
        l1_output_reg=l1_reg, weight_decay=weight_decay,
    )
    print(f"  Stage 1 GAM: NDCG@{K} = {gam_ndcg:.4f}")

    # --- Stage 2: Compute residuals, train GBDT regressor on what GAM missed ---
    print("\n  [Stage 2] Training GBDT regressor on GAM residuals...")
    from ranking_gam.training.boosting import _gam_predict, _train_gbdt_regression

    gam_scores_train = _gam_predict(gam, data["train_X"], device)
    gam_scores_eval = _gam_predict(gam, data["eval_X"], device)

    def _make_residual_labels(y_true, gam_scores):
        valid = y_true >= 0
        residuals = np.full_like(y_true, -1, dtype=np.float32)
        if valid.any():
            gs = gam_scores.copy()
            g_mean, g_std = gs[valid].mean(), max(gs[valid].std(), 1e-6)
            y_mean, y_std = y_true[valid].astype(np.float32).mean(), max(y_true[valid].astype(np.float32).std(), 1e-6)
            gs_norm = (gs - g_mean) / g_std * y_std + y_mean
            residuals[valid] = y_true[valid].astype(np.float32) - gs_norm[valid]
        return residuals

    train_residuals = _make_residual_labels(data["train_y"], gam_scores_train)
    eval_residuals = _make_residual_labels(data["eval_y"], gam_scores_eval)

    gbdt_residual = _train_gbdt_regression(
        data["train_X"], train_residuals,
        data["eval_X"], eval_residuals,
        n_estimators=300,
    )

    # --- Stage 3: Add magic curve tower (freeze D towers, train tower D+1) ---
    print("\n  [Stage 3] Training magic curve tower (D original towers frozen)...")
    train_X_boosted = compute_gbdt_residual_feature(gbdt_residual, data["train_X"])
    eval_X_boosted = compute_gbdt_residual_feature(gbdt_residual, data["eval_X"])

    boosted_gam = rg.GAM_Paper(
        num_features=137, hidden_dims=[16, 8],
        feature_transforms=transforms, residual=residual,
        activation=activation,
    )

    # Copy trained weights from Stage 1 into first 136 towers
    with torch.no_grad():
        for j in range(136):
            boosted_gam.towers[j].load_state_dict(gam.towers[j].state_dict())
        if transforms:
            for j in range(136):
                boosted_gam.feature_transforms[j].load_state_dict(
                    gam.feature_transforms[j].state_dict()
                )
            # Init transform for magic curve from boosted data
            boosted_gam.init_transforms_from_data(train_X_boosted)

    # Freeze the 136 original towers and their transforms
    for j in range(136):
        for param in boosted_gam.towers[j].parameters():
            param.requires_grad = False
        if transforms:
            for param in boosted_gam.feature_transforms[j].parameters():
                param.requires_grad = False

    def make_loaders_boosted(X, y):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        train_l = DataLoader(ds, batch_size=32, shuffle=True)
        eval_ds = TensorDataset(torch.from_numpy(eval_X_boosted), torch.from_numpy(data["eval_y"]))
        eval_l = DataLoader(eval_ds, batch_size=32)
        return train_l, eval_l

    train_loader_b, eval_loader_b = make_loaders_boosted(train_X_boosted, data["train_y"])
    boosted_ndcg = rg.train_model(
        boosted_gam, train_loader_b, eval_loader_b,
        loss_fn,
        epochs=epochs, patience=patience, device=device,
        lr_schedule=lr_schedule,
        l1_output_reg=l1_reg, weight_decay=weight_decay,
    )

    # --- Standalone GBDT baseline (for comparison only, not part of pipeline) ---
    print("\n  [Comparison] Training standalone GBDT (LambdaMART) for reference...")
    gbdt_standalone, gbdt_standalone_ndcg = train_gbdt_baseline(
        data["train_X"], data["train_y"],
        data["eval_X"], data["eval_y"],
        k=K, n_estimators=300,
    )

    print(f"\n  Summary:")
    print(f"    GAM only (136 towers):         NDCG@{K} = {gam_ndcg:.4f}")
    print(f"    GAM + magic curve (137 towers): NDCG@{K} = {boosted_ndcg:.4f}")
    print(f"    GBDT standalone (black box):    NDCG@{K} = {gbdt_standalone_ndcg:.4f}")
    delta = boosted_ndcg - gam_ndcg
    print(f"    Magic curve gain:               {delta:+.4f}")

    # Plot response curve for the magic curve tower (feature 136)
    boosted_gam.eval()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x_vals = np.linspace(0, 1, 100).astype(np.float32)
    effect = boosted_gam.get_main_effect(136, x_vals)
    ax.plot(x_vals, effect, linewidth=2, color="#e74c3c")
    ax.set_xlabel("GBDT residual score (normalized)")
    ax.set_ylabel("Tower output")
    ax.set_title("Magic Curve: f(GBDT_residual_score)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("magic_curve_response.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: magic_curve_response.png")

    return {
        "GAM_only": gam_ndcg,
        "GAM+magic_curve": boosted_ndcg,
        "GBDT_standalone": gbdt_standalone_ndcg,
    }


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
  gbdt         GBDT baseline + residual-boosted GAM (requires lightgbm)

Examples:
  python examples/demo.py --demo gam                         # best config (default)
  python examples/demo.py --demo gam --epochs 30             # production quality
  python examples/demo.py --demo gam --loss approxndcg       # compare losses
  python examples/demo.py --demo gam --context               # context-weighted GAM (higher NDCG)
  python examples/demo.py --demo gam --ga2m --ga2m-pairs 20  # GA2M with interactions
  python examples/demo.py --demo gam --tower-dropout 0.1     # regularize via tower dropout
  python examples/demo.py --demo gam --output-norm           # center tower outputs
  python examples/demo.py --demo gbdt                        # compare GAM vs GBDT vs boosted
  python examples/demo.py --demo gam --no-transforms --no-residual --no-cosine --l1-reg 0  # bare GAM
        """,
    )
    parser.add_argument(
        "--demo",
        type=str,
        default="all",
        help="comma-separated demos to run: gam,submodular,multi,all (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10,
                        help="early stopping patience (default: 10)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries-per-epoch", type=int, default=400)
    # Enhancements ON by default -- use --no-* to disable
    parser.add_argument("--no-transforms", action="store_true",
                        help="disable learnable monotone feature transforms")
    parser.add_argument("--no-residual", action="store_true",
                        help="disable residual skip connections")
    parser.add_argument("--no-cosine", action="store_true",
                        help="disable cosine annealing LR schedule")
    parser.add_argument("--lr-schedule", type=str, default=None,
                        choices=["constant", "cosine", "plateau"],
                        help="LR schedule (default: cosine unless --no-cosine)")
    parser.add_argument("--l1-reg", type=float, default=0.0001,
                        help="L1 output regularization weight (default: 0.0001, 0 to disable)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="label smoothing for ListNet (default: 0.1, 0 to disable)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay (default: 0.01, 0 to disable)")
    # New improvement flags (off by default for comparison)
    parser.add_argument("--loss", type=str, default="listnet",
                        choices=["listnet", "pairwise", "approxndcg", "ndcg2pp", "listmle"],
                        help="ranking loss function (default: listnet)")
    parser.add_argument("--tower-dropout", type=float, default=0.0,
                        help="tower output dropout rate (default: 0, e.g. 0.1)")
    parser.add_argument("--output-norm", action="store_true",
                        help="enable tower output normalization (BatchNorm)")
    parser.add_argument("--context", action="store_true",
                        help="use ContextGAM with learned per-feature importance weights")
    parser.add_argument("--ga2m", action="store_true",
                        help="use GA2M with pairwise interactions instead of GAM")
    parser.add_argument("--ga2m-pairs", type=int, default=20,
                        help="number of interaction pairs for GA2M (default: 20)")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "silu", "gelu"],
                        help="tower activation function (default: relu, try silu for smoother curves)")
    # Backward compat: keep --transforms etc. as no-ops (already default)
    parser.add_argument("--transforms", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--residual", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--cosine", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Resolve flags: --no-* overrides defaults
    use_transforms = not args.no_transforms
    use_residual = not args.no_residual
    # --lr-schedule takes precedence over --no-cosine
    if args.lr_schedule is not None:
        lr_schedule = args.lr_schedule
    elif args.no_cosine:
        lr_schedule = "constant"
    else:
        lr_schedule = "cosine"

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
    config_parts = [f"epochs={EPOCHS}", f"k={K}", f"loss={args.loss}"]
    config_parts.append(f"transforms={'ON' if use_transforms else 'OFF'}")
    config_parts.append(f"residual={'ON' if use_residual else 'OFF'}")
    config_parts.append(f"lr_schedule={lr_schedule}")
    if args.l1_reg > 0:
        config_parts.append(f"l1={args.l1_reg}")
    if args.label_smoothing > 0:
        config_parts.append(f"label_smooth={args.label_smoothing}")
    if args.weight_decay > 0:
        config_parts.append(f"wd={args.weight_decay}")
    if args.tower_dropout > 0:
        config_parts.append(f"tower_drop={args.tower_dropout}")
    if args.output_norm:
        config_parts.append("output_norm")
    if args.context:
        config_parts.append("context_weights")
    if args.ga2m:
        config_parts.append(f"ga2m(pairs={args.ga2m_pairs})")
    if args.activation != "relu":
        config_parts.append(f"activation={args.activation}")
    print(f"Config: {', '.join(config_parts)}\n")

    data = load_data()
    results = {}

    # Base training kwargs shared by all demos
    base_kw = dict(
        transforms=use_transforms, residual=use_residual,
        lr_schedule=lr_schedule, l1_reg=args.l1_reg,
        label_smoothing=args.label_smoothing, weight_decay=args.weight_decay,
        patience=args.patience,
    )
    # Extra kwargs only for GAM/GA2M demo
    gam_kw = dict(
        **base_kw,
        loss=args.loss, tower_dropout=args.tower_dropout,
        output_norm=args.output_norm, ga2m=args.ga2m,
        ga2m_pairs=args.ga2m_pairs, activation=args.activation,
        context=args.context,
    )

    if "gam" in demos:
        results.update(demo_gam(data, EPOCHS, K, **gam_kw))

    if "submodular" in demos:
        results.update(demo_submodular(data, EPOCHS, K, QPE, **base_kw,
                                       activation=args.activation, loss=args.loss))

    if "multi" in demos:
        results.update(demo_multi_objective(data, EPOCHS, K, QPE, **base_kw))

    if "gbdt" in demos:
        results.update(demo_gbdt(data, EPOCHS, K, **base_kw,
                                 activation=args.activation, loss=args.loss))

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

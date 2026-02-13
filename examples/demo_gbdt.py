#!/usr/bin/env python3
"""
GBDT Residual Boosting demo -- 3-stage pipeline.

  Stage 1: Train interpretable GAM on raw features
  Stage 2: Train GBDT regressor on GAM residuals (captures non-additive signal)
  Stage 3: Add "magic curve" tower f(GBDT_score) to GAM (freeze D towers, train tower D+1)

The magic curve distills the GBDT's black-box signal into a single interpretable
response curve, giving most of GBDT's accuracy while keeping full interpretability.

Usage:
    python examples/demo_gbdt.py
    python examples/demo_gbdt.py --loss ndcg2pp --activation silu --epochs 30
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

import ranking_gam as rg

from _common import (
    add_common_args, device, load_data, make_loss, print_config, resolve_args,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GBDT residual boosting demo")
    add_common_args(parser)
    args = resolve_args(parser.parse_args())
    print_config(args)

    data = load_data()
    results = run(data, args)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for name, ndcg in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<28} NDCG@{args.k} = {ndcg:.4f}")


def run(data, args):
    """Run 3-stage GBDT boosting pipeline. Returns results dict."""
    K = args.k

    print("\n" + "=" * 70)
    print("GAM + GBDT Magic Curve (Residual Boosting)")
    print("=" * 70)
    print("  Pipeline: GAM first -> GBDT on residuals -> add magic curve tower")

    from ranking_gam.training.boosting import (
        train_gbdt_baseline,
        compute_gbdt_residual_feature,
        _gam_predict,
        _train_gbdt_regression,
    )

    def make_loaders(X, y):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        train_l = DataLoader(ds, batch_size=32, shuffle=True)
        eval_ds = TensorDataset(torch.from_numpy(data["eval_X"]), torch.from_numpy(data["eval_y"]))
        eval_l = DataLoader(eval_ds, batch_size=32)
        return train_l, eval_l

    # --- Stage 1: Train GAM on raw features ---
    print("\n  [Stage 1] Training GAM on raw features...")
    gam = rg.GAM_Paper(
        num_features=136, hidden_dims=[16, 8],
        feature_transforms=args.transforms, residual=args.residual,
        activation=args.activation,
    )
    if args.transforms:
        gam.init_transforms_from_data(data["train_X"])

    loss_fn = make_loss(args.loss, args.label_smoothing)
    train_loader, eval_loader = make_loaders(data["train_X"], data["train_y"])
    gam_ndcg = rg.train_model(
        gam, train_loader, eval_loader, loss_fn,
        epochs=args.epochs, patience=args.patience, device=device,
        lr_schedule=args.lr_schedule,
        l1_output_reg=args.l1_reg, weight_decay=args.weight_decay,
    )
    print(f"  Stage 1 GAM: NDCG@{K} = {gam_ndcg:.4f}")

    # --- Stage 2: GBDT on residuals ---
    print("\n  [Stage 2] Training GBDT regressor on GAM residuals...")
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

    # --- Stage 3: Magic curve tower ---
    print("\n  [Stage 3] Training magic curve tower (D original towers frozen)...")
    train_X_boosted = compute_gbdt_residual_feature(gbdt_residual, data["train_X"])
    eval_X_boosted = compute_gbdt_residual_feature(gbdt_residual, data["eval_X"])

    boosted_gam = rg.GAM_Paper(
        num_features=137, hidden_dims=[16, 8],
        feature_transforms=args.transforms, residual=args.residual,
        activation=args.activation,
    )

    # Copy trained weights from Stage 1 into first 136 towers
    with torch.no_grad():
        for j in range(136):
            boosted_gam.towers[j].load_state_dict(gam.towers[j].state_dict())
        if args.transforms:
            for j in range(136):
                boosted_gam.feature_transforms[j].load_state_dict(
                    gam.feature_transforms[j].state_dict()
                )
            boosted_gam.init_transforms_from_data(train_X_boosted)

    # Freeze original towers
    for j in range(136):
        for param in boosted_gam.towers[j].parameters():
            param.requires_grad = False
        if args.transforms:
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
        epochs=args.epochs, patience=args.patience, device=device,
        lr_schedule=args.lr_schedule,
        l1_output_reg=args.l1_reg, weight_decay=args.weight_decay,
    )

    # --- Standalone GBDT baseline ---
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

    # Plot magic curve
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


if __name__ == "__main__":
    main()

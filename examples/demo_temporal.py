#!/usr/bin/env python3
"""
Temporal iTransformer × GAM demo — behavioral time series for ranking.

Demonstrates the TemporalGAMFormer architecture that combines:
  - Static GAM towers on scalar ranking features (interpretable main effects)
  - iTransformer temporal encoder on behavioral time series (learned interactions)

Since MSLR-WEB10K only has static features, this demo synthesizes temporal
behavioral signals (click rate, purchase rate, dwell time, return rate,
wishlist rate) from the static features to demonstrate the architecture.

Trains four models and compares:
  1. GAM            — static features only (interpretable baseline)
  2. TemporalGAMFormer (extract)   — Option 1: iTransformer → GAM towers
  3. TemporalGAMFormer (integrated) — Option 2: additive base + interaction
  4. TemporalGAMFormer (2D patches) — iTransformer2D with weekly patches

Usage:
    python examples/demo_temporal.py
    python examples/demo_temporal.py --loss ndcg2pp --epochs 30
    python examples/demo_temporal.py --num-variates 8 --time-steps 28 --num-patches 4
    python examples/demo_temporal.py --no-static  # temporal features only
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import ranking_gam as rg

from _common import (
    add_common_args, device, load_data, make_loss, print_config, resolve_args,
)


VARIATE_NAMES = [
    "click_rate", "purchase_rate", "dwell_time", "return_rate", "wishlist_rate",
    "cart_add_rate", "view_duration", "search_click_rate",
]


def synthesize_temporal(X_static, num_variates=5, time_steps=30, seed=42):
    """Create synthetic temporal behavioral features from static features.

    Simulates daily behavioral signals (clicks, purchases, etc.) over a
    lookback window. Uses static features as seed signals with added
    temporal patterns: recency decay, weekly seasonality, and noise.

    Args:
        X_static: [B, L, D] static features (numpy).
        num_variates: number of behavioral signal channels.
        time_steps: daily time steps in the lookback window.
        seed: random seed for reproducibility.

    Returns:
        [B, L, V * T] flattened temporal features (numpy float32).
    """
    rng = np.random.RandomState(seed)
    B, L, D = X_static.shape

    temporal = np.zeros((B, L, num_variates, time_steps), dtype=np.float32)

    for v in range(num_variates):
        # Use a different static feature as base for each variate
        base = X_static[:, :, v % D]  # [B, L]

        for t in range(time_steps):
            day = time_steps - 1 - t  # days ago (0 = most recent)

            # Recency: exponential decay (more recent = stronger signal)
            recency = np.exp(-0.05 * day)

            # Weekly seasonality (e.g., weekday vs weekend patterns)
            weekday = np.sin(2 * np.pi * day / 7) * 0.2

            # Per-variate scaling (different signals have different magnitudes)
            scale = 0.5 + 0.5 * (v / max(num_variates - 1, 1))

            # Noise
            noise = rng.randn(B, L).astype(np.float32) * 0.15

            temporal[:, :, v, t] = base * recency * scale + weekday + noise

    return temporal.reshape(B, L, num_variates * time_steps)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Temporal iTransformer × GAM for ranking"
    )
    add_common_args(parser)
    # Temporal config
    parser.add_argument("--num-variates", type=int, default=5,
                        help="number of behavioral signal channels (default: 5)")
    parser.add_argument("--time-steps", type=int, default=30,
                        help="time series length per variate (default: 30)")
    parser.add_argument("--num-patches", type=int, default=1,
                        help="time patches for iTransformer2D (default: 1, try 5 or 6)")
    # Transformer config
    parser.add_argument("--d-model", type=int, default=32,
                        help="temporal transformer hidden dim (default: 32)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="attention heads (default: 4)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="transformer layers (default: 2)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout (default: 0.1)")
    # Flags
    parser.add_argument("--no-static", action="store_true",
                        help="temporal features only (no static GAM)")
    parser.add_argument("--shared-embedding", action="store_true",
                        help="share embedding across variates (default: per-variate)")
    parser.add_argument("--no-revin", action="store_true",
                        help="disable RevIN normalization")
    args = resolve_args(parser.parse_args())

    extra = [f"variates={args.num_variates}", f"T={args.time_steps}",
             f"d_model={args.d_model}"]
    if args.num_patches > 1:
        extra.append(f"patches={args.num_patches}")
    print_config(args, extra)

    # Load static data
    data = load_data(dataset=args.dataset, data_dir=args.data_dir)
    num_static = data["num_features"]
    num_variates = args.num_variates
    time_steps = args.time_steps

    # Synthesize temporal features
    print("\n" + "=" * 70)
    print(f"Synthesizing temporal features: {num_variates} variates × {time_steps} days")
    print("=" * 70)
    variate_names = VARIATE_NAMES[:num_variates]
    print(f"  Variates: {', '.join(variate_names)}")

    train_temporal = synthesize_temporal(
        data["train_X"], num_variates, time_steps, seed=42
    )
    eval_temporal = synthesize_temporal(
        data["eval_X"], num_variates, time_steps, seed=123
    )

    # Combine static + temporal into flat tensors
    if args.no_static:
        num_static_used = 0
        train_combined = train_temporal
        eval_combined = eval_temporal
        print(f"  Mode: temporal only ({num_variates * time_steps} features)")
    else:
        num_static_used = num_static
        train_combined = np.concatenate([data["train_X"], train_temporal], axis=-1)
        eval_combined = np.concatenate([data["eval_X"], eval_temporal], axis=-1)
        print(f"  Mode: static ({num_static}) + temporal ({num_variates * time_steps})"
              f" = {train_combined.shape[-1]} total features")

    # Create DataLoaders with combined features
    train_ds = TensorDataset(
        torch.from_numpy(train_combined), torch.from_numpy(data["train_y"])
    )
    eval_ds = TensorDataset(
        torch.from_numpy(eval_combined), torch.from_numpy(data["eval_y"])
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False)

    loss_fn = make_loss(args.loss, args.label_smoothing)
    results = {}
    param_counts = {}

    # --- GAM baseline (static features only) ---
    if not args.no_static:
        print("\n" + "=" * 70)
        print("GAM baseline (static features only)")
        print("=" * 70)

        gam = rg.GAM_Paper(
            num_features=num_static, hidden_dims=[16, 8],
            feature_transforms=args.transforms, residual=args.residual,
            activation=args.activation,
        )
        if args.transforms:
            gam.init_transforms_from_data(data["train_X"])

        param_counts["GAM (static)"] = sum(p.numel() for p in gam.parameters())
        print(f"  Parameters: {param_counts['GAM (static)']:,}")

        gam_ndcg = rg.train_model(
            gam, data["train_loader"], data["eval_loader"], loss_fn,
            epochs=args.epochs, patience=args.patience, device=device,
            lr_schedule=args.lr_schedule,
            l1_output_reg=args.l1_reg, weight_decay=args.weight_decay,
        )
        results["GAM (static)"] = gam_ndcg
        print(f"\n  GAM: NDCG@{args.k} = {gam_ndcg:.4f}")

    # --- TemporalGAMFormer: Option 1 (extract) ---
    print("\n" + "=" * 70)
    print("TemporalGAMFormer — Option 1: extract (iTransformer → GAM towers)")
    print("=" * 70)

    tgf_extract = rg.TemporalGAMFormer(
        num_static_features=num_static_used,
        static_hidden_dims=[16, 8],
        static_residual=args.residual,
        static_feature_transforms=args.transforms and not args.no_static,
        activation=args.activation,
        num_variates=num_variates,
        time_steps=time_steps,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        shared_embedding=args.shared_embedding,
        use_revin=not args.no_revin,
        mode="extract",
    )
    if args.transforms and not args.no_static:
        tgf_extract.init_transforms_from_data(data["train_X"])

    param_counts["Extract (Opt 1)"] = sum(p.numel() for p in tgf_extract.parameters())
    print(f"  Parameters: {param_counts['Extract (Opt 1)']:,}"
          f"  (static: {tgf_extract.static_param_count():,}"
          f" + temporal: {tgf_extract.temporal_param_count():,})")

    ext_ndcg = rg.train_model(
        tgf_extract, train_loader, eval_loader, loss_fn,
        epochs=args.epochs, patience=args.patience, device=device,
        lr_schedule=args.lr_schedule,
        l1_output_reg=args.l1_reg, weight_decay=args.weight_decay,
    )
    results["Extract (Opt 1)"] = ext_ndcg
    print(f"\n  Extract mode: NDCG@{args.k} = {ext_ndcg:.4f}")

    # --- TemporalGAMFormer: Option 2 (integrated) ---
    print("\n" + "=" * 70)
    print("TemporalGAMFormer — Option 2: integrated (additive + interaction)")
    print("=" * 70)

    tgf_integrated = rg.TemporalGAMFormer(
        num_static_features=num_static_used,
        static_hidden_dims=[16, 8],
        static_residual=args.residual,
        static_feature_transforms=args.transforms and not args.no_static,
        activation=args.activation,
        num_variates=num_variates,
        time_steps=time_steps,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        shared_embedding=args.shared_embedding,
        use_revin=not args.no_revin,
        mode="integrated",
    )
    if args.transforms and not args.no_static:
        tgf_integrated.init_transforms_from_data(data["train_X"])

    param_counts["Integrated (Opt 2)"] = sum(p.numel() for p in tgf_integrated.parameters())
    print(f"  Parameters: {param_counts['Integrated (Opt 2)']:,}"
          f"  (static: {tgf_integrated.static_param_count():,}"
          f" + temporal: {tgf_integrated.temporal_param_count():,})")

    int_ndcg = rg.train_model(
        tgf_integrated, train_loader, eval_loader, loss_fn,
        epochs=args.epochs, patience=args.patience, device=device,
        lr_schedule=args.lr_schedule,
        l1_output_reg=args.l1_reg, weight_decay=args.weight_decay,
    )
    results["Integrated (Opt 2)"] = int_ndcg
    print(f"\n  Integrated mode: NDCG@{args.k} = {int_ndcg:.4f}")

    # After training, show decomposition for integrated mode
    if hasattr(tgf_integrated, 'interaction_weight'):
        w = tgf_integrated.interaction_weight.item()
        print(f"  Learned interaction weight: {w:.4f}")

    # --- TemporalGAMFormer: 2D patches ---
    if args.num_patches > 1 or time_steps >= 10:
        num_patches = args.num_patches if args.num_patches > 1 else 5
        # Ensure divisibility
        if time_steps % num_patches != 0:
            # Find closest valid patch count
            for p in range(num_patches, 0, -1):
                if time_steps % p == 0:
                    num_patches = p
                    break

        print("\n" + "=" * 70)
        print(f"TemporalGAMFormer — iTransformer2D ({num_patches} time patches)")
        print("=" * 70)
        patch_size = time_steps // num_patches
        print(f"  {num_variates} variates × {num_patches} patches"
              f" (each {patch_size} days) = {num_variates * num_patches} tokens")

        tgf_2d = rg.TemporalGAMFormer(
            num_static_features=num_static_used,
            static_hidden_dims=[16, 8],
            static_residual=args.residual,
            static_feature_transforms=args.transforms and not args.no_static,
            activation=args.activation,
            num_variates=num_variates,
            time_steps=time_steps,
            num_patches=num_patches,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            shared_embedding=args.shared_embedding,
            use_revin=not args.no_revin,
            mode="extract",
        )
        if args.transforms and not args.no_static:
            tgf_2d.init_transforms_from_data(data["train_X"])

        param_counts["2D patches"] = sum(p.numel() for p in tgf_2d.parameters())
        print(f"  Parameters: {param_counts['2D patches']:,}")

        d2_ndcg = rg.train_model(
            tgf_2d, train_loader, eval_loader, loss_fn,
            epochs=args.epochs, patience=args.patience, device=device,
            lr_schedule=args.lr_schedule,
            l1_output_reg=args.l1_reg, weight_decay=args.weight_decay,
        )
        results["2D patches"] = d2_ndcg
        print(f"\n  2D patches: NDCG@{args.k} = {d2_ndcg:.4f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"RESULTS — NDCG@{args.k} comparison")
    print("=" * 70)

    for name, ndcg in sorted(results.items(), key=lambda x: -x[1]):
        params = param_counts.get(name, 0)
        print(f"  {name:<28} NDCG@{args.k} = {ndcg:.4f}  ({params:>8,} params)")

    if "GAM (static)" in results:
        gam_base = results["GAM (static)"]
        print(f"\n  {'─' * 55}")
        print(f"  Gains over GAM baseline:")
        for name, ndcg in sorted(results.items(), key=lambda x: -x[1]):
            if name == "GAM (static)":
                continue
            gap = ndcg - gam_base
            print(f"    {name:<26} {gap:+.4f}")
        print(f"  {'─' * 55}")


if __name__ == "__main__":
    main()

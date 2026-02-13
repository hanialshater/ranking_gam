#!/usr/bin/env python3
"""
Transformer black-box baseline -- upper bound comparison for GAM.

Trains a self-attention Transformer ranker where each document attends to all
others in the list. This is NOT interpretable (no per-feature response curves),
but shows how much NDCG we sacrifice for interpretability.

Usage:
    python examples/demo_transformer.py
    python examples/demo_transformer.py --loss ndcg2pp --epochs 30
    python examples/demo_transformer.py --d-model 128 --num-layers 3
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ranking_gam as rg

from _common import (
    add_common_args, device, load_data, make_loss, print_config, resolve_args,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Transformer black-box baseline")
    add_common_args(parser)
    parser.add_argument("--d-model", type=int, default=64,
                        help="transformer hidden dimension (default: 64)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="number of attention heads (default: 4)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of transformer layers (default: 2)")
    parser.add_argument("--dim-feedforward", type=int, default=128,
                        help="FFN hidden dimension (default: 128)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout rate (default: 0.1)")
    parser.add_argument("--compare-gam", action="store_true", default=True,
                        help="also train a GAM for comparison (default: True)")
    parser.add_argument("--no-compare-gam", action="store_true",
                        help="skip GAM comparison")
    args = resolve_args(parser.parse_args())

    extra = [f"d_model={args.d_model}", f"layers={args.num_layers}",
             f"heads={args.nhead}"]
    print_config(args, extra)

    data = load_data()
    results = {}

    # --- Train Transformer ---
    print("\n" + "=" * 70)
    print("Transformer Ranker (black-box baseline)")
    print("=" * 70)

    transformer = rg.TransformerRanker(
        num_features=136,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in transformer.parameters())
    print(f"  Parameters: {n_params:,}")

    loss_fn = make_loss(args.loss, args.label_smoothing)
    transformer_ndcg = rg.train_model(
        transformer, data["train_loader"], data["eval_loader"], loss_fn,
        epochs=args.epochs, patience=args.patience, device=device,
        lr_schedule=args.lr_schedule,
        l1_output_reg=args.l1_reg, weight_decay=args.weight_decay,
    )
    results["Transformer"] = transformer_ndcg
    print(f"\n  Transformer: NDCG@{args.k} = {transformer_ndcg:.4f}")

    # --- Train GAM for comparison ---
    if not args.no_compare_gam:
        print("\n" + "=" * 70)
        print("GAM (interpretable comparison)")
        print("=" * 70)

        gam = rg.GAM_Paper(
            num_features=136, hidden_dims=[16, 8],
            feature_transforms=args.transforms, residual=args.residual,
            activation=args.activation,
        )
        if args.transforms:
            gam.init_transforms_from_data(data["train_X"])

        gam_params = sum(p.numel() for p in gam.parameters())
        print(f"  Parameters: {gam_params:,}")

        gam_ndcg = rg.train_model(
            gam, data["train_loader"], data["eval_loader"], loss_fn,
            epochs=args.epochs, patience=args.patience, device=device,
            lr_schedule=args.lr_schedule,
            l1_output_reg=args.l1_reg, weight_decay=args.weight_decay,
        )
        results["GAM"] = gam_ndcg
        print(f"\n  GAM: NDCG@{args.k} = {gam_ndcg:.4f}")

        # Summary
        gap = transformer_ndcg - gam_ndcg
        print(f"\n  {'=' * 50}")
        print(f"  Interpretability cost:")
        print(f"    Transformer (black-box): NDCG@{args.k} = {transformer_ndcg:.4f}  ({n_params:,} params)")
        print(f"    GAM (interpretable):     NDCG@{args.k} = {gam_ndcg:.4f}  ({gam_params:,} params)")
        print(f"    Gap:                     {gap:+.4f}")
        print(f"  {'=' * 50}")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for name, ndcg in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<28} NDCG@{args.k} = {ndcg:.4f}")


if __name__ == "__main__":
    main()

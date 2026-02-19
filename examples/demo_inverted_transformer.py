#!/usr/bin/env python3
"""
Inverted Transformer baseline -- feature-interaction upper bound for GAM.

Trains an inverted transformer where self-attention operates across *features*
(each feature is a token) rather than across documents. This captures arbitrary
feature interactions while still scoring each document independently, sitting
between GAM (no interactions) and the standard TransformerRanker (full
cross-document interactions).

Usage:
    python examples/demo_inverted_transformer.py
    python examples/demo_inverted_transformer.py --loss ndcg2pp --epochs 30
    python examples/demo_inverted_transformer.py --d-model 128 --num-layers 3
    python examples/demo_inverted_transformer.py --pooling mean
"""

import matplotlib
matplotlib.use("Agg")

import ranking_gam as rg

from _common import (
    add_common_args, device, load_data, make_loss, print_config, resolve_args,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Inverted Transformer baseline (feature-level attention)"
    )
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
    parser.add_argument("--pooling", type=str, default="cls",
                        choices=["cls", "mean"],
                        help="feature aggregation: cls token or mean pool (default: cls)")
    parser.add_argument("--compare-gam", action="store_true", default=True,
                        help="also train a GAM for comparison (default: True)")
    parser.add_argument("--no-compare-gam", action="store_true",
                        help="skip GAM comparison")
    parser.add_argument("--compare-transformer", action="store_true", default=False,
                        help="also train standard TransformerRanker for comparison")
    args = resolve_args(parser.parse_args())

    extra = [f"d_model={args.d_model}", f"layers={args.num_layers}",
             f"heads={args.nhead}", f"pooling={args.pooling}"]
    print_config(args, extra)

    data = load_data(dataset=args.dataset, data_dir=args.data_dir)
    num_features = data["num_features"]
    results = {}

    # --- Train Inverted Transformer ---
    print("\n" + "=" * 70)
    print("Inverted Transformer (feature-level attention)")
    print("=" * 70)

    inv_transformer = rg.InvertedTransformerRanker(
        num_features=num_features,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pooling=args.pooling,
    )
    n_params = sum(p.numel() for p in inv_transformer.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Scores each document independently (no cross-doc attention)")
    print(f"  Captures feature interactions via self-attention across features")

    loss_fn = make_loss(args.loss, args.label_smoothing)
    inv_ndcg = rg.train_model(
        inv_transformer, data["train_loader"], data["eval_loader"], loss_fn,
        epochs=args.epochs, patience=args.patience, device=device,
        lr_schedule=args.lr_schedule,
        l1_output_reg=args.l1_reg, weight_decay=args.weight_decay,
    )
    results["InvertedTransformer"] = inv_ndcg
    print(f"\n  Inverted Transformer: NDCG@{args.k} = {inv_ndcg:.4f}")

    # --- Train GAM for comparison ---
    if not args.no_compare_gam:
        print("\n" + "=" * 70)
        print("GAM (interpretable comparison — no feature interactions)")
        print("=" * 70)

        gam = rg.GAM_Paper(
            num_features=num_features, hidden_dims=[16, 8],
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

    # --- Train standard Transformer for comparison ---
    if args.compare_transformer:
        print("\n" + "=" * 70)
        print("Standard Transformer (cross-document attention)")
        print("=" * 70)

        transformer = rg.TransformerRanker(
            num_features=num_features,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )
        t_params = sum(p.numel() for p in transformer.parameters())
        print(f"  Parameters: {t_params:,}")

        t_ndcg = rg.train_model(
            transformer, data["train_loader"], data["eval_loader"], loss_fn,
            epochs=args.epochs, patience=args.patience, device=device,
            lr_schedule=args.lr_schedule,
            l1_output_reg=args.l1_reg, weight_decay=args.weight_decay,
        )
        results["Transformer"] = t_ndcg
        print(f"\n  Transformer: NDCG@{args.k} = {t_ndcg:.4f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for name, ndcg in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<28} NDCG@{args.k} = {ndcg:.4f}")

    if "GAM" in results:
        gap = inv_ndcg - results["GAM"]
        print(f"\n  Feature interaction gain (InvTransformer - GAM): {gap:+.4f}")
    if "Transformer" in results:
        gap = results["Transformer"] - inv_ndcg
        print(f"  Cross-doc interaction gain (Transformer - InvTransformer): {gap:+.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export a distilled GAM model to JSON for Java inference.

This script trains a small GAM, distills it to PWL, and exports
the model as JSON that can be loaded by the Java inference library.

Usage:
    # Export standard GAM model
    python examples/demo_java_export.py

    # Export submodular GAM with diversity towers
    python examples/demo_java_export.py --submodular

    # Export GA2M with interactions
    python examples/demo_java_export.py --ga2m

    # Then in Java:
    #   DistilledGamModel model = DistilledGamLoader.fromJson("gam_distilled.json");
    #   double[] scores = model.score(features);
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "examples")


def main():
    parser = argparse.ArgumentParser(description="Export distilled GAM to JSON for Java")
    parser.add_argument("--output", type=str, default="gam_distilled.json",
                        help="output JSON path (default: gam_distilled.json)")
    parser.add_argument("--num-features", type=int, default=10,
                        help="number of features for synthetic data (default: 10)")
    parser.add_argument("--num-knots", type=int, default=5,
                        help="PWL knots per feature (default: 5)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="training epochs (default: 5)")
    parser.add_argument("--hidden", type=int, nargs="+", default=[16, 8],
                        help="tower hidden dims (default: 16 8)")
    parser.add_argument("--ga2m", action="store_true",
                        help="include pairwise interactions (GA2M)")
    parser.add_argument("--ga2m-pairs", type=int, default=5,
                        help="number of interaction pairs (default: 5)")
    parser.add_argument("--submodular", action="store_true",
                        help="train SubmodularRankingGAM with diversity towers")
    parser.add_argument("--diversity-epochs", type=int, default=5,
                        help="diversity tower training epochs (default: 5)")
    args = parser.parse_args()

    import ranking_gam as rg

    num_features = args.num_features

    # Synthetic training data: [batch, list_size, num_features]
    np.random.seed(42)
    batch_size, list_size = 200, 20
    X = np.random.randn(batch_size, list_size, num_features).astype(np.float32)
    # Synthetic relevance: linear combination + noise
    weights = np.random.randn(num_features).astype(np.float32)
    y = (X * weights).sum(axis=-1) + 0.1 * np.random.randn(batch_size, list_size).astype(np.float32)
    # Discretize to 0-4 labels
    y = np.clip(np.round((y - y.min()) / (y.max() - y.min()) * 4), 0, 4).astype(np.float32)

    # Split
    n_train = int(0.8 * batch_size)
    train_X, val_X = X[:n_train], X[n_train:]
    train_y, val_y = y[:n_train], y[n_train:]

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    train_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    loss_fn = rg.LambdaLoss.ndcg2pp()

    if args.submodular:
        # ── SubmodularRankingGAM ──
        # Use first 2 features as "category" and "brand" for diversity
        # (In practice these would be actual categorical features)
        groupwise_specs = [
            {
                "name": "category_novelty",
                "type": "category_novelty",
                "column": 0,
                "x_min": 0.0,
                "x_max": 1.0,
            },
            {
                "name": "brand_novelty",
                "type": "brand_novelty",
                "column": 1,
                "x_min": 0.0,
                "x_max": 1.0,
            },
        ]

        model = rg.SubmodularRankingGAM(
            num_item_features=num_features,
            groupwise_specs=groupwise_specs,
            item_hidden=args.hidden,
            num_knots=10,
            activation="silu",
            feature_transforms=True,
            residual=True,
        )
        model.init_transforms_from_data(train_X)

        print(f"Training SubmodularRankingGAM with {num_features} features, "
              f"{len(groupwise_specs)} diversity towers, hidden={args.hidden}")

        # Phase 1: Train base towers
        print("\n--- Phase 1: Training base towers ---")
        rg.train_model(model, train_loader, val_loader, loss_fn,
                       epochs=args.epochs, lr_schedule="cosine")

        # Phase 2: Train diversity towers
        print("\n--- Phase 2: Training diversity towers ---")
        # Diversity tower training expects numpy arrays
        rg.train_diversity_towers(model, train_X, train_y, epochs=args.diversity_epochs)

    elif args.ga2m:
        pairs = rg.select_interactions_correlation(
            train_X, top_k=args.ga2m_pairs)
        model = rg.GA2M_Paper(
            num_features=num_features,
            interaction_pairs=pairs,
            hidden_dims=args.hidden,
            activation="silu",
            feature_transforms=True,
            residual=True,
        )
        model.init_transforms_from_data(train_X)
        print(f"Training GA2M with {num_features} features, "
              f"{len(pairs)} interaction pairs, hidden={args.hidden}")
        rg.train_model(model, train_loader, val_loader, loss_fn,
                       epochs=args.epochs, lr_schedule="cosine")
    else:
        model = rg.GAM_Paper(
            num_features=num_features,
            hidden_dims=args.hidden,
            activation="silu",
            feature_transforms=True,
            residual=True,
        )
        model.init_transforms_from_data(train_X)
        print(f"Training GAM with {num_features} features, hidden={args.hidden}")
        rg.train_model(model, train_loader, val_loader, loss_fn,
                       epochs=args.epochs, lr_schedule="cosine")

    # Distill
    print(f"\nDistilling to PWL with {args.num_knots} knots per feature...")
    pwl = rg.distill_to_pwl(model, num_knots=args.num_knots, train_X=train_X)

    # Evaluate
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(val_X))
    ndcg_neural = rg.compute_ndcg(val_pred, torch.from_numpy(val_y))
    ndcg_pwl = rg.evaluate_pwl(pwl, val_X, val_y)
    print(f"  Neural NDCG@10: {ndcg_neural:.4f}")
    print(f"  PWL    NDCG@10: {ndcg_pwl:.4f}")

    # Export
    rg.save_pwl_json(pwl, args.output)

    # Print summary
    with open(args.output) as f:
        data = json.load(f)
    size_kb = len(json.dumps(data)) / 1024
    print(f"\nExported to {args.output} ({size_kb:.1f} KB)")
    print(f"  Main effects: {len(data['main_effects'])}")
    print(f"  Interactions: {len(data['interactions'])}")
    print(f"  Diversity towers: {len(data.get('diversity_towers', []))}")
    print(f"  Bias: {data['bias']:.6f}")

    # Show Java usage
    if args.submodular:
        print(f"""
Java usage (submodular reranking):
    // Load model + diversity towers
    SubmodularModelData data = DistilledGamLoader.loadSubmodular("{args.output}");

    // Provide your groupwise feature computer
    GroupwiseFeatureComputer computer = new DefaultGroupwiseComputer(specs);

    // Create reranker
    SubmodularGamReranker reranker = new SubmodularGamReranker(
        data.baseModel, data.diversityTowers, computer);

    // Rerank with lazy greedy (Minoux), max 10 evals per position
    int[] order = reranker.rerank(features, k, 10);
""")
    else:
        print(f"""
Java usage:
    DistilledGamModel model = DistilledGamLoader.fromJson("{args.output}");
    double score = model.scoreDocument(features);
    double[] scores = model.score(featureMatrix);
""")


if __name__ == "__main__":
    main()

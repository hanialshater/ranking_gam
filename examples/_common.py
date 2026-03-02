"""Shared helpers for ranking_gam demos."""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import ranking_gam as rg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(dataset="mslr10k", data_dir="data", max_train=6000,
              max_eval=2000, list_size=40, **kwargs):
    """Load an L2R dataset and create augmented features with synthetic groupwise columns.

    Args:
        dataset: dataset name (default: "mslr10k"). See rg.DATASET_NAMES.
        data_dir: data directory
        max_train: max training queries
        max_eval: max evaluation queries
        list_size: documents per query
        **kwargs: extra args passed to the dataset loader

    Returns:
        dict with keys: train_X, train_y, eval_X, eval_y, train_loader,
        eval_loader, train_X_aug, eval_X_aug, X_flat, num_features
    """
    num_features = rg.get_num_features(dataset)

    print("=" * 70)
    print(f"Loading {dataset} (num_features={num_features}) + Synthetic Groupwise Features")
    print("=" * 70)

    train_X, train_y, eval_X, eval_y = rg.load_dataset(
        dataset, data_dir=data_dir, max_train=max_train,
        max_eval=max_eval, list_size=list_size, **kwargs,
    )

    train_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    eval_ds = TensorDataset(torch.from_numpy(eval_X), torch.from_numpy(eval_y))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False)

    # Synthetic groupwise features (most L2R datasets have no category/brand)
    # Use feature 0 and min(5, num_features-1) for binning
    feat_a = 0
    feat_b = min(5, num_features - 1)

    f0_flat = train_X[:, :, feat_a].flatten()
    f5_flat = train_X[:, :, feat_b].flatten()

    f0_valid = f0_flat[f0_flat != 0]
    f5_valid = f5_flat[f5_flat != 0]

    # Use percentile bins; guard against datasets with many zeros
    if len(f0_valid) > 10:
        f0_bins = np.digitize(
            f0_flat, np.percentile(f0_valid, [20, 40, 60, 80])
        )
    else:
        f0_bins = np.digitize(f0_flat, np.percentile(f0_flat, [20, 40, 60, 80]))

    if len(f5_valid) > 10:
        f5_bins = np.digitize(
            f5_flat,
            np.percentile(f5_valid, [12.5, 25, 37.5, 50, 62.5, 75, 87.5]),
        )
    else:
        f5_bins = np.digitize(
            f5_flat,
            np.percentile(f5_flat, [12.5, 25, 37.5, 50, 62.5, 75, 87.5]),
        )

    def augment(X, col0_bins, col5_bins):
        B, L, D = X.shape
        X_aug = np.zeros((B, L, D + 2), dtype=X.dtype)
        X_aug[:, :, :D] = X
        X_aug[:, :, D] = col0_bins.reshape(B, L)
        X_aug[:, :, D + 1] = col5_bins.reshape(B, L)
        return X_aug

    train_X_aug = augment(train_X, f0_bins, f5_bins)

    ef0 = eval_X[:, :, feat_a].flatten()
    ef5 = eval_X[:, :, feat_b].flatten()
    if len(f0_valid) > 10:
        ef0_bins = np.digitize(ef0, np.percentile(f0_valid, [20, 40, 60, 80]))
    else:
        ef0_bins = np.digitize(ef0, np.percentile(f0_flat, [20, 40, 60, 80]))
    if len(f5_valid) > 10:
        ef5_bins = np.digitize(
            ef5, np.percentile(f5_valid, [12.5, 25, 37.5, 50, 62.5, 75, 87.5]),
        )
    else:
        ef5_bins = np.digitize(
            ef5, np.percentile(f5_flat, [12.5, 25, 37.5, 50, 62.5, 75, 87.5]),
        )
    eval_X_aug = augment(eval_X, ef0_bins, ef5_bins)

    X_flat = train_X.reshape(-1, num_features)

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
        "num_features": num_features,
    }


def groupwise_specs(num_base=136):
    """Default groupwise feature specs for diversity demos."""
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
            "column": min(10, num_base - 1),
            "x_min": 0.0,
            "x_max": 3.0,
        },
    ]


def make_loss(loss_name, label_smoothing=0.0, k=10):
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
    elif loss_name == "diffsort":
        return rg.DiffSortNDCGLoss(k=k, regularization_strength=1.0)
    else:
        raise ValueError(
            f"Unknown loss: {loss_name}. "
            f"Choose from: {', '.join(LOSS_CHOICES)}"
        )


LOSS_CHOICES = ["listnet", "pairwise", "approxndcg", "ndcg2pp", "listmle", "diffsort"]
ACTIVATION_CHOICES = ["relu", "silu", "gelu"]
LR_SCHEDULE_CHOICES = ["constant", "cosine", "plateau"]


def add_common_args(parser):
    """Add training args shared across all demos."""
    parser.add_argument("--dataset", type=str, default="mslr10k",
                        choices=rg.DATASET_NAMES,
                        help="L2R dataset to use (default: mslr10k)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="data directory (default: data)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10,
                        help="early stopping patience (default: 10)")
    parser.add_argument("--k", type=int, default=10,
                        help="NDCG@K cutoff (default: 10)")
    parser.add_argument("--loss", type=str, default="listnet",
                        choices=LOSS_CHOICES,
                        help="ranking loss function (default: listnet)")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=ACTIVATION_CHOICES,
                        help="tower activation (default: relu, try silu)")
    parser.add_argument("--lr-schedule", type=str, default="cosine",
                        choices=LR_SCHEDULE_CHOICES,
                        help="LR schedule (default: cosine)")
    parser.add_argument("--no-transforms", action="store_true",
                        help="disable learnable monotone feature transforms")
    parser.add_argument("--no-residual", action="store_true",
                        help="disable residual skip connections")
    parser.add_argument("--l1-reg", type=float, default=0.0001,
                        help="L1 output regularization (default: 0.0001, 0 to disable)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="label smoothing for ListNet (default: 0.1)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay (default: 0.01)")
    return parser


def resolve_args(args):
    """Resolve --no-* flags into positive booleans."""
    args.transforms = not args.no_transforms
    args.residual = not args.no_residual
    # Ensure dataset attr exists (for demos that don't use add_common_args)
    if not hasattr(args, "dataset"):
        args.dataset = "mslr10k"
    if not hasattr(args, "data_dir"):
        args.data_dir = "data"
    return args


def print_config(args, extra=None):
    """Print resolved config summary."""
    parts = [
        f"dataset={args.dataset}",
        f"epochs={args.epochs}", f"k={args.k}", f"loss={args.loss}",
        f"transforms={'ON' if args.transforms else 'OFF'}",
        f"residual={'ON' if args.residual else 'OFF'}",
        f"lr_schedule={args.lr_schedule}",
    ]
    if args.activation != "relu":
        parts.append(f"activation={args.activation}")
    if args.l1_reg > 0:
        parts.append(f"l1={args.l1_reg}")
    if args.weight_decay > 0:
        parts.append(f"wd={args.weight_decay}")
    if extra:
        parts.extend(extra)
    print(f"Config: {', '.join(parts)}\n")

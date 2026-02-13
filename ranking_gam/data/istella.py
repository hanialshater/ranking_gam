"""
Istella LETOR dataset loading.

Requires manual download (non-commercial license):
    https://istella.ai/datasets/letor-dataset/

Variants:
    Istella-S (sampled): 33K queries, 220 features, ~3.4M docs
    Istella (full):      33K queries, 220 features, ~10.5M docs
    Istella-X (extended): 10K queries, 220 features, ~26.8M docs

All variants: 5-level relevance (0-4), SVMLight format, 220 features.

WARNING: Istella contains extreme feature values (up to DBL_MAX ≈ 1.8e308).
This loader clips values to [-1e6, 1e6] by default before applying log1p.
"""

import os

from .svmlight import load_svmlight_dataset

ISTELLA_NUM_FEATURES = 220

_DOWNLOAD_URL = "https://istella.ai/datasets/letor-dataset/"


def load_istella(data_dir="data", variant="s", max_train=6000,
                 max_eval=2000, list_size=40, clip_value=1e6):
    """Load an Istella LETOR dataset variant.

    Requires manual download (non-commercial license) from:
        https://istella.ai/datasets/letor-dataset/

    Expected directory structure:
        data_dir/istella-s/train.txt    (variant="s")
        data_dir/istella-s/test.txt
        data_dir/istella/train.txt      (variant="full")
        data_dir/istella-x/train.txt    (variant="x")

    Args:
        data_dir: directory containing the istella data folder
        variant: "s" (sampled, recommended), "full", or "x" (extended)
        max_train: max training queries
        max_eval: max evaluation queries
        list_size: documents per query
        clip_value: clip feature values to this range before log1p.
            Default 1e6. Istella has extreme outliers that need clipping.

    Returns:
        (train_X, train_y, eval_X, eval_y) as float32 numpy arrays
    """
    variant_dirs = {
        "s": "istella-s",
        "full": "istella",
        "x": "istella-x",
    }
    if variant not in variant_dirs:
        raise ValueError(
            f"variant must be 's', 'full', or 'x', got {variant!r}"
        )

    dir_name = variant_dirs[variant]
    display_name = f"Istella-{variant.upper()}" if variant != "full" else "Istella"

    # Search for files
    search_dirs = [
        os.path.join(data_dir, dir_name),
        os.path.join(data_dir, display_name),
        data_dir,
    ]

    train_path = test_path = None
    for d in search_dirs:
        candidate = os.path.join(d, "train.txt")
        if os.path.exists(candidate):
            train_path = candidate
            test_path = os.path.join(d, "test.txt")
            break

    if train_path is None or not os.path.exists(train_path):
        raise FileNotFoundError(
            f"{display_name} not found.\n"
            f"This dataset requires a non-commercial license. Download from:\n"
            f"  {_DOWNLOAD_URL}\n"
            f"Then place files as:\n"
            f"  {data_dir}/{dir_name}/train.txt\n"
            f"  {data_dir}/{dir_name}/test.txt"
        )

    return load_svmlight_dataset(
        train_path, test_path, ISTELLA_NUM_FEATURES,
        max_train=max_train, max_eval=max_eval, list_size=list_size,
        clip_value=clip_value, name=display_name,
    )

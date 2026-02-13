"""
Istella LETOR dataset loading.

Auto-downloads from istella.ai / quickrank mirrors.

Variants:
    Istella-S (sampled): 33K queries, 220 features, ~3.4M docs
    Istella (full):      33K queries, 220 features, ~10.5M docs
    Istella-X (extended): 10K queries, 220 features, ~26.8M docs

All variants: 5-level relevance (0-4), SVMLight format, 220 features.

WARNING: Istella contains extreme feature values (up to DBL_MAX ~ 1.8e308).
This loader clips values to [-1e6, 1e6] by default before applying log1p.
"""

import glob
import os

from .svmlight import download_and_extract, load_svmlight_dataset

ISTELLA_NUM_FEATURES = 220

# Download URLs (same as pytorchltr)
_ISTELLA_URLS = {
    "s": "http://library.istella.it/dataset/istella-s-letor.tar.gz",
    "full": "http://library.istella.it/dataset/istella-letor.tar.gz",
    "x": "http://quickrank.isti.cnr.it/istella-datasets-mirror/istella-X.tar.gz",
}
_ISTELLA_ARCHIVES = {
    "s": "istella-s-letor.tar.gz",
    "full": "istella-letor.tar.gz",
    "x": "istella-X.tar.gz",
}


def load_istella(data_dir="data", variant="s", max_train=6000,
                 max_eval=2000, list_size=40, clip_value=1e6):
    """Load an Istella LETOR dataset variant.

    Downloads automatically if not present.
    Istella-S: ~500MB, Istella: ~1.7GB, Istella-X: ~4.6GB

    Expected directory structure (after download):
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
        # Auto-download
        url = _ISTELLA_URLS[variant]
        archive = _ISTELLA_ARCHIVES[variant]
        print(f"Downloading {display_name}...")
        download_and_extract(url, data_dir, archive)

        # Find extracted files
        matches = glob.glob(
            f"{data_dir}/**/train.txt", recursive=True
        )
        if matches:
            train_path = matches[0]
            test_path = train_path.replace("train.txt", "test.txt")
        else:
            raise FileNotFoundError(
                f"Downloaded {display_name} but could not find train.txt. "
                f"Check {data_dir} for extracted files."
            )

    return load_svmlight_dataset(
        train_path, test_path, ISTELLA_NUM_FEATURES,
        max_train=max_train, max_eval=max_eval, list_size=list_size,
        clip_value=clip_value, name=display_name,
    )

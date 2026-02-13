"""Data loading utilities for L2R benchmark datasets.

Supported datasets:
    mslr10k  — MSLR-WEB10K (136 features, auto-download)
    mslr30k  — MSLR-WEB30K (136 features, manual download)
    yahoo1   — Yahoo LTRC Set 1 (699 features, license required)
    yahoo2   — Yahoo LTRC Set 2 (700 features, license required)
    istella  — Istella-S (220 features, license required)
    mq2007   — LETOR 4.0 MQ2007 (46 features, free)
    mq2008   — LETOR 4.0 MQ2008 (46 features, free)
"""

from .istella import ISTELLA_NUM_FEATURES, load_istella
from .letor import LETOR_FEATURE_NAMES, LETOR_NUM_FEATURES, load_mq2007, load_mq2008
from .mslr import MSLR_FEATURE_NAMES, MSLR_NUM_FEATURES, load_mslr, load_mslr30k
from .yahoo import YAHOO_SET1_NUM_FEATURES, YAHOO_SET2_NUM_FEATURES, load_yahoo

# Dataset registry: name -> (loader_func, num_features, kwargs)
DATASET_REGISTRY = {
    "mslr10k": {"loader": load_mslr, "num_features": 136},
    "mslr30k": {"loader": load_mslr30k, "num_features": 136},
    "yahoo1": {"loader": load_yahoo, "num_features": 699, "kwargs": {"set_name": "set1"}},
    "yahoo2": {"loader": load_yahoo, "num_features": 700, "kwargs": {"set_name": "set2"}},
    "istella": {"loader": load_istella, "num_features": 220, "kwargs": {"variant": "s"}},
    "istella_full": {"loader": load_istella, "num_features": 220, "kwargs": {"variant": "full"}},
    "istella_x": {"loader": load_istella, "num_features": 220, "kwargs": {"variant": "x"}},
    "mq2007": {"loader": load_mq2007, "num_features": 46},
    "mq2008": {"loader": load_mq2008, "num_features": 46},
}

DATASET_NAMES = list(DATASET_REGISTRY.keys())


def load_dataset(name, data_dir="data", max_train=6000, max_eval=2000,
                 list_size=40, **kwargs):
    """Load any supported L2R dataset by name.

    Args:
        name: dataset name (see DATASET_NAMES for options)
        data_dir: root data directory
        max_train: max training queries
        max_eval: max evaluation queries
        list_size: documents per query
        **kwargs: passed through to the specific loader

    Returns:
        (train_X, train_y, eval_X, eval_y) as float32 numpy arrays

    Example:
        >>> train_X, train_y, eval_X, eval_y = load_dataset("mslr10k")
        >>> train_X, train_y, eval_X, eval_y = load_dataset("mq2007", fold=2)
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name!r}. "
            f"Available: {', '.join(DATASET_NAMES)}"
        )

    entry = DATASET_REGISTRY[name]
    loader = entry["loader"]
    default_kwargs = entry.get("kwargs", {})

    # Merge: explicit kwargs override defaults
    merged = {**default_kwargs, **kwargs}

    return loader(
        data_dir=data_dir, max_train=max_train, max_eval=max_eval,
        list_size=list_size, **merged,
    )


def get_num_features(name):
    """Get the number of features for a dataset by name.

    Args:
        name: dataset name

    Returns:
        int: number of features
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name!r}. "
            f"Available: {', '.join(DATASET_NAMES)}"
        )
    return DATASET_REGISTRY[name]["num_features"]


__all__ = [
    "load_mslr",
    "load_mslr30k",
    "load_yahoo",
    "load_istella",
    "load_mq2007",
    "load_mq2008",
    "load_dataset",
    "get_num_features",
    "DATASET_REGISTRY",
    "DATASET_NAMES",
    "MSLR_NUM_FEATURES",
    "MSLR_FEATURE_NAMES",
    "YAHOO_SET1_NUM_FEATURES",
    "YAHOO_SET2_NUM_FEATURES",
    "ISTELLA_NUM_FEATURES",
    "LETOR_NUM_FEATURES",
    "LETOR_FEATURE_NAMES",
]

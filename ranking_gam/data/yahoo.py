"""
Yahoo! Learning to Rank Challenge (C14) data loading.

Requires manual download — you must accept the Yahoo Webscope license:
    https://webscope.sandbox.yahoo.com/catalog.php?datatype=c

Set 1: 699 features, ~29K queries, 5-level relevance (0-4)
Set 2: 700 features, ~6K queries, 5-level relevance (0-4)

Features are pre-normalized to [0, 1] by Yahoo. Feature descriptions are
not disclosed (proprietary search features).
"""

import os

from .svmlight import load_svmlight_dataset

YAHOO_SET1_NUM_FEATURES = 699
YAHOO_SET2_NUM_FEATURES = 700

_DOWNLOAD_URL = "https://webscope.sandbox.yahoo.com/catalog.php?datatype=c"


def load_yahoo(data_dir="data", set_name="set1", max_train=6000,
               max_eval=2000, list_size=40, log1p=False):
    """Load Yahoo LTRC dataset.

    Requires manual download from Yahoo Webscope (license required).

    Expected directory structure:
        data_dir/yahoo/set1.train.txt   (or set2.train.txt)
        data_dir/yahoo/set1.test.txt    (or set2.test.txt)

    Also checks:
        data_dir/ltrc_yahoo/set1.train.txt
        data_dir/set1.train.txt

    Args:
        data_dir: directory containing the yahoo data folder
        set_name: "set1" (699 features) or "set2" (700 features)
        max_train: max training queries
        max_eval: max evaluation queries
        list_size: documents per query
        log1p: apply log1p transform (default: False, Yahoo features are
            already normalized to [0, 1])

    Returns:
        (train_X, train_y, eval_X, eval_y) as float32 numpy arrays
    """
    if set_name == "set1":
        num_features = YAHOO_SET1_NUM_FEATURES
    elif set_name == "set2":
        num_features = YAHOO_SET2_NUM_FEATURES
    else:
        raise ValueError(f"set_name must be 'set1' or 'set2', got {set_name!r}")

    # Search for files in common layouts
    train_name = f"{set_name}.train.txt"
    test_name = f"{set_name}.test.txt"

    search_dirs = [
        os.path.join(data_dir, "yahoo"),
        os.path.join(data_dir, "ltrc_yahoo"),
        data_dir,
    ]

    train_path = test_path = None
    for d in search_dirs:
        candidate = os.path.join(d, train_name)
        if os.path.exists(candidate):
            train_path = candidate
            test_path = os.path.join(d, test_name)
            break

    if train_path is None or not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Yahoo LTRC {set_name} not found.\n"
            f"This dataset requires a license. Download from:\n"
            f"  {_DOWNLOAD_URL}\n"
            f"Then place files as:\n"
            f"  {data_dir}/yahoo/{train_name}\n"
            f"  {data_dir}/yahoo/{test_name}"
        )

    return load_svmlight_dataset(
        train_path, test_path, num_features,
        max_train=max_train, max_eval=max_eval, list_size=list_size,
        log1p=log1p, name=f"Yahoo LTRC ({set_name})",
    )

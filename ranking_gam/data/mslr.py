"""
MSLR-WEB10K / MSLR-WEB30K data loading.

Downloads MSLR-WEB10K automatically if not found locally.
MSLR-WEB30K requires manual download from Microsoft Research.

Both datasets: 136 features, 5-level relevance (0-4), SVMLight format.
"""

import glob
import os
import urllib.request
import zipfile

from .svmlight import load_svmlight_dataset

MSLR_NUM_FEATURES = 136


def _find_split_files(data_dir, dataset_name):
    """Search common directory layouts for train.txt / test.txt."""
    search_paths = [
        f"{data_dir}/{dataset_name}/Fold1",
        f"{data_dir}/Fold1",
        data_dir,
    ]
    for p in search_paths:
        if os.path.exists(f"{p}/train.txt"):
            return f"{p}/train.txt", f"{p}/test.txt"
    return None, None


def load_mslr(data_dir="data", max_train=6000, max_eval=2000, list_size=40):
    """Load MSLR-WEB10K (Fold1) for learning-to-rank experiments.

    Downloads automatically if not present.

    Args:
        data_dir: directory to store/find the data
        max_train: max number of training queries
        max_eval: max number of evaluation queries
        list_size: pad/truncate each query to this many documents

    Returns:
        (train_X, train_y, eval_X, eval_y) as float32 numpy arrays
        Shapes: X = [num_queries, list_size, 136], y = [num_queries, list_size]
        Padding uses -1.0 labels to mark invalid positions.
    """
    os.makedirs(data_dir, exist_ok=True)

    train_path, test_path = _find_split_files(data_dir, "MSLR-WEB10K")

    if not train_path:
        print("Downloading MSLR-WEB10K...")
        zip_path = f"{data_dir}/MSLR-WEB10K.zip"
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/personalization-takehome/MSLR-WEB10K.zip",
            zip_path,
        )
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        os.remove(zip_path)
        train_path = glob.glob(f"{data_dir}/**/train.txt", recursive=True)[0]
        test_path = train_path.replace("train.txt", "test.txt")

    return load_svmlight_dataset(
        train_path, test_path, MSLR_NUM_FEATURES,
        max_train=max_train, max_eval=max_eval, list_size=list_size,
        name="MSLR-WEB10K",
    )


def load_mslr30k(data_dir="data", fold=1, max_train=6000, max_eval=2000,
                 list_size=40):
    """Load MSLR-WEB30K for learning-to-rank experiments.

    Requires manual download from:
        https://www.microsoft.com/en-us/research/project/mslr/

    Expected directory structure:
        data_dir/MSLR-WEB30K/Fold{fold}/train.txt
        data_dir/MSLR-WEB30K/Fold{fold}/test.txt

    Args:
        data_dir: directory containing the MSLR-WEB30K folder
        fold: which fold to use (1-5, default: 1)
        max_train: max number of training queries
        max_eval: max number of evaluation queries
        list_size: pad/truncate each query to this many documents

    Returns:
        (train_X, train_y, eval_X, eval_y) as float32 numpy arrays
        Same format as load_mslr().
    """
    fold_dir = os.path.join(data_dir, "MSLR-WEB30K", f"Fold{fold}")
    train_path = os.path.join(fold_dir, "train.txt")
    test_path = os.path.join(fold_dir, "test.txt")

    if not os.path.exists(train_path):
        # Also check flat layout
        train_path2, test_path2 = _find_split_files(data_dir, "MSLR-WEB30K")
        if train_path2:
            train_path, test_path = train_path2, test_path2
        else:
            raise FileNotFoundError(
                f"MSLR-WEB30K not found at {fold_dir}/.\n"
                f"Download from: https://www.microsoft.com/en-us/research/project/mslr/\n"
                f"Extract so that {fold_dir}/train.txt exists."
            )

    return load_svmlight_dataset(
        train_path, test_path, MSLR_NUM_FEATURES,
        max_train=max_train, max_eval=max_eval, list_size=list_size,
        name=f"MSLR-WEB30K (Fold{fold})",
    )


# Abbreviated feature names for MSLR-WEB10K/30K
MSLR_FEATURE_NAMES = [
    "BM25_body",
    "BM25_anchor",
    "BM25_title",
    "BM25_url",
    "BM25_wholedoc",
    "TF_body",
    "TF_anchor",
    "TF_title",
    "TF_url",
    "TF_wholedoc",
    "IDF_body",
    "IDF_anchor",
    "IDF_title",
    "IDF_url",
    "IDF_wholedoc",
    "TFIDF_body",
    "TFIDF_anchor",
    "TFIDF_title",
    "TFIDF_url",
    "TFIDF_wholedoc",
    "DL_body",
    "DL_anchor",
    "DL_title",
    "DL_url",
    "DL_wholedoc",
] + [f"f_{i}" for i in range(25, 136)]

"""
LETOR 4.0 (MQ2007, MQ2008) data loading.

Freely available from Microsoft Research:
    https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/letor-4-0/

MQ2007: 1,692 queries, 46 features, 3-level relevance (0-2)
MQ2008:   784 queries, 46 features, 3-level relevance (0-2)

Both use 5-fold cross-validation with train/val/test per fold.
Note: LETOR uses 3-level relevance (0, 1, 2), not 5-level like MSLR/Yahoo.
"""

import os

from .svmlight import load_svmlight_dataset

LETOR_NUM_FEATURES = 46

_DOWNLOAD_URL = (
    "https://www.microsoft.com/en-us/research/project/"
    "letor-learning-rank-information-retrieval/letor-4-0/"
)


def _find_letor_files(data_dir, dataset, fold):
    """Find LETOR train/test files in common directory layouts.

    LETOR uses different filenames: train.txt, test.txt or
    Fold{N}/train.txt, Fold{N}/test.txt.
    """
    search_dirs = [
        os.path.join(data_dir, dataset, f"Fold{fold}"),
        os.path.join(data_dir, f"Fold{fold}"),
        os.path.join(data_dir, dataset),
        data_dir,
    ]
    for d in search_dirs:
        candidate = os.path.join(d, "train.txt")
        if os.path.exists(candidate):
            return candidate, os.path.join(d, "test.txt")
    return None, None


def load_mq2007(data_dir="data", fold=1, max_train=None, max_eval=None,
                list_size=40):
    """Load MQ2007 from LETOR 4.0.

    Freely available (no license required):
        https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/letor-4-0/

    Expected directory structure:
        data_dir/MQ2007/Fold{fold}/train.txt
        data_dir/MQ2007/Fold{fold}/test.txt

    Args:
        data_dir: directory containing the MQ2007 folder
        fold: which fold to use (1-5, default: 1)
        max_train: max training queries (default: None = all ~1200)
        max_eval: max evaluation queries (default: None = all ~300)
        list_size: documents per query

    Returns:
        (train_X, train_y, eval_X, eval_y) as float32 numpy arrays
        X shape: [num_queries, list_size, 46]
        Note: relevance labels are 0, 1, or 2 (3-level, not 5-level).
    """
    train_path, test_path = _find_letor_files(data_dir, "MQ2007", fold)

    if train_path is None:
        raise FileNotFoundError(
            f"MQ2007 not found.\n"
            f"Download from (free, no license):\n"
            f"  {_DOWNLOAD_URL}\n"
            f"Then place files as:\n"
            f"  {data_dir}/MQ2007/Fold{fold}/train.txt\n"
            f"  {data_dir}/MQ2007/Fold{fold}/test.txt"
        )

    # MQ2007 is small (~1700 queries), so default to loading all
    mt = max_train if max_train is not None else 2000
    me = max_eval if max_eval is not None else 500

    return load_svmlight_dataset(
        train_path, test_path, LETOR_NUM_FEATURES,
        max_train=mt, max_eval=me, list_size=list_size,
        name=f"MQ2007 (Fold{fold})",
    )


def load_mq2008(data_dir="data", fold=1, max_train=None, max_eval=None,
                list_size=40):
    """Load MQ2008 from LETOR 4.0.

    Freely available (no license required):
        https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/letor-4-0/

    Expected directory structure:
        data_dir/MQ2008/Fold{fold}/train.txt
        data_dir/MQ2008/Fold{fold}/test.txt

    Args:
        data_dir: directory containing the MQ2008 folder
        fold: which fold to use (1-5, default: 1)
        max_train: max training queries (default: None = all ~500)
        max_eval: max evaluation queries (default: None = all ~150)
        list_size: documents per query

    Returns:
        (train_X, train_y, eval_X, eval_y) as float32 numpy arrays
        X shape: [num_queries, list_size, 46]
        Note: relevance labels are 0, 1, or 2 (3-level, not 5-level).
    """
    train_path, test_path = _find_letor_files(data_dir, "MQ2008", fold)

    if train_path is None:
        raise FileNotFoundError(
            f"MQ2008 not found.\n"
            f"Download from (free, no license):\n"
            f"  {_DOWNLOAD_URL}\n"
            f"Then place files as:\n"
            f"  {data_dir}/MQ2008/Fold{fold}/train.txt\n"
            f"  {data_dir}/MQ2008/Fold{fold}/test.txt"
        )

    # MQ2008 is very small (~784 queries), load all
    mt = max_train if max_train is not None else 1000
    me = max_eval if max_eval is not None else 300

    return load_svmlight_dataset(
        train_path, test_path, LETOR_NUM_FEATURES,
        max_train=mt, max_eval=me, list_size=list_size,
        name=f"MQ2008 (Fold{fold})",
    )


# LETOR 4.0 feature names (46 features from Gov2 collection)
LETOR_FEATURE_NAMES = [
    # TF features (body, anchor, title, URL, whole doc)
    "TF_body", "TF_anchor", "TF_title", "TF_url", "TF_wholedoc",
    # IDF features
    "IDF_body", "IDF_anchor", "IDF_title", "IDF_url", "IDF_wholedoc",
    # TF-IDF features
    "TFIDF_body", "TFIDF_anchor", "TFIDF_title", "TFIDF_url", "TFIDF_wholedoc",
    # DL (document length)
    "DL_body", "DL_anchor", "DL_title", "DL_url", "DL_wholedoc",
    # BM25
    "BM25_body", "BM25_anchor", "BM25_title", "BM25_url", "BM25_wholedoc",
    # LMIR.ABS (Absolute Discount LM)
    "LMIR_ABS_body", "LMIR_ABS_anchor", "LMIR_ABS_title", "LMIR_ABS_url", "LMIR_ABS_wholedoc",
    # LMIR.DIR (Dirichlet LM)
    "LMIR_DIR_body", "LMIR_DIR_anchor", "LMIR_DIR_title", "LMIR_DIR_url", "LMIR_DIR_wholedoc",
    # LMIR.JM (Jelinek-Mercer LM)
    "LMIR_JM_body", "LMIR_JM_anchor", "LMIR_JM_title", "LMIR_JM_url", "LMIR_JM_wholedoc",
    # Page-level features
    "PageRank", "InlinkNum", "OutlinkNum", "NumSlash", "URLLen", "NumChild",
]

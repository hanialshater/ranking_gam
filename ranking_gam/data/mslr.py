"""
MSLR-WEB10K data loading and preprocessing.

Downloads the dataset if not found locally, parses the SVMLight format,
pads/truncates lists to a fixed size, and applies log1p transform.
"""

import glob
import os
import urllib.request
import zipfile

import numpy as np


def load_mslr(data_dir="data", max_train=6000, max_eval=2000, list_size=40):
    """
    Load MSLR-WEB10K (Fold1) for learning-to-rank experiments.

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

    train_path = test_path = None
    for p in [f"{data_dir}/MSLR-WEB10K/Fold1", f"{data_dir}/Fold1", data_dir]:
        if os.path.exists(f"{p}/train.txt"):
            train_path, test_path = f"{p}/train.txt", f"{p}/test.txt"
            break

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

    def parse(path, max_q):
        queries = {}
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                label = float(parts[0])
                qid = next(
                    (p.split(":")[1] for p in parts if p.startswith("qid:")), None
                )
                if not qid:
                    continue
                feats = {
                    int(p.split(":")[0]): float(p.split(":")[1])
                    for p in parts[1:]
                    if ":" in p and not p.startswith("qid:")
                }
                if qid not in queries:
                    if len(queries) >= max_q:
                        continue
                    queries[qid] = {"f": [], "l": []}
                queries[qid]["l"].append(label)
                queries[qid]["f"].append([feats.get(i, 0.0) for i in range(1, 137)])
        return queries

    def prepare(queries):
        X, y = [], []
        for q in queries.values():
            f = np.array(q["f"], dtype=np.float32)
            l = np.array(q["l"], dtype=np.float32)
            n = len(l)
            if n == 0:
                continue
            if n < list_size:
                f = np.vstack(
                    [f, np.zeros((list_size - n, 136), dtype=np.float32)]
                )
                l = np.concatenate([l, np.full(list_size - n, -1.0)])
            else:
                idx = np.argsort(-l)[:list_size]
                f, l = f[idx], l[idx]
            X.append(f)
            y.append(l)
        X = np.stack(X)
        y = np.stack(y)
        X = np.sign(X) * np.log1p(np.abs(X))
        return X.astype(np.float32), y.astype(np.float32)

    print("Loading data...")
    train_X, train_y = prepare(parse(train_path, max_train))
    eval_X, eval_y = prepare(parse(test_path, max_eval))
    print(f"Train: {train_X.shape}, Eval: {eval_X.shape}")
    return train_X, train_y, eval_X, eval_y


# Abbreviated feature names for MSLR-WEB10K
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

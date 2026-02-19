"""
Generic SVMLight/LibSVM parser for learning-to-rank datasets.

All standard L2R benchmarks (MSLR, Yahoo, Istella, LETOR) use SVMLight format:
    <label> qid:<query_id> <fid>:<value> <fid>:<value> ... [# comment]

This module provides a shared parser so each dataset loader only needs to
handle download/path logic and call the generic routines.
"""

import os
import tarfile
import urllib.request
import zipfile

import numpy as np


def parse_svmlight_file(path, max_queries=None, num_features=136):
    """Parse a SVMLight file into per-query groups.

    Args:
        path: path to SVMLight file
        max_queries: stop after this many queries (None = all)
        num_features: number of features to extract (1-indexed in file)

    Returns:
        dict mapping qid -> {"features": list[list[float]], "labels": list[float]}
    """
    queries = {}
    with open(path) as f:
        for line in f:
            # Strip comments (LETOR 4.0 appends # docid=...)
            line = line.split("#")[0].strip()
            parts = line.split()
            if len(parts) < 2:
                continue

            label = float(parts[0])
            qid = None
            feats = {}
            for p in parts[1:]:
                if p.startswith("qid:"):
                    qid = p.split(":")[1]
                elif ":" in p:
                    idx_str, val_str = p.split(":", 1)
                    feats[int(idx_str)] = float(val_str)

            if qid is None:
                continue
            if qid not in queries:
                if max_queries is not None and len(queries) >= max_queries:
                    continue
                queries[qid] = {"features": [], "labels": []}

            queries[qid]["labels"].append(label)
            queries[qid]["features"].append(
                [feats.get(i, 0.0) for i in range(1, num_features + 1)]
            )

    return queries


def prepare_arrays(queries, num_features, list_size=40, log1p=True,
                   clip_value=None):
    """Convert parsed query dicts to padded numpy arrays.

    Args:
        queries: output of parse_svmlight_file()
        num_features: feature dimensionality
        list_size: pad/truncate each query to this many documents
        log1p: apply sign(x)*log1p(|x|) transform
        clip_value: clip feature values to [-clip_value, clip_value] before
            transforms. Useful for Istella which has extreme outliers.

    Returns:
        (X, y) as float32 numpy arrays
        X shape: [num_queries, list_size, num_features]
        y shape: [num_queries, list_size]
        Padding uses -1.0 labels to mark invalid positions.
    """
    X_list, y_list = [], []
    for q in queries.values():
        f = np.array(q["features"], dtype=np.float32)
        labels = np.array(q["labels"], dtype=np.float32)
        n = len(labels)
        if n == 0:
            continue
        if n < list_size:
            f = np.vstack(
                [f, np.zeros((list_size - n, num_features), dtype=np.float32)]
            )
            labels = np.concatenate([labels, np.full(list_size - n, -1.0)])
        else:
            idx = np.argsort(-labels)[:list_size]
            f, labels = f[idx], labels[idx]
        X_list.append(f)
        y_list.append(labels)

    X = np.stack(X_list)
    y = np.stack(y_list)

    if clip_value is not None:
        X = np.clip(X, -clip_value, clip_value)
    if log1p:
        X = np.sign(X) * np.log1p(np.abs(X))

    return X.astype(np.float32), y.astype(np.float32)


def load_svmlight_dataset(train_path, test_path, num_features,
                          max_train=6000, max_eval=2000, list_size=40,
                          log1p=True, clip_value=None, name="dataset"):
    """Load train/test splits from SVMLight files.

    Args:
        train_path: path to training SVMLight file
        test_path: path to test SVMLight file
        num_features: number of features in the dataset
        max_train: max training queries to load
        max_eval: max evaluation queries to load
        list_size: documents per query (pad/truncate)
        log1p: apply log1p feature transform
        clip_value: clip extreme feature values (None = no clipping)
        name: dataset name for print messages

    Returns:
        (train_X, train_y, eval_X, eval_y) as float32 numpy arrays
    """
    print(f"Loading {name}...")
    train_q = parse_svmlight_file(train_path, max_train, num_features)
    test_q = parse_svmlight_file(test_path, max_eval, num_features)

    train_X, train_y = prepare_arrays(
        train_q, num_features, list_size, log1p, clip_value
    )
    eval_X, eval_y = prepare_arrays(
        test_q, num_features, list_size, log1p, clip_value
    )
    print(f"  Train: {train_X.shape}, Eval: {eval_X.shape}")
    return train_X, train_y, eval_X, eval_y


def _download_file(url, dest_path):
    """Download a URL to a local file with a browser-like User-Agent.

    Many CDNs and cloud storage services (OneDrive, etc.) reject requests
    from bare urllib without a User-Agent header.
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (ranking_gam dataset downloader)"},
    )
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)  # 1MB chunks
            if not chunk:
                break
            f.write(chunk)


def download_and_extract(url, data_dir, target_filename, extract_dir=None):
    """Download a file and extract it (zip or tar.gz).

    Args:
        url: download URL (or list of URLs to try as fallbacks)
        data_dir: directory to save/extract into
        target_filename: filename for the downloaded archive
        extract_dir: subdirectory name after extraction (None = data_dir)

    Returns:
        path to the extracted directory
    """
    os.makedirs(data_dir, exist_ok=True)
    archive_path = os.path.join(data_dir, target_filename)

    if not os.path.exists(archive_path):
        urls = url if isinstance(url, (list, tuple)) else [url]
        print(f"  Downloading {target_filename}...")
        last_err = None
        for u in urls:
            try:
                _download_file(u, archive_path)
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"  Download failed ({e}), trying next mirror...")
                if os.path.exists(archive_path):
                    os.remove(archive_path)
        if last_err is not None:
            raise RuntimeError(
                f"All download URLs failed for {target_filename}. "
                f"Last error: {last_err}\n"
                f"You can manually download the file and place it at:\n"
                f"  {archive_path}"
            ) from last_err

    print(f"  Extracting {target_filename}...")
    if target_filename.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(data_dir)
    elif target_filename.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as t:
            t.extractall(data_dir)
    else:
        raise ValueError(f"Unsupported archive format: {target_filename}")

    os.remove(archive_path)

    if extract_dir:
        return os.path.join(data_dir, extract_dir)
    return data_dir

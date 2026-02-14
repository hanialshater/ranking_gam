"""
FINN.no RecSys Slates dataset loader for GA2M slate optimization.

Downloads ~37.4M interactions from FINN.no marketplace and transforms
them into [B, L, D] feature matrices suitable for GA2M training with
position-item interaction pairs.

Dataset: Eide et al. "Dynamic Slate Recommendation with GRU and Thompson
Sampling" (Data Mining and Knowledge Discovery, 2022).

Source: https://github.com/finn-no/recsys_slates_dataset

Engineered features per item slot:
    0  position          — normalized slot position in slate (0..1)
    1  category          — item category ID normalized to (0..1) over 290 groups
    2  item_popularity   — item click frequency (log-scaled)
    3  category_popularity — category click frequency (log-scaled)
    4  interaction_type  — source: search=0.5, recommendation=1.0, undefined=0.0
    5  slate_length      — visible slate length normalized to (0..1) over max 20
    6  same_cat_before   — count of same-category items above this position (0..1)
    7  unique_cats_so_far — category diversity up to this position (0..1)

Recommended GA2M interaction pairs:
    (0, 1)  position x category       — which categories need top slots
    (0, 2)  position x item_pop       — popular items tolerate lower slots
    (0, 4)  position x interaction    — search vs rec position sensitivity
    (1, 6)  category x same_cat_before — category redundancy (submodularity)
"""

import json
import os

import numpy as np

FINN_NUM_FEATURES = 8

FINN_FEATURE_NAMES = [
    "position",
    "category",
    "item_popularity",
    "category_popularity",
    "interaction_type",
    "slate_length",
    "same_cat_before",
    "unique_cats_so_far",
]

# Recommended interaction pairs for GA2M
FINN_INTERACTION_PAIRS = [
    (0, 1),  # position x category
    (0, 2),  # position x item_popularity
    (0, 4),  # position x interaction_type
    (1, 6),  # category x same_cat_before
]

# HuggingFace dataset repo (most reliable)
_HF_REPO = "simeneide/recsys_slates_dataset"

# Google Drive file IDs from official repo (fallback)
_GDRIVE_IDS = {
    "data_int32": "1XHqyk01qi9qnvBTfWWwqgDzrdjv1eBVV",
    "data_full": "1VXKXIvPCJ7z4BCa4G_5-Q2XMAD7nXOc7",
    "ind2val": "1WOCKfuttMacCb84yQYcRjxjEtgPp6F4N",
    "itemattr": "1rKKyMQZqWp8vQ-Pl1SeHrQxzc5dXldnR",
}

_FINN_FILES = ["data.npz", "ind2val.json", "itemattr.npz"]


def _download_from_huggingface(data_dir):
    """Download FINN data files from HuggingFace Hub.

    Repo: https://huggingface.co/datasets/simeneide/recsys_slates_dataset

    Returns:
        True if all files downloaded successfully, False otherwise.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False

    os.makedirs(data_dir, exist_ok=True)

    for fname in _FINN_FILES:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            continue
        print(f"  Downloading {fname} from HuggingFace...")
        try:
            downloaded = hf_hub_download(
                repo_id=_HF_REPO,
                filename=fname,
                repo_type="dataset",
                local_dir=data_dir,
            )
            # hf_hub_download may place file in a subdir; copy if needed
            if downloaded != fpath and os.path.exists(downloaded):
                import shutil
                shutil.copy2(downloaded, fpath)
        except Exception as e:
            print(f"  HuggingFace download failed for {fname}: {e}")
            return False

    return all(os.path.exists(os.path.join(data_dir, f)) for f in _FINN_FILES)


def _download_from_gdrive(data_dir, use_int32=True):
    """Download FINN data files from Google Drive (fallback).

    File IDs from https://github.com/finn-no/recsys_slates_dataset
    """
    try:
        import gdown
    except ImportError:
        return False

    os.makedirs(data_dir, exist_ok=True)

    data_fileid = _GDRIVE_IDS["data_int32" if use_int32 else "data_full"]
    files = {
        "data.npz": data_fileid,
        "ind2val.json": _GDRIVE_IDS["ind2val"],
        "itemattr.npz": _GDRIVE_IDS["itemattr"],
    }

    for fname, gdrive_id in files.items():
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            continue

        url = f"https://drive.google.com/uc?id={gdrive_id}"
        print(f"  Downloading {fname} from Google Drive...")

        try:
            gdown.download(url, fpath, quiet=False, fuzzy=True)
        except TypeError:
            gdown.download(url, fpath, quiet=False)
        except Exception as e:
            print(f"  Google Drive download failed for {fname}: {e}")
            return False

    return all(os.path.exists(os.path.join(data_dir, f)) for f in _FINN_FILES)


def _download_finn(data_dir, use_int32=True):
    """Download FINN slate data files.

    Tries HuggingFace Hub first (most reliable), falls back to Google Drive.

    Args:
        data_dir: directory to save files
        use_int32: if True, download int32 data.npz from Google Drive
            (HuggingFace hosts the default version)
    """
    # Try HuggingFace first
    if _download_from_huggingface(data_dir):
        return True

    print("  HuggingFace unavailable, trying Google Drive...")
    if _download_from_gdrive(data_dir, use_int32=use_int32):
        return True

    raise RuntimeError(
        "Failed to download FINN dataset from both HuggingFace and Google Drive.\n"
        "Manual options:\n"
        "  1. pip install huggingface_hub && huggingface-cli download "
        f"--repo-type dataset {_HF_REPO} --local-dir {data_dir}\n"
        "  2. Download data.npz, itemattr.npz, ind2val.json from:\n"
        f"     https://huggingface.co/datasets/{_HF_REPO}\n"
        f"     and place them in {data_dir}/\n"
        "  3. pip install gdown && pip install recsys-slates-dataset"
    )


def _compute_item_stats(slates, clicks, item_categories, n_items):
    """Pre-compute item and category popularity from click data.

    Returns:
        item_click_count: [n_items] click frequency per item
        cat_click_count: [n_cats] click frequency per category
    """
    n_cats = int(item_categories.max()) + 1
    item_click_count = np.zeros(n_items, dtype=np.float32)
    cat_click_count = np.zeros(n_cats, dtype=np.float32)

    for c in clicks.flatten():
        if c > 0:
            item_click_count[c] += 1
            cat_click_count[item_categories[c]] += 1

    return item_click_count, cat_click_count


def _engineer_features(
    slates, clicks, click_idx, slate_lengths, interaction_types,
    item_categories, item_pop, cat_pop, max_slate=20,
):
    """Transform raw FINN data into [B, L, D] feature matrices.

    Args:
        slates: [N, max_slate] item IDs per interaction
        clicks: [N] clicked item IDs
        click_idx: [N] position of click in slate
        slate_lengths: [N] visible slate lengths
        interaction_types: [N] 0=undefined, 1=search, 2=recommendation
        item_categories: [n_items] category per item
        item_pop: [n_items] log-scaled item popularity
        cat_pop: [n_cats] log-scaled category popularity
        max_slate: pad/truncate to this slate size

    Returns:
        X: [N, max_slate, 8] feature matrix
        y: [N, max_slate] labels (1=clicked, 0=not clicked, -1=padding)
    """
    N = len(slates)
    D = FINN_NUM_FEATURES
    n_cats = len(cat_pop)

    X = np.zeros((N, max_slate, D), dtype=np.float32)
    y = np.full((N, max_slate), -1.0, dtype=np.float32)

    for i in range(N):
        sl = int(slate_lengths[i])
        sl = min(sl, max_slate)
        itype = int(interaction_types[i])
        itype_val = {0: 0.0, 1: 0.5, 2: 1.0}.get(itype, 0.0)

        seen_cats = set()
        same_cat_counts = np.zeros(max_slate, dtype=np.float32)

        for pos in range(sl):
            item_id = int(slates[i, pos])
            if item_id <= 0:
                continue

            cat_id = int(item_categories[item_id]) if item_id < len(item_categories) else 0

            # Count same-category items before this position
            same_cat_counts[pos] = sum(1 for c in seen_cats if c == cat_id)
            seen_cats.add(cat_id)

            X[i, pos, 0] = pos / max(sl - 1, 1)                     # position (0..1)
            X[i, pos, 1] = cat_id / max(n_cats - 1, 1)              # category (0..1)
            X[i, pos, 2] = item_pop[item_id]                        # item popularity
            X[i, pos, 3] = cat_pop[cat_id]                          # category popularity
            X[i, pos, 4] = itype_val                                 # interaction type
            X[i, pos, 5] = sl / max_slate                            # slate length (0..1)
            X[i, pos, 6] = same_cat_counts[pos] / max(pos, 1)       # same_cat_before (0..1)

            unique_so_far = len(set(
                int(item_categories[int(slates[i, p])])
                for p in range(pos + 1)
                if int(slates[i, p]) > 0 and int(slates[i, p]) < len(item_categories)
            ))
            X[i, pos, 7] = unique_so_far / max(pos + 1, 1)          # unique_cats_so_far (0..1)

            # Label: 1 if this position was clicked, 0 otherwise
            y[i, pos] = 1.0 if pos == int(click_idx[i]) else 0.0

    return X, y


def _load_from_files(data_dir):
    """Load raw arrays from FINN npz/json files.

    Returns:
        dict with raw arrays, or None if files missing
    """
    data_path = os.path.join(data_dir, "data.npz")
    itemattr_path = os.path.join(data_dir, "itemattr.npz")
    ind2val_path = os.path.join(data_dir, "ind2val.json")

    if not all(os.path.exists(p) for p in [data_path, itemattr_path, ind2val_path]):
        return None

    data = np.load(data_path, allow_pickle=True)
    itemattr = np.load(itemattr_path, allow_pickle=True)

    with open(ind2val_path) as f:
        ind2val = json.load(f)

    return {
        "slates": data["slate"],
        "clicks": data["click"],
        "click_idx": data["click_idx"],
        "slate_lengths": data["slate_lengths"],
        "interaction_types": data["interaction_type"],
        "item_categories": itemattr["itemattr"].flatten(),
        "ind2val": ind2val,
    }


def load_finn(data_dir="data/finn", max_queries=10000, list_size=20,
              train_frac=0.8, seed=42):
    """Load FINN RecSys Slates dataset for GA2M training.

    Downloads automatically if not present.

    Each "query" is one slate presentation. Features are engineered
    per-item to support GA2M with position-item interactions.

    Args:
        data_dir: directory to store/find FINN data files
        max_queries: max total interactions to use (subsampled for speed)
        list_size: pad/truncate slates to this size (max 20)
        train_frac: fraction of data for training (rest for eval)
        seed: random seed for train/eval split

    Returns:
        dict with keys:
            train_X: [B_train, list_size, 8] float32
            train_y: [B_train, list_size] float32 (1=click, 0=no-click, -1=pad)
            eval_X: [B_eval, list_size, 8] float32
            eval_y: [B_eval, list_size] float32
            num_features: 8
            feature_names: list of feature name strings
            interaction_pairs: recommended GA2M pairs
            metadata: dict with item_categories, ind2val, etc.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Try loading existing files first
    raw = _load_from_files(data_dir)

    if raw is None:
        # Try downloading
        print(f"FINN data not found in {data_dir}. Downloading...")
        _download_finn(data_dir)
        raw = _load_from_files(data_dir)

    if raw is None:
        raise FileNotFoundError(
            f"FINN data files not found in {data_dir} after download attempt.\n"
            "Please manually download data.npz, itemattr.npz, ind2val.json from:\n"
            "  https://github.com/finn-no/recsys_slates_dataset\n"
            f"and place them in {data_dir}/"
        )

    print("Loading FINN RecSys Slates dataset...")
    slates = raw["slates"]
    clicks = raw["clicks"]
    click_idx = raw["click_idx"]
    slate_lengths = raw["slate_lengths"]
    interaction_types = raw["interaction_types"]
    item_categories = raw["item_categories"]
    ind2val = raw["ind2val"]
    n_items = len(item_categories)

    N_total = len(slates)
    print(f"  Total interactions: {N_total:,}")
    print(f"  Items: {n_items:,}, Categories: {int(item_categories.max()) + 1}")

    # Subsample for tractability
    rng = np.random.RandomState(seed)
    if max_queries < N_total:
        indices = rng.choice(N_total, max_queries, replace=False)
        indices.sort()
    else:
        indices = np.arange(N_total)
        max_queries = N_total

    slates_sub = slates[indices]
    clicks_sub = clicks[indices]
    click_idx_sub = click_idx[indices]
    slate_lengths_sub = slate_lengths[indices]
    interaction_types_sub = interaction_types[indices]

    # Compute popularity stats from ALL data (not just subsample)
    print("  Computing item/category popularity stats...")
    item_click_count, cat_click_count = _compute_item_stats(
        slates, clicks, item_categories, n_items,
    )

    # Log-scale and normalize to [0, 1]
    item_pop = np.log1p(item_click_count)
    item_pop = item_pop / max(item_pop.max(), 1e-6)
    cat_pop = np.log1p(cat_click_count)
    cat_pop = cat_pop / max(cat_pop.max(), 1e-6)

    # Engineer features
    print(f"  Engineering features for {max_queries:,} interactions...")
    X, y = _engineer_features(
        slates_sub, clicks_sub, click_idx_sub, slate_lengths_sub,
        interaction_types_sub, item_categories, item_pop, cat_pop,
        max_slate=list_size,
    )

    # Train/eval split
    n_train = int(len(X) * train_frac)
    perm = rng.permutation(len(X))
    train_idx = perm[:n_train]
    eval_idx = perm[n_train:]

    train_X = X[train_idx]
    train_y = y[train_idx]
    eval_X = X[eval_idx]
    eval_y = y[eval_idx]

    # Stats
    valid_train = train_y >= 0
    valid_eval = eval_y >= 0
    click_rate_train = train_y[valid_train].mean()
    click_rate_eval = eval_y[valid_eval].mean()

    print(f"  Train: {len(train_X):,} queries, click rate: {click_rate_train:.4f}")
    print(f"  Eval:  {len(eval_X):,} queries, click rate: {click_rate_eval:.4f}")
    print(f"  Features: {FINN_NUM_FEATURES} ({', '.join(FINN_FEATURE_NAMES)})")

    return {
        "train_X": train_X,
        "train_y": train_y,
        "eval_X": eval_X,
        "eval_y": eval_y,
        "num_features": FINN_NUM_FEATURES,
        "feature_names": FINN_FEATURE_NAMES,
        "interaction_pairs": FINN_INTERACTION_PAIRS,
        "metadata": {
            "item_categories": item_categories,
            "ind2val": ind2val,
            "item_popularity": item_pop,
            "category_popularity": cat_pop,
        },
    }

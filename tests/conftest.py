"""Shared test fixtures -- all synthetic, no data download needed."""

import numpy as np
import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def synthetic_data():
    """Synthetic ranking data: 16 queries, list_size=10, 8 features."""
    B, L, D = 16, 10, 8
    rng = np.random.RandomState(42)
    X = rng.randn(B, L, D).astype(np.float32)
    y = rng.randint(0, 5, size=(B, L)).astype(np.float32)
    # Add some padding
    y[:, -2:] = -1.0
    return X, y


@pytest.fixture
def synthetic_tensors(synthetic_data):
    X, y = synthetic_data
    return torch.from_numpy(X), torch.from_numpy(y)


@pytest.fixture
def synthetic_augmented(synthetic_data):
    """Augmented data with 2 extra groupwise columns (cat, brand)."""
    X, y = synthetic_data
    B, L, D = X.shape
    rng = np.random.RandomState(123)
    cats = rng.randint(0, 4, size=(B, L)).astype(np.float32)
    brands = rng.randint(0, 6, size=(B, L)).astype(np.float32)
    X_aug = np.concatenate([X, cats[:, :, None], brands[:, :, None]], axis=-1)
    return X_aug, y

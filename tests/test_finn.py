"""Tests for FINN slate data loader and feature engineering."""

import numpy as np

from ranking_gam.data.finn import (
    FINN_FEATURE_NAMES,
    FINN_INTERACTION_PAIRS,
    FINN_NUM_FEATURES,
    _compute_item_stats,
    _engineer_features,
)


def _make_synthetic_finn(n_interactions=50, n_items=100, n_cats=10, max_slate=20):
    """Create synthetic FINN-like data for testing without download."""
    rng = np.random.RandomState(42)

    item_categories = rng.randint(0, n_cats, size=n_items)
    item_categories[0] = 0  # item 0 is special (padding)

    slates = np.zeros((n_interactions, max_slate), dtype=np.int64)
    clicks = np.zeros(n_interactions, dtype=np.int64)
    click_idx = np.zeros(n_interactions, dtype=np.int64)
    slate_lengths = np.zeros(n_interactions, dtype=np.int64)
    interaction_types = rng.randint(0, 3, size=n_interactions)

    for i in range(n_interactions):
        sl = rng.randint(3, max_slate + 1)
        slate_lengths[i] = sl
        items = rng.choice(np.arange(1, n_items), size=sl, replace=False)
        slates[i, :sl] = items
        ci = rng.randint(0, sl)
        click_idx[i] = ci
        clicks[i] = items[ci]

    return {
        "slates": slates,
        "clicks": clicks,
        "click_idx": click_idx,
        "slate_lengths": slate_lengths,
        "interaction_types": interaction_types,
        "item_categories": item_categories,
        "n_items": n_items,
        "n_cats": n_cats,
        "max_slate": max_slate,
    }


class TestFinnConstants:
    def test_num_features(self):
        assert FINN_NUM_FEATURES == 8

    def test_feature_names_length(self):
        assert len(FINN_FEATURE_NAMES) == FINN_NUM_FEATURES

    def test_interaction_pairs_valid(self):
        for f1, f2 in FINN_INTERACTION_PAIRS:
            assert 0 <= f1 < FINN_NUM_FEATURES
            assert 0 <= f2 < FINN_NUM_FEATURES
            assert f1 != f2


class TestComputeItemStats:
    def test_basic(self):
        syn = _make_synthetic_finn()
        item_count, cat_count = _compute_item_stats(
            syn["slates"], syn["clicks"],
            syn["item_categories"], syn["n_items"],
        )
        assert item_count.shape == (syn["n_items"],)
        assert cat_count.shape == (syn["n_cats"],)
        # Each interaction has exactly one click
        assert item_count.sum() == len(syn["clicks"])
        assert cat_count.sum() == len(syn["clicks"])


class TestEngineerFeatures:
    def test_output_shape(self):
        syn = _make_synthetic_finn(n_interactions=30)
        item_count, cat_count = _compute_item_stats(
            syn["slates"], syn["clicks"],
            syn["item_categories"], syn["n_items"],
        )
        item_pop = np.log1p(item_count)
        item_pop /= max(item_pop.max(), 1e-6)
        cat_pop = np.log1p(cat_count)
        cat_pop /= max(cat_pop.max(), 1e-6)

        X, y = _engineer_features(
            syn["slates"], syn["clicks"], syn["click_idx"],
            syn["slate_lengths"], syn["interaction_types"],
            syn["item_categories"], item_pop, cat_pop,
            max_slate=syn["max_slate"],
        )
        assert X.shape == (30, syn["max_slate"], FINN_NUM_FEATURES)
        assert y.shape == (30, syn["max_slate"])

    def test_labels_correct(self):
        syn = _make_synthetic_finn(n_interactions=20)
        item_count, cat_count = _compute_item_stats(
            syn["slates"], syn["clicks"],
            syn["item_categories"], syn["n_items"],
        )
        item_pop = np.log1p(item_count)
        item_pop /= max(item_pop.max(), 1e-6)
        cat_pop = np.log1p(cat_count)
        cat_pop /= max(cat_pop.max(), 1e-6)

        X, y = _engineer_features(
            syn["slates"], syn["clicks"], syn["click_idx"],
            syn["slate_lengths"], syn["interaction_types"],
            syn["item_categories"], item_pop, cat_pop,
            max_slate=syn["max_slate"],
        )
        # Each query should have exactly one click (label=1)
        for i in range(20):
            sl = int(syn["slate_lengths"][i])
            valid = y[i, :sl]
            assert valid.sum() == 1.0, f"Query {i} should have exactly 1 click"
            # Padding should be -1
            if sl < syn["max_slate"]:
                assert (y[i, sl:] == -1.0).all()

    def test_features_in_range(self):
        syn = _make_synthetic_finn(n_interactions=20)
        item_count, cat_count = _compute_item_stats(
            syn["slates"], syn["clicks"],
            syn["item_categories"], syn["n_items"],
        )
        item_pop = np.log1p(item_count)
        item_pop /= max(item_pop.max(), 1e-6)
        cat_pop = np.log1p(cat_count)
        cat_pop /= max(cat_pop.max(), 1e-6)

        X, y = _engineer_features(
            syn["slates"], syn["clicks"], syn["click_idx"],
            syn["slate_lengths"], syn["interaction_types"],
            syn["item_categories"], item_pop, cat_pop,
            max_slate=syn["max_slate"],
        )
        # All features should be in [0, 1] for valid positions
        for i in range(20):
            sl = int(syn["slate_lengths"][i])
            for pos in range(sl):
                if syn["slates"][i, pos] > 0:
                    for d in range(FINN_NUM_FEATURES):
                        val = X[i, pos, d]
                        assert 0.0 <= val <= 1.0 + 1e-6, (
                            f"Feature {FINN_FEATURE_NAMES[d]} out of range: {val}"
                        )

    def test_position_feature_monotone(self):
        """Position feature should increase along the slate."""
        syn = _make_synthetic_finn(n_interactions=10)
        item_count, cat_count = _compute_item_stats(
            syn["slates"], syn["clicks"],
            syn["item_categories"], syn["n_items"],
        )
        item_pop = np.log1p(item_count)
        item_pop /= max(item_pop.max(), 1e-6)
        cat_pop = np.log1p(cat_count)
        cat_pop /= max(cat_pop.max(), 1e-6)

        X, y = _engineer_features(
            syn["slates"], syn["clicks"], syn["click_idx"],
            syn["slate_lengths"], syn["interaction_types"],
            syn["item_categories"], item_pop, cat_pop,
            max_slate=syn["max_slate"],
        )
        for i in range(10):
            sl = min(int(syn["slate_lengths"][i]), syn["max_slate"])
            positions = X[i, :sl, 0]
            # Position should be non-decreasing
            for p in range(1, len(positions)):
                if syn["slates"][i, p] > 0:
                    assert positions[p] >= positions[p - 1]

"""Tests for interaction selection."""

import numpy as np

from ranking_gam.interactions import select_interactions_correlation


class TestSelectInteractions:
    def test_returns_correct_count(self):
        rng = np.random.RandomState(0)
        X = rng.randn(10, 5, 6).astype(np.float32)
        y = rng.randint(0, 4, (10, 5)).astype(np.float32)
        pairs = select_interactions_correlation(X, y, top_k=3)
        assert len(pairs) == 3
        for f1, f2 in pairs:
            assert 0 <= f1 < 6
            assert 0 <= f2 < 6
            assert f1 != f2

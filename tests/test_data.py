"""Tests for data loading utilities: SVMLight parser, dataset registry, loaders."""

import os
import tempfile

import numpy as np
import pytest

from ranking_gam.data.svmlight import (
    parse_svmlight_file,
    prepare_arrays,
    load_svmlight_dataset,
)
from ranking_gam.data import (
    load_dataset,
    get_num_features,
    DATASET_REGISTRY,
    DATASET_NAMES,
    MSLR_NUM_FEATURES,
    YAHOO_SET1_NUM_FEATURES,
    YAHOO_SET2_NUM_FEATURES,
    ISTELLA_NUM_FEATURES,
    LETOR_NUM_FEATURES,
)


# ---- SVMLight format test data ----

SVMLIGHT_TRAIN = """\
4 qid:1 1:0.5 2:1.0 3:0.3
3 qid:1 1:0.2 2:0.8 3:0.1
0 qid:1 1:0.0 2:0.1 3:0.0
2 qid:2 1:0.9 2:0.5 3:0.7
1 qid:2 1:0.4 2:0.3 3:0.2
3 qid:2 1:0.7 2:0.9 3:0.5
4 qid:3 1:1.0 2:1.0 3:1.0
2 qid:3 1:0.3 2:0.4 3:0.5
"""

SVMLIGHT_TEST = """\
3 qid:10 1:0.6 2:0.7 3:0.4
1 qid:10 1:0.1 2:0.2 3:0.1
2 qid:11 1:0.5 2:0.5 3:0.5
0 qid:11 1:0.0 2:0.0 3:0.0
"""

# LETOR-style with comments
SVMLIGHT_LETOR = """\
2 qid:1 1:0.5 2:1.0 3:0.3 # docid=GX000-00-001
1 qid:1 1:0.2 2:0.8 3:0.1 # docid=GX000-00-002
0 qid:1 1:0.0 2:0.1 3:0.0 # docid=GX000-00-003
"""


@pytest.fixture
def svmlight_dir():
    """Create a temp directory with train.txt and test.txt SVMLight files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, "train.txt")
        test_path = os.path.join(tmpdir, "test.txt")
        with open(train_path, "w") as f:
            f.write(SVMLIGHT_TRAIN)
        with open(test_path, "w") as f:
            f.write(SVMLIGHT_TEST)
        yield tmpdir, train_path, test_path


@pytest.fixture
def letor_file():
    """Create a temp SVMLight file with LETOR-style comments."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SVMLIGHT_LETOR)
        path = f.name
    yield path
    os.unlink(path)


# ---- SVMLight parser tests ----

class TestParseSvmlightFile:
    def test_basic_parse(self, svmlight_dir):
        _, train_path, _ = svmlight_dir
        queries = parse_svmlight_file(train_path, num_features=3)
        assert len(queries) == 3
        assert "1" in queries
        assert "2" in queries
        assert "3" in queries

    def test_query_contents(self, svmlight_dir):
        _, train_path, _ = svmlight_dir
        queries = parse_svmlight_file(train_path, num_features=3)
        # Query 1 has 3 docs
        assert len(queries["1"]["labels"]) == 3
        assert queries["1"]["labels"] == [4.0, 3.0, 0.0]
        # Check features (1-indexed in file, 0-indexed in output)
        assert len(queries["1"]["features"][0]) == 3
        assert queries["1"]["features"][0][0] == pytest.approx(0.5)
        assert queries["1"]["features"][0][1] == pytest.approx(1.0)
        assert queries["1"]["features"][0][2] == pytest.approx(0.3)

    def test_max_queries(self, svmlight_dir):
        _, train_path, _ = svmlight_dir
        queries = parse_svmlight_file(train_path, max_queries=2, num_features=3)
        assert len(queries) == 2

    def test_missing_features_default_zero(self, svmlight_dir):
        """When num_features > actual features in file, missing ones default to 0."""
        _, train_path, _ = svmlight_dir
        queries = parse_svmlight_file(train_path, num_features=5)
        # Feature 4 and 5 (1-indexed) not in file, should be 0.0
        assert queries["1"]["features"][0][3] == 0.0
        assert queries["1"]["features"][0][4] == 0.0

    def test_letor_comments_stripped(self, letor_file):
        """LETOR-style comments (# docid=...) should be stripped."""
        queries = parse_svmlight_file(letor_file, num_features=3)
        assert len(queries) == 1
        assert len(queries["1"]["labels"]) == 3
        # Verify features were parsed correctly despite comments
        assert queries["1"]["features"][0][0] == pytest.approx(0.5)


class TestPrepareArrays:
    def test_output_shapes(self, svmlight_dir):
        _, train_path, _ = svmlight_dir
        queries = parse_svmlight_file(train_path, num_features=3)
        X, y = prepare_arrays(queries, num_features=3, list_size=5)
        assert X.shape == (3, 5, 3)
        assert y.shape == (3, 5)
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_padding(self, svmlight_dir):
        _, train_path, _ = svmlight_dir
        queries = parse_svmlight_file(train_path, num_features=3)
        X, y = prepare_arrays(queries, num_features=3, list_size=5)
        # Query 3 has only 2 docs, so positions 2-4 are padded
        # After sorting by label descending: [4, 2, pad, pad, pad]
        # Padded labels should be -1
        q3_idx = None
        for i in range(3):
            if y[i, 0] == 4.0 and y[i, 1] == 2.0:
                q3_idx = i
                break
        if q3_idx is not None:
            assert y[q3_idx, 2] == -1.0

    def test_truncation(self, svmlight_dir):
        _, train_path, _ = svmlight_dir
        queries = parse_svmlight_file(train_path, num_features=3)
        X, y = prepare_arrays(queries, num_features=3, list_size=2)
        assert X.shape == (3, 2, 3)
        assert y.shape == (3, 2)
        # Top-2 by label should be selected
        for i in range(3):
            assert y[i, 0] >= y[i, 1]  # sorted descending

    def test_log1p_transform(self, svmlight_dir):
        _, train_path, _ = svmlight_dir
        queries = parse_svmlight_file(train_path, num_features=3)
        X_log, _ = prepare_arrays(queries, num_features=3, list_size=5, log1p=True)
        X_raw, _ = prepare_arrays(queries, num_features=3, list_size=5, log1p=False)
        # log1p should compress values
        np.testing.assert_array_less(np.abs(X_log), np.abs(X_raw) + 1.0)

    def test_no_log1p(self, svmlight_dir):
        _, train_path, _ = svmlight_dir
        queries = parse_svmlight_file(train_path, num_features=3)
        X, _ = prepare_arrays(queries, num_features=3, list_size=5, log1p=False)
        # Raw values should be preserved
        # Check that at least some values are non-zero
        assert np.any(X != 0)

    def test_clip_value(self, svmlight_dir):
        _, train_path, _ = svmlight_dir
        queries = parse_svmlight_file(train_path, num_features=3)
        X, _ = prepare_arrays(queries, num_features=3, list_size=5,
                              log1p=False, clip_value=0.5)
        assert X.max() <= 0.5
        assert X.min() >= -0.5


class TestLoadSvmlightDataset:
    def test_full_pipeline(self, svmlight_dir):
        tmpdir, train_path, test_path = svmlight_dir
        train_X, train_y, eval_X, eval_y = load_svmlight_dataset(
            train_path, test_path, num_features=3,
            max_train=100, max_eval=100, list_size=4,
        )
        assert train_X.shape[0] == 3  # 3 train queries
        assert eval_X.shape[0] == 2  # 2 test queries
        assert train_X.shape[1] == 4  # list_size
        assert train_X.shape[2] == 3  # num_features
        assert eval_X.shape[1] == 4
        assert eval_X.shape[2] == 3

    def test_max_train_limit(self, svmlight_dir):
        _, train_path, test_path = svmlight_dir
        train_X, _, _, _ = load_svmlight_dataset(
            train_path, test_path, num_features=3,
            max_train=1, max_eval=100, list_size=4,
        )
        assert train_X.shape[0] == 1


# ---- Dataset registry tests ----

class TestDatasetRegistry:
    def test_registry_has_all_datasets(self):
        expected = {"mslr10k", "mslr30k", "yahoo1", "yahoo2",
                    "istella", "istella_full", "istella_x",
                    "mq2007", "mq2008"}
        assert set(DATASET_NAMES) == expected

    def test_registry_entries_have_required_keys(self):
        for name, entry in DATASET_REGISTRY.items():
            assert "loader" in entry, f"{name} missing 'loader'"
            assert "num_features" in entry, f"{name} missing 'num_features'"
            assert callable(entry["loader"]), f"{name} loader not callable"

    def test_feature_counts(self):
        assert MSLR_NUM_FEATURES == 136
        assert YAHOO_SET1_NUM_FEATURES == 699
        assert YAHOO_SET2_NUM_FEATURES == 700
        assert ISTELLA_NUM_FEATURES == 220
        assert LETOR_NUM_FEATURES == 46

    def test_get_num_features(self):
        assert get_num_features("mslr10k") == 136
        assert get_num_features("mslr30k") == 136
        assert get_num_features("yahoo1") == 699
        assert get_num_features("yahoo2") == 700
        assert get_num_features("istella") == 220
        assert get_num_features("mq2007") == 46
        assert get_num_features("mq2008") == 46

    def test_get_num_features_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_num_features("nonexistent")

    def test_load_dataset_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent")


# ---- Dataset-specific error message tests ----

class TestDatasetNotFoundErrors:
    """Each loader should raise FileNotFoundError with helpful download info."""

    def test_mslr30k_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="MSLR-WEB30K"):
            from ranking_gam.data.mslr import load_mslr30k
            load_mslr30k(data_dir=str(tmp_path))

    def test_yahoo_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Yahoo LTRC"):
            from ranking_gam.data.yahoo import load_yahoo
            load_yahoo(data_dir=str(tmp_path))

    def test_yahoo_invalid_set(self):
        with pytest.raises(ValueError, match="set_name must be"):
            from ranking_gam.data.yahoo import load_yahoo
            load_yahoo(set_name="set3")

    def test_istella_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Istella"):
            from ranking_gam.data.istella import load_istella
            load_istella(data_dir=str(tmp_path))

    def test_istella_invalid_variant(self):
        with pytest.raises(ValueError, match="variant must be"):
            from ranking_gam.data.istella import load_istella
            load_istella(variant="invalid")

    def test_mq2007_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="MQ2007"):
            from ranking_gam.data.letor import load_mq2007
            load_mq2007(data_dir=str(tmp_path))

    def test_mq2008_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="MQ2008"):
            from ranking_gam.data.letor import load_mq2008
            load_mq2008(data_dir=str(tmp_path))


# ---- Feature names tests ----

class TestFeatureNames:
    def test_mslr_feature_names_length(self):
        from ranking_gam.data.mslr import MSLR_FEATURE_NAMES
        assert len(MSLR_FEATURE_NAMES) == 136 - 25 + 25  # 25 named + 111 generic = 136

    def test_letor_feature_names_length(self):
        from ranking_gam.data.letor import LETOR_FEATURE_NAMES
        assert len(LETOR_FEATURE_NAMES) == 46


# ---- Load from synthetic SVMLight files via load_dataset() ----

class TestLoadDatasetWithSyntheticFiles:
    """Test the unified load_dataset() with synthetic SVMLight files
    placed in the expected directory layout."""

    def test_mslr10k_layout(self, tmp_path):
        """Test MSLR-WEB10K file discovery with synthetic data."""
        fold_dir = tmp_path / "MSLR-WEB10K" / "Fold1"
        fold_dir.mkdir(parents=True)
        (fold_dir / "train.txt").write_text(
            _make_svmlight(num_queries=5, num_features=136, docs_per_query=3)
        )
        (fold_dir / "test.txt").write_text(
            _make_svmlight(num_queries=2, num_features=136, docs_per_query=3,
                           qid_start=100)
        )
        train_X, train_y, eval_X, eval_y = load_dataset(
            "mslr10k", data_dir=str(tmp_path), max_train=10, max_eval=10,
        )
        assert train_X.shape[0] == 5
        assert eval_X.shape[0] == 2
        assert train_X.shape[2] == 136

    def test_mq2007_layout(self, tmp_path):
        """Test MQ2007 file discovery with synthetic data."""
        fold_dir = tmp_path / "MQ2007" / "Fold1"
        fold_dir.mkdir(parents=True)
        (fold_dir / "train.txt").write_text(
            _make_svmlight(num_queries=3, num_features=46, docs_per_query=4,
                           max_label=2)
        )
        (fold_dir / "test.txt").write_text(
            _make_svmlight(num_queries=2, num_features=46, docs_per_query=4,
                           qid_start=50, max_label=2)
        )
        train_X, train_y, eval_X, eval_y = load_dataset(
            "mq2007", data_dir=str(tmp_path), max_train=10, max_eval=10,
        )
        assert train_X.shape[0] == 3
        assert train_X.shape[2] == 46
        # Labels should be 0, 1, or 2 (3-level)
        valid = train_y >= 0
        assert train_y[valid].max() <= 2

    def test_yahoo_layout(self, tmp_path):
        """Test Yahoo file discovery with synthetic data."""
        yahoo_dir = tmp_path / "yahoo"
        yahoo_dir.mkdir()
        (yahoo_dir / "set1.train.txt").write_text(
            _make_svmlight(num_queries=3, num_features=699, docs_per_query=2)
        )
        (yahoo_dir / "set1.test.txt").write_text(
            _make_svmlight(num_queries=2, num_features=699, docs_per_query=2,
                           qid_start=100)
        )
        train_X, train_y, eval_X, eval_y = load_dataset(
            "yahoo1", data_dir=str(tmp_path), max_train=10, max_eval=10,
        )
        assert train_X.shape[0] == 3
        assert train_X.shape[2] == 699

    def test_istella_layout(self, tmp_path):
        """Test Istella-S file discovery with synthetic data."""
        istella_dir = tmp_path / "istella-s"
        istella_dir.mkdir()
        (istella_dir / "train.txt").write_text(
            _make_svmlight(num_queries=3, num_features=220, docs_per_query=2)
        )
        (istella_dir / "test.txt").write_text(
            _make_svmlight(num_queries=2, num_features=220, docs_per_query=2,
                           qid_start=100)
        )
        train_X, train_y, eval_X, eval_y = load_dataset(
            "istella", data_dir=str(tmp_path), max_train=10, max_eval=10,
        )
        assert train_X.shape[0] == 3
        assert train_X.shape[2] == 220

    def test_mq2008_layout(self, tmp_path):
        """Test MQ2008 file discovery with synthetic data."""
        fold_dir = tmp_path / "MQ2008" / "Fold1"
        fold_dir.mkdir(parents=True)
        (fold_dir / "train.txt").write_text(
            _make_svmlight(num_queries=3, num_features=46, docs_per_query=3,
                           max_label=2)
        )
        (fold_dir / "test.txt").write_text(
            _make_svmlight(num_queries=2, num_features=46, docs_per_query=3,
                           qid_start=50, max_label=2)
        )
        train_X, train_y, eval_X, eval_y = load_dataset(
            "mq2008", data_dir=str(tmp_path), max_train=10, max_eval=10,
        )
        assert train_X.shape[0] == 3
        assert train_X.shape[2] == 46

    def test_mslr30k_layout(self, tmp_path):
        """Test MSLR-WEB30K file discovery with synthetic data."""
        fold_dir = tmp_path / "MSLR-WEB30K" / "Fold1"
        fold_dir.mkdir(parents=True)
        (fold_dir / "train.txt").write_text(
            _make_svmlight(num_queries=3, num_features=136, docs_per_query=3)
        )
        (fold_dir / "test.txt").write_text(
            _make_svmlight(num_queries=2, num_features=136, docs_per_query=3,
                           qid_start=100)
        )
        train_X, train_y, eval_X, eval_y = load_dataset(
            "mslr30k", data_dir=str(tmp_path), max_train=10, max_eval=10,
        )
        assert train_X.shape[0] == 3
        assert train_X.shape[2] == 136


# ---- Helpers ----

def _make_svmlight(num_queries, num_features, docs_per_query=3,
                   max_label=4, qid_start=1):
    """Generate synthetic SVMLight-format data."""
    rng = np.random.RandomState(42 + qid_start)
    lines = []
    for qi in range(num_queries):
        qid = qid_start + qi
        for di in range(docs_per_query):
            label = rng.randint(0, max_label + 1)
            feats = " ".join(
                f"{j+1}:{rng.uniform(-1, 1):.4f}"
                for j in range(num_features)
            )
            lines.append(f"{label} qid:{qid} {feats}")
    return "\n".join(lines) + "\n"

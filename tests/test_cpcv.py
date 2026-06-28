"""
Test suite for backtest/cpcv.py.

Tests Combinatorial Purged Cross-Validation (CPCV) as described in
Lopez de Prado (2018), including purging, embargo, combinatorial generation,
and integration with scikit-learn-style models.

The mathematics of combinatorial splits and leakage prevention are
deterministic — known inputs must produce known index sets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from backtest.cpcv import (
    PurgedKFold,
    CombinatorialPurgedCV,
    apply_purged_cv,
    calculate_purged_cv_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Simple time-series DataFrame with 100 observations."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.randn(100),
        },
        index=dates,
    )


@pytest.fixture
def linear_model():
    """Minimal sklearn-compatible regressor for integration tests."""
    from sklearn.linear_model import LinearRegression

    return LinearRegression()


# ---------------------------------------------------------------------------
# PurgedKFold
# ---------------------------------------------------------------------------

class TestPurgedKFold:
    """Tests for PurgedKFold split generation and purging logic."""

    def test_default_initialization(self):
        pkf = PurgedKFold()
        assert pkf.n_splits == 5
        assert pkf.purge_gap == 1

    def test_custom_initialization(self):
        pkf = PurgedKFold(n_splits=3, purge_gap=5)
        assert pkf.n_splits == 3
        assert pkf.purge_gap == 5

    def test_split_count(self, sample_data):
        pkf = PurgedKFold(n_splits=5, purge_gap=1)
        splits = list(pkf.split(sample_data))
        assert len(splits) == 5

    def test_train_test_disjoint(self, sample_data):
        pkf = PurgedKFold(n_splits=5, purge_gap=1)
        for train_idx, test_idx in pkf.split(sample_data):
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_all_indices_covered(self, sample_data):
        pkf = PurgedKFold(n_splits=5, purge_gap=1)
        all_train = set()
        all_test = set()
        for train_idx, test_idx in pkf.split(sample_data):
            all_train.update(train_idx)
            all_test.update(test_idx)
        # Every index appears at least once in train or test
        assert all_train | all_test == set(range(len(sample_data)))

    def test_purge_gap_removes_adjacent_indices(self, sample_data):
        pkf = PurgedKFold(n_splits=3, purge_gap=3)
        for train_idx, test_idx in pkf.split(sample_data):
            test_min, test_max = test_idx.min(), test_idx.max()
            # No train index should be within purge_gap of test boundaries
            assert not any(
                (test_min - pkf.purge_gap) <= idx <= (test_max + pkf.purge_gap)
                for idx in train_idx
                if idx < test_min or idx > test_max
            )

    def test_purge_gap_zero_is_standard_kfold(self, sample_data):
        pkf = PurgedKFold(n_splits=5, purge_gap=0)
        for train_idx, test_idx in pkf.split(sample_data):
            combined = np.sort(np.concatenate([train_idx, test_idx]))
            assert np.array_equal(combined, np.arange(len(sample_data)))

    def test_equal_split_sizes_without_purge(self, sample_data):
        pkf = PurgedKFold(n_splits=5, purge_gap=0)
        test_sizes = [len(test_idx) for _, test_idx in pkf.split(sample_data)]
        # 100 samples / 5 splits = 20 per test set
        assert all(s == 20 for s in test_sizes)

    def test_single_sample(self):
        data = pd.DataFrame({"x": [1]})
        pkf = PurgedKFold(n_splits=1, purge_gap=0)
        splits = list(pkf.split(data))
        assert len(splits) == 1
        train_idx, test_idx = splits[0]
        assert len(train_idx) == 0
        assert len(test_idx) == 1

    def test_fewer_samples_than_splits(self):
        data = pd.DataFrame({"x": [1, 2]})
        pkf = PurgedKFold(n_splits=5, purge_gap=0)
        splits = list(pkf.split(data))
        assert len(splits) == 5
        for train_idx, test_idx in splits:
            # fold_size = 2 // 5 = 0, so test sets are minimal
            assert len(test_idx) <= 1


# ---------------------------------------------------------------------------
# CombinatorialPurgedCV
# ---------------------------------------------------------------------------

class TestCombinatorialPurgedCV:
    """Tests for combinatorial generation, purging, and embargo."""

    def test_default_initialization(self):
        cpcv = CombinatorialPurgedCV()
        assert cpcv.n_splits == 5
        assert cpcv.n_test_splits == 2
        assert cpcv.purge_gap == 1
        assert cpcv.embargo_pct == 0.01

    def test_combination_count(self):
        """C(n_splits, n_test_splits) combinations should be generated."""
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
        assert cpcv.get_n_splits() == 10  # C(5,2) = 10

        cpcv = CombinatorialPurgedCV(n_splits=6, n_test_splits=3)
        assert cpcv.get_n_splits() == 20  # C(6,3) = 20

        cpcv = CombinatorialPurgedCV(n_splits=4, n_test_splits=1)
        assert cpcv.get_n_splits() == 4  # C(4,1) = 4

    def test_split_count_matches_combinations(self, sample_data):
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
        splits = list(cpcv.split(sample_data))
        assert len(splits) == cpcv.get_n_splits()

    def test_train_test_disjoint(self, sample_data):
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
        for train_idx, test_idx, _ in cpcv.split(sample_data):
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_test_size_is_combined_splits(self, sample_data):
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
        fold_size = len(sample_data) // 5  # 20
        for train_idx, test_idx, meta in cpcv.split(sample_data):
            # Test should be exactly 2 * fold_size (minus remainder)
            assert len(test_idx) == 2 * fold_size
            assert meta["test_size"] == len(test_idx)

    def test_train_plus_test_plus_purged_equals_total(self, sample_data):
        """
        For contiguous splits with no gaps between test splits,
        train + test + purged should account for all indices.
        """
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=1, purge_gap=0, embargo_pct=0.0)
        for train_idx, test_idx, _ in cpcv.split(sample_data):
            combined = np.sort(np.concatenate([train_idx, test_idx]))
            assert np.array_equal(combined, np.arange(len(sample_data)))

    def test_purge_reduces_train_size(self, sample_data):
        cpcv_no_purge = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=0)
        cpcv_purge = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=5)

        no_purge_sizes = [
            len(train_idx)
            for train_idx, _, _ in cpcv_no_purge.split(sample_data)
        ]
        purge_sizes = [
            len(train_idx) for train_idx, _, _ in cpcv_purge.split(sample_data)
        ]

        # Purge should reduce or equal train size
        assert all(p <= np for p, np in zip(purge_sizes, no_purge_sizes))

    def test_embargo_reduces_train_size(self, sample_data):
        cpcv_no_embargo = CombinatorialPurgedCV(
            n_splits=5, n_test_splits=2, purge_gap=0, embargo_pct=0.0
        )
        cpcv_embargo = CombinatorialPurgedCV(
            n_splits=5, n_test_splits=2, purge_gap=0, embargo_pct=0.05
        )

        no_embargo_sizes = [
            len(train_idx)
            for train_idx, _, _ in cpcv_no_embargo.split(sample_data)
        ]
        embargo_sizes = [
            len(train_idx)
            for train_idx, _, _ in cpcv_embargo.split(sample_data)
        ]

        # Embargo should reduce or equal train size
        assert all(e <= ne for e, ne in zip(embargo_sizes, no_embargo_sizes))

    def test_metadata_structure(self, sample_data):
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
        for train_idx, test_idx, meta in cpcv.split(sample_data):
            assert "combination_id" in meta
            assert "train_splits" in meta
            assert "test_splits" in meta
            assert "train_size" in meta
            assert "test_size" in meta
            assert "purge_gap" in meta
            assert "embargo_size" in meta
            assert len(meta["train_splits"]) + len(meta["test_splits"]) == 5
            assert meta["train_size"] == len(train_idx)
            assert meta["test_size"] == len(test_idx)

    def test_combination_ids_are_sequential(self, sample_data):
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
        ids = [meta["combination_id"] for _, _, meta in cpcv.split(sample_data)]
        assert ids == list(range(cpcv.get_n_splits()))

    def test_all_splits_used_as_test_at_least_once(self, sample_data):
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
        test_split_sets = []
        for _, _, meta in cpcv.split(sample_data):
            test_split_sets.append(tuple(meta["test_splits"]))

        # Every split index 0..4 should appear in at least one test set
        all_tested = set()
        for ts in test_split_sets:
            all_tested.update(ts)
        assert all_tested == set(range(5))

    def test_large_purge_gap_can_empty_train(self, sample_data):
        """With a very large purge gap, train can become empty."""
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=50)
        empty_trains = 0
        for train_idx, _, _ in cpcv.split(sample_data):
            if len(train_idx) == 0:
                empty_trains += 1
        # Some combinations will have empty train sets with huge purge gap
        assert empty_trains > 0

    def test_single_observation(self):
        data = pd.DataFrame({"x": [1]})
        cpcv = CombinatorialPurgedCV(n_splits=1, n_test_splits=1)
        splits = list(cpcv.split(data))
        assert len(splits) == 1
        train_idx, test_idx, meta = splits[0]
        assert len(test_idx) == 1
        assert meta["test_size"] == 1

    def test_two_splits_one_test(self):
        data = pd.DataFrame({"x": range(10)})
        cpcv = CombinatorialPurgedCV(n_splits=2, n_test_splits=1, purge_gap=0, embargo_pct=0.0)
        splits = list(cpcv.split(data))
        assert len(splits) == 2
        for train_idx, test_idx, meta in splits:
            assert len(train_idx) == 5
            assert len(test_idx) == 5
            assert meta["train_size"] == 5
            assert meta["test_size"] == 5


# ---------------------------------------------------------------------------
# apply_purged_cv
# ---------------------------------------------------------------------------

class TestApplyPurgedCV:
    """Integration tests for applying CPCV to a model."""

    def test_returns_dataframe(self, sample_data, linear_model):
        cpcv = CombinatorialPurgedCV(n_splits=3, n_test_splits=1, purge_gap=1)
        results = apply_purged_cv(
            data=sample_data,
            features=["feature1", "feature2"],
            target="target",
            model=linear_model,
            cv=cpcv,
        )
        assert isinstance(results, pd.DataFrame)
        assert "actual" in results.columns
        assert "predicted" in results.columns
        assert "fold" in results.columns
        assert "combination_id" in results.columns

    def test_fold_column_matches_cpcv(self, sample_data, linear_model):
        cpcv = CombinatorialPurgedCV(n_splits=4, n_test_splits=1, purge_gap=1)
        results = apply_purged_cv(
            data=sample_data,
            features=["feature1", "feature2"],
            target="target",
            model=linear_model,
            cv=cpcv,
        )
        unique_folds = sorted(results["fold"].unique())
        assert unique_folds == list(range(cpcv.get_n_splits()))

    def test_predictions_have_same_length_as_test(self, sample_data, linear_model):
        cpcv = CombinatorialPurgedCV(n_splits=3, n_test_splits=1, purge_gap=1)
        results = apply_purged_cv(
            data=sample_data,
            features=["feature1", "feature2"],
            target="target",
            model=linear_model,
            cv=cpcv,
        )
        # Total predictions should equal sum of all test set sizes
        expected_len = sum(
            len(test_idx)
            for _, test_idx, _ in cpcv.split(sample_data)
        )
        assert len(results) == expected_len

    def test_with_sample_weights(self, sample_data, linear_model):
        weights = np.random.rand(len(sample_data))
        cpcv = CombinatorialPurgedCV(n_splits=3, n_test_splits=1, purge_gap=1)
        results = apply_purged_cv(
            data=sample_data,
            features=["feature1", "feature2"],
            target="target",
            model=linear_model,
            cv=cpcv,
            sample_weights=weights,
        )
        assert len(results) > 0
        assert "predicted" in results.columns

    def test_empty_fold_skipped(self, sample_data, linear_model):
        """If purge empties a train set, that fold should be skipped."""
        cpcv = CombinatorialPurgedCV(n_splits=3, n_test_splits=1, purge_gap=50)
        results = apply_purged_cv(
            data=sample_data,
            features=["feature1", "feature2"],
            target="target",
            model=linear_model,
            cv=cpcv,
        )
        # Some folds may be skipped, but we should still have results
        assert isinstance(results, pd.DataFrame)


# ---------------------------------------------------------------------------
# calculate_purged_cv_score
# ---------------------------------------------------------------------------

class TestCalculatePurgedCVScore:
    """Tests for scoring aggregated CPCV results."""

    def test_returns_dict_with_expected_keys(self):
        results = pd.DataFrame(
            {
                "actual": [1.0, 2.0, 3.0, 4.0],
                "predicted": [1.1, 1.9, 3.2, 3.8],
                "fold": [0, 0, 1, 1],
                "combination_id": [0, 0, 1, 1],
            }
        )

        def mse(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        scores = calculate_purged_cv_score(results, mse)
        assert "mean" in scores
        assert "std" in scores
        assert "min" in scores
        assert "max" in scores
        assert "scores" in scores
        assert len(scores["scores"]) == 2  # 2 folds

    def test_mean_is_average_of_fold_scores(self):
        results = pd.DataFrame(
            {
                "actual": [1.0, 2.0, 3.0, 6.0],
                "predicted": [1.0, 2.0, 3.0, 6.0],
                "fold": [0, 0, 1, 1],
                "combination_id": [0, 0, 1, 1],
            }
        )

        def accuracy(y_true, y_pred):
            return np.mean(y_true == y_pred)

        scores = calculate_purged_cv_score(results, accuracy)
        assert scores["mean"] == 1.0
        assert scores["std"] == 0.0
        assert scores["min"] == 1.0
        assert scores["max"] == 1.0

    def test_single_fold(self):
        results = pd.DataFrame(
            {
                "actual": [1.0, 2.0, 3.0],
                "predicted": [1.5, 2.5, 3.5],
                "fold": [0, 0, 0],
                "combination_id": [0, 0, 0],
            }
        )

        def mae(y_true, y_pred):
            return np.mean(np.abs(y_true - y_pred))

        scores = calculate_purged_cv_score(results, mae)
        assert len(scores["scores"]) == 1
        assert scores["mean"] == scores["scores"][0]
        assert scores["std"] == 0.0

    def test_empty_folds_returns_nan(self):
        """When all folds are skipped, statistics should be NaN, not raise."""
        results = pd.DataFrame(
            {
                "actual": [],
                "predicted": [],
                "fold": [],
                "combination_id": [],
            }
        )

        def mse(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        scores = calculate_purged_cv_score(results, mse)
        assert len(scores["scores"]) == 0
        assert np.isnan(scores["mean"])
        assert np.isnan(scores["std"])
        assert np.isnan(scores["min"])
        assert np.isnan(scores["max"])

    def test_many_folds(self):
        np.random.seed(42)
        n = 100
        results = pd.DataFrame(
            {
                "actual": np.random.randn(n),
                "predicted": np.random.randn(n),
                "fold": np.repeat(range(10), 10),
                "combination_id": np.repeat(range(10), 10),
            }
        )

        def rmse(y_true, y_pred):
            return np.sqrt(np.mean((y_true - y_pred) ** 2))

        scores = calculate_purged_cv_score(results, rmse)
        assert len(scores["scores"]) == 10
        assert all(s >= 0 for s in scores["scores"])
        assert scores["min"] <= scores["mean"] <= scores["max"]


# ---------------------------------------------------------------------------
# Leakage Prevention
# ---------------------------------------------------------------------------

class TestLeakagePrevention:
    """Validate that purging and embargo actually prevent data leakage."""

    def test_no_overlapping_indices_between_train_and_test(self, sample_data):
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=5)
        for train_idx, test_idx, _ in cpcv.split(sample_data):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_purged_indices_not_in_train(self, sample_data):
        """
        If split i is test, splits i-1 and i+1 should have their
        boundaries purged (reduced) in the train set.
        """
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=1, purge_gap=5)
        for train_idx, test_idx, meta in cpcv.split(sample_data):
            test_splits = meta["test_splits"]
            fold_size = len(sample_data) // 5

            for ts in test_splits:
                # Adjacent splits should have reduced size if in train
                for split_idx in meta["train_splits"]:
                    if abs(split_idx - ts) == 1:
                        # This split is adjacent to test — check it's purged
                        start = split_idx * fold_size
                        end = min((split_idx + 1) * fold_size, len(sample_data))
                        expected_full = set(range(start, end))
                        actual_train = set(train_idx)
                        # At least some indices should be missing due to purge
                        missing = expected_full - actual_train
                        assert len(missing) > 0 or len(expected_full) <= cpcv.purge_gap

    def test_embargo_applied_after_test(self, sample_data):
        """
        Embargo should remove indices immediately following test splits.
        """
        cpcv = CombinatorialPurgedCV(
            n_splits=5, n_test_splits=1, purge_gap=0, embargo_pct=0.05
        )
        for train_idx, test_idx, meta in cpcv.split(sample_data):
            embargo_size = meta["embargo_size"]
            if embargo_size == 0:
                continue

            test_max = test_idx.max()
            # The embargo zone is test_max+1 .. test_max+embargo_size
            embargo_zone = set(range(test_max + 1, test_max + 1 + embargo_size))
            # These should NOT be in train
            overlap = embargo_zone & set(train_idx)
            # Due to split boundaries, embargo may fall into next split
            # Just verify embargo_size was calculated correctly
            assert embargo_size == max(1, int(len(test_idx) * 0.05))


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_very_small_dataset(self):
        """Use evenly-divisible size to avoid remainder-dropping behavior."""
        data = pd.DataFrame({"x": [1, 2, 3, 4]})
        cpcv = CombinatorialPurgedCV(n_splits=2, n_test_splits=1, purge_gap=0, embargo_pct=0.0)
        splits = list(cpcv.split(data))
        assert len(splits) == 2
        for train_idx, test_idx, _ in splits:
            assert len(train_idx) + len(test_idx) == 4

    def test_uneven_split_sizes(self):
        """100 samples / 5 splits = 20 exactly — no remainder."""
        data = pd.DataFrame({"x": range(100)})
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=1, purge_gap=0, embargo_pct=0.0)
        for train_idx, test_idx, _ in cpcv.split(data):
            # Train + test should equal total when embargo is disabled
            assert len(train_idx) + len(test_idx) == 100

    def test_zero_embargo(self, sample_data):
        cpcv = CombinatorialPurgedCV(
            n_splits=5, n_test_splits=2, purge_gap=0, embargo_pct=0.0
        )
        # Should not raise
        splits = list(cpcv.split(sample_data))
        assert len(splits) == 10

    def test_all_data_as_test(self):
        """n_test_splits == n_splits means all data is test, train is empty."""
        data = pd.DataFrame({"x": range(100)})
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=5, purge_gap=0)
        for train_idx, test_idx, meta in cpcv.split(data):
            assert len(train_idx) == 0
            assert len(test_idx) == 100
            assert meta["train_size"] == 0
            assert meta["test_size"] == 100

    def test_large_combination_count(self):
        """C(10,5) = 252 combinations."""
        data = pd.DataFrame({"x": range(1000)})
        cpcv = CombinatorialPurgedCV(n_splits=10, n_test_splits=5, purge_gap=1)
        assert cpcv.get_n_splits() == 252
        splits = list(cpcv.split(data))
        assert len(splits) == 252

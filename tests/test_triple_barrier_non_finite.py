"""
Regression tests for non-finite inputs in the triple barrier module.

A price stream from yfinance can contain NaN, zero, or missing ticks. These tests
ensure that the triple barrier labeling pipeline returns None or finite aggregates
instead of emitting RuntimeWarnings or propagating NaN/Inf into downstream
backtest reports.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from backtest.triple_barrier import (
    BarrierConfig,
    BarrierType,
    TripleBarrierLabel,
    analyze_barrier_distribution,
    apply_triple_barrier,
    get_barrier_levels,
    label_events,
)


def test_get_barrier_levels_rejects_zero_entry_price():
    """Zero entry price collapses barriers and must be rejected."""
    cfg = BarrierConfig.symmetric()
    upper, lower = get_barrier_levels(0.0, 0.01, cfg)
    assert np.isnan(upper)
    assert np.isnan(lower)


def test_get_barrier_levels_rejects_nan_entry_price():
    """NaN entry price must not produce finite barriers."""
    cfg = BarrierConfig.symmetric()
    upper, lower = get_barrier_levels(np.nan, 0.01, cfg)
    assert np.isnan(upper)
    assert np.isnan(lower)


def test_get_barrier_levels_rejects_negative_volatility():
    """Negative volatility would invert the barriers and must be rejected."""
    cfg = BarrierConfig.symmetric()
    upper, lower = get_barrier_levels(100.0, -0.01, cfg)
    assert np.isnan(upper)
    assert np.isnan(lower)


def test_get_barrier_levels_rejects_nan_volatility():
    """NaN volatility must not produce finite barriers."""
    cfg = BarrierConfig.symmetric()
    upper, lower = get_barrier_levels(100.0, np.nan, cfg)
    assert np.isnan(upper)
    assert np.isnan(lower)


def test_get_barrier_levels_keeps_valid_inputs():
    """Valid inputs still produce correct symmetric barriers."""
    cfg = BarrierConfig.symmetric()
    upper, lower = get_barrier_levels(100.0, 0.01, cfg)
    assert upper == pytest.approx(102.0)
    assert lower == pytest.approx(98.0)


class TestApplyTripleBarrierNonFinite:
    """Tests for apply_triple_barrier with degenerate prices."""

    def _make_prices(self, values):
        return pd.Series(
            values, index=pd.date_range("2024-01-01", periods=len(values), freq="D")
        )

    def test_zero_entry_price_returns_none(self):
        prices = self._make_prices([0.0, 0.0, 0.0])
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            result = apply_triple_barrier(prices, 0, 0.01, BarrierConfig())
        runtime = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert not runtime
        assert result is None

    def test_negative_entry_price_is_allowed(self):
        """Negative prices are non-physical but existing tests expect no crash."""
        prices = self._make_prices([-100.0, -102.0])
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            result = apply_triple_barrier(prices, 0, 0.02, BarrierConfig(max_holding=5))
        runtime = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert not runtime
        assert result is not None

    def test_nan_entry_price_returns_none(self):
        prices = self._make_prices([np.nan, 100.0, 101.0])
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            result = apply_triple_barrier(prices, 0, 0.01, BarrierConfig())
        runtime = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert not runtime
        assert result is None

    def test_nan_volatility_returns_none(self):
        prices = self._make_prices([100.0, 101.0, 102.0])
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            result = apply_triple_barrier(prices, 0, np.nan, BarrierConfig())
        runtime = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert not runtime
        assert result is None

    def test_negative_volatility_returns_none(self):
        prices = self._make_prices([100.0, 101.0, 102.0])
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            result = apply_triple_barrier(prices, 0, -0.01, BarrierConfig())
        runtime = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert not runtime
        assert result is None

    def test_valid_series_still_labels(self):
        prices = self._make_prices([100.0, 101.0, 102.5])
        result = apply_triple_barrier(
            prices,
            0,
            0.02,
            BarrierConfig(profit_take_std=1.0, stop_loss_std=1.0, max_holding=10),
        )
        assert result is not None
        assert result.barrier_type == BarrierType.UPPER
        assert result.return_pct == pytest.approx(0.025)


class TestLabelEventsNonFinite:
    """Tests for label_events with dirty price streams."""

    def _make_prices(self, values):
        return pd.Series(
            values, index=pd.date_range("2024-01-01", periods=len(values), freq="D")
        )

    def test_zero_price_in_series_skips_event(self):
        """A zero tick makes volatility NaN; the event should be skipped."""
        prices = self._make_prices([100.0, 0.0, 101.0, 102.0])
        events = [prices.index[2]]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            labels = label_events(prices, events, volatility_window=2)
        runtime = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert not runtime
        assert len(labels) == 0

    def test_nan_price_in_series_skips_event(self):
        """An event whose entry price is NaN cannot produce a valid label."""
        prices = self._make_prices([100.0, np.nan, 101.0, 102.0])
        events = [prices.index[1]]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            labels = label_events(prices, events, volatility_window=2)
        runtime = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert not runtime
        assert len(labels) == 0

    def test_valid_series_still_labels(self):
        prices = self._make_prices([100.0, 101.0, 102.0, 103.0])
        events = [prices.index[0]]
        labels = label_events(prices, events, volatility_window=2)
        assert len(labels) == 1


class TestAnalyzeBarrierDistributionNonFinite:
    """Tests for analyze_barrier_distribution with non-finite returns."""

    def test_inf_return_is_excluded_from_aggregates(self):
        labels = [
            TripleBarrierLabel(
                entry_time=pd.Timestamp("2024-01-01"),
                exit_time=pd.Timestamp("2024-01-02"),
                barrier_type=BarrierType.UPPER,
                return_pct=float("inf"),
                label=1,
                holding_periods=1,
            ),
            TripleBarrierLabel(
                entry_time=pd.Timestamp("2024-01-02"),
                exit_time=pd.Timestamp("2024-01-03"),
                barrier_type=BarrierType.UPPER,
                return_pct=0.02,
                label=1,
                holding_periods=1,
            ),
        ]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            stats = analyze_barrier_distribution(labels)
        runtime = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert not runtime
        assert stats["avg_return"] == pytest.approx(2.0)
        assert stats["median_return"] == pytest.approx(2.0)
        assert stats["total_return"] == pytest.approx(0.02)
        # The vertical win counter is not affected because both labels are upper.
        assert stats["win_rate"] == 100.0

    def test_nan_return_is_excluded_from_aggregates(self):
        labels = [
            TripleBarrierLabel(
                entry_time=pd.Timestamp("2024-01-01"),
                exit_time=pd.Timestamp("2024-01-02"),
                barrier_type=BarrierType.VERTICAL,
                return_pct=float("nan"),
                label=0,
                holding_periods=1,
            ),
            TripleBarrierLabel(
                entry_time=pd.Timestamp("2024-01-02"),
                exit_time=pd.Timestamp("2024-01-03"),
                barrier_type=BarrierType.VERTICAL,
                return_pct=0.01,
                label=0,
                holding_periods=1,
            ),
        ]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            stats = analyze_barrier_distribution(labels)
        runtime = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert not runtime
        assert stats["total_events"] == 2
        assert stats["vertical_touches"] == 2
        # Only the finite positive vertical return counts as a win.
        assert stats["win_rate"] == 50.0
        assert stats["avg_return"] == pytest.approx(1.0)
        assert stats["total_return"] == pytest.approx(0.01)

    def test_all_non_finite_returns_returns_zeroed_stats(self):
        labels = [
            TripleBarrierLabel(
                entry_time=pd.Timestamp("2024-01-01"),
                exit_time=pd.Timestamp("2024-01-02"),
                barrier_type=BarrierType.LOWER,
                return_pct=float("inf"),
                label=-1,
                holding_periods=1,
            ),
        ]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            stats = analyze_barrier_distribution(labels)
        runtime = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert not runtime
        assert stats["avg_return"] == 0.0
        assert stats["median_return"] == 0.0
        assert stats["total_return"] == 0.0


class TestLabelEventsNoRuntimeWarning:
    """End-to-end guard: a dirty stream should not raise RuntimeWarning."""

    def test_dirty_stream_no_runtime_warning(self):
        prices = pd.Series(
            [100.0, 0.0, np.nan, 101.0, 102.0, 103.0],
            index=pd.date_range("2024-01-01", periods=6, freq="D"),
        )
        events = [prices.index[3], prices.index[4]]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            labels = label_events(prices, events, volatility_window=2)
        runtime = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert not runtime
        assert len(labels) >= 0

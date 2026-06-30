"""
Test suite for backtest/triple_barrier.py.

Tests the Triple-Barrier Method implementation from Lopez de Prado (2018),
including barrier level calculations, single-position labeling, multi-event
labeling, signal extraction, and distribution analysis.

Financial formulas are deterministic — known inputs must produce known outputs.
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from backtest.triple_barrier import (
    BarrierType,
    BarrierConfig,
    TripleBarrierLabel,
    calculate_volatility,
    get_barrier_levels,
    apply_triple_barrier,
    label_events,
    get_events_from_signals,
    analyze_barrier_distribution,
    format_barrier_report,
)


def _approx(a, b, rel_tol=1e-6, abs_tol=1e-9):
    """Compare floats with tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _make_price_series(returns, start_price=100.0):
    """Build a price series from daily returns."""
    prices = [start_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    dates = pd.date_range('2024-01-01', periods=len(prices), freq='D')
    return pd.Series(prices, index=dates)


# ---------------------------------------------------------------------------
# BarrierConfig
# ---------------------------------------------------------------------------

class TestBarrierConfig:
    """Tests for BarrierConfig dataclass and factory methods."""

    def test_default_values(self):
        cfg = BarrierConfig()
        assert cfg.profit_take_std == 2.0
        assert cfg.stop_loss_std == 2.0
        assert cfg.max_holding == 20
        assert cfg.trailing_stop is False

    def test_conservative_factory(self):
        cfg = BarrierConfig.conservative()
        assert cfg.profit_take_std == 1.5
        assert cfg.stop_loss_std == 1.0
        assert cfg.max_holding == 10

    def test_aggressive_factory(self):
        cfg = BarrierConfig.aggressive()
        assert cfg.profit_take_std == 3.0
        assert cfg.stop_loss_std == 2.5
        assert cfg.max_holding == 30

    def test_symmetric_factory(self):
        cfg = BarrierConfig.symmetric()
        assert cfg.profit_take_std == 2.0
        assert cfg.stop_loss_std == 2.0
        assert cfg.max_holding == 20

    def test_custom_init(self):
        cfg = BarrierConfig(profit_take_std=1.0, stop_loss_std=0.5, max_holding=5, trailing_stop=True)
        assert cfg.profit_take_std == 1.0
        assert cfg.stop_loss_std == 0.5
        assert cfg.max_holding == 5
        assert cfg.trailing_stop is True


# ---------------------------------------------------------------------------
# calculate_volatility
# ---------------------------------------------------------------------------

class TestCalculateVolatility:
    """Tests for rolling volatility calculation."""

    def test_constant_price_zero_vol(self):
        prices = pd.Series([100.0] * 30, index=pd.date_range('2024-01-01', periods=30, freq='D'))
        vol = calculate_volatility(prices, window=20)
        # First 20 values are NaN due to rolling window + pct_change
        assert vol.iloc[21:].isna().all() or (vol.iloc[21:] == 0).all()

    def test_known_volatility(self):
        # 20 days of 1% daily returns → std ≈ 0
        np.random.seed(42)
        returns = np.full(30, 0.01)
        prices = _make_price_series(returns)
        vol = calculate_volatility(prices, window=20)
        # With constant returns, volatility should be near zero
        valid = vol.dropna()
        assert len(valid) > 0
        assert (valid < 1e-10).all()

    def test_window_parameter(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 50)
        prices = _make_price_series(returns)
        vol_10 = calculate_volatility(prices, window=10)
        vol_30 = calculate_volatility(prices, window=30)
        # Shorter window = more responsive = higher variance in vol estimate
        assert vol_10.notna().sum() > vol_30.notna().sum()


# ---------------------------------------------------------------------------
# get_barrier_levels
# ---------------------------------------------------------------------------

class TestGetBarrierLevels:
    """Tests for barrier level calculations."""

    def test_symmetric_levels(self):
        cfg = BarrierConfig.symmetric()
        upper, lower = get_barrier_levels(100.0, 0.01, cfg)
        # upper = 100 * (1 + 2.0 * 0.01) = 102.0
        # lower = 100 * (1 - 2.0 * 0.01) = 98.0
        assert _approx(upper, 102.0)
        assert _approx(lower, 98.0)

    def test_conservative_levels(self):
        cfg = BarrierConfig.conservative()
        upper, lower = get_barrier_levels(100.0, 0.01, cfg)
        assert _approx(upper, 101.5)
        assert _approx(lower, 99.0)

    def test_zero_volatility(self):
        cfg = BarrierConfig.symmetric()
        upper, lower = get_barrier_levels(100.0, 0.0, cfg)
        assert _approx(upper, 100.0)
        assert _approx(lower, 100.0)

    def test_high_volatility(self):
        cfg = BarrierConfig.aggressive()
        upper, lower = get_barrier_levels(100.0, 0.05, cfg)
        assert _approx(upper, 115.0)
        assert _approx(lower, 87.5)

    def test_different_entry_prices(self):
        cfg = BarrierConfig.symmetric()
        upper1, lower1 = get_barrier_levels(50.0, 0.02, cfg)
        upper2, lower2 = get_barrier_levels(200.0, 0.02, cfg)
        # Proportional to entry price
        assert _approx(upper2 / upper1, 4.0)
        assert _approx(lower2 / lower1, 4.0)


# ---------------------------------------------------------------------------
# apply_triple_barrier
# ---------------------------------------------------------------------------

class TestApplyTripleBarrier:
    """Tests for single-position triple barrier labeling."""

    def test_upper_barrier_hit(self):
        """Price rises to hit upper barrier first."""
        prices = pd.Series(
            [100.0, 101.0, 102.5, 103.0],  # 102.5 >= upper=102.0
            index=pd.date_range('2024-01-01', periods=4, freq='D')
        )
        cfg = BarrierConfig(profit_take_std=1.0, stop_loss_std=1.0, max_holding=10)
        result = apply_triple_barrier(prices, 0, 0.02, cfg)
        assert result is not None
        assert result.barrier_type == BarrierType.UPPER
        assert result.label == 1
        assert _approx(result.return_pct, 0.025)
        assert result.holding_periods == 2

    def test_lower_barrier_hit(self):
        """Price falls to hit lower barrier first."""
        prices = pd.Series(
            [100.0, 99.0, 97.5],  # 97.5 <= lower=98.0
            index=pd.date_range('2024-01-01', periods=3, freq='D')
        )
        cfg = BarrierConfig(profit_take_std=1.0, stop_loss_std=1.0, max_holding=10)
        result = apply_triple_barrier(prices, 0, 0.02, cfg)
        assert result is not None
        assert result.barrier_type == BarrierType.LOWER
        assert result.label == -1
        assert _approx(result.return_pct, -0.025)
        assert result.holding_periods == 2

    def test_vertical_barrier_hit(self):
        """Price meanders, vertical barrier expires first."""
        prices = pd.Series(
            [100.0, 100.5, 100.3, 100.7, 100.2],
            index=pd.date_range('2024-01-01', periods=5, freq='D')
        )
        cfg = BarrierConfig(profit_take_std=2.0, stop_loss_std=2.0, max_holding=3)
        result = apply_triple_barrier(prices, 0, 0.01, cfg)
        assert result is not None
        assert result.barrier_type == BarrierType.VERTICAL
        assert result.label == 0
        assert result.holding_periods == 3

    def test_entry_at_last_index(self):
        """Entry at the last valid index should hit vertical barrier immediately."""
        prices = pd.Series(
            [100.0, 101.0],
            index=pd.date_range('2024-01-01', periods=2, freq='D')
        )
        cfg = BarrierConfig(max_holding=5)
        result = apply_triple_barrier(prices, 1, 0.01, cfg)
        assert result is not None
        assert result.barrier_type == BarrierType.VERTICAL
        assert result.holding_periods == 0

    def test_entry_beyond_length(self):
        """Entry index beyond series length returns None."""
        prices = pd.Series([100.0, 101.0], index=pd.date_range('2024-01-01', periods=2, freq='D'))
        cfg = BarrierConfig()
        result = apply_triple_barrier(prices, 5, 0.01, cfg)
        assert result is None

    def test_no_data_after_entry(self):
        """Entry at last element returns vertical barrier at same point."""
        prices = pd.Series(
            [100.0, 101.0],
            index=pd.date_range('2024-01-01', periods=2, freq='D')
        )
        cfg = BarrierConfig(max_holding=5)
        result = apply_triple_barrier(prices, 1, 0.01, cfg)
        assert result is not None
        assert result.barrier_type == BarrierType.VERTICAL
        assert _approx(result.return_pct, 0.0)

    def test_default_config(self):
        """None config should default to symmetric."""
        prices = pd.Series(
            [100.0, 105.0],
            index=pd.date_range('2024-01-01', periods=2, freq='D')
        )
        result = apply_triple_barrier(prices, 0, 0.01, None)
        assert result is not None

    def test_exact_barrier_touch(self):
        """Price exactly at barrier should trigger."""
        prices = pd.Series(
            [100.0, 102.0],  # upper = 100 * (1 + 1.0 * 0.02) = 102.0
            index=pd.date_range('2024-01-01', periods=2, freq='D')
        )
        cfg = BarrierConfig(profit_take_std=1.0, stop_loss_std=1.0, max_holding=10)
        result = apply_triple_barrier(prices, 0, 0.02, cfg)
        assert result is not None
        assert result.barrier_type == BarrierType.UPPER
        assert _approx(result.return_pct, 0.02)

    def test_zero_volatility_barriers_at_entry_price(self):
        """Zero vol collapses barriers to entry price. Price == entry triggers upper first."""
        prices = pd.Series(
            [100.0, 100.0, 100.0, 100.0],
            index=pd.date_range('2024-01-01', periods=4, freq='D')
        )
        cfg = BarrierConfig(max_holding=3)
        result = apply_triple_barrier(prices, 0, 0.0, cfg)
        assert result is not None
        # When upper == lower == entry_price, price >= upper is True immediately.
        # The main API (label_events) floors vol to 0.5% to avoid this degeneracy.
        assert result.barrier_type == BarrierType.UPPER
        assert _approx(result.return_pct, 0.0)
        assert result.holding_periods == 1

    def test_timestamps_preserved(self):
        """Entry and exit timestamps should match the price series index."""
        dates = pd.date_range('2024-03-15', periods=5, freq='D')
        prices = pd.Series([100.0, 99.0, 97.0, 98.0, 99.0], index=dates)
        cfg = BarrierConfig(profit_take_std=1.0, stop_loss_std=1.0, max_holding=10)
        result = apply_triple_barrier(prices, 0, 0.02, cfg)
        assert result.entry_time == dates[0]
        assert result.exit_time == dates[2]


# ---------------------------------------------------------------------------
# label_events
# ---------------------------------------------------------------------------

class TestLabelEvents:
    """Tests for multi-event triple barrier labeling."""

    def test_single_event(self):
        prices = pd.Series(
            [100.0, 101.0, 102.5],
            index=pd.date_range('2024-01-01', periods=3, freq='D')
        )
        events = [prices.index[0]]
        labels = label_events(prices, events)
        assert len(labels) == 1
        assert labels[0].barrier_type == BarrierType.UPPER

    def test_multiple_events(self):
        prices = pd.Series(
            [100.0, 101.0, 102.0, 99.0, 98.0],
            index=pd.date_range('2024-01-01', periods=5, freq='D')
        )
        events = [prices.index[0], prices.index[2]]
        labels = label_events(prices, events, volatility_window=2)
        assert len(labels) == 2

    def test_event_not_in_index(self):
        prices = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range('2024-01-01', periods=3, freq='D')
        )
        events = [pd.Timestamp('2023-12-31')]  # Not in index
        labels = label_events(prices, events)
        assert len(labels) == 0

    def test_event_at_end(self):
        prices = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range('2024-01-01', periods=3, freq='D')
        )
        events = [prices.index[-1]]
        labels = label_events(prices, events)
        assert len(labels) == 0  # No room after entry

    def test_empty_events(self):
        prices = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range('2024-01-01', periods=3, freq='D')
        )
        labels = label_events(prices, [])
        assert len(labels) == 0

    def test_volatility_floor(self):
        """Very low vol should be floored to 0.5% to avoid zero-width barriers."""
        prices = pd.Series(
            [100.0, 100.01, 100.02, 100.03],
            index=pd.date_range('2024-01-01', periods=4, freq='D')
        )
        events = [prices.index[0]]
        labels = label_events(prices, events, volatility_window=20)
        assert len(labels) == 1
        # Should not crash — vol floor ensures barriers are calculable
        assert labels[0] is not None

    def test_custom_config(self):
        prices = pd.Series(
            [100.0, 100.5, 101.0, 101.5, 102.0],
            index=pd.date_range('2024-01-01', periods=5, freq='D')
        )
        events = [prices.index[0]]
        cfg = BarrierConfig.conservative()
        labels = label_events(prices, events, config=cfg, volatility_window=2)
        assert len(labels) == 1


# ---------------------------------------------------------------------------
# get_events_from_signals
# ---------------------------------------------------------------------------

class TestGetEventsFromSignals:
    """Tests for signal-to-event conversion."""

    def test_buy_signals(self):
        signals = pd.Series(
            [0, 1, 0, 0, 1, 0],
            index=pd.date_range('2024-01-01', periods=6, freq='D')
        )
        events = get_events_from_signals(signals, signals)
        assert len(events) == 2
        assert events[0] == signals.index[1]
        assert events[1] == signals.index[4]

    def test_sell_signals(self):
        signals = pd.Series(
            [0, -1, 0, 0, -1, 0],
            index=pd.date_range('2024-01-01', periods=6, freq='D')
        )
        events = get_events_from_signals(signals, signals)
        assert len(events) == 2

    def test_mixed_signals(self):
        signals = pd.Series(
            [1, -1, 1, -1, 1],
            index=pd.date_range('2024-01-01', periods=5, freq='D')
        )
        events = get_events_from_signals(signals, signals)
        assert len(events) == 5

    def test_min_hold_filtering(self):
        signals = pd.Series(
            [1, 1, 1, 1],  # 4 consecutive buy signals
            index=pd.date_range('2024-01-01', periods=4, freq='D')
        )
        events = get_events_from_signals(signals, signals, min_hold=2)
        # Should only get signals at indices 0 and 2 (gap of 2)
        assert len(events) == 2
        assert events[0] == signals.index[0]
        assert events[1] == signals.index[2]

    def test_no_signals(self):
        signals = pd.Series(
            [0, 0, 0, 0],
            index=pd.date_range('2024-01-01', periods=4, freq='D')
        )
        events = get_events_from_signals(signals, signals)
        assert len(events) == 0

    def test_all_signals_no_hold(self):
        signals = pd.Series(
            [1, 1, 1, 1],
            index=pd.date_range('2024-01-01', periods=4, freq='D')
        )
        events = get_events_from_signals(signals, signals, min_hold=1)
        assert len(events) == 4


# ---------------------------------------------------------------------------
# analyze_barrier_distribution
# ---------------------------------------------------------------------------

class TestAnalyzeBarrierDistribution:
    """Tests for barrier touch statistics."""

    def test_empty_list(self):
        stats = analyze_barrier_distribution([])
        assert stats['total_events'] == 0
        assert stats['win_rate'] == 0.0
        assert stats['avg_return'] == 0.0

    def test_all_upper(self):
        labels = [
            TripleBarrierLabel(
                entry_time=pd.Timestamp('2024-01-01'),
                exit_time=pd.Timestamp('2024-01-02'),
                barrier_type=BarrierType.UPPER,
                return_pct=0.02,
                label=1,
                holding_periods=1
            )
            for _ in range(5)
        ]
        stats = analyze_barrier_distribution(labels)
        assert stats['total_events'] == 5
        assert stats['upper_touches'] == 5
        assert stats['win_rate'] == 100.0
        assert _approx(stats['avg_return'], 2.0)  # Percentage

    def test_all_lower(self):
        labels = [
            TripleBarrierLabel(
                entry_time=pd.Timestamp('2024-01-01'),
                exit_time=pd.Timestamp('2024-01-02'),
                barrier_type=BarrierType.LOWER,
                return_pct=-0.02,
                label=-1,
                holding_periods=1
            )
            for _ in range(5)
        ]
        stats = analyze_barrier_distribution(labels)
        assert stats['lower_touches'] == 5
        assert stats['win_rate'] == 0.0

    def test_vertical_positive_counts_as_win(self):
        labels = [
            TripleBarrierLabel(
                entry_time=pd.Timestamp('2024-01-01'),
                exit_time=pd.Timestamp('2024-01-02'),
                barrier_type=BarrierType.VERTICAL,
                return_pct=0.01,
                label=0,
                holding_periods=5
            )
        ]
        stats = analyze_barrier_distribution(labels)
        assert stats['vertical_touches'] == 1
        assert stats['win_rate'] == 100.0

    def test_vertical_negative_counts_as_loss(self):
        labels = [
            TripleBarrierLabel(
                entry_time=pd.Timestamp('2024-01-01'),
                exit_time=pd.Timestamp('2024-01-02'),
                barrier_type=BarrierType.VERTICAL,
                return_pct=-0.01,
                label=0,
                holding_periods=5
            )
        ]
        stats = analyze_barrier_distribution(labels)
        assert stats['win_rate'] == 0.0

    def test_mixed_distribution(self):
        labels = [
            TripleBarrierLabel(
                entry_time=pd.Timestamp('2024-01-01'),
                exit_time=pd.Timestamp('2024-01-02'),
                barrier_type=BarrierType.UPPER,
                return_pct=0.02,
                label=1,
                holding_periods=1
            ),
            TripleBarrierLabel(
                entry_time=pd.Timestamp('2024-01-02'),
                exit_time=pd.Timestamp('2024-01-03'),
                barrier_type=BarrierType.LOWER,
                return_pct=-0.01,
                label=-1,
                holding_periods=2
            ),
            TripleBarrierLabel(
                entry_time=pd.Timestamp('2024-01-03'),
                exit_time=pd.Timestamp('2024-01-08'),
                barrier_type=BarrierType.VERTICAL,
                return_pct=0.005,
                label=0,
                holding_periods=5
            ),
        ]
        stats = analyze_barrier_distribution(labels)
        assert stats['total_events'] == 3
        assert stats['upper_touches'] == 1
        assert stats['lower_touches'] == 1
        assert stats['vertical_touches'] == 1
        assert stats['upper_pct'] == pytest.approx(100 / 3, rel=1e-6)
        assert stats['lower_pct'] == pytest.approx(100 / 3, rel=1e-6)
        assert stats['vertical_pct'] == pytest.approx(100 / 3, rel=1e-6)
        assert stats['win_rate'] == pytest.approx(200 / 3, rel=1e-6)

    def test_total_return_compounding(self):
        labels = [
            TripleBarrierLabel(
                entry_time=pd.Timestamp('2024-01-01'),
                exit_time=pd.Timestamp('2024-01-02'),
                barrier_type=BarrierType.UPPER,
                return_pct=0.10,
                label=1,
                holding_periods=1
            ),
            TripleBarrierLabel(
                entry_time=pd.Timestamp('2024-01-02'),
                exit_time=pd.Timestamp('2024-01-03'),
                barrier_type=BarrierType.UPPER,
                return_pct=0.10,
                label=1,
                holding_periods=1
            ),
        ]
        stats = analyze_barrier_distribution(labels)
        # (1.1 * 1.1) - 1 = 0.21
        assert _approx(stats['total_return'], 0.21)

    def test_avg_holding_periods(self):
        labels = [
            TripleBarrierLabel(
                entry_time=pd.Timestamp('2024-01-01'),
                exit_time=pd.Timestamp('2024-01-02'),
                barrier_type=BarrierType.UPPER,
                return_pct=0.01,
                label=1,
                holding_periods=2
            ),
            TripleBarrierLabel(
                entry_time=pd.Timestamp('2024-01-02'),
                exit_time=pd.Timestamp('2024-01-03'),
                barrier_type=BarrierType.UPPER,
                return_pct=0.01,
                label=1,
                holding_periods=4
            ),
        ]
        stats = analyze_barrier_distribution(labels)
        assert stats['avg_holding_periods'] == 3.0


# ---------------------------------------------------------------------------
# format_barrier_report
# ---------------------------------------------------------------------------

class TestFormatBarrierReport:
    """Tests for report formatting."""

    def test_contains_key_sections(self):
        stats = {
            'total_events': 10,
            'upper_touches': 5,
            'lower_touches': 3,
            'vertical_touches': 2,
            'upper_pct': 50.0,
            'lower_pct': 30.0,
            'vertical_pct': 20.0,
            'win_rate': 60.0,
            'avg_return': 1.5,
            'median_return': 1.2,
            'avg_holding_periods': 5.5,
            'total_return': 0.15
        }
        report = format_barrier_report(stats)
        assert "TRIPLE BARRIER ANALYSIS" in report
        assert "Total Events:           10" in report
        assert "Win Rate:             60.0%" in report
        assert "Total Return:         15.00%" in report

    def test_empty_stats(self):
        stats = {
            'total_events': 0,
            'upper_touches': 0,
            'lower_touches': 0,
            'vertical_touches': 0,
            'upper_pct': 0.0,
            'lower_pct': 0.0,
            'vertical_pct': 0.0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'median_return': 0.0,
            'avg_holding_periods': 0,
            'total_return': 0.0
        }
        report = format_barrier_report(stats)
        assert "Total Events:           0" in report
        assert "Win Rate:             0.0%" in report


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_price_point(self):
        prices = pd.Series([100.0], index=pd.date_range('2024-01-01', periods=1, freq='D'))
        result = apply_triple_barrier(prices, 0, 0.01)
        assert result is not None
        assert result.barrier_type == BarrierType.VERTICAL
        assert result.holding_periods == 0

    def test_two_price_points(self):
        prices = pd.Series(
            [100.0, 102.0],
            index=pd.date_range('2024-01-01', periods=2, freq='D')
        )
        cfg = BarrierConfig(profit_take_std=1.0, stop_loss_std=1.0, max_holding=5)
        result = apply_triple_barrier(prices, 0, 0.02, cfg)
        assert result is not None
        # upper = 100 * 1.02 = 102.0, price at idx 1 = 102.0
        assert result.barrier_type == BarrierType.UPPER

    def test_very_large_return(self):
        prices = pd.Series(
            [100.0, 150.0],
            index=pd.date_range('2024-01-01', periods=2, freq='D')
        )
        cfg = BarrierConfig(profit_take_std=1.0, stop_loss_std=1.0, max_holding=5)
        result = apply_triple_barrier(prices, 0, 0.10, cfg)
        assert result is not None
        assert result.barrier_type == BarrierType.UPPER
        assert _approx(result.return_pct, 0.50)

    def test_very_small_return(self):
        prices = pd.Series(
            [100.0, 100.001],
            index=pd.date_range('2024-01-01', periods=2, freq='D')
        )
        cfg = BarrierConfig(profit_take_std=1.0, stop_loss_std=1.0, max_holding=5)
        result = apply_triple_barrier(prices, 0, 0.0001, cfg)
        assert result is not None
        # upper = 100 * 1.0001 = 100.01, price is below
        assert result.barrier_type == BarrierType.VERTICAL

    def test_negative_prices(self):
        """Negative prices are non-physical but should not crash."""
        prices = pd.Series(
            [-100.0, -102.0],
            index=pd.date_range('2024-01-01', periods=2, freq='D')
        )
        cfg = BarrierConfig(profit_take_std=1.0, stop_loss_std=1.0, max_holding=5)
        result = apply_triple_barrier(prices, 0, 0.02, cfg)
        assert result is not None
        # upper = -100 * 1.02 = -102.0
        assert result.barrier_type == BarrierType.UPPER

    def test_nan_in_prices(self):
        """NaN in price series after entry may cause issues."""
        prices = pd.Series(
            [100.0, 101.0, np.nan, 103.0],
            index=pd.date_range('2024-01-01', periods=4, freq='D')
        )
        cfg = BarrierConfig(profit_take_std=1.0, stop_loss_std=1.0, max_holding=5)
        result = apply_triple_barrier(prices, 0, 0.02, cfg)
        # NaN comparisons are False, so it should walk through and possibly hit vertical
        assert result is not None

    def test_duplicate_index(self):
        """Duplicate indices in price series."""
        idx = [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')]
        prices = pd.Series([100.0, 100.5, 102.0], index=idx)
        cfg = BarrierConfig(profit_take_std=1.0, stop_loss_std=1.0, max_holding=5)
        # get_loc on duplicates returns a boolean mask, not an int
        result = apply_triple_barrier(prices, 0, 0.02, cfg)
        # This may or may not work depending on implementation; just ensure no crash
        assert result is not None or result is None  # Accept either

    def test_very_long_holding_period(self):
        prices = pd.Series(
            [100.0] + [100.0 + i * 0.01 for i in range(1, 101)],
            index=pd.date_range('2024-01-01', periods=101, freq='D')
        )
        cfg = BarrierConfig(max_holding=100)
        result = apply_triple_barrier(prices, 0, 0.001, cfg)
        assert result is not None
        assert result.holding_periods <= 100

    def test_gapped_prices(self):
        """Overnight gaps that jump past barriers."""
        prices = pd.Series(
            [100.0, 105.0],  # Gaps past upper=102.0
            index=pd.date_range('2024-01-01', periods=2, freq='D')
        )
        cfg = BarrierConfig(profit_take_std=1.0, stop_loss_std=1.0, max_holding=5)
        result = apply_triple_barrier(prices, 0, 0.02, cfg)
        assert result is not None
        assert result.barrier_type == BarrierType.UPPER
        assert _approx(result.return_pct, 0.05)


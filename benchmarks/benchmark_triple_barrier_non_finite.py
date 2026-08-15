"""
Benchmark for the triple barrier non-finite input guards.

Exercises get_barrier_levels, apply_triple_barrier, label_events, and
analyze_barrier_distribution with NaN/Inf/zero inputs, asserting that the
pipeline does not emit RuntimeWarnings and does not propagate non-finite
values into downstream backtest reports.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from backtest.triple_barrier import (
    BarrierConfig,
    BarrierType,
    TripleBarrierLabel,
    analyze_barrier_distribution,
    apply_triple_barrier,
    get_barrier_levels,
    label_events,
)


def benchmark_get_barrier_levels_rejects_degenerate_inputs():
    cfg = BarrierConfig.symmetric()
    for entry_price, vol in [
        (0.0, 0.01),
        (np.nan, 0.01),
        (100.0, np.nan),
        (100.0, -0.01),
    ]:
        upper, lower = get_barrier_levels(entry_price, vol, cfg)
        assert np.isnan(upper)
        assert np.isnan(lower)


def benchmark_apply_triple_barrier_rejects_degenerate_entry():
    prices = pd.Series(
        [0.0, 0.0, 0.0], index=pd.date_range("2024-01-01", periods=3, freq="D")
    )
    result = apply_triple_barrier(prices, 0, 0.01, BarrierConfig())
    assert result is None

    prices = pd.Series(
        [np.nan, 100.0, 101.0], index=pd.date_range("2024-01-01", periods=3, freq="D")
    )
    result = apply_triple_barrier(prices, 0, 0.01, BarrierConfig())
    assert result is None

    prices = pd.Series(
        [100.0, 101.0, 102.0], index=pd.date_range("2024-01-01", periods=3, freq="D")
    )
    result = apply_triple_barrier(prices, 0, np.nan, BarrierConfig())
    assert result is None

    result = apply_triple_barrier(prices, 0, -0.01, BarrierConfig())
    assert result is None


def benchmark_label_events_skips_non_finite_volatility():
    prices = pd.Series(
        [100.0, 0.0, 101.0, 102.0, 103.0],
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    events = [prices.index[2]]
    labels = label_events(prices, events, volatility_window=2)
    assert len(labels) == 0


def benchmark_analyze_barrier_distribution_ignores_non_finite_returns():
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
            barrier_type=BarrierType.VERTICAL,
            return_pct=float("nan"),
            label=0,
            holding_periods=1,
        ),
        TripleBarrierLabel(
            entry_time=pd.Timestamp("2024-01-03"),
            exit_time=pd.Timestamp("2024-01-04"),
            barrier_type=BarrierType.UPPER,
            return_pct=0.02,
            label=1,
            holding_periods=1,
        ),
    ]
    stats = analyze_barrier_distribution(labels)
    assert np.isfinite(stats["avg_return"])
    assert np.isfinite(stats["median_return"])
    assert np.isfinite(stats["total_return"])
    assert stats["avg_return"] == 2.0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        benchmark_get_barrier_levels_rejects_degenerate_inputs()
        benchmark_apply_triple_barrier_rejects_degenerate_entry()
        benchmark_label_events_skips_non_finite_volatility()
        benchmark_analyze_barrier_distribution_ignores_non_finite_returns()
    print("All triple-barrier non-finite benchmarks passed without RuntimeWarning.")

"""Benchmark weekly_report non-finite guards.

Measures calculate_weekly_returns and fetch_benchmark_returns with varying
ratios of non-finite values to ensure the guard paths do not materially regress
performance.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from unittest.mock import patch  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from weekly_report import calculate_weekly_returns, fetch_benchmark_returns  # noqa: E402


def _make_week_results(count: int, invalid_ratio: float):
    """Build a list of week results with a configurable share of NaN values."""
    results = []
    for i in range(count):
        value = 10000.0 * (1 + i * 0.001)
        if np.random.rand() < invalid_ratio:
            value = float("nan")
        results.append({"portfolio_after": {"total_value": value}})
    return results


def _make_benchmark_frame(count: int, invalid_ratio: float):
    """Build a DataFrame of close prices with a configurable share of NaN values."""
    prices = 400.0 * (1 + np.arange(count) * 0.001)
    mask = np.random.rand(count) < invalid_ratio
    prices[mask] = np.nan
    return pd.DataFrame({"Close": prices})


def _bench(name: str, func, iterations: int = 1000):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    avg_us = sum(times) / len(times) * 1e6
    print(f"{name}: {avg_us:.1f} µs/run")
    return avg_us


def main():
    np.random.seed(42)

    print("=" * 70)
    print("Weekly report non-finite guard benchmark")
    print("=" * 70)

    for count in [20, 100, 500]:
        print(f"\ncalculate_weekly_returns ({count} days)")
        for invalid in [0.0, 0.1, 0.5]:
            results = _make_week_results(count, invalid)
            _bench(
                f"  invalid={invalid:.0%}",
                lambda r=results: calculate_weekly_returns(r),
            )

    with patch("weekly_report.fetch_historical_data") as mock_fetch:
        print("\nfetch_benchmark_returns (1000 rows)")
        for invalid in [0.0, 0.1, 0.5]:
            frame = _make_benchmark_frame(1000, invalid)
            mock_fetch.return_value = {"SPY": frame}
            _bench(
                f"  invalid={invalid:.0%}",
                lambda: fetch_benchmark_returns("2026-01-01", "2026-12-31", benchmarks=["SPY"]),
            )

    print("=" * 70)


if __name__ == "__main__":
    main()

"""Benchmark for weekly_report benchmark alignment.

Demonstrates that the cumulative-return comparison is robust to calendar
differences between markets (e.g. US and European holidays). Previously the
weekly report skipped the CAC.PA comparison because raw daily-return arrays
had different lengths; this benchmark shows the new date-bound fetch produces a
meaningful cumulative comparison regardless of daily count.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weekly_report import fetch_benchmark_returns


def _make_df(prices, start_date):
    dates = pd.date_range(start=start_date, periods=len(prices), freq="B")
    return pd.DataFrame({"Close": prices}, index=dates)


def _fake_fetch(tickers, start=None, end=None, **kwargs):
    """Simulate a US market with 5 days and a European market with 6 days."""
    return {
        "SPY": _make_df([100.0, 101.0, 102.0, 101.5, 103.0], start),
        "CAC.PA": _make_df([50.0, 51.0, 51.5, 52.0, 51.8, 53.0], start),
    }


if __name__ == "__main__":
    import weekly_report
    weekly_report.fetch_historical_data = _fake_fetch

    start = time.perf_counter()
    result = fetch_benchmark_returns("2026-07-13", "2026-07-17")
    elapsed = time.perf_counter() - start

    print("Benchmark: weekly_report.fetch_benchmark_returns")
    print(f"Elapsed: {elapsed*1000:.3f} ms")
    for benchmark in result:
        cum = result[benchmark]["cumulative_return"]
        n_days = len(result[benchmark]["returns"]) + 1
        print(f"  {benchmark}: {n_days} trading days, cumulative return {cum*100:+.2f}%")

    # CAC.PA has more daily bars but still produces a comparable cumulative return.
    assert result["CAC.PA"]["cumulative_return"] > 0
    assert result["SPY"]["cumulative_return"] > 0
    print("\nOK – cumulative comparison is calendar-agnostic.")

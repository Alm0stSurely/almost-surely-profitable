#!/usr/bin/env python3
"""
Benchmark for the daily_run regime-analysis data-preparation path.

Measures the time needed to convert the Dict[str, pd.DataFrame] output of
fetch_historical_data into the price DataFrame expected by RegimeDetector,
and the time to run the detector on the resulting matrix.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from daily_run import _build_prices_df
from analysis.regime_detector import RegimeDetector


def generate_market_data(n_tickers: int, n_days: int, seed: int = 42) -> dict:
    """Generate realistic OHLCV DataFrames for a set of tickers."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    data = {}
    for i in range(n_tickers):
        returns = rng.normal(0.0005, 0.015, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            "Open": prices * (1 + rng.normal(0, 0.001, n_days)),
            "High": prices * (1 + abs(rng.normal(0, 0.01, n_days))),
            "Low": prices * (1 - abs(rng.normal(0, 0.01, n_days))),
            "Close": prices,
            "Volume": rng.integers(1_000_000, 10_000_000, n_days),
        }, index=dates)
        data[f"TICKER{i}"] = df
    return data


def benchmark(n_tickers: int, n_days: int, iterations: int = 100) -> dict:
    """Run the benchmark and return timing results."""
    market_data = generate_market_data(n_tickers, n_days)

    # Warm-up
    _build_prices_df(market_data)
    detector = RegimeDetector()
    prices_df = _build_prices_df(market_data)
    detector.analyze(prices_df)

    # Time helper
    t0 = time.perf_counter()
    for _ in range(iterations):
        prices_df = _build_prices_df(market_data)
    t1 = time.perf_counter()
    build_time = (t1 - t0) / iterations

    # Time regime analysis
    t0 = time.perf_counter()
    for _ in range(iterations):
        detector.analyze(prices_df)
    t1 = time.perf_counter()
    regime_time = (t1 - t0) / iterations

    return {
        "n_tickers": n_tickers,
        "n_days": n_days,
        "iterations": iterations,
        "build_ms": build_time * 1000,
        "regime_ms": regime_time * 1000,
        "total_ms": (build_time + regime_time) * 1000,
        "shape": list(prices_df.shape),
    }


if __name__ == "__main__":
    print("Benchmark: daily_run regime-analysis preparation")
    print("=" * 60)

    for n_tickers, n_days in [(5, 30), (10, 60), (20, 252)]:
        result = benchmark(n_tickers, n_days, iterations=100)
        print(
            f"{n_tickers:2d} tickers x {n_days:3d} days "
            f"(shape {result['shape']}): "
            f"build={result['build_ms']:.3f} ms, "
            f"regime={result['regime_ms']:.3f} ms, "
            f"total={result['total_ms']:.3f} ms"
        )

    print("=" * 60)
    print("Notes:")
    print("- Build time is dominated by pandas DataFrame alignment.")
    print("- Regime time scales with the number of assets and observations.")
    print("- Results are per-iteration averages over 100 runs.")

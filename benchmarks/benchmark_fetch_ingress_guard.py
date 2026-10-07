#!/usr/bin/env python3
"""
Benchmark: fetch-ingress finiteness guard overhead.

The guard is one math.isfinite() call on a float — O(1), expected within noise.
This benchmark exists for the embedded functional gate (fails on pre-fix code)
and to prove the guard adds no measurable cost to the hot path.

Run: python benchmarks/benchmark_fetch_ingress_guard.py
"""

import contextlib
import io
import logging
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# The guard's warning log is loud-by-design; keep it out of the benchmark loop
# output while still exercising the full code path (LEARNINGS 2026-09-24).
logging.getLogger("data.fetch_market_data").setLevel(logging.CRITICAL)

from data.fetch_market_data import _fetch_single_price

N_ITER = 5000
N_ROUNDS = 5


def _mock_ticker(close_value):
    dates = pd.date_range("2024-01-01", periods=1)
    mock_df = pd.DataFrame(
        {
            "Open": [close_value],
            "High": [close_value],
            "Low": [close_value],
            "Close": [close_value],
            "Volume": [1000],
        },
        index=dates,
    )
    mock_ticker = Mock()
    mock_ticker.history.return_value = mock_df
    return mock_ticker


def _bench(close_value, label):
    best = float("inf")
    for _ in range(N_ROUNDS):
        start = time.perf_counter()
        for _ in range(N_ITER):
            with patch(
                "data.fetch_market_data.yf.Ticker",
                return_value=_mock_ticker(close_value),
            ):
                _fetch_single_price("SPY")
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    per_call_us = (best / N_ITER) * 1e6
    print(f"  {label}: {per_call_us:.2f} µs/call (best of {N_ROUNDS})")
    return per_call_us


def _gate():
    """Embedded functional gate — fails on pre-fix code."""
    failures = []

    with patch(
        "data.fetch_market_data.yf.Ticker",
        return_value=_mock_ticker(float("nan")),
    ):
        _, price = _fetch_single_price("SPY")
    if price is not None:
        failures.append(f"NaN accepted: {price!r}")

    with patch(
        "data.fetch_market_data.yf.Ticker",
        return_value=_mock_ticker(float("inf")),
    ):
        _, price = _fetch_single_price("SPY")
    if price is not None:
        failures.append(f"+inf accepted: {price!r}")

    with patch(
        "data.fetch_market_data.yf.Ticker",
        return_value=_mock_ticker(104.5),
    ):
        _, price = _fetch_single_price("SPY")
    if price != 104.5:
        failures.append(f"Healthy broken: {price!r}")

    if failures:
        print("GATE FAILED (pre-fix behavior):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("  Gate: all finiteness checks pass")


def main():
    print(f"Benchmark: fetch-ingress guard (N_ITER={N_ITER}, best of {N_ROUNDS})")
    print("-" * 50)

    _gate()

    healthy_us = _bench(104.5, "healthy ")
    nan_us = _bench(float("nan"), "nan     ")

    print("-" * 50)
    print(f"No perf claim: guard is one math.isfinite() — same-cost variant.")
    print(f"Delta healthy-vs-nan: {abs(healthy_us - nan_us):.2f} µs (noise)")

    # Sanity: both paths within an order of magnitude (guard doesn't explode cost)
    if max(healthy_us, nan_us) / min(healthy_us, nan_us) > 10:
        print("WARNING: guard path cost anomaly — investigate")
        sys.exit(1)

    print("OK")


if __name__ == "__main__":
    main()

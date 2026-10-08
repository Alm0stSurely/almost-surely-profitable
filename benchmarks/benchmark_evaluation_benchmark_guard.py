#!/usr/bin/env python3
"""
Benchmark: evaluation benchmark-return guard overhead.

The guard is two _is_finite_number calls plus one comparison on scalars —
O(1), expected within noise against the dominant cost (the fetch). This
benchmark exists for the embedded functional gate (fails on pre-fix code)
and to prove the guard adds no measurable cost to the healthy path.

Run: python benchmarks/benchmark_evaluation_benchmark_guard.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import evaluation
from evaluation import _get_benchmark_return

N_ITER = 20000
N_ROUNDS = 5


def _frame(closes):
    dates = pd.date_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"Close": closes}, index=dates)


def _bench(closes, label):
    """Time _get_benchmark_return with the fetch stubbed at module level.

    Direct attribute substitution (not unittest.mock.patch) so the measured
    cost is the function itself, not the patching machinery.
    """
    real_fetch = evaluation.fetch_historical_data
    frame = _frame(closes)
    evaluation.fetch_historical_data = lambda *a, **k: {"SPY": frame}
    try:
        best = float("inf")
        for _ in range(N_ROUNDS):
            start = time.perf_counter()
            for _ in range(N_ITER):
                _get_benchmark_return("2024-01-01", "2024-01-05")
            best = min(best, (time.perf_counter() - start) / N_ITER)
    finally:
        evaluation.fetch_historical_data = real_fetch
    print(f"{label:<28} {best * 1e6:8.2f} µs/call")
    return best


def _guarded_result(closes):
    real_fetch = evaluation.fetch_historical_data
    frame = _frame(closes)
    evaluation.fetch_historical_data = lambda *a, **k: {"SPY": frame}
    try:
        return _get_benchmark_return("2024-01-01", "2024-01-05")
    finally:
        evaluation.fetch_historical_data = real_fetch


def main():
    # Embedded functional gate — fails on pre-fix code.
    failures = []
    for name, closes in [
        ("NaN first close", [float("nan"), 110.0]),
        ("NaN last close", [100.0, float("nan")]),
        ("+inf last close", [100.0, float("inf")]),
        ("-inf first close", [float("-inf"), 110.0]),
    ]:
        result = _guarded_result(closes)
        if result is not None:
            failures.append(f"{name}: expected None, got {result!r}")
    if failures:
        print("GATE FAILED — pre-fix behavior detected:")
        for f in failures:
            print(f"  FAIL: {f}")
        sys.exit(1)
    print("Gate: non-finite/invalid windows rejected (4/4)")
    print()

    healthy = _bench([100.0, 110.0], "healthy window (guarded)")
    print()
    print("Guard overhead is two scalar isfinite checks on a path dominated by")
    print("the data fetch — no perf claim, gate is the embedded functional proof.")
    print(f"Healthy-path absolute cost: {healthy * 1e6:.2f} µs/call (mocked fetch).")

if __name__ == "__main__":
    main()

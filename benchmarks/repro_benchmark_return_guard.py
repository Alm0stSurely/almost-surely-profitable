#!/usr/bin/env python3
"""
Pre-fix reproduction: a NaN/±inf Close at either edge of the fetched window
flows through evaluation._get_benchmark_return as a "valid" benchmark return,
violating the Optional[float] contract — NaN is neither None (unavailable)
nor a usable scalar. The twin producer in reporting.py already guards this
(_is_finite_number + start_price <= 0); this site was missed.

Run: python benchmarks/repro_benchmark_return_guard.py
Exit 0 = fix live (non-finite/invalid window rejected, None returned).
Exit 1 = pre-fix behavior (non-finite ratio flows through).
"""

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import _get_benchmark_return


def _frame(closes):
    dates = pd.date_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"Close": closes}, index=dates)


def main():
    failures = []

    cases = [
        ("NaN first close", [float("nan"), 110.0]),
        ("NaN last close", [100.0, float("nan")]),
        ("+inf last close", [100.0, float("inf")]),
        ("-inf first close", [float("-inf"), 110.0]),
    ]
    for name, closes in cases:
        with patch("evaluation.fetch_historical_data") as mock_fetch:
            mock_fetch.return_value = {"SPY": _frame(closes)}
            result = _get_benchmark_return("2024-01-01", "2024-01-05")
        if result is not None:
            failures.append(f"{name}: expected None, got {result!r}")

    # Healthy window must survive unchanged
    with patch("evaluation.fetch_historical_data") as mock_fetch:
        mock_fetch.return_value = {"SPY": _frame([100.0, 110.0])}
        result = _get_benchmark_return("2024-01-01", "2024-01-05")
    if result is None or not math.isclose(result, 0.1, rel_tol=1e-12):
        failures.append(f"healthy window broken: expected ~0.1, got {result!r}")

    if failures:
        print("REPRODUCTION — pre-fix behavior detected:")
        for f in failures:
            print(f"  FAIL: {f}")
        sys.exit(1)

    print("FIX LIVE — non-finite/invalid benchmark windows rejected, healthy window unchanged")
    sys.exit(0)


if __name__ == "__main__":
    main()

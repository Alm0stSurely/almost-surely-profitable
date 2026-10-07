#!/usr/bin/env python3
"""
Pre-fix reproduction: NaN current price from yfinance flows through
fetch_current_prices as a "valid" price, violating the Optional[float] contract.

Run: python benchmarks/repro_fetch_nan_ingress.py
Exit 0 = fix live (non-finite rejected at ingress).
Exit 1 = pre-fix behavior (non-finite flows through as valid price).
"""

import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.fetch_market_data import _fetch_single_price


def _mock_history(close_value):
    """Build a mock yfinance history returning a single Close of *close_value*."""
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


def main():
    failures = []

    # Case 1: NaN close must be rejected (return None)
    with patch("data.fetch_market_data.yf.Ticker", return_value=_mock_history(float("nan"))):
        ticker, price = _fetch_single_price("SPY")
    if price is not None:
        failures.append(f"NaN close accepted as valid price: {price!r}")

    # Case 2: +inf close must be rejected
    with patch("data.fetch_market_data.yf.Ticker", return_value=_mock_history(float("inf"))):
        ticker, price = _fetch_single_price("SPY")
    if price is not None:
        failures.append(f"+inf close accepted as valid price: {price!r}")

    # Case 3: -inf close must be rejected
    with patch("data.fetch_market_data.yf.Ticker", return_value=_mock_history(float("-inf"))):
        ticker, price = _fetch_single_price("SPY")
    if price is not None:
        failures.append(f"-inf close accepted as valid price: {price!r}")

    # Case 4: healthy price must survive unchanged
    with patch("data.fetch_market_data.yf.Ticker", return_value=_mock_history(104.5)):
        ticker, price = _fetch_single_price("SPY")
    if price != 104.5:
        failures.append(f"Healthy price broken: expected 104.5, got {price!r}")

    if failures:
        print("REPRODUCTION — pre-fix behavior detected:")
        for f in failures:
            print(f"  FAIL: {f}")
        sys.exit(1)

    print("FIX LIVE — all non-finite closes rejected at ingress, healthy prices unchanged")
    sys.exit(0)


if __name__ == "__main__":
    main()

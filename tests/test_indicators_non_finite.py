"""
Regression tests for non-finite value handling in technical indicators.

These tests ensure that yfinance-style bad ticks (NaN, Inf) cannot propagate
non-finite values into the JSON-serialized indicator summary consumed by the
LLM prompt and the monitor.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.indicators import (
    calculate_all_indicators,
    get_latest_indicators,
    analyze_market_data,
)


def _all_finite(mapping: dict) -> bool:
    """Return True if every scalar value in *mapping* is finite."""
    for value in mapping.values():
        try:
            if not np.isfinite(value):
                return False
        except (TypeError, ValueError):
            return False
    return True


def test_non_finite_close_rows_are_dropped():
    """Rows with NaN or Inf Close prices must be removed before indicator calc."""
    df = pd.DataFrame({
        "Close": [100.0, 101.0, np.nan, np.inf, 102.0, -np.inf],
    })

    result = calculate_all_indicators(df)

    assert len(result) == 3
    assert result["Close"].tolist() == [100.0, 101.0, 102.0]
    assert np.isfinite(result["Close"]).all()


def test_latest_indicators_default_on_non_finite_output():
    """get_latest_indicators must coerce non-finite indicator values to defaults."""
    df = pd.DataFrame({
        "Close": [100.0],
        "SMA_20": [np.nan],
        "SMA_50": [np.inf],
        "SMA_200": [-np.inf],
        "RSI_14": [np.nan],
        "BB_upper": [np.inf],
        "BB_lower": [-np.inf],
        "BB_position": [np.nan],
        "Volatility_20": [np.inf],
        "Drawdown": [-np.inf],
        "Max_Drawdown": [-np.inf],
        "Daily_Return": [np.nan],
    })

    latest = get_latest_indicators(df)

    assert latest == {
        "price": 100.0,
        "sma_20": 0.0,
        "sma_50": 0.0,
        "sma_200": 0.0,
        "rsi_14": 50.0,
        "bb_upper": 0.0,
        "bb_lower": 0.0,
        "bb_position": 0.5,
        "volatility_annual": 0.0,
        "drawdown": 0.0,
        "max_drawdown": 0.0,
        "daily_return": 0.0,
    }
    assert _all_finite(latest)


def test_analyze_market_data_is_json_safe_with_bad_ticks():
    """A asset with Inf/NaN Close ticks must still produce JSON-safe output."""
    df = pd.DataFrame({
        "Close": [100.0, np.nan, np.inf, 101.0, 102.0, np.nan],
    })

    result = analyze_market_data({"BAD": df})

    assert "BAD" in result["assets"]
    latest = result["assets"]["BAD"]["latest"]
    assert _all_finite(latest)
    assert np.isfinite(result["assets"]["BAD"]["total_return"])
    assert all(np.isfinite(r) for r in result["assets"]["BAD"]["returns"])

    # This must not raise; the entire analysis output is JSON-safe.
    json.dumps(result["assets"]["BAD"]["latest"], allow_nan=False)


def test_analyze_market_data_skips_asset_with_all_non_finite_closes():
    """An asset whose Close column is entirely non-finite must be skipped."""
    df = pd.DataFrame({"Close": [np.nan, np.inf, -np.inf]})

    result = analyze_market_data({"EMPTY": df})

    assert "EMPTY" not in result["assets"]


def test_calculate_all_indicators_no_runtime_warnings_on_bad_ticks():
    """Non-finite Close ticks must not emit RuntimeWarnings during indicator calc."""
    df = pd.DataFrame({
        "Close": [100.0, np.nan, np.inf, 101.0, 102.0, -np.inf],
    })

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = calculate_all_indicators(df)

    assert len(result) == 3
    assert np.isfinite(result["Close"]).all()


def test_get_latest_indicators_handles_missing_columns():
    """Missing indicator columns should fall back to defaults without errors."""
    df = pd.DataFrame({"Close": [100.0]})

    latest = get_latest_indicators(df)

    assert _all_finite(latest)
    assert latest["rsi_14"] == 50.0
    assert latest["bb_position"] == 0.5
    assert latest["price"] == 100.0

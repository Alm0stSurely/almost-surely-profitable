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


def test_calculate_all_indicators_guards_none_and_no_close_column():
    """None input or a DataFrame without 'Close' must return an empty frame.

    Regression 2026-09-30: yfinance fetch gaps can propagate None frames to
    ad-hoc callers, previously raising AttributeError (silent failure
    documented in the 2026-09-29/30 intraday monitor sessions).
    """
    result_none = calculate_all_indicators(None)
    assert isinstance(result_none, pd.DataFrame)
    assert result_none.empty

    result_no_close = calculate_all_indicators(pd.DataFrame({"Open": [1.0, 2.0]}))
    assert isinstance(result_no_close, pd.DataFrame)
    assert result_no_close.empty


def test_get_latest_indicators_guards_none():
    """None input must return an empty dict, not raise."""
    assert get_latest_indicators(None) == {}
    assert get_latest_indicators(pd.DataFrame()) == {}


# --- Primitive-level contract (regression 2026-10-02) -----------------------
# Raw callers (e.g. monitor.check_bollinger_breakouts, backtest examples)
# feed yfinance series directly into the primitives, bypassing the cleaning
# convention that lives in calculate_all_indicators. Without a guard at this
# layer the primitives silently disagree on dirty input: rolling-based ones
# skip NaN but poison forever on Inf, and calculate_drawdown propagates NaN
# into the last value. The contract is: non-finite ticks are dropped at the
# primitive boundary, for every consumer, by provenance of the data.

from data.indicators import (  # noqa: E402
    calculate_bollinger_bands,
    calculate_drawdown,
    calculate_rsi,
    calculate_sma,
    calculate_volatility,
)


def _dirty_series():
    """20-observation series mimicking a yfinance 20d fetch during the
    session: one mid-series NaN tick and one Inf tick, plus a trailing NaN
    pre-close bar."""
    idx = pd.date_range("2026-09-13", periods=20, freq="D")
    values = list(np.linspace(100.0, 118.0, 19)) + [np.nan]
    values[5] = np.inf
    values[11] = np.nan
    return pd.Series(values, index=idx)


def test_primitives_drop_non_finite_ticks():
    """Every primitive must drop non-finite input ticks: output on dirty
    input must equal output on the explicitly cleaned input, and the final
    value (what raw callers read via .iloc[-1]) must be finite."""
    s = _dirty_series()
    clean = s[np.isfinite(s.to_numpy(dtype=float))]

    for name, dirty_out, clean_out in [
        ("sma", calculate_sma(s.copy(), 20), calculate_sma(clean.copy(), 20)),
        ("rsi", calculate_rsi(s.copy(), 14), calculate_rsi(clean.copy(), 14)),
    ]:
        assert len(dirty_out) == len(clean_out), f"{name}: row count differs"
        assert np.allclose(
            dirty_out.to_numpy(dtype=float), clean_out.to_numpy(dtype=float),
            equal_nan=True,
        ), f"{name}: dirty input changed the result"
        assert np.isfinite(dirty_out.iloc[-1]), f"{name}: non-finite final value"

    for name, dirty_bands, clean_bands in [(
        "bollinger",
        calculate_bollinger_bands(s.copy()),
        calculate_bollinger_bands(clean.copy()),
    )]:
        for d, c in zip(dirty_bands, clean_bands):
            assert len(d) == len(c), f"{name}: row count differs"
            assert np.allclose(
                d.to_numpy(dtype=float), c.to_numpy(dtype=float), equal_nan=True,
            ), f"{name}: dirty input changed the result"
            # The first band row is NaN by definition (std of one observation),
            # on clean data too; the final row is what callers consume.
            assert np.isfinite(d.iloc[-1]), f"{name}: non-finite final band"

    for name, dirty_out, clean_out in [
        ("volatility", calculate_volatility(s.copy(), 20), calculate_volatility(clean.copy(), 20)),
        ("drawdown", calculate_drawdown(s.copy()), calculate_drawdown(clean.copy())),
    ]:
        assert len(dirty_out) == len(clean_out), f"{name}: row count differs"
        assert np.allclose(
            dirty_out.to_numpy(dtype=float), clean_out.to_numpy(dtype=float),
            equal_nan=True,
        ), f"{name}: dirty input changed the result"
        assert np.isfinite(dirty_out.iloc[-1]), f"{name}: non-finite final value"


def test_drawdown_no_longer_propagates_trailing_nan():
    """Regression: a trailing NaN price produced a NaN final drawdown on the
    raw-caller path (wrapper path was cleaned since 0c0ebe4)."""
    s = pd.Series([100.0, 101.0, 102.0, 101.5, np.nan])
    dd = calculate_drawdown(s)
    assert np.isfinite(dd.iloc[-1])
    assert dd.iloc[-1] == pytest.approx((101.5 / 102.0) - 1)


def test_primitives_all_non_finite_input_returns_empty():
    """An input with no finite tick must yield empty output, not raise."""
    s = pd.Series([np.nan, np.inf, -np.inf])

    assert calculate_sma(s, 20).empty
    assert calculate_rsi(s, 14).empty
    upper, middle, lower = calculate_bollinger_bands(s)
    assert upper.empty and middle.empty and lower.empty
    assert calculate_volatility(s, 20).empty
    assert calculate_drawdown(s).empty


def test_primitives_clean_data_unchanged():
    """Clean input must pass through untouched (bounded side of the change)."""
    s = pd.Series(np.linspace(100.0, 119.0, 20))

    sma_before = calculate_sma(s.copy(), 20)
    rsi_before = calculate_rsi(s.copy(), 14)
    upper_b, middle_b, lower_b = calculate_bollinger_bands(s.copy())
    vol_before = calculate_volatility(s.copy(), 20)
    dd_before = calculate_drawdown(s.copy())

    assert calculate_sma(s.copy(), 20).equals(sma_before)
    assert calculate_rsi(s.copy(), 14).equals(rsi_before)
    u2, m2, l2 = calculate_bollinger_bands(s.copy())
    assert u2.equals(upper_b) and m2.equals(middle_b) and l2.equals(lower_b)
    assert calculate_volatility(s.copy(), 20).equals(vol_before)
    assert calculate_drawdown(s.copy()).equals(dd_before)


def test_primitives_no_runtime_warnings_on_dirty_input():
    """Dirty ticks must not emit RuntimeWarnings inside the primitives."""
    s = _dirty_series()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        calculate_sma(s.copy(), 20)
        calculate_rsi(s.copy(), 14)
        calculate_bollinger_bands(s.copy())
        calculate_volatility(s.copy(), 20)
        calculate_drawdown(s.copy())

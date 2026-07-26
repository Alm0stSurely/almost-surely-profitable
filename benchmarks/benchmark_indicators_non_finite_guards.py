"""Benchmark for indicators non-finite input guards.

Demonstrates that calculate_all_indicators and analyze_market_data silently
drop or coerce non-finite yfinance ticks and that the resulting indicator
summary can be serialized with ``allow_nan=False``.
"""

import json
import sys
import timeit
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from data.indicators import (
    calculate_all_indicators,
    get_latest_indicators,
    analyze_market_data,
)


def _build_bad_df(rows: int = 30) -> pd.DataFrame:
    """Build a DataFrame where every third Close value is non-finite."""
    prices = np.linspace(100.0, 130.0, rows)
    prices[::3] = np.nan
    prices[1::3] = np.inf
    return pd.DataFrame({"Close": prices})


def _benchmark(name: str, stmt, setup="pass", number: int = 1000):
    full_setup = f"import warnings; warnings.simplefilter('error', RuntimeWarning); {setup}"
    elapsed = timeit.timeit(stmt, setup=full_setup, number=number, globals=globals())
    mean_us = elapsed / number * 1e6
    print(f"{name:40s} | {number:6d} runs | {mean_us:8.2f} µs/run")
    return elapsed


if __name__ == "__main__":
    print("Benchmark: indicators finite-value guards")
    print("-" * 75)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        # 1. calculate_all_indicators drops non-finite Close rows.
        bad_df = _build_bad_df()
        result = calculate_all_indicators(bad_df)
        assert len(result) < len(bad_df)
        assert np.isfinite(result["Close"]).all()
        _benchmark(
            "calculate_all_indicators (bad ticks)",
            "calculate_all_indicators(bad_df)",
        )

        # 2. get_latest_indicators coerces non-finite outputs to defaults.
        dirty_df = pd.DataFrame({
            "Close": [100.0],
            "SMA_20": [np.nan],
            "SMA_50": [np.inf],
            "RSI_14": [np.inf],
            "BB_position": [np.nan],
            "Volatility_20": [-np.inf],
            "Drawdown": [np.nan],
            "Max_Drawdown": [-np.inf],
            "Daily_Return": [np.nan],
        })
        latest = get_latest_indicators(dirty_df)
        assert all(np.isfinite(v) for v in latest.values() if isinstance(v, (int, float)))
        json.dumps(latest, allow_nan=False)
        _benchmark(
            "get_latest_indicators (coerce)",
            "get_latest_indicators(dirty_df)",
        )

        # 3. analyze_market_data produces JSON-safe output for bad data.
        data_dict = {"BAD": bad_df, "OK": pd.DataFrame({"Close": np.linspace(100.0, 110.0, 30)})}
        analysis = analyze_market_data(data_dict)
        assert "BAD" in analysis["assets"]
        bad_latest = analysis["assets"]["BAD"]["latest"]
        assert all(np.isfinite(v) for v in bad_latest.values() if isinstance(v, (int, float)))
        assert all(np.isfinite(r) for r in analysis["assets"]["BAD"]["returns"])
        json.dumps(analysis["assets"]["BAD"]["latest"], allow_nan=False)
        _benchmark(
            "analyze_market_data (mixed quality)",
            "analyze_market_data(data_dict)",
        )

    print("-" * 75)
    print("OK - indicators silently drop/coerce non-finite inputs and emit valid JSON.")

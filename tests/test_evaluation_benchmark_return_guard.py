"""Regression tests for the evaluation benchmark-return producer guard.

Contract: evaluation._get_benchmark_return returns Optional[float] where
None means unavailable/invalid and the float is always finite. Before the
guard, a NaN/±inf Close at either window edge flowed through as a "valid"
return — NaN/inf outright, or finite garbage (-inf edge -> -1.0, i.e.
"-100% benchmark") that passes downstream math.isfinite checks.

The twin producer reporting.py::_get_benchmark_return already carried this
guard; this file pins the same behavior for the evaluation site.
"""

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import _get_benchmark_return


def _frame(closes):
    dates = pd.date_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"Close": closes}, index=dates)


def _patch_fetch(monkeypatch, closes, ticker="SPY"):
    monkeypatch.setattr(
        "evaluation.fetch_historical_data",
        lambda *a, **k: {ticker: _frame(closes)},
    )


class TestBenchmarkReturnGuard:
    def test_nan_first_close_returns_none(self, monkeypatch):
        _patch_fetch(monkeypatch, [float("nan"), 110.0])
        assert _get_benchmark_return("2024-01-01", "2024-01-05") is None

    def test_nan_last_close_returns_none(self, monkeypatch):
        _patch_fetch(monkeypatch, [100.0, float("nan")])
        assert _get_benchmark_return("2024-01-01", "2024-01-05") is None

    def test_inf_last_close_returns_none(self, monkeypatch):
        _patch_fetch(monkeypatch, [100.0, float("inf")])
        assert _get_benchmark_return("2024-01-01", "2024-01-05") is None

    def test_negative_inf_first_close_returns_none(self, monkeypatch):
        # Pre-fix this produced -1.0: finite garbage that passes isfinite.
        _patch_fetch(monkeypatch, [float("-inf"), 110.0])
        result = _get_benchmark_return("2024-01-01", "2024-01-05")
        assert result is None

    def test_zero_start_price_returns_none(self, monkeypatch):
        # Pre-fix: ZeroDivisionError, caught loud by the except path -> None.
        # Guard makes it a quiet, deliberate None — behavior pinned either way.
        _patch_fetch(monkeypatch, [0.0, 110.0])
        assert _get_benchmark_return("2024-01-01", "2024-01-05") is None

    def test_healthy_window_unchanged(self, monkeypatch):
        _patch_fetch(monkeypatch, [100.0, 110.0])
        result = _get_benchmark_return("2024-01-01", "2024-01-05")
        assert result is not None and math.isclose(result, 0.1, rel_tol=1e-12)

    def test_healthy_declining_window_unchanged(self, monkeypatch):
        _patch_fetch(monkeypatch, [200.0, 150.0])
        result = _get_benchmark_return("2024-01-01", "2024-01-05")
        assert result is not None and math.isclose(result, -0.25, rel_tol=1e-12)

    def test_short_window_returns_none(self, monkeypatch):
        # len(closes) < 2 — pre-existing branch, pinned as healthy-path guard.
        _patch_fetch(monkeypatch, [100.0])
        assert _get_benchmark_return("2024-01-01", "2024-01-05") is None

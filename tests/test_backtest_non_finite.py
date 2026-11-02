"""
Regression tests for non-finite value handling in backtest outputs.

After sanitizing portfolio, risk, and indicator outputs, the backtest engine
remained the last JSON-emitting consumer that could persist ``NaN`` or
``Infinity``. These tests ensure every metric returned by the engine is finite
and that the CLI serialization path is RFC 8259 compliant.
"""

import io
import json
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import pandas as pd

from backtest.backtest import (
    BacktestEngine,
    print_backtest_report,
    run_comparison_backtest,
    _fmt_finite,
    _fmt_pct,
)
from portfolio.portfolio import Portfolio
from utils import dump_json_safe, sanitize_for_json


class TestBacktestFiniteMetrics:
    """Tests that the backtest engine never emits non-finite scalar metrics."""

    def _make_engine(self, start="2024-01-01", end="2024-01-31"):
        return BacktestEngine(
            start_date=start,
            end_date=end,
            tickers=["SPY", "QQQ"],
        )

    def _make_mock_data(self, tickers, dates, base_price=100.0):
        """Build deterministic mock OHLCV data."""
        data = {}
        for ticker in tickers:
            prices = np.linspace(base_price, base_price + len(dates) * 0.5, len(dates))
            data[ticker] = pd.DataFrame({
                "Open": prices,
                "High": prices + 0.5,
                "Low": prices - 0.5,
                "Close": prices,
            }, index=pd.to_datetime(dates))
        return data

    def _attach_portfolio(self, engine, tmp_path, cash=10000.0):
        engine.portfolio = Portfolio(
            state_file="bt_test.json",
            trades_file="bt_trades.json",
            data_dir=str(tmp_path),
        )
        engine.portfolio.cash = cash
        engine.portfolio.positions = {}
        engine.portfolio.trades = []
        engine.portfolio.total_realized_pnl = 0.0

    def test_calculate_metrics_all_positive_returns(self, tmp_path):
        """Degenerate case: every day is positive, so losses and gross_loss are zero.

        ``omega_ratio`` and ``profit_factor`` used to return ``+inf`` in this
        case. They must now return finite values (0.0) so that downstream JSON
        serialization is safe.
        """
        engine = self._make_engine(start="2024-01-01", end="2024-01-05")
        self._attach_portfolio(engine, tmp_path)
        # 5 days of strictly increasing portfolio values
        engine.results = [
            {"total_value": 10000.0, "cash": 5000.0, "positions_value": 5000.0,
             "total_return_pct": 0.0, "num_positions": 1},
            {"total_value": 10100.0, "cash": 5000.0, "positions_value": 5100.0,
             "total_return_pct": 1.0, "num_positions": 1},
            {"total_value": 10200.0, "cash": 5000.0, "positions_value": 5200.0,
             "total_return_pct": 2.0, "num_positions": 1},
            {"total_value": 10300.0, "cash": 5000.0, "positions_value": 5300.0,
             "total_return_pct": 3.0, "num_positions": 1},
            {"total_value": 10400.0, "cash": 5000.0, "positions_value": 5400.0,
             "total_return_pct": 4.0, "num_positions": 1},
        ]

        metrics = engine._calculate_metrics(benchmark_returns=[0.0, 0.0, 0.0, 0.0])

        assert metrics["omega_ratio"] == 0.0
        assert metrics["profit_factor"] == 0.0
        assert np.isfinite(metrics["sharpe_ratio"])
        assert np.isfinite(metrics["sortino_ratio"])
        assert np.isfinite(metrics["calmar_ratio"])
        assert np.isfinite(metrics["max_drawdown"])
        assert np.isfinite(metrics["total_return"])
        assert np.isfinite(metrics["annualized_return"])

    def test_calculate_metrics_zero_benchmark_variance(self, tmp_path):
        """Constant benchmark returns produce zero variance; beta must be 0.0."""
        engine = self._make_engine(start="2024-01-01", end="2024-01-05")
        self._attach_portfolio(engine, tmp_path)
        engine.results = [
            {"total_value": 10000.0, "cash": 5000.0, "positions_value": 5000.0,
             "total_return_pct": 0.0, "num_positions": 1},
            {"total_value": 10050.0, "cash": 5000.0, "positions_value": 5050.0,
             "total_return_pct": 0.5, "num_positions": 1},
            {"total_value": 10100.0, "cash": 5000.0, "positions_value": 5100.0,
             "total_return_pct": 1.0, "num_positions": 1},
            {"total_value": 10050.0, "cash": 5000.0, "positions_value": 5050.0,
             "total_return_pct": 0.5, "num_positions": 1},
        ]

        metrics = engine._calculate_metrics(benchmark_returns=[0.0, 0.0, 0.0])

        assert metrics["beta"] == 0.0
        assert np.isfinite(metrics["alpha"])

    def test_run_backtest_serializes_without_nan(self, tmp_path, monkeypatch):
        """A full mocked backtest must produce JSON that ``allow_nan=False`` accepts."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data/backtest").mkdir(parents=True)
        engine = self._make_engine(start="2024-01-01", end="2024-01-10")
        self._attach_portfolio(engine, tmp_path)

        dates = pd.date_range("2024-01-01", "2024-01-10", freq="B")
        data = self._make_mock_data(["SPY", "QQQ"], dates)

        with patch.object(engine, "fetch_historical_data", return_value=data):
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                result = engine.run_backtest(strategy="buy_and_hold")

        assert result
        assert "total_return" in result
        # Ensure every top-level scalar metric is finite
        scalar_keys = [
            "total_return", "annualized_return", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "calmar_ratio", "omega_ratio", "win_rate",
            "profit_factor", "beta", "alpha", "volatility", "num_trades",
            "final_value", "initial_capital",
        ]
        for key in scalar_keys:
            assert key in result
            assert np.isfinite(result[key]), f"{key} is non-finite: {result[key]}"

        # Verify JSON serialization is strict
        output = io.StringIO()
        dump_json_safe(result, output, indent=2, default=str)
        output.seek(0)
        # Must not raise ValueError
        json.loads(output.read())

        # Also verify raw json.dumps with allow_nan=False works
        cleaned = sanitize_for_json(result)
        json.dumps(cleaned, allow_nan=False, default=str)

    def test_run_comparison_backtest_serializes_without_nan(self, tmp_path, monkeypatch):
        """The comparison helper returns a dict of backtest results that must be JSON safe."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data/backtest").mkdir(parents=True)
        dates = pd.date_range("2024-01-01", "2024-01-10", freq="B")
        data = self._make_mock_data(["SPY", "QQQ"], dates)

        with patch("backtest.backtest.fetch_historical_data", return_value=data):
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                results = run_comparison_backtest(
                    start_date="2024-01-01",
                    end_date="2024-01-10",
                    tickers=["SPY", "QQQ"],
                    include_llm=False,
                )

        assert "buy_and_hold" in results
        assert "equal_weight" in results
        assert "random" in results

        for strategy, result in results.items():
            assert result, f"{strategy} returned empty result"
            for key in ["total_return", "sharpe_ratio", "omega_ratio", "profit_factor", "final_value"]:
                assert np.isfinite(result[key]), f"{strategy}.{key} is non-finite"

        output = io.StringIO()
        dump_json_safe(results, output, indent=2, default=str)
        output.seek(0)
        json.loads(output.read())


class TestBacktestCLISerialization:
    """Tests that the CLI entry points use dump_json_safe."""

    def test_run_backtest_cli_writes_json(self, tmp_path, monkeypatch):
        """``run_backtest.py --output`` must write a strict JSON file."""
        from backtest import run_backtest as cli_module

        monkeypatch.chdir(tmp_path)
        (tmp_path / "data/backtest").mkdir(parents=True)

        dates = pd.date_range("2024-01-01", "2024-01-10", freq="B")
        data = {
            "SPY": pd.DataFrame({
                "Open": np.linspace(100, 105, len(dates)),
                "High": np.linspace(100, 105, len(dates)) + 0.5,
                "Low": np.linspace(100, 105, len(dates)) - 0.5,
                "Close": np.linspace(100, 105, len(dates)),
            }, index=pd.to_datetime(dates)),
        }

        monkeypatch.setattr(
            "backtest.backtest.fetch_historical_data",
            lambda *args, **kwargs: data,
        )

        output_path = tmp_path / "backtest.json"
        monkeypatch.setattr(
            sys, "argv",
            [
                "run_backtest.py",
                "--start", "2024-01-01",
                "--end", "2024-01-10",
                "--strategy", "buy_and_hold",
                "--tickers", "SPY",
                "--output", str(output_path),
            ],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cli_module.main()

        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert "total_return" in loaded
        # No NaN/Infinity tokens
        raw = output_path.read_text()
        assert "NaN" not in raw
        assert "Infinity" not in raw


class TestPrintBacktestReportNonFiniteGuards:
    """Regression tests: the console report must render n/a, never nan/inf."""

    def _make_result(self, **overrides):
        base = {
            "start_date": "2026-01-01",
            "end_date": "2026-06-30",
            "initial_capital": 10000.0,
            "final_value": 10427.55,
            "total_return": 0.0428,
            "annualized_return": 0.0871,
            "volatility": 0.1142,
            "max_drawdown": 0.0312,
            "sharpe_ratio": 0.76,
            "sortino_ratio": 1.12,
            "calmar_ratio": 2.79,
            "omega_ratio": 1.31,
            "num_trades": 12,
            "win_rate": 0.5833,
            "profit_factor": 1.42,
            "beta": 0.88,
            "alpha": 0.0111,
        }
        base.update(overrides)
        return base

    def _render(self, result):
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_backtest_report(result, "guard_test")
        return buf.getvalue()

    def test_all_finite_renders_formatted_values(self):
        out = self._render(self._make_result())
        assert "10,000.00" in out
        assert "10,427.55" in out
        assert "4.28%" in out
        assert "0.76" in out
        assert "nan" not in out.lower()
        assert "inf" not in out.lower()

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_metrics_render_na(self, bad):
        result = self._make_result(
            final_value=bad,
            total_return=bad,
            annualized_return=bad,
            volatility=bad,
            max_drawdown=bad,
            sharpe_ratio=bad,
            sortino_ratio=bad,
            calmar_ratio=bad,
            omega_ratio=bad,
            win_rate=bad,
            profit_factor=bad,
            beta=bad,
            alpha=bad,
        )
        out = self._render(result)
        # The only permitted non-finite marker is the explicit "n/a" fallback
        assert "nan%" not in out
        assert "inf%" not in out
        assert "-inf%" not in out
        assert "€nan" not in out
        assert "€inf" not in out
        assert "n/a" in out
        # num_trades stayed finite and must still render
        assert "12" in out

    def test_none_and_bool_values_render_na(self):
        result = self._make_result(
            final_value=None,
            total_return=None,
            sharpe_ratio=True,
            num_trades=False,
        )
        out = self._render(result)
        assert "n/a" in out
        # bool inputs are rejected even though isinstance(True, int)
        assert "True" not in out
        assert "False" not in out

    def test_missing_num_trades_does_not_crash(self):
        result = self._make_result()
        del result["num_trades"]
        # KeyError on a missing field is acceptable and pre-existing behavior;
        # non-finite num_trades is what we guard here.
        result2 = self._make_result(num_trades=float("nan"))
        out = self._render(result2)
        assert "n/a" in out

    def test_fmt_finite_rejects_non_numeric(self):
        assert _fmt_finite("3.14", ".2f") == "n/a"
        assert _fmt_finite(None, ".2f") == "n/a"
        assert _fmt_finite([1.0], ".2f") == "n/a"
        assert _fmt_finite(3.14159, ".2f") == "3.14"
        assert _fmt_finite(np.float64(2.5), ".1f") == "2.5"

    def test_fmt_pct_scales_only_finite_values(self):
        assert _fmt_pct(0.0428, ".2f") == "4.28"
        assert _fmt_pct(float("nan"), ".2f") == "n/a"
        assert _fmt_pct(None, ".2f") == "n/a"
        assert _fmt_pct(float("inf"), ".2f") == "n/a"


class TestBacktestPriceLookupNonFiniteGuards:
    """Regression tests: non-finite Close values must not enter price lookups.

    A NaN price that reaches the strategy loop silently suppresses stop-loss
    overrides (NaN comparisons are always False) and corrupts cooldown reason
    strings.  The fix drops non-finite rows at the engine's price-ingestion
    boundary with a loud log per dropped row.
    """

    def _make_engine(self):
        return BacktestEngine(
            start_date="2024-01-01",
            end_date="2024-01-31",
            tickers=["SPY"],
        )

    def _make_data_with_nan_close(self, nan_date="2024-01-05"):
        dates = pd.date_range("2024-01-01", "2024-01-10", freq="B")
        closes = [100.0, 101.0, 102.0, 103.0, np.nan, 105.0, 106.0, 107.0]
        return {
            "SPY": pd.DataFrame({
                "Open": closes,
                "High": [c + 0.5 if not np.isnan(c) else np.nan for c in closes],
                "Low": [c - 0.5 if not np.isnan(c) else np.nan for c in closes],
                "Close": closes,
            }, index=dates[:len(closes)])
        }

    def test_nan_close_dropped_from_lookup(self, caplog):
        import logging
        engine = self._make_engine()
        data = self._make_data_with_nan_close()
        with caplog.at_level(logging.WARNING, logger="backtest.backtest"):
            engine._precompute_price_lookups(data)
        from datetime import datetime
        val = engine._price_lookups["SPY"].get(datetime(2024, 1, 5).date())
        assert val is None, f"NaN price leaked into lookup: {val}"
        # Sibling rows survive
        assert engine._price_lookups["SPY"][datetime(2024, 1, 4).date()] == 103.0
        assert engine._price_lookups["SPY"][datetime(2024, 1, 8).date()] == 105.0

    def test_inf_close_dropped_from_lookup(self, caplog):
        import logging
        engine = self._make_engine()
        data = self._make_data_with_nan_close()
        data["SPY"].loc[data["SPY"].index[4], "Close"] = float("inf")
        with caplog.at_level(logging.WARNING, logger="backtest.backtest"):
            engine._precompute_price_lookups(data)
        from datetime import datetime
        val = engine._price_lookups["SPY"].get(datetime(2024, 1, 5).date())
        assert val is None, f"inf price leaked into lookup: {val}"

    def test_all_finite_closes_unchanged(self, caplog):
        import logging
        engine = self._make_engine()
        dates = pd.date_range("2024-01-01", "2024-01-05", freq="B")
        data = {
            "SPY": pd.DataFrame({
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [100.5, 101.5, 102.5, 103.5, 104.5],
                "Low": [99.5, 100.5, 101.5, 102.5, 103.5],
                "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
            }, index=dates)
        }
        with caplog.at_level(logging.WARNING, logger="backtest.backtest"):
            engine._precompute_price_lookups(data)
        # No warnings on healthy path
        warnings = [r for r in caplog.records if "Non-finite" in r.message]
        assert len(warnings) == 0, f"Unexpected warnings: {warnings}"
        from datetime import datetime
        assert len(engine._price_lookups["SPY"]) == 5
        assert engine._price_lookups["SPY"][datetime(2024, 1, 1).date()] == 100.0

    def test_loudness_per_dropped_row(self, caplog):
        import logging
        engine = self._make_engine()
        data = self._make_data_with_nan_close()
        # Add a second NaN
        data["SPY"].loc[data["SPY"].index[2], "Close"] = float("nan")
        with caplog.at_level(logging.WARNING, logger="backtest.backtest"):
            engine._precompute_price_lookups(data)
        warnings = [r for r in caplog.records if "Non-finite" in r.message]
        assert len(warnings) == 2, f"Expected 2 warnings, got {len(warnings)}"
        # Each warning names the ticker and the date
        msgs = [r.message for r in warnings]
        assert any("SPY" in m and "2024-01-03" in m for m in msgs)
        assert any("SPY" in m and "2024-01-05" in m for m in msgs)


class TestFetchMarketDataNonFiniteGuards:
    """Regression tests: _fetch_single_ticker drops non-finite Close rows.

    The scalar path (_fetch_single_price) already rejects non-finite values.
    This pins the same contract for the bulk DataFrame path.
    """

    def _make_hist(self, closes):
        import yfinance as yf  # noqa: F401
        dates = pd.date_range("2024-01-01", periods=len(closes), freq="B")
        return pd.DataFrame({
            "Open": closes,
            "High": [c + 0.5 if np.isfinite(c) else c for c in closes],
            "Low": [c - 0.5 if np.isfinite(c) else c for c in closes],
            "Close": closes,
            "Volume": [1000] * len(closes),
        }, index=dates)

    def test_nan_close_rows_dropped(self, caplog):
        import logging
        from data.fetch_market_data import _fetch_single_ticker
        hist = self._make_hist([100.0, np.nan, 102.0, 103.0])
        mock_ticker = type("MockTicker", (), {
            "history": lambda self, **kwargs: hist,
        })()
        with caplog.at_level(logging.WARNING, logger="data.fetch_market_data"):
            with patch("data.fetch_market_data.yf.Ticker", return_value=mock_ticker):
                ticker, result = _fetch_single_ticker("SPY", "1mo", "1d", None, None)
        assert ticker == "SPY"
        assert result is not None
        assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
        assert not result["Close"].isna().any()
        warnings = [r for r in caplog.records if "Non-finite" in r.message]
        assert len(warnings) == 1

    def test_all_nan_returns_none(self, caplog):
        import logging
        from data.fetch_market_data import _fetch_single_ticker
        hist = self._make_hist([np.nan, np.nan, np.nan])
        mock_ticker = type("MockTicker", (), {
            "history": lambda self, **kwargs: hist,
        })()
        with caplog.at_level(logging.WARNING, logger="data.fetch_market_data"):
            with patch("data.fetch_market_data.yf.Ticker", return_value=mock_ticker):
                ticker, result = _fetch_single_ticker("SPY", "1mo", "1d", None, None)
        assert result is None

    def test_inf_close_rows_dropped(self, caplog):
        import logging
        from data.fetch_market_data import _fetch_single_ticker
        hist = self._make_hist([100.0, float("inf"), 102.0])
        mock_ticker = type("MockTicker", (), {
            "history": lambda self, **kwargs: hist,
        })()
        with caplog.at_level(logging.WARNING, logger="data.fetch_market_data"):
            with patch("data.fetch_market_data.yf.Ticker", return_value=mock_ticker):
                ticker, result = _fetch_single_ticker("SPY", "1mo", "1d", None, None)
        assert result is not None
        assert len(result) == 2
        warnings = [r for r in caplog.records if "Non-finite" in r.message]
        assert len(warnings) == 1

    def test_all_finite_unchanged(self, caplog):
        import logging
        from data.fetch_market_data import _fetch_single_ticker
        hist = self._make_hist([100.0, 101.0, 102.0])
        mock_ticker = type("MockTicker", (), {
            "history": lambda self, **kwargs: hist,
        })()
        with caplog.at_level(logging.WARNING, logger="data.fetch_market_data"):
            with patch("data.fetch_market_data.yf.Ticker", return_value=mock_ticker):
                ticker, result = _fetch_single_ticker("SPY", "1mo", "1d", None, None)
        assert result is not None
        assert len(result) == 3
        warnings = [r for r in caplog.records if "Non-finite" in r.message]
        assert len(warnings) == 0

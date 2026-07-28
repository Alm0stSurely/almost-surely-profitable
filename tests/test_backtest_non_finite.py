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
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import pandas as pd

from backtest.backtest import BacktestEngine, run_comparison_backtest
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

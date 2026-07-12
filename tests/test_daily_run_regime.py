"""
Tests for the daily_run.py regime-analysis data preparation.

The pipeline's regime-analysis block consumes the raw output of
fetch_historical_data, which is Dict[str, pd.DataFrame] with OHLCV
columns and a DatetimeIndex. These tests verify that the helper which
builds the price DataFrame handles the real API contract correctly,
including mismatched trading calendars and missing data.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daily_run import _build_prices_df


class TestBuildPricesDf:
    """Test suite for _build_prices_df helper."""

    def test_builds_dataframe_from_close_columns(self):
        """Should extract Close columns and align by DatetimeIndex."""
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        market_data = {
            "SPY": pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates),
            "TLT": pd.DataFrame({"Close": [90.0, 91.0, 92.0, 93.0, 94.0]}, index=dates),
        }

        df = _build_prices_df(market_data)

        assert list(df.columns) == ["SPY", "TLT"]
        assert len(df) == 5
        assert df["SPY"].iloc[-1] == 104.0
        assert df["TLT"].iloc[-1] == 94.0

    def test_aligns_mismatched_trading_calendars(self):
        """Should align dates when calendars differ (e.g. US vs Euronext holidays)."""
        us_dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"])
        eu_dates = pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-06"])  # Jan 2 holiday
        market_data = {
            "SPY": pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0]}, index=us_dates),
            "MC.PA": pd.DataFrame({"Close": [200.0, 201.0, 202.0]}, index=eu_dates),
        }

        df = _build_prices_df(market_data)

        assert isinstance(df.index, pd.DatetimeIndex)
        # Jan 2 is present only for SPY, so MC.PA should be NaN on that date
        assert "2026-01-02" in df.index.strftime("%Y-%m-%d")
        assert pd.isna(df.loc["2026-01-02", "MC.PA"])
        assert df.loc["2026-01-02", "SPY"] == 101.0
        # Common dates should have values for both assets
        assert df.loc["2026-01-06", "SPY"] == 103.0
        assert df.loc["2026-01-06", "MC.PA"] == 202.0

    def test_skips_empty_or_missing_close(self):
        """Should ignore assets without a Close column or with empty data."""
        dates = pd.date_range("2026-01-01", periods=3, freq="B")
        market_data = {
            "SPY": pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=dates),
            "NO_CLOSE": pd.DataFrame({"Open": [1.0, 2.0, 3.0]}, index=dates),
            "EMPTY": pd.DataFrame(),
            "BAD": {"history": {"close": pd.Series([1.0, 2.0])}},  # wrong type
        }

        df = _build_prices_df(market_data)

        assert list(df.columns) == ["SPY"]
        assert "NO_CLOSE" not in df.columns
        assert "EMPTY" not in df.columns
        assert "BAD" not in df.columns

    def test_returns_empty_when_no_valid_data(self):
        """Should return an empty DataFrame when no valid Close series exist."""
        market_data = {
            "NO_CLOSE": pd.DataFrame({"Open": [1.0, 2.0]}),
            "BAD": {"history": {"close": pd.Series([1.0, 2.0])}},
        }

        df = _build_prices_df(market_data)

        assert df.empty

    def test_preserves_datetime_index(self):
        """The resulting index should remain a DatetimeIndex."""
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        market_data = {
            "SPY": pd.DataFrame({"Close": np.linspace(100, 110, 10)}, index=dates),
        }

        df = _build_prices_df(market_data)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0] == dates[0]


class TestDailyRunRegimeIntegration:
    """
    Integration tests that exercise the regime-analysis path in run_daily_pipeline.

    These tests mock the network boundary (fetch_historical_data) with realistic
    DataFrame-shaped data and assert that the RegimeDetector is invoked.
    """

    def test_regime_detector_invoked_with_realistic_data(self, tmp_path, monkeypatch):
        """The pipeline should run regime analysis on DataFrame-shaped market data."""
        from unittest.mock import MagicMock, patch
        from daily_run import run_daily_pipeline

        results_dir = tmp_path / "results" / "daily"
        results_dir.mkdir(parents=True)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        monkeypatch.chdir(tmp_path)

        dates = pd.date_range("2026-06-01", periods=70, freq="B")
        np.random.seed(42)
        spy = pd.DataFrame({"Close": 100 + np.cumsum(np.random.randn(70) * 0.5)}, index=dates)
        tlt = pd.DataFrame({"Close": 90 + np.cumsum(np.random.randn(70) * 0.3)}, index=dates)
        market_data = {"SPY": spy, "TLT": tlt}

        market_analysis = {
            "assets": {
                "SPY": {"latest": {"price": 105.0}, "returns": [0.0] * 69},
                "TLT": {"latest": {"price": 91.0}, "returns": [0.0] * 69},
            },
            "analysis_date": "2026-08-12",
        }

        mock_portfolio = MagicMock()
        mock_portfolio.positions = {}
        mock_portfolio.trades = []
        mock_portfolio.get_summary.return_value = {
            "cash": 10000.0,
            "positions_value": 0.0,
            "total_value": 10000.0,
            "total_return_pct": 0.0,
            "total_realized_pnl": 0.0,
            "total_unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "num_positions": 0,
            "positions": [],
        }

        mock_agent = MagicMock()
        mock_agent.get_trading_decision.return_value = {
            "reasoning": "HOLD for test",
            "actions": [{"ticker": "SPY", "action": "hold", "pct": 0}],
            "error": False,
        }

        mock_cooldown_mgr = MagicMock()
        mock_cooldown_mgr.get_status.return_value = {
            "trades_this_week": 0,
            "weekly_cap": 5,
            "adaptive_stop_loss": 5.0,
            "active_entries": {},
        }

        mock_regime_state = MagicMock()
        mock_regime_state.summary.return_value = "test-regime"

        mock_regime_detector = MagicMock()
        mock_regime_detector.analyze.return_value = mock_regime_state
        mock_regime_detector.get_strategy_recommendation.return_value = {
            "position_sizing": "normal",
            "mean_reversion_opportunities": False,
            "trend_following": False,
        }

        mock_benchmark = MagicMock()
        mock_benchmark.rebalance.return_value = {
            "total_value": 10000.0,
            "total_return_pct": 0.0,
            "num_positions": 0,
        }

        class _FixedNow:
            def __init__(self, when):
                self._when = when

            def now(self):
                return self._when

            def strftime(self, fmt):
                return self._when.strftime(fmt)

        fixed_date = datetime(2026, 8, 12, 10, 30, 0)

        patches = {
            "REPO_ROOT": tmp_path,
            "DATA_DIR": data_dir,
            "DAILY_RESULTS_DIR": results_dir,
            "fetch_historical_data": MagicMock(return_value=market_data),
            "analyze_market_data": MagicMock(return_value=market_analysis),
            "fetch_current_prices": MagicMock(return_value={}),
            "Portfolio": MagicMock(return_value=mock_portfolio),
            "TradingAgent": MagicMock(return_value=mock_agent),
            "PositionCooldownManager": MagicMock(return_value=mock_cooldown_mgr),
            "CooldownConfig": MagicMock(),
            "calculate_portfolio_cvar": MagicMock(),
            "tail_risk_analysis": MagicMock(return_value={}),
            "calculate_all_metrics": MagicMock(),
            "LiveEqualWeightBenchmark": MagicMock(return_value=mock_benchmark),
            "RegimeDetector": mock_regime_detector,
            "format_regime_for_llm": MagicMock(),
            "datetime": _FixedNow(fixed_date),
        }

        with patch.multiple("daily_run", **patches):
            run_daily_pipeline(dry_run=False, no_overwrite=False)

        # RegimeDetector should have been called with a non-empty price DataFrame
        mock_regime_detector.assert_called_once()
        call_args = mock_regime_detector.return_value.analyze.call_args
        prices_df = call_args[0][0]
        assert isinstance(prices_df, pd.DataFrame)
        assert not prices_df.empty
        assert "SPY" in prices_df.columns
        assert "TLT" in prices_df.columns

    def test_regime_skipped_when_no_valid_data(self, tmp_path, monkeypatch):
        """The pipeline should skip regime analysis when fetch returns no usable data."""
        from unittest.mock import MagicMock, patch
        from daily_run import run_daily_pipeline

        results_dir = tmp_path / "results" / "daily"
        results_dir.mkdir(parents=True)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        monkeypatch.chdir(tmp_path)

        market_data = {
            "SPY": pd.DataFrame({"Open": [1.0, 2.0]}),  # no Close column
            "TLT": pd.DataFrame(),  # empty
        }
        market_analysis = {
            "assets": {
                "SPY": {"latest": {"price": 100.0}, "returns": [0.0]},
            },
            "analysis_date": "2026-08-12",
        }

        mock_portfolio = MagicMock()
        mock_portfolio.positions = {}
        mock_portfolio.trades = []
        mock_portfolio.get_summary.return_value = {
            "cash": 10000.0,
            "positions_value": 0.0,
            "total_value": 10000.0,
            "total_return_pct": 0.0,
            "total_realized_pnl": 0.0,
            "total_unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "num_positions": 0,
            "positions": [],
        }

        mock_agent = MagicMock()
        mock_agent.get_trading_decision.return_value = {
            "reasoning": "HOLD",
            "actions": [{"ticker": "SPY", "action": "hold", "pct": 0}],
            "error": False,
        }

        mock_cooldown_mgr = MagicMock()
        mock_cooldown_mgr.get_status.return_value = {
            "trades_this_week": 0,
            "weekly_cap": 5,
            "adaptive_stop_loss": 5.0,
            "active_entries": {},
        }

        mock_regime_detector = MagicMock()

        mock_benchmark = MagicMock()
        mock_benchmark.rebalance.return_value = {
            "total_value": 10000.0,
            "total_return_pct": 0.0,
            "num_positions": 0,
        }

        class _FixedNow:
            def __init__(self, when):
                self._when = when

            def now(self):
                return self._when

        fixed_date = datetime(2026, 8, 12, 10, 30, 0)

        patches = {
            "REPO_ROOT": tmp_path,
            "DATA_DIR": data_dir,
            "DAILY_RESULTS_DIR": results_dir,
            "fetch_historical_data": MagicMock(return_value=market_data),
            "analyze_market_data": MagicMock(return_value=market_analysis),
            "fetch_current_prices": MagicMock(return_value={}),
            "Portfolio": MagicMock(return_value=mock_portfolio),
            "TradingAgent": MagicMock(return_value=mock_agent),
            "PositionCooldownManager": MagicMock(return_value=mock_cooldown_mgr),
            "CooldownConfig": MagicMock(),
            "calculate_portfolio_cvar": MagicMock(),
            "tail_risk_analysis": MagicMock(return_value={}),
            "calculate_all_metrics": MagicMock(),
            "LiveEqualWeightBenchmark": MagicMock(return_value=mock_benchmark),
            "RegimeDetector": mock_regime_detector,
            "format_regime_for_llm": MagicMock(),
            "datetime": _FixedNow(fixed_date),
        }

        with patch.multiple("daily_run", **patches):
            run_daily_pipeline(dry_run=False, no_overwrite=False)

        mock_regime_detector.analyze.assert_not_called()

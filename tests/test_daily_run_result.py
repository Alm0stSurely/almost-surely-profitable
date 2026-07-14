"""
Tests for the daily_run result file and dry-run isolation.

These tests guard the result-persistence layer: a dry run must leave the
stored portfolio and cooldown state untouched, and the pre-trade tail-risk
context computed for the LLM must be saved in the result log for offline
analysis.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daily_run import run_daily_pipeline


class _FixedNow:
    """Simple stand-in for datetime in the pipeline."""

    def __init__(self, when: datetime):
        self._when = when

    def now(self):
        return self._when

    def strftime(self, fmt):
        return self._when.strftime(fmt)

    def isoformat(self):
        return self._when.isoformat()


def _patch_pipeline(tmp_path, monkeypatch, fixed_date: datetime, include_risk_metrics: bool = False):
    """Patch the network/IO surface of run_daily_pipeline so it runs offline."""
    results_dir = tmp_path / "results" / "daily"
    results_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    monkeypatch.chdir(tmp_path)

    dates = pd.date_range("2026-07-01", periods=3, freq="B")
    mock_market_data = {
        "SPY": pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=dates)
    }
    market_analysis = {
        "assets": {
            "SPY": {
                "latest": {"price": 400.0},
                "returns": [0.0, 0.0, 0.0],
            }
        },
        "analysis_date": fixed_date.strftime("%Y-%m-%d"),
    }

    positions_summary = [
        {
            "ticker": "SPY",
            "quantity": 20.0,
            "avg_price": 100.0,
            "current_price": 103.0,
            "market_value": 2000.0,
            "unrealized_pnl_pct": 3.0,
        }
    ] if include_risk_metrics else []

    risk_metrics = {
        "cvar_95": -0.02,
        "cvar_99": -0.03,
        "var_95": -0.015,
        "var_99": -0.025,
        "max_drawdown": -0.04,
        "sortino_ratio": 0.8,
        "skewness": -0.2,
        "kurtosis": 3.2,
    } if include_risk_metrics else None

    portfolio_summary = {
        "cash": 8000.0 if include_risk_metrics else 10000.0,
        "positions_value": 2000.0 if include_risk_metrics else 0.0,
        "total_value": 10000.0,
        "total_return_pct": 0.0,
        "total_realized_pnl": 0.0,
        "total_unrealized_pnl": 0.0,
        "total_pnl": 0.0,
        "num_positions": 1 if include_risk_metrics else 0,
        "positions": positions_summary,
    }
    if include_risk_metrics:
        portfolio_summary["risk_metrics"] = risk_metrics

    mock_portfolio = MagicMock()
    mock_portfolio.positions = {"SPY": MagicMock()} if include_risk_metrics else {}
    mock_portfolio.trades = []
    mock_portfolio.get_summary.return_value = portfolio_summary

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
    mock_cooldown_mgr.can_buy.return_value = (True, "")
    mock_cooldown_mgr.can_sell.return_value = (True, "")

    mock_regime_state = MagicMock()
    mock_regime_state.summary.return_value = "test-regime"

    mock_regime_detector = MagicMock()
    mock_regime_detector.analyze.return_value = mock_regime_state
    mock_regime_detector.get_strategy_recommendation.return_value = {
        "position_sizing": "normal",
        "mean_reversion_opportunities": False,
        "trend_following": False,
    }

    mock_cvar = MagicMock()
    mock_cvar.cvar_95 = -0.02
    mock_cvar.cvar_99 = -0.03
    mock_cvar.var_95 = -0.015
    mock_cvar.var_99 = -0.025

    mock_tail_risk = MagicMock(return_value={
        "max_drawdown": -0.04,
        "sortino_ratio": 0.8,
        "skewness": -0.2,
        "kurtosis": 3.2,
    })

    mock_perf_metrics = MagicMock()
    mock_perf_metrics.sharpe_ratio = 1.5
    mock_perf_metrics.sortino_ratio = 1.8
    mock_perf_metrics.calmar_ratio = 0.9
    mock_perf_metrics.volatility = 0.12
    mock_perf_metrics.beta = 0.95
    mock_perf_metrics.alpha = 0.01
    mock_perf_metrics.treynor_ratio = 1.2
    mock_perf_metrics.information_ratio = 0.5
    mock_perf_metrics.tracking_error = 0.02
    mock_perf_metrics.max_drawdown = -0.04
    mock_perf_metrics.annualized_return = 0.08

    mock_benchmark = MagicMock()
    mock_benchmark.rebalance.return_value = {
        "total_value": 10000.0,
        "total_return_pct": 0.0,
        "num_positions": 0,
    }

    patches = {
        "REPO_ROOT": tmp_path,
        "DATA_DIR": data_dir,
        "DAILY_RESULTS_DIR": results_dir,
        "fetch_historical_data": MagicMock(return_value=mock_market_data),
        "analyze_market_data": MagicMock(return_value=market_analysis),
        "fetch_current_prices": MagicMock(return_value={"SPY": 103.0}),
        "Portfolio": MagicMock(return_value=mock_portfolio),
        "TradingAgent": MagicMock(return_value=mock_agent),
        "PositionCooldownManager": MagicMock(return_value=mock_cooldown_mgr),
        "CooldownConfig": MagicMock(),
        "calculate_portfolio_cvar": MagicMock(return_value=mock_cvar),
        "tail_risk_analysis": mock_tail_risk,
        "calculate_all_metrics": MagicMock(return_value=mock_perf_metrics),
        "LiveEqualWeightBenchmark": MagicMock(return_value=mock_benchmark),
        "RegimeDetector": mock_regime_detector,
        "format_regime_for_llm": MagicMock(),
        "datetime": _FixedNow(fixed_date),
    }

    return patches, mock_portfolio


def test_dry_run_does_not_save_portfolio_state(tmp_path, monkeypatch):
    """A dry run must not persist updated prices or trades to portfolio state."""
    fixed_date = datetime(2026, 7, 14, 10, 30, 0)
    patches, mock_portfolio = _patch_pipeline(tmp_path, monkeypatch, fixed_date)
    result_file = tmp_path / "results" / "daily" / "2026-07-14_dry_run.json"

    with patch.multiple("daily_run", **patches):
        run_daily_pipeline(dry_run=True, no_overwrite=False)

    assert result_file.exists()
    result = json.loads(result_file.read_text())
    assert result["dry_run"] is True

    mock_portfolio.save_state.assert_not_called()
    mock_portfolio.update_prices.assert_called_once()


def test_dry_run_does_not_save_cooldown_state(tmp_path, monkeypatch):
    """A dry run must not persist cooldown-manager state."""
    fixed_date = datetime(2026, 7, 14, 10, 30, 0)
    patches, _ = _patch_pipeline(tmp_path, monkeypatch, fixed_date)
    result_file = tmp_path / "results" / "daily" / "2026-07-14_dry_run.json"

    with patch.multiple("daily_run", **patches):
        run_daily_pipeline(dry_run=True, no_overwrite=False)

    assert result_file.exists()
    mock_cooldown_mgr = patches["PositionCooldownManager"].return_value
    mock_cooldown_mgr.save_state.assert_not_called()


def test_risk_metrics_persisted_in_result_file(tmp_path, monkeypatch):
    """Pre-trade CVaR/VaR context must be written to the daily result log."""
    fixed_date = datetime(2026, 7, 14, 10, 30, 0)
    patches, _ = _patch_pipeline(
        tmp_path, monkeypatch, fixed_date, include_risk_metrics=True
    )
    result_file = tmp_path / "results" / "daily" / "2026-07-14.json"

    with patch.multiple("daily_run", **patches):
        run_daily_pipeline(dry_run=False, no_overwrite=False)

    assert result_file.exists()
    result = json.loads(result_file.read_text())
    assert result["dry_run"] is False

    portfolio_before = result["portfolio_before"]
    assert "risk_metrics" in portfolio_before
    risk_metrics = portfolio_before["risk_metrics"]
    assert risk_metrics["cvar_95"] == -0.02
    assert risk_metrics["var_95"] == -0.015
    assert risk_metrics["max_drawdown"] == -0.04


def test_no_risk_metrics_when_no_positions(tmp_path, monkeypatch):
    """When there are no positions, risk_metrics should be absent from the log."""
    fixed_date = datetime(2026, 7, 14, 10, 30, 0)
    patches, _ = _patch_pipeline(
        tmp_path, monkeypatch, fixed_date, include_risk_metrics=False
    )
    result_file = tmp_path / "results" / "daily" / "2026-07-14.json"

    with patch.multiple("daily_run", **patches):
        run_daily_pipeline(dry_run=False, no_overwrite=False)

    assert result_file.exists()
    result = json.loads(result_file.read_text())
    assert result["portfolio_before"].get("risk_metrics") is None

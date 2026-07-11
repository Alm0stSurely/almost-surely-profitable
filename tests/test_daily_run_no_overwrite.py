"""
Tests for the daily_run.py no-overwrite safeguard.

These tests focus on the result-persistence logic: when the --no-overwrite
flag is active, an existing daily result file must not be overwritten.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daily_run import run_daily_pipeline


class _FixedNow:
    """Simple stand-in for datetime in the pipeline."""

    def __init__(self, when: datetime):
        self._when = when

    def now(self):
        return self._when


def _patch_pipeline(tmp_path, monkeypatch, fixed_date: datetime):
    """Patch the network/IO surface of run_daily_pipeline so it runs offline."""
    results_dir = tmp_path / "results" / "daily"
    results_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Change cwd so files are written into the temporary directory
    monkeypatch.chdir(tmp_path)

    # Minimal market data that satisfies the DataFrame creation in regime analysis
    mock_market_data = {
        "SPY": {"history": {"close": pd.Series([1.0, 2.0, 3.0])}}
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

    mock_portfolio = MagicMock()
    mock_portfolio.positions = {}
    mock_portfolio.trades = []
    mock_portfolio.cash = 10000.0
    mock_portfolio.total_value = 10000.0
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

    mock_benchmark = MagicMock()
    mock_benchmark.rebalance.return_value = {
        "total_value": 10000.0,
        "total_return_pct": 0.0,
        "num_positions": 0,
    }

    patches = {
        "fetch_historical_data": MagicMock(return_value=mock_market_data),
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

    return patches, mock_portfolio


def test_no_overwrite_skips_existing_real_result(tmp_path, monkeypatch):
    """When no_overwrite=True, an existing real result file is preserved."""
    fixed_date = datetime(2026, 7, 6, 10, 30, 0)
    patches, _ = _patch_pipeline(tmp_path, monkeypatch, fixed_date)
    original = {"marker": "original-real", "dry_run": False}
    result_file = tmp_path / "results" / "daily" / "2026-07-06.json"
    result_file.write_text(json.dumps(original))

    with patch.multiple("daily_run", **patches):
        run_daily_pipeline(dry_run=False, no_overwrite=True)

    assert json.loads(result_file.read_text()) == original


def test_no_overwrite_skips_existing_dry_run_result(tmp_path, monkeypatch):
    """When no_overwrite=True, an existing dry-run result file is preserved."""
    fixed_date = datetime(2026, 7, 6, 10, 30, 0)
    patches, _ = _patch_pipeline(tmp_path, monkeypatch, fixed_date)
    original = {"marker": "original-dry", "dry_run": True}
    result_file = tmp_path / "results" / "daily" / "2026-07-06_dry_run.json"
    result_file.write_text(json.dumps(original))

    with patch.multiple("daily_run", **patches):
        run_daily_pipeline(dry_run=True, no_overwrite=True)

    assert json.loads(result_file.read_text()) == original


def test_default_overwrites_existing_real_result(tmp_path, monkeypatch):
    """By default, an existing result file is overwritten (backward compatible)."""
    fixed_date = datetime(2026, 7, 6, 10, 30, 0)
    patches, _ = _patch_pipeline(tmp_path, monkeypatch, fixed_date)
    original = {"marker": "original-real", "dry_run": False}
    result_file = tmp_path / "results" / "daily" / "2026-07-06.json"
    result_file.write_text(json.dumps(original))

    with patch.multiple("daily_run", **patches):
        run_daily_pipeline(dry_run=False, no_overwrite=False)

    result = json.loads(result_file.read_text())
    assert result["dry_run"] is False
    assert result.get("marker") != "original-real"


def test_no_overwrite_writes_when_file_missing(tmp_path, monkeypatch):
    """no_overwrite still writes the file if it does not yet exist."""
    fixed_date = datetime(2026, 7, 6, 10, 30, 0)
    patches, _ = _patch_pipeline(tmp_path, monkeypatch, fixed_date)
    result_file = tmp_path / "results" / "daily" / "2026-07-06.json"

    with patch.multiple("daily_run", **patches):
        run_daily_pipeline(dry_run=False, no_overwrite=True)

    assert result_file.exists()
    result = json.loads(result_file.read_text())
    assert result["dry_run"] is False

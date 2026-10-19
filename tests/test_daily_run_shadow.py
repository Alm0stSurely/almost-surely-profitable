"""
Wiring tests: the daily pipeline invokes the H4 shadow arm correctly.

Shadow mode (docs/experiments/2026-10-16-prompt-midband-deployment.md) runs
the variant prompt alongside the live prompt each evening and logs variant
actions to data/shadow_decisions.json. These tests verify the pipeline
integration contract:
  - enabled config (mode == "shadow") -> shadow arm runs after the live call
  - dry_run -> shadow arm skipped (tests must not burn API calls)
  - disabled/missing config -> shadow arm skipped
  - shadow arm raising -> pipeline continues (best-effort arm)
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daily_run import run_daily_pipeline


class _FixedNow:
    def __init__(self, when: datetime):
        self._when = when

    def now(self):
        return self._when


def _patch_pipeline(tmp_path, monkeypatch, fixed_date: datetime, experiments_config=None):
    """Patch the network/IO surface of run_daily_pipeline so it runs offline."""
    results_dir = tmp_path / "results" / "daily"
    results_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    if experiments_config is not None:
        (config_dir / "experiments.json").write_text(json.dumps(experiments_config))

    monkeypatch.chdir(tmp_path)

    dates = pd.date_range("2026-07-01", periods=3, freq="B")
    mock_market_data = {
        "SPY": pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=dates)
    }
    market_analysis = {
        "assets": {
            "SPY": {"latest": {"price": 400.0}, "returns": [0.0, 0.0, 0.0]}
        },
        "analysis_date": fixed_date.strftime("%Y-%m-%d"),
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
        "weekly_cap": 3,
        "adaptive_stop_loss": 5.0,
        "active_entries": {},
    }
    mock_cooldown_mgr.can_buy.return_value = (True, "")
    mock_cooldown_mgr.can_sell.return_value = (True, "")

    mock_regime_detector = MagicMock()
    mock_regime_state = MagicMock()
    mock_regime_state.summary.return_value = "test-regime"
    mock_regime_state.volatility_regime = "normal"
    mock_regime_state.trend_regime = "neutral"
    mock_regime_state.correlation_regime = "normal"
    mock_regime_state.volatility_percentile = 0.5
    mock_regime_state.adx_value = 20.0
    mock_regime_state.avg_correlation = 0.3
    # The pipeline calls RegimeDetector() then .analyze() on the RESULT —
    # wire the state through the constructor's return_value.
    mock_regime_detector.return_value.analyze.return_value = mock_regime_state
    mock_regime_detector.return_value.get_strategy_recommendation.return_value = {
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
        "REPO_ROOT": tmp_path,
        "DATA_DIR": data_dir,
        "DAILY_RESULTS_DIR": results_dir,
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

    applied = []
    import daily_run
    for name, value in patches.items():
        applied.append(monkeypatch.setattr(daily_run, name, value))

    return mock_agent


@pytest.fixture
def fixed_date():
    return datetime(2026, 10, 19, 21, 6, 0)


SHADOW_ENABLED = {"midband_deployment": {"mode": "shadow"}}


class TestShadowArmWiring:
    def test_shadow_arm_runs_when_enabled(self, tmp_path, monkeypatch, fixed_date):
        _patch_pipeline(tmp_path, monkeypatch, fixed_date, experiments_config=SHADOW_ENABLED)
        import daily_run
        mock_shadow = MagicMock()
        monkeypatch.setattr(daily_run, "run_shadow_arm", mock_shadow)

        run_daily_pipeline()

        assert mock_shadow.call_count == 1
        kwargs = mock_shadow.call_args.kwargs
        assert kwargs["agent"] is not None
        assert kwargs["log_path"] == tmp_path / "data" / "shadow_decisions.json"
        assert kwargs["vol_regime"] == "normal"
        # Live decision summary is forwarded for the diff audit.
        assert "SPY hold" in kwargs["live_actions_summary"]

    def test_shadow_arm_skipped_in_dry_run(self, tmp_path, monkeypatch, fixed_date):
        _patch_pipeline(tmp_path, monkeypatch, fixed_date, experiments_config=SHADOW_ENABLED)
        import daily_run
        mock_shadow = MagicMock()
        monkeypatch.setattr(daily_run, "run_shadow_arm", mock_shadow)

        run_daily_pipeline(dry_run=True)

        assert mock_shadow.call_count == 0

    def test_shadow_arm_skipped_when_config_missing(self, tmp_path, monkeypatch, fixed_date):
        _patch_pipeline(tmp_path, monkeypatch, fixed_date, experiments_config=None)
        import daily_run
        mock_shadow = MagicMock()
        monkeypatch.setattr(daily_run, "run_shadow_arm", mock_shadow)

        run_daily_pipeline()

        assert mock_shadow.call_count == 0

    def test_shadow_arm_skipped_when_mode_off(self, tmp_path, monkeypatch, fixed_date):
        _patch_pipeline(
            tmp_path, monkeypatch, fixed_date,
            experiments_config={"midband_deployment": {"mode": "off"}},
        )
        import daily_run
        mock_shadow = MagicMock()
        monkeypatch.setattr(daily_run, "run_shadow_arm", mock_shadow)

        run_daily_pipeline()

        assert mock_shadow.call_count == 0

    def test_shadow_arm_failure_does_not_abort_pipeline(self, tmp_path, monkeypatch, fixed_date):
        _patch_pipeline(tmp_path, monkeypatch, fixed_date, experiments_config=SHADOW_ENABLED)
        import daily_run

        def _boom(**kwargs):
            raise RuntimeError("shadow backend down")

        monkeypatch.setattr(daily_run, "run_shadow_arm", _boom)

        # Must complete without raising.
        run_daily_pipeline()

        # The daily result file was still written.
        result_files = list((tmp_path / "results" / "daily").glob("*.json"))
        assert len(result_files) == 1

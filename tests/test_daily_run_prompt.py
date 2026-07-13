"""
Smoke tests for the LLM prompt assembled in the daily pipeline.

These tests verify that the prompt sent to the trading agent actually contains
the Market Regime Analysis and Tail Risk sections that the pipeline computes.
They also guard against a regression where the portfolio_summary passed to the
agent was re-fetched after risk metrics were computed, silently dropping the
CVaR/VaR context from the LLM.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daily_run import run_daily_pipeline
from llm.trading_agent import TradingAgent


class TestTradingAgentPrompt:
    """Unit tests for TradingAgent.build_prompt()."""

    @pytest.fixture
    def agent(self):
        return TradingAgent(api_key="dummy-key")

    @pytest.fixture
    def market_data(self):
        return {
            "assets": {
                "SPY": {
                    "latest": {
                        "price": 450.0,
                        "sma_20": 445.0,
                        "sma_50": 440.0,
                        "rsi_14": 55.0,
                        "bb_position": 0.6,
                        "volatility_annual": 0.15,
                        "drawdown": -0.02,
                        "daily_return": 0.005,
                    }
                }
            },
            "correlations": {},
        }

    @pytest.fixture
    def portfolio_summary(self):
        return {
            "cash": 5000.0,
            "total_value": 10000.0,
            "total_return_pct": 0.0,
            "total_pnl": 0.0,
            "positions": [
                {
                    "ticker": "SPY",
                    "quantity": 10.0,
                    "avg_price": 440.0,
                    "current_price": 450.0,
                    "market_value": 4500.0,
                    "unrealized_pnl_pct": 2.27,
                }
            ],
            "risk_metrics": {
                "cvar_95": -0.015,
                "cvar_99": -0.025,
                "var_95": -0.01,
                "var_99": -0.02,
                "max_drawdown": -0.05,
                "sortino_ratio": 1.2,
                "skewness": -0.1,
                "kurtosis": 3.1,
            },
        }

    def test_prompt_contains_regime_section(self, agent, market_data, portfolio_summary):
        """The formatted regime block should appear in the prompt."""
        market_data["regime"] = {
            "formatted": "\n## Market Regime Analysis\n\n**Current Regime:** ..."
        }
        prompt = agent.build_prompt(market_data, portfolio_summary)
        assert "Market Regime Analysis" in prompt

    def test_prompt_contains_risk_metrics(self, agent, market_data, portfolio_summary):
        """Tail-risk metrics computed by the pipeline should appear in the prompt."""
        prompt = agent.build_prompt(market_data, portfolio_summary)
        assert "=== RISK METRICS (Tail Risk Analysis) ===" in prompt
        assert "CVaR 95%" in prompt
        assert "VaR 95%" in prompt

    def test_prompt_omits_regime_when_missing(self, agent, market_data, portfolio_summary):
        """Without a regime block, the prompt should not mention it."""
        prompt = agent.build_prompt(market_data, portfolio_summary)
        assert "Market Regime Analysis" not in prompt

    def test_prompt_omits_risk_metrics_when_missing(self, agent, market_data, portfolio_summary):
        """Without risk metrics, the prompt should not include the section."""
        portfolio_summary.pop("risk_metrics")
        prompt = agent.build_prompt(market_data, portfolio_summary)
        assert "=== RISK METRICS (Tail Risk Analysis) ===" not in prompt
        assert "CVaR 95%" not in prompt


class TestDailyRunRiskMetricsPreserved:
    """
    Integration test that the risk_metrics computed in run_daily_pipeline are
    passed to the LLM agent instead of being overwritten by a fresh summary.
    """

    def test_risk_metrics_and_regime_reach_agent(self, tmp_path, monkeypatch):
        results_dir = tmp_path / "results" / "daily"
        results_dir.mkdir(parents=True)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        monkeypatch.chdir(tmp_path)

        dates = pd.date_range("2026-06-01", periods=70, freq="B")
        market_data = {
            "SPY": pd.DataFrame({"Close": [100.0 + i * 0.05 for i in range(70)]}, index=dates),
            "TLT": pd.DataFrame({"Close": [90.0 + i * 0.03 for i in range(70)]}, index=dates),
        }
        market_analysis = {
            "assets": {
                "SPY": {"latest": {"price": 103.0}, "returns": [0.0] * 69},
                "TLT": {"latest": {"price": 92.0}, "returns": [0.0] * 69},
            },
            "analysis_date": "2026-08-12",
        }

        mock_portfolio = MagicMock()
        mock_portfolio.positions = {}
        mock_portfolio.trades = []
        mock_portfolio.get_summary.return_value = {
            "cash": 8000.0,
            "positions_value": 2000.0,
            "total_value": 10000.0,
            "total_return_pct": 0.0,
            "total_realized_pnl": 0.0,
            "total_unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "num_positions": 1,
            "positions": [
                {
                    "ticker": "SPY",
                    "quantity": 20.0,
                    "avg_price": 100.0,
                    "current_price": 103.0,
                    "market_value": 2000.0,
                    "unrealized_pnl_pct": 3.0,
                }
            ],
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
        mock_regime_state.volatility_regime = "normal"
        mock_regime_state.trend_regime = "neutral"
        mock_regime_state.correlation_regime = "normal"
        mock_regime_state.volatility_percentile = 50.0
        mock_regime_state.adx_value = 20.0
        mock_regime_state.avg_correlation = 0.5

        mock_regime_detector = MagicMock()
        mock_regime_detector.return_value.analyze.return_value = mock_regime_state
        mock_regime_detector.return_value.get_strategy_recommendation.return_value = {
            "position_sizing": "normal",
            "stop_loss_tightening": False,
            "mean_reversion_opportunities": False,
            "trend_following": False,
            "reduce_correlated_exposure": False,
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

        class _FixedNow:
            def __init__(self, when):
                self._when = when

            def now(self):
                return self._when

            def strftime(self, fmt):
                return self._when.strftime(fmt)

            def isoformat(self):
                return self._when.isoformat()

        fixed_date = datetime(2026, 8, 12, 10, 30, 0)

        patches = {
            "REPO_ROOT": tmp_path,
            "DATA_DIR": data_dir,
            "DAILY_RESULTS_DIR": results_dir,
            "fetch_historical_data": MagicMock(return_value=market_data),
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
            "format_regime_for_llm": MagicMock(return_value="\n## Market Regime Analysis\n"),
            "datetime": _FixedNow(fixed_date),
        }

        with patch.multiple("daily_run", **patches):
            run_daily_pipeline(dry_run=False, no_overwrite=False)

        # Agent must have been called exactly once
        mock_agent.get_trading_decision.assert_called_once()
        call_args = mock_agent.get_trading_decision.call_args
        passed_market_analysis = call_args[0][0]
        passed_portfolio_summary = call_args[0][1]

        # Regime analysis must be present in the market analysis passed to the LLM
        assert "regime" in passed_market_analysis
        assert passed_market_analysis["regime"] is not None

        # Risk metrics computed in the pipeline must survive into the agent call
        assert "risk_metrics" in passed_portfolio_summary
        risk_metrics = passed_portfolio_summary["risk_metrics"]
        assert risk_metrics["cvar_95"] == -0.02
        assert risk_metrics["var_95"] == -0.015
        assert risk_metrics["max_drawdown"] == -0.04

    def test_risk_metrics_omitted_when_no_positions(self, tmp_path, monkeypatch):
        """When there are no positions, risk_metrics are not added and should not be present."""
        results_dir = tmp_path / "results" / "daily"
        results_dir.mkdir(parents=True)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        monkeypatch.chdir(tmp_path)

        market_data = {"SPY": pd.DataFrame({"Close": [100.0, 101.0]})}
        market_analysis = {
            "assets": {"SPY": {"latest": {"price": 101.0}, "returns": [0.01]}},
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

            def strftime(self, fmt):
                return self._when.strftime(fmt)

            def isoformat(self):
                return self._when.isoformat()

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

        call_args = mock_agent.get_trading_decision.call_args
        passed_portfolio_summary = call_args[0][1]
        assert "risk_metrics" not in passed_portfolio_summary

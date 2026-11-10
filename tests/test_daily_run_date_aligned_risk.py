"""Date-aligned portfolio risk/performance wiring in run_daily_pipeline.

The 'returns' producer (analyze_market_data) emits per-ticker returns as a
date-stripped dropna() list.  Pairing those lists positionally silently
misprices VaR/CVaR/tail and Sharpe/beta when the underlying calendars
diverge (NYSE vs Euronext holidays in a mixed US/EUR book).  These tests pin
the date-aligned path: real calculate_portfolio_cvar / tail_risk_analysis run
on constructed divergent-calendar data, and the numbers that reach the LLM
prompt must be the date-joined ones — not the positional zip.

Discriminator doctrine (PR #84): tests drive the production wiring through
run_daily_pipeline and assert on computed values, so they fail on the old
positional code with a visible symptom instead of erroring at import time.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(REPO_SRC))

from daily_run import run_daily_pipeline  # noqa: E402
from risk.cvar import calculate_portfolio_cvar, tail_risk_analysis  # noqa: E402


def _helper():
    """Lazy import (PR #84 doctrine): helper tests must not kill collection
    of the integration discriminators on pre-fix code."""
    from daily_run import _date_aligned_portfolio_series
    return _date_aligned_portfolio_series


# ---------------------------------------------------------------------------
# Deterministic divergent-calendar fixtures
# ---------------------------------------------------------------------------

BASE_DATES = pd.date_range("2026-06-01", periods=40, freq="B")
# MC.PA (Euronext) misses three scattered sessions that SPY/TLT trade.
PA_MISSING = [BASE_DATES[10], BASE_DATES[20], BASE_DATES[30]]
PA_DATES = BASE_DATES.drop(PA_MISSING)

SPY_VALUES = np.array([0.010, -0.004, 0.006, 0.002, -0.008] * 8)          # 40
PA_VALUES = np.array(
    [(0.020 + 0.001 * (i % 7)) if i % 4 != 2 else -0.015 for i in range(37)]
)
TLT_VALUES = np.array([0.003, 0.001, -0.002, 0.004, 0.0005] * 8)          # 40


def _frame(dates, values):
    return pd.DataFrame({"Daily_Return": values}, index=pd.DatetimeIndex(dates))


SPY_FRAME = _frame(BASE_DATES, SPY_VALUES)
PA_FRAME = _frame(PA_DATES, PA_VALUES)
TLT_FRAME = _frame(BASE_DATES, TLT_VALUES)

# SPY benchmark frame that misses ONE day of the TLT∩MC.PA joint calendar.
SPY_BENCH_MISSING = BASE_DATES[15]
SPY_BENCH_DATES = BASE_DATES.drop([SPY_BENCH_MISSING])
SPY_BENCH_VALUES = np.array([0.008, -0.003, 0.005, 0.001, -0.006] * 8)[:39]
SPY_BENCH_FRAME = _frame(SPY_BENCH_DATES, SPY_BENCH_VALUES)


def _asset(df):
    series = df["Daily_Return"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "latest": {"price": 100.0},
        "returns": series.tolist(),
        "dataframe": df,
    }


def _expected_aligned(frames, weights):
    """Independently compute the date-aligned raw weighted portfolio series."""
    cols = {t: f["Daily_Return"].replace([np.inf, -np.inf], np.nan) for t, f in frames.items()}
    joined = pd.DataFrame(cols).dropna()
    return (joined * pd.Series(weights)).sum(axis=1).to_numpy(dtype=float), joined.index


def _legacy_zip(position_returns, weights):
    min_len = min(len(r) for r in position_returns.values())
    series = np.zeros(min_len)
    for t, r in position_returns.items():
        series += np.asarray(r)[-min_len:] * weights[t]
    return series


class _FixedNow:
    def __init__(self, when):
        self._when = when

    def now(self):
        return self._when

    def strftime(self, fmt):
        return self._when.strftime(fmt)

    def isoformat(self):
        return self._when.isoformat()


def _run_pipeline(tmp_path, monkeypatch, market_analysis, market_data, positions,
                  patch_perf=False):
    """Run run_daily_pipeline with proven mock scaffolding.

    Uses the REAL calculate_portfolio_cvar / tail_risk_analysis so computed
    risk numbers are asserted end-to-end.  Returns (mock_agent, calc_mock).
    """
    results_dir = tmp_path / "results" / "daily"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    total_value = 10000.0
    portfolio_summary = {
        "cash": 1000.0,
        "positions_value": 9000.0,
        "total_value": total_value,
        "total_return_pct": 0.0,
        "total_realized_pnl": 0.0,
        "total_unrealized_pnl": 0.0,
        "total_pnl": 0.0,
        "num_positions": len(positions),
        "positions": positions,
    }

    mock_portfolio = MagicMock()
    mock_portfolio.positions = {}
    mock_portfolio.trades = []
    mock_portfolio.get_summary.return_value = portfolio_summary

    mock_agent = MagicMock()
    mock_agent.get_trading_decision.return_value = {
        "reasoning": "HOLD for test",
        "actions": [],
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

    mock_benchmark = MagicMock()
    mock_benchmark.rebalance.return_value = {
        "total_value": 10000.0,
        "total_return_pct": 0.0,
        "num_positions": 0,
    }

    fixed_date = datetime(2026, 8, 12, 10, 30, 0)

    patches = {
        "REPO_ROOT": tmp_path,
        "DATA_DIR": data_dir,
        "DAILY_RESULTS_DIR": results_dir,
        "fetch_historical_data": MagicMock(return_value=market_data),
        "analyze_market_data": MagicMock(return_value=market_analysis),
        "fetch_current_prices": MagicMock(return_value={"SPY": 100.0}),
        "Portfolio": MagicMock(return_value=mock_portfolio),
        "TradingAgent": MagicMock(return_value=mock_agent),
        "PositionCooldownManager": MagicMock(return_value=mock_cooldown_mgr),
        "CooldownConfig": MagicMock(),
        "LiveEqualWeightBenchmark": MagicMock(return_value=mock_benchmark),
        "RegimeDetector": mock_regime_detector,
        "format_regime_for_llm": MagicMock(return_value="\n## Market Regime Analysis\n"),
        "datetime": _FixedNow(fixed_date),
    }
    calc_mock = None
    if patch_perf:
        calc_mock = MagicMock()
        perf = MagicMock()
        perf.sharpe_ratio = 1.0
        perf.sortino_ratio = 1.0
        perf.calmar_ratio = 1.0
        perf.volatility = 0.1
        perf.beta = 1.0
        perf.alpha = 0.0
        perf.treynor_ratio = 1.0
        perf.information_ratio = 0.5
        perf.tracking_error = 0.02
        perf.max_drawdown = -0.05
        perf.annualized_return = 0.08
        calc_mock.return_value = perf
        patches["calculate_all_metrics"] = calc_mock

    with patch.multiple("daily_run", **patches):
        run_daily_pipeline(dry_run=False, no_overwrite=False)

    return mock_agent, calc_mock


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestDateAlignedPortfolioSeries:
    def test_returns_none_for_legacy_dateless_assets(self):
        assets = {
            "SPY": {"latest": {"price": 100.0}, "returns": [0.01, 0.02]},
            "TLT": {"latest": {"price": 100.0}, "returns": [0.01, 0.02]},
        }
        result = _helper()(
            assets,
            {"SPY": np.array([0.01, 0.02]), "TLT": np.array([0.01, 0.02])},
            {"SPY": 0.5, "TLT": 0.5},
        )
        assert result is None

    def test_returns_none_for_non_datetime_index(self):
        df = pd.DataFrame({"Daily_Return": [0.01, 0.02]}, index=[0, 1])
        assets = {"SPY": {"dataframe": df, "returns": [0.01, 0.02]}}
        result = _helper()(
            assets, {"SPY": np.array([0.01, 0.02])}, {"SPY": 1.0}
        )
        assert result is None

    def test_inner_joins_and_weight_sums_concrete_values(self):
        dates_a = pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"])
        dates_b = pd.to_datetime(["2026-06-02", "2026-06-03", "2026-06-04"])
        assets = {
            "A": {"dataframe": _frame(dates_a, [0.01, 0.02, 0.03])},
            "B": {"dataframe": _frame(dates_b, [0.10, 0.20, 0.30])},
        }
        result = _helper()(
            assets,
            {"A": np.array([0.01, 0.02, 0.03]), "B": np.array([0.10, 0.20, 0.30])},
            {"A": 0.5, "B": 0.5},
        )
        assert result is not None
        assert list(result.index) == list(pd.to_datetime(["2026-06-02", "2026-06-03"]))
        np.testing.assert_allclose(result.to_numpy(), [0.06, 0.115])

    def test_disjoint_calendars_return_empty_series(self):
        dates_a = pd.to_datetime(["2026-06-01", "2026-06-02"])
        dates_b = pd.to_datetime(["2026-06-03", "2026-06-04"])
        assets = {
            "A": {"dataframe": _frame(dates_a, [0.01, 0.02])},
            "B": {"dataframe": _frame(dates_b, [0.10, 0.20])},
        }
        result = _helper()(
            assets,
            {"A": np.array([0.01, 0.02]), "B": np.array([0.10, 0.20])},
            {"A": 0.5, "B": 0.5},
        )
        assert result is not None
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Integration discriminators — real risk functions through the pipeline
# ---------------------------------------------------------------------------

class TestDateAlignedRiskReachesLLM:
    def test_cvar_uses_date_alignment_not_positional_zip(self, tmp_path, monkeypatch):
        weights = {"SPY": 0.54, "MC.PA": 0.36}  # €9000 invested of €10000
        positions = [
            {"ticker": "SPY", "quantity": 54.0, "avg_price": 100.0,
             "current_price": 100.0, "market_value": 5400.0, "unrealized_pnl_pct": 0.0},
            {"ticker": "MC.PA", "quantity": 36.0, "avg_price": 100.0,
             "current_price": 100.0, "market_value": 3600.0, "unrealized_pnl_pct": 0.0},
        ]
        market_analysis = {
            "assets": {"SPY": _asset(SPY_FRAME), "MC.PA": _asset(PA_FRAME)},
            "analysis_date": "2026-08-12",
        }
        market_data = {"SPY": SPY_FRAME, "MC.PA": PA_FRAME}

        mock_agent, _ = _run_pipeline(
            tmp_path, monkeypatch, market_analysis, market_data, positions
        )

        passed_summary = mock_agent.get_trading_decision.call_args[0][1]
        risk_metrics = passed_summary["risk_metrics"]

        expected_aligned, _ = _expected_aligned(
            {"SPY": SPY_FRAME, "MC.PA": PA_FRAME}, weights
        )
        invested = sum(weights.values())
        expected_cvar = calculate_portfolio_cvar(
            {"portfolio": expected_aligned / invested}, {"portfolio": 1.0}
        )
        expected_tail = tail_risk_analysis(expected_aligned, None)

        # Vacuity guard: the two pairings must actually diverge on this data.
        legacy_series = _legacy_zip(
            {"SPY": np.array(_asset(SPY_FRAME)["returns"]),
             "MC.PA": np.array(_asset(PA_FRAME)["returns"])},
            weights,
        )
        assert not np.allclose(legacy_series, expected_aligned)

        legacy_position_returns = {
            "SPY": np.array(_asset(SPY_FRAME)["returns"]),
            "MC.PA": np.array(_asset(PA_FRAME)["returns"]),
        }
        legacy_cvar = calculate_portfolio_cvar(legacy_position_returns, weights)
        legacy_tail = tail_risk_analysis(legacy_series, None)
        # The 5% tail on n=36 can coincide between pairings (same worst day
        # dominates), so cvar_95 alone cannot discriminate — skewness and
        # sortino are whole-distribution fingerprints that must differ.
        assert risk_metrics["skewness"] != pytest.approx(legacy_tail["skewness"], abs=1e-9)
        assert risk_metrics["sortino_ratio"] != pytest.approx(legacy_tail["sortino_ratio"], abs=1e-9)

        # The numbers reaching the LLM must be the date-aligned ones.
        assert risk_metrics["cvar_95"] == pytest.approx(expected_cvar.cvar_95, abs=1e-12)
        assert risk_metrics["var_95"] == pytest.approx(expected_cvar.var_95, abs=1e-12)
        assert risk_metrics["skewness"] == pytest.approx(expected_tail["skewness"], abs=1e-12)
        assert risk_metrics["sortino_ratio"] == pytest.approx(expected_tail["sortino_ratio"], abs=1e-12)
        assert risk_metrics["max_drawdown"] == pytest.approx(
            expected_tail["max_drawdown"], abs=1e-12
        )

    def test_same_calendar_parity_with_legacy_zip(self, tmp_path, monkeypatch):
        """Healthy pin: identical calendars must reproduce legacy numbers."""
        weights = {"SPY": 0.54, "TLT": 0.36}
        positions = [
            {"ticker": "SPY", "quantity": 54.0, "avg_price": 100.0,
             "current_price": 100.0, "market_value": 5400.0, "unrealized_pnl_pct": 0.0},
            {"ticker": "TLT", "quantity": 36.0, "avg_price": 100.0,
             "current_price": 100.0, "market_value": 3600.0, "unrealized_pnl_pct": 0.0},
        ]
        market_analysis = {
            "assets": {"SPY": _asset(SPY_FRAME), "TLT": _asset(TLT_FRAME)},
            "analysis_date": "2026-08-12",
        }
        market_data = {"SPY": SPY_FRAME, "TLT": TLT_FRAME}

        mock_agent, _ = _run_pipeline(
            tmp_path, monkeypatch, market_analysis, market_data, positions
        )
        risk_metrics = mock_agent.get_trading_decision.call_args[0][1]["risk_metrics"]

        spy_list = np.array(_asset(SPY_FRAME)["returns"])
        tlt_list = np.array(_asset(TLT_FRAME)["returns"])
        legacy_cvar = calculate_portfolio_cvar(
            {"SPY": spy_list, "TLT": tlt_list}, weights
        )
        assert risk_metrics["cvar_95"] == pytest.approx(legacy_cvar.cvar_95, abs=1e-12)

    def test_performance_metrics_use_full_joint_series_no_benchmark(self, tmp_path, monkeypatch):
        """Relative metrics moved off the nightly window (2026-11-10): the
        ~22-session fetch can never satisfy the estimators' n>=30 guards, so
        calculate_all_metrics must receive NO benchmark here, and the
        absolute metrics must keep the FULL joint series — the SPY pairwise
        mask that used to drop SPY-missing days only existed for beta/alpha
        pairing and is gone with them."""
        weights = {"TLT": 0.45, "MC.PA": 0.45}
        positions = [
            {"ticker": "TLT", "quantity": 45.0, "avg_price": 100.0,
             "current_price": 100.0, "market_value": 4500.0, "unrealized_pnl_pct": 0.0},
            {"ticker": "MC.PA", "quantity": 45.0, "avg_price": 100.0,
             "current_price": 100.0, "market_value": 4500.0, "unrealized_pnl_pct": 0.0},
        ]
        market_analysis = {
            "assets": {
                "TLT": _asset(TLT_FRAME),
                "MC.PA": _asset(PA_FRAME),
                "SPY": _asset(SPY_BENCH_FRAME),
            },
            "analysis_date": "2026-08-12",
        }
        market_data = {"TLT": TLT_FRAME, "MC.PA": PA_FRAME, "SPY": SPY_BENCH_FRAME}

        _, calc_mock = _run_pipeline(
            tmp_path, monkeypatch, market_analysis, market_data, positions,
            patch_perf=True,
        )
        assert calc_mock is not None and calc_mock.called

        portfolio_arg = calc_mock.call_args[0][0]
        benchmark_arg = calc_mock.call_args.kwargs["benchmark_returns"]

        expected_aligned, joint_index = _expected_aligned(
            {"TLT": TLT_FRAME, "MC.PA": PA_FRAME}, weights
        )
        # SPY misses one day of the joint calendar — with the benchmark gone
        # from this path, that day must STAY in the absolute-metrics series
        # (full joint calendar, consistent with the 3.5 risk block).
        assert SPY_BENCH_MISSING in joint_index
        assert benchmark_arg is None
        np.testing.assert_allclose(portfolio_arg, expected_aligned)
        assert len(portfolio_arg) == len(joint_index)

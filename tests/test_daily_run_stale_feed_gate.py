"""Regression tests: stale-feed gate in the nightly pipeline decision layer.

Background — the 2026-11-05 production incident (see monitor.py freshness
guard and memory/trading-2026-11-13.md): the yfinance Euronext feed went
stale overnight ("possibly delisted" errors on AI.PA / MC.PA / ^FCHI, 4th
occurrence in the week of 2026-11-09..13) while the intraday monitor's
freshness guard correctly excluded the stale tickers and kept the saved
references intact. The nightly pipeline (``run_daily_pipeline``, 21:00 UTC)
had NO equivalent check: a fetch failure silently shrinks the analyzed
universe, and a frozen-but-served frame looks like a normal close. The LLM
would then decide — and the executor trade — on prices that miss required
trading sessions. The monitor's doctrine was explicit: *"le guard protège
le monitor, pas le pipeline"* (2026-11-13 morning session note).

The gate: after the market-data fetch, every expected universe ticker is
classified fresh/stale by the same session arithmetic the monitor uses
(shared ``data.feed_freshness`` module). Stale non-position tickers get a
loud banner and a result-log record only. If ANY held position's feed is
stale or missing, the decision layer degrades to a forced HOLD: the LLM is
not consulted, the H4 shadow arm is skipped (no API burn on poisoned
data), no order reaches the executor, and the run is marked
``stale_feed.gate_active`` in the result JSON for offline audit.

These tests follow the PR #101 clock-freezing pattern: ``daily_run.datetime``
is replaced by a stand-in whose ``now()`` is pinned, so the gate sees the
same wall clock as the rest of the pipeline.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.feed_freshness import (  # noqa: E402
    classify_daily_frames,
    session_date_of_daily_bar,
)
from daily_run import run_daily_pipeline  # noqa: E402


# --- Unit tests: daily-bar session mapping ---------------------------------

class TestSessionDateOfDailyBar:
    """Euronext daily bars are anchored at local midnight: 23:00 UTC the
    previous day in winter (CET), 22:00 UTC in summer (CEST). Both anchors
    are evening stamps, so any stamp at/after 20:00 UTC belongs to the NEXT
    calendar day's session. US daily bars are anchored at midnight ET
    (04:00/05:00 UTC of the session date itself) — always their own date.
    """

    def test_euronext_winter_anchor_2300_rolls_to_next_day(self):
        ts = pd.Timestamp("2026-11-12 23:00:00")
        assert session_date_of_daily_bar(ts, "euronext").isoformat() == "2026-11-13"

    def test_euronext_summer_anchor_2200_rolls_to_next_day(self):
        # Summer DST anchor — the intraday-quote 22:30 boundary in monitor.py
        # would misclassify this one; the daily-frame boundary must be wider.
        ts = pd.Timestamp("2026-07-14 22:00:00")
        assert session_date_of_daily_bar(ts, "euronext").isoformat() == "2026-07-15"

    def test_euronext_morning_stamp_maps_to_own_date(self):
        ts = pd.Timestamp("2026-11-12 09:00:00")
        assert session_date_of_daily_bar(ts, "euronext").isoformat() == "2026-11-12"

    def test_us_daily_bar_is_own_date(self):
        ts = pd.Timestamp("2026-11-13 05:00:00")
        assert session_date_of_daily_bar(ts, "us").isoformat() == "2026-11-13"


class TestClassifyDailyFrames:
    """classify_daily_frames mirrors the monitor's split_fresh_and_stale
    doctrine for bulk daily OHLCV frames: missing frame, empty frame, and
    stale last bar are all stale; each carries a human-readable reason."""

    def _frame(self, last_ts: str) -> pd.DataFrame:
        last = pd.Timestamp(last_ts)
        idx = [last - pd.Timedelta(days=2), last - pd.Timedelta(days=1), last]
        return pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)

    def test_missing_frame_is_stale(self):
        stale = classify_daily_frames(["SPY"], {}, datetime(2026, 11, 13, 21, 0))
        assert stale == {"SPY": "missing_frame"}

    def test_empty_frame_is_stale(self):
        stale = classify_daily_frames(
            ["SPY"], {"SPY": pd.DataFrame()}, datetime(2026, 11, 13, 21, 0)
        )
        assert stale == {"SPY": "empty_frame"}

    def test_non_frame_value_is_stale(self):
        # A present-but-unusable entry (not a DataFrame) is an empty frame;
        # a None/absent entry is a missing frame (both stale, reasons differ).
        stale = classify_daily_frames(
            ["SPY"], {"SPY": pd.Series(dtype=float)}, datetime(2026, 11, 13, 21, 0)
        )
        assert stale == {"SPY": "empty_frame"}

    def test_frozen_euronext_feed_is_stale_us_fresh(self):
        # Incident replay, 2026-11-13 21:00 UTC Friday: Paris frozen since
        # Wednesday's close (last bar anchored Tue 23:00 UTC -> session Wed
        # Nov 11); US served Friday's bar. Expected: AI.PA stale, SPY fresh.
        frames = {
            "AI.PA": self._frame("2026-11-11 23:00"),
            "SPY": self._frame("2026-11-13 05:00"),
        }
        stale = classify_daily_frames(
            ["AI.PA", "SPY"], frames, datetime(2026, 11, 13, 21, 0)
        )
        assert "AI.PA" in stale and "stale_bar" in stale["AI.PA"]
        assert "SPY" not in stale

    def test_friday_evening_both_markets_fresh(self):
        frames = {
            "AI.PA": self._frame("2026-11-12 23:00"),  # session Fri Nov 13
            "SPY": self._frame("2026-11-13 05:00"),
        }
        stale = classify_daily_frames(
            ["AI.PA", "SPY"], frames, datetime(2026, 11, 13, 21, 0)
        )
        assert stale == {}

    def test_reason_carries_last_bar_timestamp(self):
        frames = {"SPY": self._frame("2026-11-10 05:00")}
        stale = classify_daily_frames(["SPY"], frames, datetime(2026, 11, 13, 21, 0))
        assert "2026-11-10" in stale["SPY"]


# --- Integration tests: the gate inside run_daily_pipeline -----------------

class _FixedNow:
    """Simple stand-in for datetime in the pipeline (PR #101 pattern)."""

    def __init__(self, when: datetime):
        self._when = when

    def now(self):
        return self._when

    def strftime(self, fmt):
        return self._when.strftime(fmt)

    def isoformat(self):
        return self._when.isoformat()


def _frame(last_ts: str) -> pd.DataFrame:
    last = pd.Timestamp(last_ts)
    idx = [last - pd.Timedelta(days=2), last - pd.Timedelta(days=1), last]
    return pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)


def _patch_pipeline(tmp_path, monkeypatch, market_data, position_tickers,
                    expected_tickers=None):
    """Patch the network/IO surface of run_daily_pipeline so it runs offline.

    Friday 2026-11-13 21:00 UTC — the canonical nightly run time. With a
    healthy feed, Friday's bars are fresh for both markets (Euronext via
    the Thu-23:00 anchor mapping, US directly). ``expected_tickers`` is the
    universe the freshness gate classifies; it defaults to the served
    frames, pass a wider list to simulate a fetch that DROPPED tickers.
    """
    fixed_date = datetime(2026, 11, 13, 21, 0, 0)
    results_dir = tmp_path / "results" / "daily"
    results_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "experiments.json").write_text(
        json.dumps({"midband_deployment": {"mode": "shadow"}})
    )

    monkeypatch.chdir(tmp_path)

    assets = {}
    for ticker in market_data:
        assets[ticker] = {"latest": {"price": 100.0}, "returns": [0.0, 0.0, 0.0]}
    market_analysis = {
        "assets": assets,
        "analysis_date": fixed_date.strftime("%Y-%m-%d"),
    }

    positions = {t: MagicMock() for t in position_tickers}
    mock_portfolio = MagicMock()
    mock_portfolio.positions = positions
    mock_portfolio.trades = []
    mock_portfolio.get_summary.return_value = {
        "cash": 10000.0,
        "positions_value": 0.0,
        "total_value": 10000.0,
        "total_return_pct": 0.0,
        "total_realized_pnl": 0.0,
        "total_unrealized_pnl": 0.0,
        "total_pnl": 0.0,
        "num_positions": len(positions),
        "positions": [],
    }

    mock_agent = MagicMock()
    mock_agent.get_trading_decision.return_value = {
        "reasoning": "HOLD for test",
        "actions": [],
        "error": False,
    }

    mock_cooldown_mgr = MagicMock()
    mock_cooldown_mgr.get_status.return_value = {
        "trades_this_week": 0,
        "weekly_cap": 3,
        "adaptive_stop_loss": 5.0,
        "active_entries": {},
    }

    mock_regime_detector = MagicMock()
    mock_regime_state = MagicMock()
    mock_regime_state.summary.return_value = "test-regime"
    mock_regime_state.volatility_regime = "normal"
    mock_regime_state.trend_regime = "neutral"
    mock_regime_state.correlation_regime = "normal"
    mock_regime_state.volatility_percentile = 0.5
    mock_regime_state.adx_value = 20.0
    mock_regime_state.avg_correlation = 0.3
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
        # Classify exactly the served universe (unless the test widens it)
        # so "missing" reflects the scenario under test, not the full
        # production ticker list.
        "ALL_TICKERS": list(expected_tickers or market_data.keys()),
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

    import daily_run
    for name, value in patches.items():
        monkeypatch.setattr(daily_run, name, value)

    mock_shadow = MagicMock()
    monkeypatch.setattr(daily_run, "run_shadow_arm", mock_shadow)
    return mock_agent, mock_shadow, results_dir


def _load_result(results_dir):
    files = list(results_dir.glob("*.json"))
    assert len(files) == 1, f"expected exactly one result file, got {files}"
    return json.loads(files[0].read_text())


class TestStaleFeedGate:
    FRESH_FRIDAY = {
        "SPY": _frame("2026-11-13 05:00"),
        "AI.PA": _frame("2026-11-12 23:00"),
    }

    def test_fresh_feed_asks_llm_and_no_gate(self, tmp_path, monkeypatch):
        mock_agent, mock_shadow, results_dir = _patch_pipeline(
            tmp_path, monkeypatch, dict(self.FRESH_FRIDAY), ["SPY"]
        )
        run_daily_pipeline(dry_run=True)
        assert mock_agent.get_trading_decision.call_count == 1
        result = _load_result(results_dir)
        assert result["stale_feed"]["gate_active"] is False
        assert result["stale_feed"]["stale_tickers"] == {}

    def test_stale_position_frame_forces_hold_and_skips_llm(self, tmp_path, monkeypatch):
        market_data = dict(self.FRESH_FRIDAY)
        market_data["SPY"] = _frame("2026-11-11 05:00")  # session Wed, run is Fri
        mock_agent, mock_shadow, results_dir = _patch_pipeline(
            tmp_path, monkeypatch, market_data, ["SPY"]
        )
        run_daily_pipeline(dry_run=True)
        assert mock_agent.get_trading_decision.call_count == 0
        assert mock_shadow.call_count == 0
        result = _load_result(results_dir)
        assert result["stale_feed"]["gate_active"] is True
        assert result["stale_feed"]["gate_tickers"] == ["SPY"]
        assert "stale_bar" in result["stale_feed"]["stale_tickers"]["SPY"]
        assert "STALE FEED" in result["decision"]["reasoning"]
        assert result["executed_trades"] == []

    def test_missing_position_frame_forces_hold(self, tmp_path, monkeypatch):
        # The live incident signature: "possibly delisted" -> no frame at all.
        market_data = {"AI.PA": self.FRESH_FRIDAY["AI.PA"]}  # SPY absent
        mock_agent, _, results_dir = _patch_pipeline(
            tmp_path, monkeypatch, market_data, ["SPY"],
            expected_tickers=["SPY", "AI.PA"],
        )
        run_daily_pipeline(dry_run=True)
        assert mock_agent.get_trading_decision.call_count == 0
        result = _load_result(results_dir)
        assert result["stale_feed"]["gate_active"] is True
        assert result["stale_feed"]["stale_tickers"]["SPY"] == "missing_frame"
        assert result["executed_trades"] == []

    def test_stale_euronext_position_gates_while_us_fresh(self, tmp_path, monkeypatch):
        market_data = {
            "SPY": self.FRESH_FRIDAY["SPY"],
            "AI.PA": _frame("2026-11-11 23:00"),  # frozen since Wed close
        }
        mock_agent, _, results_dir = _patch_pipeline(
            tmp_path, monkeypatch, market_data, ["AI.PA"]
        )
        run_daily_pipeline(dry_run=True)
        assert mock_agent.get_trading_decision.call_count == 0
        result = _load_result(results_dir)
        assert result["stale_feed"]["gate_active"] is True
        assert result["stale_feed"]["gate_tickers"] == ["AI.PA"]

    def test_stale_non_position_ticker_banners_without_gating(self, tmp_path, monkeypatch):
        market_data = {
            "SPY": self.FRESH_FRIDAY["SPY"],
            "AI.PA": _frame("2026-11-11 23:00"),  # stale but NOT held
        }
        mock_agent, mock_shadow, results_dir = _patch_pipeline(
            tmp_path, monkeypatch, market_data, ["SPY"]
        )
        run_daily_pipeline(dry_run=True)
        assert mock_agent.get_trading_decision.call_count == 1
        result = _load_result(results_dir)
        assert result["stale_feed"]["gate_active"] is False
        assert result["stale_feed"]["gate_tickers"] == []
        assert "AI.PA" in result["stale_feed"]["stale_tickers"]

    def test_gate_skips_shadow_arm_in_live_run(self, tmp_path, monkeypatch):
        market_data = {"SPY": _frame("2026-11-11 05:00")}
        _, mock_shadow, results_dir = _patch_pipeline(
            tmp_path, monkeypatch, market_data, ["SPY"]
        )
        run_daily_pipeline(dry_run=False)
        assert mock_shadow.call_count == 0
        result = _load_result(results_dir)
        assert result["stale_feed"]["gate_active"] is True

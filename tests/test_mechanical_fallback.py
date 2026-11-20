"""Tests for the mechanical fallback rulebook (src/mechanical_fallback.py).

The rulebook is the codified 2026-11-20 exit-rule review: on LLM outage
days it must close only positions that have breached the pre-registered
adaptive stop on a fresh price, flag everything else, and never buy.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mechanical_fallback import (
    CASH_BAND_FRACTION,
    DEFAULT_WARNING_BAND_PCT,
    run_mechanical_rulebook,
)


def _pos(ticker, avg_price, market_value=1000.0):
    return {"ticker": ticker, "avg_price": avg_price, "market_value": market_value}


class TestMF1HardStopExit:
    def test_drawdown_at_stop_boundary_exits(self):
        # drawdown exactly == -stop is a breach (<= comparison, same as
        # PositionCooldownManager.can_sell's stop-loss override).
        result = run_mechanical_rulebook(
            positions=[_pos("SPY", 100.0)],
            current_prices={"SPY": 95.0},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        assert [a["ticker"] for a in result["actions"]] == ["SPY"]
        assert result["actions"] == [{"ticker": "SPY", "action": "sell", "pct": 100}]
        assert result["stop_exits"][0]["drawdown_pct"] == pytest.approx(-5.0)

    def test_drawdown_beyond_stop_exits(self):
        result = run_mechanical_rulebook(
            positions=[_pos("IJR", 139.02)],
            current_prices={"IJR": 131.0},  # ~ -5.77 %
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        assert result["actions"][0]["ticker"] == "IJR"
        assert result["stop_exits"][0]["drawdown_pct"] == pytest.approx(-5.769, abs=1e-3)

    def test_drawdown_above_stop_holds(self):
        result = run_mechanical_rulebook(
            positions=[_pos("SPY", 100.0)],
            current_prices={"SPY": 96.0},  # -4 %, inside the band
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        assert result["actions"] == []
        assert result["stop_exits"] == []

    def test_winner_never_exits(self):
        result = run_mechanical_rulebook(
            positions=[_pos("AI.PA", 170.0)],
            current_prices={"AI.PA": 187.0},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        assert result["actions"] == []
        assert result["watch"] == []

    def test_stale_feed_ticker_is_excluded(self):
        # The 2026-11-05 doctrine: a frozen close must never trigger an
        # exit, even when the frozen price is below the stop.
        result = run_mechanical_rulebook(
            positions=[_pos("SPY", 100.0)],
            current_prices={"SPY": 90.0},
            stale_tickers=["SPY"],
            adaptive_stop_pct=5.0,
        )
        assert result["actions"] == []
        assert result["stop_exits"] == []
        assert "stale" in result["skipped"]["SPY"]

    def test_missing_price_skips_loudly(self):
        result = run_mechanical_rulebook(
            positions=[_pos("SPY", 100.0)],
            current_prices={"SPY": None},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        assert result["actions"] == []
        assert "SPY" in result["skipped"]

    def test_absent_price_key_skips_loudly(self):
        result = run_mechanical_rulebook(
            positions=[_pos("SPY", 100.0)],
            current_prices={},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        assert result["actions"] == []
        assert "SPY" in result["skipped"]

    def test_non_finite_price_skips(self):
        for bad in (math.nan, math.inf, -math.inf, 0.0, -5.0, "95.0", True):
            result = run_mechanical_rulebook(
                positions=[_pos("SPY", 100.0)],
                current_prices={"SPY": bad},
                stale_tickers=[],
                adaptive_stop_pct=5.0,
            )
            assert result["actions"] == [], f"price {bad!r} must not trigger an exit"
            assert "SPY" in result["skipped"]

    def test_non_finite_avg_price_skips(self):
        result = run_mechanical_rulebook(
            positions=[_pos("SPY", math.nan)],
            current_prices={"SPY": 90.0},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        assert result["actions"] == []
        assert "SPY" in result["skipped"]

    def test_bool_positions_guarded(self):
        # bool is an int subclass in Python; must not be treated as a price.
        result = run_mechanical_rulebook(
            positions=[_pos("SPY", 100.0)],
            current_prices={"SPY": True},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        assert result["actions"] == []

    def test_multiple_positions_only_breaching_exit(self):
        result = run_mechanical_rulebook(
            positions=[
                _pos("AAA", 100.0),
                _pos("BBB", 200.0),
                _pos("CCC", 50.0),
            ],
            current_prices={"AAA": 94.0, "BBB": 210.0, "CCC": 47.0},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        # AAA: -6 % breach; BBB: winner; CCC: -6 % breach.
        assert sorted(a["ticker"] for a in result["actions"]) == ["AAA", "CCC"]
        assert all(a["action"] == "sell" and a["pct"] == 100 for a in result["actions"])

    def test_full_exit_pct_is_100(self):
        result = run_mechanical_rulebook(
            positions=[_pos("SPY", 100.0)],
            current_prices={"SPY": 90.0},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        assert result["actions"][0]["pct"] == 100

    def test_non_finite_stop_disables_exits_but_stays_active(self):
        result = run_mechanical_rulebook(
            positions=[_pos("SPY", 100.0)],
            current_prices={"SPY": 90.0},
            stale_tickers=[],
            adaptive_stop_pct=math.nan,
        )
        assert result["active"] is True
        assert result["actions"] == []
        assert result["stop_exits"] == []
        assert result["rules"]["mf1_hard_stop"]["threshold_pct"] is None


class TestMF2WatchBand:
    def test_position_inside_watch_band_reported_not_sold(self):
        result = run_mechanical_rulebook(
            positions=[_pos("SPY", 100.0)],
            current_prices={"SPY": 96.0},  # -4 %, 1 pp above the 5 % stop
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        assert result["actions"] == []
        assert [w["ticker"] for w in result["watch"]] == ["SPY"]
        assert result["watch"][0]["distance_to_stop_pp"] == pytest.approx(1.0)

    def test_watch_band_boundary_belongs_to_exit(self):
        # At exactly -stop the position is an exit (<=), not a watch item.
        result = run_mechanical_rulebook(
            positions=[_pos("SPY", 100.0)],
            current_prices={"SPY": 95.0},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        assert [w["ticker"] for w in result["watch"]] == []
        assert [a["ticker"] for a in result["actions"]] == ["SPY"]

    def test_custom_warning_band(self):
        result = run_mechanical_rulebook(
            positions=[_pos("SPY", 100.0)],
            current_prices={"SPY": 96.5},  # -3.5 %, 1.5 pp above the 5 % stop
            stale_tickers=[],
            adaptive_stop_pct=5.0,
            warning_band_pct=3.0,
        )
        assert result["actions"] == []
        assert [w["ticker"] for w in result["watch"]] == ["SPY"]
        assert result["watch"][0]["distance_to_stop_pp"] == pytest.approx(1.5)

    def test_outside_watch_band_not_reported(self):
        result = run_mechanical_rulebook(
            positions=[_pos("SPY", 100.0)],
            current_prices={"SPY": 90.0},  # -10 %, well past the stop
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        assert result["watch"] == []


class TestMF3CashMandate:
    def test_cash_above_band_flags_mandate(self):
        result = run_mechanical_rulebook(
            positions=[],
            current_prices={},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
            cash_pct=0.4166,
        )
        mandate = result["cash_mandate"]
        assert mandate["active"] is True
        assert mandate["side"] == "above"
        assert mandate["executed"] is False

    def test_cash_below_band_flags_mandate(self):
        result = run_mechanical_rulebook(
            positions=[],
            current_prices={},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
            cash_pct=0.10,
        )
        assert result["cash_mandate"]["active"] is True
        assert result["cash_mandate"]["side"] == "below"

    def test_cash_inside_band_no_mandate(self):
        result = run_mechanical_rulebook(
            positions=[],
            current_prices={},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
            cash_pct=0.218,
        )
        assert result["cash_mandate"]["active"] is False
        assert result["cash_mandate"]["side"] is None

    def test_mandate_never_produces_actions(self):
        # The rulebook must never buy: even a deep cash-band breach with
        # breaching positions yields sells only.
        result = run_mechanical_rulebook(
            positions=[_pos("SPY", 100.0)],
            current_prices={"SPY": 90.0},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
            cash_pct=0.45,
        )
        assert result["cash_mandate"]["active"] is True
        assert all(a["action"] != "buy" for a in result["actions"])
        assert [a["ticker"] for a in result["actions"]] == ["SPY"]

    def test_non_finite_cash_disables_mandate(self):
        for bad in (math.nan, math.inf, None, "0.4"):
            result = run_mechanical_rulebook(
                positions=[],
                current_prices={},
                stale_tickers=[],
                adaptive_stop_pct=5.0,
                cash_pct=bad,
            )
            assert result["cash_mandate"]["active"] is False
            assert result["cash_mandate"]["cash_pct"] is None

    def test_cash_band_constant_matches_prompt_rule(self):
        assert CASH_BAND_FRACTION == (0.15, 0.30)


class TestInvariants:
    def test_no_position_data_is_safe(self):
        result = run_mechanical_rulebook(
            positions=[],
            current_prices={},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        assert result["active"] is True
        assert result["actions"] == []
        assert result["stop_exits"] == []
        assert result["watch"] == []

    def test_none_positions_safe(self):
        result = run_mechanical_rulebook(
            positions=None,
            current_prices={},
            stale_tickers=None,
            adaptive_stop_pct=5.0,
        )
        assert result["actions"] == []

    def test_position_without_ticker_skipped_loudly(self):
        result = run_mechanical_rulebook(
            positions=[{"avg_price": 100.0}],
            current_prices={},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
        )
        assert result["actions"] == []
        assert any("ticker" in reason for reason in result["skipped"].values())

    def test_actions_contain_only_sell_pct100(self):
        result = run_mechanical_rulebook(
            positions=[_pos("AAA", 100.0), _pos("BBB", 50.0)],
            current_prices={"AAA": 90.0, "BBB": 47.0},
            stale_tickers=[],
            adaptive_stop_pct=5.0,
            cash_pct=0.5,
        )
        assert len(result["actions"]) == 2
        for action in result["actions"]:
            assert action["action"] == "sell"
            assert action["pct"] == 100
            assert set(action.keys()) == {"ticker", "action", "pct"}

    def test_default_warning_band_constant(self):
        assert DEFAULT_WARNING_BAND_PCT == 1.0

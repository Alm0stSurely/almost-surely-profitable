"""Comprehensive tests for churn_analysis.py.

Tests round-trip matching, churn metrics, holding period bucketing,
and action flip detection.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.churn_analysis import (
    RoundTrip,
    load_trades,
    load_decisions,
    match_round_trips,
    analyze_churn,
    print_report,
    _parse_trade_timestamp,
    _bucket_metrics,
    analyze_cohort,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trade(ticker, action, price, realized_pnl=0, timestamp=None, qty=10):
    if timestamp is None:
        timestamp = datetime(2026, 1, 1, 10, 0, 0).isoformat()
    return {
        "ticker": ticker,
        "action": action,
        "price": price,
        "realized_pnl": realized_pnl,
        "timestamp": timestamp,
        "qty": qty,
    }


def make_decision(timestamp, actions):
    return {
        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
        "actions": actions,
    }


# ---------------------------------------------------------------------------
# load_trades / load_decisions
# ---------------------------------------------------------------------------

class TestLoadTrades:
    def test_missing_file_returns_empty(self, tmp_path):
        with patch("analysis.churn_analysis.Path") as MockPath:
            MockPath.return_value.__truediv__ = lambda self, other: tmp_path / other
            MockPath.return_value.exists.return_value = False
            result = load_trades(str(tmp_path))
            assert result == []

    def test_loads_valid_json(self, tmp_path):
        trades = [make_trade("AAPL", "buy", 150.0)]
        path = tmp_path / "trades_history.json"
        path.write_text(json.dumps(trades))
        result = load_trades(str(tmp_path))
        assert result == trades

    def test_loads_empty_list(self, tmp_path):
        path = tmp_path / "trades_history.json"
        path.write_text("[]")
        result = load_trades(str(tmp_path))
        assert result == []


class TestLoadDecisions:
    def test_missing_file_returns_empty(self, tmp_path):
        with patch("analysis.churn_analysis.Path") as MockPath:
            MockPath.return_value.__truediv__ = lambda self, other: tmp_path / other
            MockPath.return_value.exists.return_value = False
            result = load_decisions(str(tmp_path))
            assert result == []

    def test_loads_valid_json(self, tmp_path):
        decisions = [make_decision(datetime(2026, 1, 1), [{"ticker": "AAPL", "action": "buy"}])]
        path = tmp_path / "decision_history.json"
        path.write_text(json.dumps(decisions))
        result = load_decisions(str(tmp_path))
        assert result == decisions


# ---------------------------------------------------------------------------
# match_round_trips
# ---------------------------------------------------------------------------

class TestMatchRoundTrips:
    def test_empty_trades_returns_empty(self):
        assert match_round_trips([]) == []

    def test_single_buy_no_sell_returns_empty(self):
        trades = [make_trade("AAPL", "buy", 150.0)]
        assert match_round_trips(trades) == []

    def test_buy_sell_pair(self):
        buy_dt = datetime(2026, 1, 1, 10, 0, 0)
        sell_dt = datetime(2026, 1, 5, 10, 0, 0)
        trades = [
            make_trade("AAPL", "buy", 150.0, timestamp=buy_dt.isoformat()),
            make_trade("AAPL", "sell", 160.0, realized_pnl=100.0, timestamp=sell_dt.isoformat()),
        ]
        rts = match_round_trips(trades)
        assert len(rts) == 1
        rt = rts[0]
        assert rt.ticker == "AAPL"
        assert rt.buy_price == 150.0
        assert rt.sell_price == 160.0
        assert rt.pnl == 100.0
        assert rt.hold_days == 4.0

    def test_multiple_tickers(self):
        trades = [
            make_trade("AAPL", "buy", 150.0, timestamp="2026-01-01T10:00:00"),
            make_trade("TSLA", "buy", 200.0, timestamp="2026-01-01T10:00:00"),
            make_trade("AAPL", "sell", 160.0, realized_pnl=10.0, timestamp="2026-01-02T10:00:00"),
            make_trade("TSLA", "sell", 210.0, realized_pnl=10.0, timestamp="2026-01-03T10:00:00"),
        ]
        rts = match_round_trips(trades)
        assert len(rts) == 2
        tickers = {rt.ticker for rt in rts}
        assert tickers == {"AAPL", "TSLA"}

    def test_fifo_matching(self):
        """FIFO: first buy matched with first sell."""
        trades = [
            make_trade("AAPL", "buy", 100.0, timestamp="2026-01-01T10:00:00"),
            make_trade("AAPL", "buy", 110.0, timestamp="2026-01-02T10:00:00"),
            make_trade("AAPL", "sell", 120.0, realized_pnl=20.0, timestamp="2026-01-03T10:00:00"),
            make_trade("AAPL", "sell", 130.0, realized_pnl=20.0, timestamp="2026-01-04T10:00:00"),
        ]
        rts = match_round_trips(trades)
        assert len(rts) == 2
        assert rts[0].buy_price == 100.0  # first buy
        assert rts[1].buy_price == 110.0  # second buy

    def test_more_sells_than_buys_ignored(self):
        """Extra sells without matching buys are skipped."""
        trades = [
            make_trade("AAPL", "buy", 150.0, timestamp="2026-01-01T10:00:00"),
            make_trade("AAPL", "sell", 160.0, realized_pnl=10.0, timestamp="2026-01-02T10:00:00"),
            make_trade("AAPL", "sell", 170.0, realized_pnl=10.0, timestamp="2026-01-03T10:00:00"),
        ]
        rts = match_round_trips(trades)
        assert len(rts) == 1

    def test_missing_realized_pnl_defaults_zero(self):
        trades = [
            make_trade("AAPL", "buy", 150.0, timestamp="2026-01-01T10:00:00"),
            {"ticker": "AAPL", "action": "sell", "price": 160.0, "timestamp": "2026-01-02T10:00:00"},
        ]
        rts = match_round_trips(trades)
        assert rts[0].pnl == 0

    def test_very_short_hold(self):
        buy_dt = datetime(2026, 1, 1, 10, 0, 0)
        sell_dt = datetime(2026, 1, 1, 22, 0, 0)  # 12 hours later
        trades = [
            make_trade("AAPL", "buy", 150.0, timestamp=buy_dt.isoformat()),
            make_trade("AAPL", "sell", 160.0, realized_pnl=10.0, timestamp=sell_dt.isoformat()),
        ]
        rts = match_round_trips(trades)
        assert rts[0].hold_days == 0.5

    def test_zero_second_hold(self):
        dt = datetime(2026, 1, 1, 10, 0, 0)
        trades = [
            make_trade("AAPL", "buy", 150.0, timestamp=dt.isoformat()),
            make_trade("AAPL", "sell", 160.0, realized_pnl=10.0, timestamp=dt.isoformat()),
        ]
        rts = match_round_trips(trades)
        assert rts[0].hold_days == 0.0


# ---------------------------------------------------------------------------
# analyze_churn
# ---------------------------------------------------------------------------

class TestAnalyzeChurnEmpty:
    def test_all_empty_returns_zeroed_metrics(self):
        metrics = analyze_churn([], [], [])
        assert metrics["total_round_trips"] == 0
        assert metrics["winning_round_trips"] == 0
        assert metrics["losing_round_trips"] == 0
        assert metrics["win_rate_pct"] == 0.0
        assert metrics["total_realized_pnl"] == 0.0
        assert metrics["avg_hold_days"] == 0.0
        assert metrics["action_flips"] == 0
        assert metrics["days_active"] == 1
        assert metrics["trades_per_week"] == 0.0
        assert metrics["annualized_turnover"] == 0.0


class TestAnalyzeChurnWinning:
    def test_all_winning(self):
        rt = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 5), 4.0, 100.0, 150.0, 160.0)
        metrics = analyze_churn([rt], [], [])
        assert metrics["winning_round_trips"] == 1
        assert metrics["losing_round_trips"] == 0
        assert metrics["win_rate_pct"] == 100.0
        assert metrics["total_realized_pnl"] == 100.0

    def test_all_losing(self):
        rt = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 5), 4.0, -50.0, 150.0, 140.0)
        metrics = analyze_churn([rt], [], [])
        assert metrics["winning_round_trips"] == 0
        assert metrics["losing_round_trips"] == 1
        assert metrics["win_rate_pct"] == 0.0
        assert metrics["total_realized_pnl"] == -50.0

    def test_mixed(self):
        rt_win = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 5), 4.0, 100.0, 150.0, 160.0)
        rt_loss = RoundTrip("TSLA", datetime(2026, 1, 1), datetime(2026, 1, 5), 4.0, -30.0, 200.0, 190.0)
        metrics = analyze_churn([rt_win, rt_loss], [], [])
        assert metrics["win_rate_pct"] == 50.0
        assert metrics["total_realized_pnl"] == 70.0


class TestAnalyzeChurnHoldingBuckets:
    def test_short_term_boundary(self):
        """Exactly 3 days → short term (<= 3)."""
        rt = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 4), 3.0, 10.0, 100.0, 110.0)
        metrics = analyze_churn([rt], [], [])
        assert metrics["short_term_count"] == 1
        assert metrics["medium_term_count"] == 0
        assert metrics["long_term_count"] == 0

    def test_medium_term_boundary_low(self):
        """Just over 3 days → medium term."""
        rt = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 5), 4.0, 10.0, 100.0, 110.0)
        metrics = analyze_churn([rt], [], [])
        assert metrics["short_term_count"] == 0
        assert metrics["medium_term_count"] == 1
        assert metrics["long_term_count"] == 0

    def test_medium_term_boundary_high(self):
        """Exactly 14 days → medium term (<= 14)."""
        rt = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 15), 14.0, 10.0, 100.0, 110.0)
        metrics = analyze_churn([rt], [], [])
        assert metrics["short_term_count"] == 0
        assert metrics["medium_term_count"] == 1
        assert metrics["long_term_count"] == 0

    def test_long_term_boundary(self):
        """Just over 14 days → long term."""
        rt = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 16), 15.0, 10.0, 100.0, 110.0)
        metrics = analyze_churn([rt], [], [])
        assert metrics["short_term_count"] == 0
        assert metrics["medium_term_count"] == 0
        assert metrics["long_term_count"] == 1

    def test_all_buckets_mixed(self):
        rt_short = RoundTrip("S", datetime(2026, 1, 1), datetime(2026, 1, 2), 1.0, 10.0, 100.0, 110.0)
        rt_med = RoundTrip("M", datetime(2026, 1, 1), datetime(2026, 1, 10), 9.0, -5.0, 100.0, 95.0)
        rt_long = RoundTrip("L", datetime(2026, 1, 1), datetime(2026, 1, 20), 19.0, 20.0, 100.0, 120.0)
        metrics = analyze_churn([rt_short, rt_med, rt_long], [], [])
        assert metrics["short_term_count"] == 1
        assert metrics["short_term_pnl"] == 10.0
        assert metrics["medium_term_count"] == 1
        assert metrics["medium_term_pnl"] == -5.0
        assert metrics["long_term_count"] == 1
        assert metrics["long_term_pnl"] == 20.0


class TestAnalyzeChurnWinRatesByBucket:
    def test_bucket_win_rate_calculation(self):
        rt1 = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 2), 1.0, 10.0, 100.0, 110.0)
        rt2 = RoundTrip("AAPL", datetime(2026, 1, 3), datetime(2026, 1, 4), 1.0, -5.0, 100.0, 95.0)
        metrics = analyze_churn([rt1, rt2], [], [])
        assert metrics["short_term_win_rate"] == 50.0
        assert metrics["short_term_pnl"] == 5.0

    def test_empty_bucket_win_rate(self):
        rt = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 20), 19.0, 10.0, 100.0, 110.0)
        metrics = analyze_churn([rt], [], [])
        assert metrics["short_term_win_rate"] == 0.0  # no short-term trips
        assert metrics["medium_term_win_rate"] == 0.0
        assert metrics["long_term_win_rate"] == 100.0


class TestAnalyzeChurnFlips:
    def test_no_flips_same_action(self):
        decisions = [
            make_decision("2026-01-01", [{"ticker": "AAPL", "action": "buy"}]),
            make_decision("2026-01-02", [{"ticker": "AAPL", "action": "buy"}]),
        ]
        metrics = analyze_churn([], [], decisions)
        assert metrics["action_flips"] == 0

    def test_single_flip(self):
        decisions = [
            make_decision("2026-01-01", [{"ticker": "AAPL", "action": "buy"}]),
            make_decision("2026-01-02", [{"ticker": "AAPL", "action": "sell"}]),
        ]
        metrics = analyze_churn([], [], decisions)
        assert metrics["action_flips"] == 1

    def test_multiple_flips(self):
        decisions = [
            make_decision("2026-01-01", [{"ticker": "AAPL", "action": "buy"}]),
            make_decision("2026-01-02", [{"ticker": "AAPL", "action": "sell"}]),
            make_decision("2026-01-03", [{"ticker": "AAPL", "action": "buy"}]),
        ]
        metrics = analyze_churn([], [], decisions)
        assert metrics["action_flips"] == 2

    def test_hold_ignored_in_flips(self):
        decisions = [
            make_decision("2026-01-01", [{"ticker": "AAPL", "action": "buy"}]),
            make_decision("2026-01-02", [{"ticker": "AAPL", "action": "hold"}]),
            make_decision("2026-01-03", [{"ticker": "AAPL", "action": "sell"}]),
        ]
        metrics = analyze_churn([], [], decisions)
        assert metrics["action_flips"] == 1  # buy → sell (hold skipped)

    def test_multiple_tickers_independent(self):
        decisions = [
            make_decision("2026-01-01", [
                {"ticker": "AAPL", "action": "buy"},
                {"ticker": "TSLA", "action": "sell"},
            ]),
            make_decision("2026-01-02", [
                {"ticker": "AAPL", "action": "sell"},
                {"ticker": "TSLA", "action": "buy"},
            ]),
        ]
        metrics = analyze_churn([], [], decisions)
        assert metrics["action_flips"] == 2  # one per ticker

    def test_multiple_actions_same_day_same_ticker(self):
        """Only first non-hold action per day should count (current implementation
        appends all, so both count toward sequence). Documenting behavior."""
        decisions = [
            make_decision("2026-01-01", [
                {"ticker": "AAPL", "action": "buy"},
                {"ticker": "AAPL", "action": "sell"},  # same day
            ]),
        ]
        metrics = analyze_churn([], [], decisions)
        # Both actions are recorded; they differ → counts as 1 flip
        assert metrics["action_flips"] == 1


class TestAnalyzeChurnActivity:
    def test_days_active_from_decisions(self):
        decisions = [
            make_decision("2026-01-01T10:00:00", []),
            make_decision("2026-01-10T10:00:00", []),
        ]
        metrics = analyze_churn([], [], decisions)
        assert metrics["days_active"] == 9

    def test_trades_per_week(self):
        trades = [make_trade("AAPL", "buy", 150.0) for _ in range(10)]
        decisions = [
            make_decision("2026-01-01T10:00:00", []),
            make_decision("2026-01-08T10:00:00", []),
        ]
        metrics = analyze_churn([], trades, decisions)
        assert metrics["days_active"] == 7
        assert metrics["trades_per_week"] == 10.0

    def test_annualized_turnover(self):
        trades = [make_trade("AAPL", "buy", 150.0) for _ in range(10)]
        decisions = [
            make_decision("2026-01-01T10:00:00", []),
            make_decision("2026-01-08T10:00:00", []),
        ]
        metrics = analyze_churn([], trades, decisions)
        assert metrics["annualized_turnover"] == pytest.approx(10 * 365 / 7, rel=0.01)


class TestAnalyzeChurnEdgeCases:
    def test_zero_pnl_counts_as_losing(self):
        """Zero P&L is non-winning → classified as losing."""
        rt = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 5), 4.0, 0.0, 150.0, 150.0)
        metrics = analyze_churn([rt], [], [])
        assert metrics["winning_round_trips"] == 0
        assert metrics["losing_round_trips"] == 1
        assert metrics["win_rate_pct"] == 0.0

    def test_avg_hold_days_calculation(self):
        rt1 = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 3), 2.0, 10.0, 100.0, 110.0)
        rt2 = RoundTrip("TSLA", datetime(2026, 1, 1), datetime(2026, 1, 5), 4.0, 20.0, 100.0, 120.0)
        metrics = analyze_churn([rt1, rt2], [], [])
        assert metrics["avg_hold_days"] == 3.0

    def test_large_pnl_values(self):
        rt = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 5), 4.0, 1e6, 100.0, 200.0)
        metrics = analyze_churn([rt], [], [])
        assert metrics["total_realized_pnl"] == 1e6

    def test_negative_pnl_values(self):
        rt = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 5), 4.0, -1e6, 100.0, 50.0)
        metrics = analyze_churn([rt], [], [])
        assert metrics["total_realized_pnl"] == -1e6


# ---------------------------------------------------------------------------
# print_report
# ---------------------------------------------------------------------------

class TestPrintReport:
    def test_prints_without_error(self, capsys):
        metrics = analyze_churn([], [], [])
        print_report(metrics)
        captured = capsys.readouterr()
        assert "PORTFOLIO CHURN ANALYSIS" in captured.out
        assert "Round Trips:" in captured.out
        assert "Winning:" in captured.out
        assert "Losing:" in captured.out
        assert "Short" in captured.out
        assert "Medium" in captured.out
        assert "Long" in captured.out
        assert "Action Flips:" in captured.out

    def test_prints_with_data(self, capsys):
        rt = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 5), 4.0, 100.0, 150.0, 160.0)
        metrics = analyze_churn([rt], [], [])
        print_report(metrics)
        captured = capsys.readouterr()
        assert "1" in captured.out
        assert "100.0%" in captured.out
        assert "€+100.00" in captured.out


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestMainIntegration:
    def test_main_with_files(self, tmp_path, capsys):
        trades = [
            make_trade("AAPL", "buy", 150.0, timestamp="2026-01-01T10:00:00"),
            make_trade("AAPL", "sell", 160.0, realized_pnl=100.0, timestamp="2026-01-05T10:00:00"),
        ]
        decisions = [
            make_decision("2026-01-01T10:00:00", [{"ticker": "AAPL", "action": "buy"}]),
            make_decision("2026-01-05T10:00:00", [{"ticker": "AAPL", "action": "sell"}]),
        ]
        (tmp_path / "trades_history.json").write_text(json.dumps(trades))
        (tmp_path / "decision_history.json").write_text(json.dumps(decisions))

        with patch("analysis.churn_analysis.load_trades", side_effect=lambda d="data": trades), \
             patch("analysis.churn_analysis.load_decisions", side_effect=lambda d="data": decisions):
            from analysis.churn_analysis import main
            main()
            captured = capsys.readouterr()
            assert "PORTFOLIO CHURN ANALYSIS" in captured.out
            assert "1" in captured.out


# ---------------------------------------------------------------------------
# _parse_trade_timestamp
# ---------------------------------------------------------------------------

class TestParseTradeTimestamp:
    def test_parses_iso_datetime(self):
        t = {"timestamp": "2026-01-01T10:00:00"}
        assert _parse_trade_timestamp(t) == datetime(2026, 1, 1, 10, 0, 0)

    def test_parses_iso_date_fallback(self):
        t = {"timestamp": "2026-01-01"}
        assert _parse_trade_timestamp(t) == datetime(2026, 1, 1)

    def test_missing_timestamp_returns_datetime_min(self):
        assert _parse_trade_timestamp({}) == datetime.min

    def test_invalid_timestamp_returns_datetime_min(self):
        t = {"timestamp": "not-a-date"}
        assert _parse_trade_timestamp(t) == datetime.min


# ---------------------------------------------------------------------------
# match_round_trips sorting
# ---------------------------------------------------------------------------

class TestMatchRoundTripsSorting:
    def test_unsorted_trades_still_fifo_by_timestamp(self):
        """Trades are reordered by timestamp before matching."""
        trades = [
            make_trade("AAPL", "sell", 160.0, realized_pnl=10.0, timestamp="2026-01-04T10:00:00"),
            make_trade("AAPL", "buy", 150.0, timestamp="2026-01-01T10:00:00"),
            make_trade("AAPL", "buy", 120.0, timestamp="2026-01-02T10:00:00"),
            make_trade("AAPL", "sell", 130.0, realized_pnl=10.0, timestamp="2026-01-05T10:00:00"),
        ]
        rts = match_round_trips(trades)
        assert len(rts) == 2
        assert rts[0].buy_price == 150.0
        assert rts[0].sell_price == 160.0
        assert rts[1].buy_price == 120.0
        assert rts[1].sell_price == 130.0
        assert all(rt.hold_days >= 0 for rt in rts)

    def test_sell_before_its_buy_in_input_sorted_out(self):
        """A sell that appears earlier in the file than its matching buy must still pair with the oldest buy."""
        trades = [
            make_trade("AAPL", "sell", 160.0, realized_pnl=10.0, timestamp="2026-01-02T10:00:00"),
            make_trade("AAPL", "buy", 150.0, timestamp="2026-01-01T10:00:00"),
        ]
        rts = match_round_trips(trades)
        assert len(rts) == 1
        assert rts[0].hold_days == 1.0


# ---------------------------------------------------------------------------
# _bucket_metrics
# ---------------------------------------------------------------------------

class TestBucketMetrics:
    def test_empty_returns_zeroed(self):
        metrics = _bucket_metrics([])
        assert metrics["total_round_trips"] == 0
        assert metrics["win_rate_pct"] == 0.0
        assert metrics["avg_hold_days"] == 0.0

    def test_basic_metrics(self):
        rt1 = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 5), 4.0, 100.0, 150.0, 160.0)
        rt2 = RoundTrip("TSLA", datetime(2026, 1, 1), datetime(2026, 1, 20), 19.0, -20.0, 200.0, 190.0)
        metrics = _bucket_metrics([rt1, rt2])
        assert metrics["total_round_trips"] == 2
        assert metrics["winning_round_trips"] == 1
        assert metrics["win_rate_pct"] == 50.0
        assert metrics["total_realized_pnl"] == 80.0
        assert metrics["avg_hold_days"] == pytest.approx(11.5)
        assert metrics["short_term_count"] == 0
        assert metrics["medium_term_count"] == 1
        assert metrics["long_term_count"] == 1


# ---------------------------------------------------------------------------
# analyze_cohort
# ---------------------------------------------------------------------------

class TestAnalyzeCohort:
    def test_cohort_split_by_buy_date(self):
        """Cohort attribution is based on the entry (buy) date, not the sell date."""
        trades = [
            make_trade("AAPL", "buy", 150.0, timestamp="2026-01-01T10:00:00"),
            make_trade("AAPL", "sell", 160.0, realized_pnl=10.0, timestamp="2026-02-02T10:00:00"),
            make_trade("AAPL", "buy", 150.0, timestamp="2026-02-01T10:00:00"),
            make_trade("AAPL", "sell", 170.0, realized_pnl=20.0, timestamp="2026-02-03T10:00:00"),
        ]
        cutoff = datetime(2026, 2, 1)
        pre, post = analyze_cohort(trades, cutoff)
        # The first buy is pre-cutoff, so its round trip is pre-cutoff even though the sell is post-cutoff.
        # The second buy is post-cutoff, so its round trip is post-cutoff.
        assert pre["total_round_trips"] == 1
        assert pre["win_rate_pct"] == 100.0
        assert post["total_round_trips"] == 1
        assert post["win_rate_pct"] == 100.0

    def test_no_pre_round_trips(self):
        trades = [
            make_trade("AAPL", "buy", 150.0, timestamp="2026-02-01T10:00:00"),
            make_trade("AAPL", "sell", 160.0, realized_pnl=10.0, timestamp="2026-02-02T10:00:00"),
        ]
        cutoff = datetime(2026, 2, 1)
        pre, post = analyze_cohort(trades, cutoff)
        assert pre["total_round_trips"] == 0
        assert post["total_round_trips"] == 1

    def test_no_post_round_trips(self):
        trades = [
            make_trade("AAPL", "buy", 150.0, timestamp="2026-01-01T10:00:00"),
            make_trade("AAPL", "sell", 160.0, realized_pnl=10.0, timestamp="2026-01-02T10:00:00"),
        ]
        cutoff = datetime(2026, 2, 1)
        pre, post = analyze_cohort(trades, cutoff)
        assert pre["total_round_trips"] == 1
        assert post["total_round_trips"] == 0

    def test_sell_before_cutoff_buy_after_cutoff_not_counted(self):
        """A sell before cutoff with a buy after cutoff should not create a round trip."""
        trades = [
            make_trade("AAPL", "sell", 160.0, realized_pnl=10.0, timestamp="2026-01-01T10:00:00"),
            make_trade("AAPL", "buy", 150.0, timestamp="2026-02-02T10:00:00"),
        ]
        cutoff = datetime(2026, 2, 1)
        pre, post = analyze_cohort(trades, cutoff)
        # The sell has no matching pre-cutoff buy, so no pre-cutoff round trip.
        # The buy has no matching post-cutoff sell, so no post-cutoff round trip.
        assert pre["total_round_trips"] == 0
        assert post["total_round_trips"] == 0

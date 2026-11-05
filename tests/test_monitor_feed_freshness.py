"""Regression tests: feed freshness guard (monitor).

Production incident 2026-11-05, run 08:06 UTC: the yfinance Euronext feed
went stale overnight (Paris series frozen at the 2026-11-03 close) while the
US feed stayed healthy. The monitor treated stale closes as live prices,
fired three spurious POSITION_MOVEMENT alerts (+3.1 % / +3.3 % / -2.3 %), and
persisted the stale closes as ``previous_close`` references — so the 12:15
run, after the feed had recovered, still compared live prices against
two-day-old closes and re-fired on the same phantom movements.

The guard: every quote carries the UTC timestamp of the bar it was read
from; a ticker whose last bar misses a required trading session is excluded
from alerting AND keeps its previous reference instead of overwriting it.

Clock freezing follows the PR #101 pattern: ``monitor.datetime`` is replaced
by a subclass whose ``now()`` is pinned, so both ``run_monitor`` and the
guard see the same wall clock.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from portfolio.portfolio import Portfolio, Position
import monitor
from monitor import (
    _expected_session_date,
    _market_of,
    _session_date_from_bar,
    is_feed_stale,
    run_monitor,
    split_fresh_and_stale,
)


# 2026-11-05 is a Wednesday; 2026-11-03 Monday, 2026-11-04 Tuesday,
# 2026-11-07 Saturday, 2026-11-09 Monday.

class TestMarketClassification:
    def test_pa_tickers_are_euronext(self):
        assert _market_of('MC.PA') == 'euronext'
        assert _market_of('AI.PA') == 'euronext'

    def test_fchi_is_euronext(self):
        assert _market_of('^FCHI') == 'euronext'

    def test_us_tickers(self):
        assert _market_of('SPY') == 'us'
        assert _market_of('PDBC') == 'us'
        assert _market_of('REET') == 'us'


class TestSessionDateFromBar:
    def test_euronext_daily_bar_stamped_previous_day_2300(self):
        # yfinance stamps the Euronext daily bar at 23:00 UTC of the day
        # before the session (midnight CET). Session of Wed 11-05 is stamped
        # Tue 11-04 23:00 UTC.
        assert _session_date_from_bar('2026-11-04T23:00:00', 'euronext') \
            == datetime(2026, 11, 5).date()

    def test_euronext_intraday_bar_stamped_same_day(self):
        assert _session_date_from_bar('2026-11-05T09:30:00', 'euronext') \
            == datetime(2026, 11, 5).date()

    def test_us_bar_maps_to_own_date(self):
        assert _session_date_from_bar('2026-11-04T00:00:00', 'us') \
            == datetime(2026, 11, 4).date()
        assert _session_date_from_bar('2026-11-05T14:35:00', 'us') \
            == datetime(2026, 11, 5).date()

    def test_missing_or_garbage_timestamp(self):
        assert _session_date_from_bar(None, 'us') is None
        assert _session_date_from_bar('', 'euronext') is None
        assert _session_date_from_bar('not-a-timestamp', 'us') is None


class TestExpectedSessionDate:
    def test_during_euronext_session_today_is_required(self):
        now = datetime(2026, 11, 5, 8, 6)  # Wed, just after Paris open
        assert _expected_session_date(now, 'euronext') == datetime(2026, 11, 5).date()

    def test_before_euronext_open_previous_session_required(self):
        now = datetime(2026, 11, 5, 6, 0)  # Wed, pre-open
        assert _expected_session_date(now, 'euronext') == datetime(2026, 11, 4).date()

    def test_during_us_session_today_is_required(self):
        now = datetime(2026, 11, 5, 14, 35)  # Wed, US open
        assert _expected_session_date(now, 'us') == datetime(2026, 11, 5).date()

    def test_before_us_open_previous_session_required(self):
        now = datetime(2026, 11, 5, 12, 0)  # Wed, US pre-market
        assert _expected_session_date(now, 'us') == datetime(2026, 11, 4).date()

    def test_saturday_rolls_back_to_friday(self):
        now = datetime(2026, 11, 7, 12, 0)
        assert _expected_session_date(now, 'euronext') == datetime(2026, 11, 6).date()
        assert _expected_session_date(now, 'us') == datetime(2026, 11, 6).date()

    def test_monday_pre_open_rolls_back_to_friday(self):
        # 06:30 UTC is outside even the generous summer bound (07:00).
        now = datetime(2026, 11, 9, 6, 30)
        assert _expected_session_date(now, 'euronext') == datetime(2026, 11, 6).date()

    def test_monday_morning_inside_generous_window_requires_today(self):
        # 07:30 UTC is inside the generous (summer-covering) Euronext window,
        # so Monday's forming bar is required even though Paris (winter) only
        # opens at 08:00 UTC. The monitor's first run is 08:05, safely inside.
        now = datetime(2026, 11, 9, 7, 30)
        assert _expected_session_date(now, 'euronext') == datetime(2026, 11, 9).date()


class TestIsFeedStale:
    """Incident replay: Wed 2026-11-05 08:06 UTC."""

    NOW = datetime(2026, 11, 5, 8, 6)

    def test_incident_euronext_bar_two_days_old_is_stale(self):
        # The actual stale feed: last Paris bar = Mon 11-03 23:00 UTC stamp
        # (session Tue 11-04 missing entirely).
        assert is_feed_stale('2026-11-03T23:00:00', 'AI.PA', self.NOW) is True
        assert is_feed_stale('2026-11-03T23:00:00', 'MC.PA', self.NOW) is True

    def test_healthy_euronext_feed_is_fresh(self):
        # Healthy feed serves the forming bar of Wed 11-05, stamped
        # Tue 11-04 23:00 UTC.
        assert is_feed_stale('2026-11-04T23:00:00', 'AI.PA', self.NOW) is False

    def test_frozen_us_premarket_price_is_fresh(self):
        # At 08:06 UTC the US market is closed; yesterday's close is the
        # legitimate reference (the documented "prix figé" case). Must NOT
        # be flagged stale.
        assert is_feed_stale('2026-11-04T00:00:00', 'SPY', self.NOW) is False

    def test_missing_timestamp_is_stale(self):
        assert is_feed_stale(None, 'SPY', self.NOW) is True

    def test_us_feed_missing_today_during_session_is_stale(self):
        now = datetime(2026, 11, 5, 16, 36)  # US open
        assert is_feed_stale('2026-11-04T00:00:00', 'SPY', now) is True
        assert is_feed_stale('2026-11-05T15:00:00', 'SPY', now) is False


class TestSplitFreshAndStale:
    NOW = datetime(2026, 11, 5, 8, 6)

    def test_partitions_quotes(self):
        quotes = {
            'AI.PA': {'price': 182.56, 'last_bar_ts': '2026-11-03T23:00:00'},  # stale
            'MC.PA': {'price': 401.50, 'last_bar_ts': '2026-11-03T23:00:00'},  # stale
            'SPY': {'price': 745.83, 'last_bar_ts': '2026-11-04T00:00:00'},    # fresh
            'REET': {'price': None, 'last_bar_ts': None},                      # failed
            'IJR': {'price': float('nan'), 'last_bar_ts': '2026-11-05T05:00:00'},  # NaN
        }
        fresh, stale = split_fresh_and_stale(quotes, self.NOW)
        assert fresh == {'SPY': 745.83}
        assert stale == {'AI.PA', 'MC.PA', 'REET', 'IJR'}


def _freeze_clock(monkeypatch, frozen):
    """Pin ``monitor.datetime.now()`` to ``frozen`` (PR #101 pattern)."""

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen

    monkeypatch.setattr(monitor, "datetime", _FrozenDateTime)


def _isolate_paths(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(monitor, "DATA_DIR", data_dir)
    monkeypatch.setattr(monitor, "MARKET_STATE_PATH", data_dir / "market_state.json")
    monkeypatch.setattr(monitor, "ALERT_HISTORY_PATH", data_dir / "alert_history.json")
    return data_dir


class TestRunMonitorFreshnessGuard:
    """Full-run regression for the 2026-11-05 08:06 incident.

    Stale Euronext quotes must produce no position-movement alerts and must
    not be persisted as references; fresh US quotes alert and save normally;
    stale tickers keep their previous reference.
    """

    def test_stale_feed_excluded_and_reference_preserved(
        self, tmp_path, monkeypatch, capsys
    ):
        frozen = datetime(2026, 11, 5, 8, 6)
        _freeze_clock(monkeypatch, frozen)
        data_dir = _isolate_paths(tmp_path, monkeypatch)

        # References as they stood after the last healthy run.
        (data_dir / "market_state.json").write_text(json.dumps({
            'timestamp': '2026-11-04T17:45:56',
            'previous_close': {'AI.PA': 188.26, 'MC.PA': 418.00, 'SPY': 740.45},
        }))
        (data_dir / "alert_history.json").write_text(json.dumps({
            'alerts': [], 'last_reset': '2026-11-04T08:06:00',
        }))

        quotes = {
            # Stale Euronext feed: frozen at the Mon 11-03 close.
            'AI.PA': {'price': 182.56, 'last_bar_ts': '2026-11-03T23:00:00'},
            'MC.PA': {'price': 401.50, 'last_bar_ts': '2026-11-03T23:00:00'},
            # Fresh US feed.
            'SPY': {'price': 733.25, 'last_bar_ts': '2026-11-04T00:00:00'},
        }

        portfolio = Portfolio(data_dir=str(data_dir))
        portfolio.positions['AI.PA'] = Position(
            ticker='AI.PA', quantity=1.0, avg_price=170.88, current_price=182.56
        )
        portfolio.positions['MC.PA'] = Position(
            ticker='MC.PA', quantity=1.0, avg_price=385.40, current_price=401.50
        )
        portfolio.positions['SPY'] = Position(
            ticker='SPY', quantity=1.0, avg_price=746.64, current_price=733.25
        )

        with patch("monitor.Portfolio", return_value=portfolio), \
             patch("monitor.fetch_current_quotes", return_value=quotes), \
             patch("data.fetch_market_data.fetch_historical_data", return_value={}):
            alerts, _ = run_monitor()

        out = capsys.readouterr().out
        assert "STALE FEED" in out
        assert "AI.PA" in out and "MC.PA" in out

        # No alert may reference a stale ticker.
        alert_tickers = {a['ticker'] for a in alerts}
        assert 'AI.PA' not in alert_tickers
        assert 'MC.PA' not in alert_tickers

        # References: stale tickers preserved, fresh ticker updated.
        saved = json.loads((data_dir / "market_state.json").read_text())
        refs = saved['previous_close']
        assert refs['AI.PA'] == 188.26  # preserved, NOT overwritten with 182.56
        assert refs['MC.PA'] == 418.00  # preserved, NOT overwritten with 401.50
        assert refs['SPY'] == 733.25    # fresh price saved

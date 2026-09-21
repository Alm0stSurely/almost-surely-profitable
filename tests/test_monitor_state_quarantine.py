"""
Regression tests for corrupt-state quarantine in the intraday monitor.

Sites: ``monitor.load_alert_history`` (alert_history.json) and
``monitor.load_previous_close`` (market_state.json).

Pre-fix behaviour: a bare ``except`` returned a fresh fallback dict, and the
same monitor run then *overwrote the corrupt file* via ``save_alert_history``
/ ``save_market_state`` — the evidence of the corruption was destroyed without
a trace. This is the same data-loss class as the trade ledger (PR #60):
the swallowed read feeds an overwrite of the very file that failed.

Post-fix: the corrupt file is quarantined (renamed ``<name>.corrupt-<ts>``),
an error is logged, and the run proceeds on a fresh state.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import monitor
from monitor import load_alert_history, load_previous_close, record_alert, save_alert_history


def _quarantine_siblings(path: Path):
    return sorted(path.parent.glob(f"{path.name}.corrupt-*"))


@pytest.fixture
def alert_history_path(tmp_path, monkeypatch):
    path = tmp_path / "alert_history.json"
    monkeypatch.setattr(monitor, "ALERT_HISTORY_PATH", path)
    return path


@pytest.fixture
def market_state_path(tmp_path, monkeypatch):
    path = tmp_path / "market_state.json"
    monkeypatch.setattr(monitor, "MARKET_STATE_PATH", path)
    return path


class TestAlertHistoryQuarantine:
    """Corrupt alert history must be quarantined, not silently overwritten."""

    def test_corrupt_history_quarantined_and_fresh_returned(self, alert_history_path, caplog):
        alert_history_path.write_text("{not valid json")

        with caplog.at_level(logging.ERROR):
            history = load_alert_history()

        assert set(history.keys()) == {'alerts', 'last_reset'}
        assert history['alerts'] == []

        quarantines = _quarantine_siblings(alert_history_path)
        assert len(quarantines) == 1
        assert quarantines[0].read_text() == "{not valid json"

        assert any("quarantined" in r.message and "alert history" in r.message
                   for r in caplog.records)

    @pytest.mark.parametrize("payload", [
        '[1, 2, 3]',
        '"a string, not a dict"',
        'null',
    ])
    def test_wrong_shape_history_quarantined(self, alert_history_path, caplog, payload):
        """Valid JSON that is not a dict must be quarantined, not kept."""
        alert_history_path.write_text(payload)

        with caplog.at_level(logging.ERROR):
            history = load_alert_history()

        assert history['alerts'] == []
        assert len(_quarantine_siblings(alert_history_path)) == 1
        assert any("wrong shape" in r.message for r in caplog.records)

    def test_healthy_history_returned_unchanged(self, alert_history_path, caplog):
        saved = {
            'alerts': [{'ticker': 'SPY', 'type': 'position_movement',
                        'movement_pct': -2.5, 'severity': 'high',
                        'timestamp': '2026-09-21T09:00:00'}],
            'last_reset': '2026-09-21T08:00:00',
        }
        alert_history_path.write_text(json.dumps(saved))

        with caplog.at_level(logging.ERROR):
            history = load_alert_history()

        assert history == saved
        assert _quarantine_siblings(alert_history_path) == []
        assert caplog.records == []

    def test_missing_history_starts_fresh_silent(self, alert_history_path, caplog):
        with caplog.at_level(logging.ERROR):
            history = load_alert_history()

        assert history['alerts'] == []
        assert 'last_reset' in history
        assert _quarantine_siblings(alert_history_path) == []
        assert caplog.records == []

    def test_dict_without_alerts_list_logged_fresh(self, alert_history_path, caplog):
        """A dict with no 'alerts' list has no records to preserve.

        It starts fresh (the run will overwrite the file), but the failure is
        logged so the overwrite is loud, not silent.
        """
        alert_history_path.write_text(json.dumps({'foo': 'bar'}))

        with caplog.at_level(logging.ERROR):
            history = load_alert_history()

        assert history['alerts'] == []
        # Nothing recoverable to preserve — no quarantine, but a loud log.
        assert _quarantine_siblings(alert_history_path) == []
        assert any("no 'alerts' list" in r.message for r in caplog.records)

    def test_data_loss_flow_preserves_evidence(self, alert_history_path, caplog):
        """The behavioural pin: load -> record -> save keeps the corrupt file.

        Pre-fix, the fresh fallback let ``save_alert_history`` overwrite the
        corrupt file within the same run — the evidence was destroyed. Post-fix
        the quarantine copy survives and the new alert lands in a fresh file.
        """
        alert_history_path.write_text("{corrupted mid-write")

        with caplog.at_level(logging.ERROR):
            history = load_alert_history()
            record_alert('SPY', -3.1, 'position_movement', 'high', history)
            save_alert_history(history)

        quarantines = _quarantine_siblings(alert_history_path)
        assert len(quarantines) == 1
        assert quarantines[0].read_text() == "{corrupted mid-write"

        saved = json.loads(alert_history_path.read_text())
        assert [a['ticker'] for a in saved['alerts']] == ['SPY']


class TestMarketStateQuarantine:
    """Corrupt market state must be quarantined, not silently overwritten."""

    def test_corrupt_state_quarantined_and_empty_returned(self, market_state_path, caplog):
        market_state_path.write_text("{not valid json")

        with caplog.at_level(logging.ERROR):
            references = load_previous_close(portfolio=None)

        assert references == {}

        quarantines = _quarantine_siblings(market_state_path)
        assert len(quarantines) == 1
        assert quarantines[0].read_text() == "{not valid json"

        assert any("quarantined" in r.message and "market state" in r.message
                   for r in caplog.records)

    @pytest.mark.parametrize("payload", [
        '[1, 2, 3]',
        '"a string, not a dict"',
        'null',
    ])
    def test_wrong_shape_state_quarantined(self, market_state_path, caplog, payload):
        market_state_path.write_text(payload)

        with caplog.at_level(logging.ERROR):
            references = load_previous_close(portfolio=None)

        assert references == {}
        assert len(_quarantine_siblings(market_state_path)) == 1
        assert any("wrong shape" in r.message for r in caplog.records)

    def test_healthy_state_returned_with_finite_filter(self, market_state_path, caplog):
        # json.load accepts the non-standard `NaN` literal; it must be filtered.
        market_state_path.write_text(
            '{"timestamp": "2026-09-21T08:00:00", '
            '"previous_close": {"SPY": 600.1, "GLD": NaN, "QQQ": 520.0}}'
        )

        with caplog.at_level(logging.ERROR):
            references = load_previous_close(portfolio=None)

        assert references == {"SPY": 600.1, "QQQ": 520.0}
        assert _quarantine_siblings(market_state_path) == []
        assert caplog.records == []

    def test_missing_state_returns_empty_silent(self, market_state_path, caplog):
        with caplog.at_level(logging.ERROR):
            references = load_previous_close(portfolio=None)

        assert references == {}
        assert _quarantine_siblings(market_state_path) == []
        assert caplog.records == []

    def test_non_dict_previous_close_logged_empty(self, market_state_path, caplog):
        """Valid dict with a non-dict 'previous_close' is ignored loudly.

        Pre-fix this crashed ``.items()`` inside the bare except and fell back
        silently; post-fix it is an explicit, logged empty reference map.
        """
        market_state_path.write_text(
            json.dumps({'timestamp': '2026-09-21T08:00:00',
                        'previous_close': [1, 2, 3]})
        )

        with caplog.at_level(logging.ERROR):
            references = load_previous_close(portfolio=None)

        assert references == {}
        assert _quarantine_siblings(market_state_path) == []
        assert any("non-dict 'previous_close'" in r.message for r in caplog.records)


class TestCooldownStateQuarantine:
    """Corrupt cooldown state must be quarantined, not silently reset+overwritten."""

    def _manager(self, tmp_path):
        from risk.position_cooldown import PositionCooldownManager
        return PositionCooldownManager(data_dir=str(tmp_path))

    def test_corrupt_state_quarantined_and_reset(self, tmp_path, caplog):
        state_file = tmp_path / "position_cooldowns.json"
        state_file.write_text("{not valid json")

        with caplog.at_level(logging.ERROR):
            mgr = self._manager(tmp_path)

        assert mgr.entries == {} and mgr.exits == {} and mgr.weekly_trades == []

        quarantines = _quarantine_siblings(state_file)
        assert len(quarantines) == 1
        assert quarantines[0].read_text() == "{not valid json"
        assert any("quarantined" in r.message and "cooldown state" in r.message
                   for r in caplog.records)

    def test_wrong_shape_state_quarantined(self, tmp_path, caplog):
        state_file = tmp_path / "position_cooldowns.json"
        state_file.write_text('null')

        with caplog.at_level(logging.ERROR):
            mgr = self._manager(tmp_path)

        assert mgr.entries == {}
        assert len(_quarantine_siblings(state_file)) == 1
        assert any("wrong shape" in r.message for r in caplog.records)

    def test_inner_unparseable_reset_loud(self, tmp_path, caplog):
        """Top-level dict with garbage timestamps: reset loudly, no quarantine."""
        state_file = tmp_path / "position_cooldowns.json"
        state_file.write_text(json.dumps({"entries": {"SPY": "not-a-date"}}))

        with caplog.at_level(logging.ERROR):
            mgr = self._manager(tmp_path)

        assert mgr.entries == {}
        # No trustworthy records to preserve; the reset is the loud failure.
        assert _quarantine_siblings(state_file) == []
        assert any("unparseable contents" in r.message for r in caplog.records)

    def test_healthy_state_loaded(self, tmp_path, caplog):
        state_file = tmp_path / "position_cooldowns.json"
        state_file.write_text(json.dumps({
            "entries": {"SPY": "2026-09-15T09:00:00"},
            "exits": {},
            "weekly_trades": ["2026-09-15T09:00:00"],
        }))

        with caplog.at_level(logging.ERROR):
            mgr = self._manager(tmp_path)

        assert "SPY" in mgr.entries
        assert len(mgr.weekly_trades) == 1
        assert _quarantine_siblings(state_file) == []
        assert caplog.records == []


class TestBenchmarkStateQuarantine:
    """Corrupt benchmark state must be quarantined, not silently reset+overwritten."""

    def _benchmark(self, tmp_path):
        from benchmark import LiveEqualWeightBenchmark
        return LiveEqualWeightBenchmark(data_dir=str(tmp_path))

    def test_corrupt_state_quarantined_and_defaults(self, tmp_path, caplog):
        state_file = tmp_path / "equalweight_benchmark_state.json"
        state_file.write_text("{not valid json")

        with caplog.at_level(logging.ERROR):
            bench = self._benchmark(tmp_path)

        assert bench.shares == {}
        assert bench.cash == bench.initial_capital

        quarantines = _quarantine_siblings(state_file)
        assert len(quarantines) == 1
        assert quarantines[0].read_text() == "{not valid json"
        assert any("quarantined" in r.message and "benchmark state" in r.message
                   for r in caplog.records)

    def test_wrong_shape_state_quarantined_not_crashed(self, tmp_path, caplog):
        """Valid JSON list pre-fix crashed .get() outside the try; quarantine now."""
        state_file = tmp_path / "equalweight_benchmark_state.json"
        state_file.write_text('[1, 2, 3]')

        with caplog.at_level(logging.ERROR):
            bench = self._benchmark(tmp_path)

        assert bench.shares == {}
        assert len(_quarantine_siblings(state_file)) == 1
        assert any("wrong shape" in r.message for r in caplog.records)

    def test_healthy_state_loaded(self, tmp_path, caplog):
        state_file = tmp_path / "equalweight_benchmark_state.json"
        state_file.write_text(json.dumps({
            "shares": {"SPY": 10.0},
            "cash": 1000.0,
            "start_date": "2026-01-02",
            "last_rebalanced": "2026-09-01",
        }))

        with caplog.at_level(logging.ERROR):
            bench = self._benchmark(tmp_path)

        assert bench.shares == {"SPY": 10.0}
        assert bench.cash == 1000.0
        assert bench.last_rebalanced == "2026-09-01"
        assert _quarantine_siblings(state_file) == []
        assert caplog.records == []

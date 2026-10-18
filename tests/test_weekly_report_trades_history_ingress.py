"""Ingress loudness pins for weekly_report._load_week_trades_from_history.

The weekly report supplements daily-result trades with the authoritative
trade ledger (data/trades_history.json) so intraday/monitor executions are
not missed (W40-2026 class). That reader used to raw-load the ledger and
silently return [] on any parse error — indistinguishable from an empty
week, hiding exactly the trades the helper exists to surface — and crashed
the whole report on a parseable-but-malformed record (non-dict element or
non-string timestamp) via attribute access outside any try.

The reader now routes through the same quarantine ingress as the ledger's
writer (Portfolio.save_trade): corrupt files are renamed aside with bytes
preserved and an ERROR log; parseable files with malformed records drop
those records loudly per element, siblings untouched.

Discriminators (fail on pre-fix code) assert the mechanism per the PR #83
doctrine: the quarantine rename, the ERROR log record itself (count, level,
content), the per-record print, and sibling survival. Bounded healthy pins
(missing / valid / empty ledger) must pass on BOTH sides of the fix.
"""

import json
import logging

import pytest

from weekly_report import _load_week_trades_from_history

WEEK = ("2026-10-12", "2026-10-16")

GOOD_GLD = {
    "timestamp": "2026-10-14T21:00:00", "ticker": "GLD",
    "action": "buy", "price": 210.0,
}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setattr("weekly_report.DATA_DIR", d)
    return d


def _ledger(data_dir, content):
    p = data_dir / "trades_history.json"
    p.write_text(content)
    return p


def _quarantines(data_dir):
    return sorted(data_dir.glob("trades_history.json.corrupt-*"))


class TestCorruptLedgerQuarantine:
    """File-level malformation: quarantine whole, bytes preserved, loud."""

    def test_corrupt_bytes_quarantined_with_log_and_print(self, data_dir, caplog, capsys):
        original = "{not json"
        _ledger(data_dir, original)

        with caplog.at_level(logging.ERROR, logger="utils"):
            result = _load_week_trades_from_history(*WEEK, [])

        assert result == []
        quarantines = _quarantines(data_dir)
        assert len(quarantines) == 1
        assert not (data_dir / "trades_history.json").exists()
        assert quarantines[0].read_text() == original  # bytes preserved

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "trade ledger" in errors[0].getMessage()

        out = capsys.readouterr().out
        assert "quarantined" in out and "trades_history" in out

    def test_nan_token_quarantined_by_strict_hook(self, data_dir, caplog):
        # No writer in the repo can emit NaN (allow_nan=False everywhere);
        # a NaN token is a corrupted/hand-edited file and must be rejected
        # at the read boundary (PR #84), then quarantined as unreadable.
        original = json.dumps([{**GOOD_GLD, "price": float("nan")}]).replace(
            "NaN", "NaN")  # json.dumps already emits the bare token here
        assert "NaN" in original
        _ledger(data_dir, original)

        with caplog.at_level(logging.ERROR, logger="utils"):
            result = _load_week_trades_from_history(*WEEK, [])

        assert result == []
        assert len(_quarantines(data_dir)) == 1
        assert any("non-standard JSON constant" in str(r.getMessage())
                   or "unreadable" in r.getMessage()
                   for r in caplog.records if r.levelno == logging.ERROR)

    def test_wrong_shape_quarantined(self, data_dir, caplog):
        original = '{"timestamp": "2026-10-13T09:00:00"}'  # parses, not a list
        _ledger(data_dir, original)

        with caplog.at_level(logging.ERROR, logger="utils"):
            result = _load_week_trades_from_history(*WEEK, [])

        assert result == []
        assert len(_quarantines(data_dir)) == 1
        assert any("wrong shape" in r.getMessage()
                   for r in caplog.records if r.levelno == logging.ERROR)


class TestMalformedRecordDrops:
    """Element-level malformation on a parseable file: drop loud, keep siblings."""

    def test_non_dict_element_dropped_loudly_sibling_survives(self, data_dir, capsys):
        _ledger(data_dir, json.dumps(["garbage", 12345, GOOD_GLD]))

        extra = _load_week_trades_from_history(*WEEK, [])

        assert [t["ticker"] for t in extra] == ["GLD"]
        out = capsys.readouterr().out
        assert out.count("Skipping malformed trade record") == 2  # one per dropped record
        # Parseable file: never quarantined.
        assert (data_dir / "trades_history.json").exists()
        assert not _quarantines(data_dir)

    def test_non_string_timestamp_dropped_loudly_sibling_survives(self, data_dir, capsys):
        bad = {"timestamp": 20261014, "ticker": "X", "action": "buy", "price": 1.0}
        _ledger(data_dir, json.dumps([bad, GOOD_GLD]))

        extra = _load_week_trades_from_history(*WEEK, [])

        assert [t["ticker"] for t in extra] == ["GLD"]
        out = capsys.readouterr().out
        assert "non-string timestamp" in out
        assert (data_dir / "trades_history.json").exists()
        assert not _quarantines(data_dir)


class TestHealthyPathsStayQuiet:
    """Bounded pins: expected absence and healthy files must not get noisy."""

    def test_missing_ledger_quiet(self, data_dir, caplog, capsys):
        with caplog.at_level(logging.WARNING):
            assert _load_week_trades_from_history(*WEEK, []) == []
        assert capsys.readouterr().out == ""
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_valid_ledger_merges_quietly(self, data_dir, caplog, capsys):
        _ledger(data_dir, json.dumps([GOOD_GLD]))

        with caplog.at_level(logging.WARNING):
            extra = _load_week_trades_from_history(*WEEK, [])

        assert [t["ticker"] for t in extra] == ["GLD"]
        assert capsys.readouterr().out == ""
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert (data_dir / "trades_history.json").exists()
        assert not _quarantines(data_dir)

    def test_empty_ledger_no_spurious_quarantine(self, data_dir, caplog, capsys):
        _ledger(data_dir, "[]")

        with caplog.at_level(logging.WARNING):
            assert _load_week_trades_from_history(*WEEK, []) == []

        assert (data_dir / "trades_history.json").exists()  # intact
        assert not _quarantines(data_dir)
        assert capsys.readouterr().out == ""
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

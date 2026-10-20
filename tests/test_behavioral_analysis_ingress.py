"""Ingress loudness pins for behavioral_analysis._load_analysis_histories.

The behavioral analysis used to read its two ledgers (data/decision_history.json,
data/trades_history.json) with raw ``json.load`` in ``main()``: a corrupt file
crashed the run before any section rendered, non-standard tokens no writer can
emit (``NaN``/``Infinity``) were silently ingested, a single malformed record
voided every section via attribute/item access outside any try (non-dict
element; missing/non-string/unparseable/timezone-aware timestamp for decisions;
non-dict or missing string ticker/action key for trades), and an empty decision
history — a legitimate fresh-account state — died on ``decisions[0]`` while the
sibling churn_analysis guards the same state.

The reader now routes through the same quarantine ingress as the ledgers'
writers (``save_decision`` / ``save_trade``): corrupt files are renamed aside
with bytes preserved and one ERROR log; parseable files with malformed records
drop those records loudly one-per-record, siblings untouched; degradation notes
are embedded in the report artifact itself, not only on stderr (PR #87
doctrine).

Discriminators (fail on pre-fix code) assert the mechanism per the PR #83
doctrine: the quarantine rename, the ERROR log record itself (count, level,
content), the per-record print, the artifact warning line, and sibling
survival with blast-radius checks. Bounded healthy pins (all-valid files,
empty trade ledger) must pass on BOTH sides of the fix.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis import behavioral_analysis

GOOD_DECISION = {
    "timestamp": "2026-10-14T21:00:00",
    "error": False,
    "actions": [],
    "reasoning": "hold steady",
}

GOOD_TRADE_BUY = {
    "timestamp": "2026-10-13T21:00:00",
    "ticker": "GLD",
    "action": "buy",
    "price": 210.0,
}

GOOD_TRADE_SELL = {
    "timestamp": "2026-10-14T21:00:00",
    "ticker": "GLD",
    "action": "sell",
    "price": 215.0,
    "realized_pnl": 5.0,
}


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    data = tmp_path / "data"
    results = tmp_path / "results"
    out = tmp_path / "out"
    data.mkdir()
    monkeypatch.setattr(behavioral_analysis, "DATA_DIR", data)
    monkeypatch.setattr(behavioral_analysis, "RESULTS_DIR", results)
    monkeypatch.setattr(behavioral_analysis, "OUTPUT_DIR", out)
    return data, results, out


def _decisions_file(data_dir, content):
    p = data_dir / "decision_history.json"
    p.write_text(content)
    return p


def _trades_file(data_dir, content):
    p = data_dir / "trades_history.json"
    p.write_text(content)
    return p


def _artifact(out_dir):
    paths = list(out_dir.glob("behavioral_analysis_*.txt"))
    assert len(paths) == 1
    return paths[0].read_text()


def _run_main():
    behavioral_analysis.main()


class TestCorruptDecisionHistoryQuarantine:
    """File-level malformation of decision history: quarantine, bytes, loud."""

    def test_corrupt_bytes_quarantined_with_log_and_artifact(self, dirs, caplog, capsys):
        data_dir, _, out_dir = dirs
        original = "{not json"
        _decisions_file(data_dir, original)

        with caplog.at_level(logging.ERROR, logger="utils"):
            _run_main()

        quarantines = sorted(data_dir.glob("decision_history.json.corrupt-*"))
        assert len(quarantines) == 1
        assert not (data_dir / "decision_history.json").exists()
        assert quarantines[0].read_text() == original  # bytes preserved

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "decision history" in errors[0].getMessage()

        report = _artifact(out_dir)
        assert "quarantined" in report
        assert "Total decisions: 0" in report  # report still rendered

    def test_nan_token_quarantined_by_strict_hook(self, dirs, caplog):
        # No writer in the repo can emit NaN (allow_nan=False everywhere); a
        # NaN token is a corrupted/hand-edited file and must be rejected at
        # the read boundary (PR #84), then quarantined as unreadable.
        data_dir, _, _ = dirs
        original = json.dumps([{**GOOD_DECISION, "cash_pct": float("nan")}])
        assert "NaN" in original
        _decisions_file(data_dir, original)

        with caplog.at_level(logging.ERROR, logger="utils"):
            _run_main()

        quarantines = list(data_dir.glob("decision_history.json.corrupt-*"))
        assert len(quarantines) == 1
        assert quarantines[0].read_text() == original
        assert any(
            "non-standard JSON constant" in str(r.getMessage())
            or "unreadable" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.ERROR
        )

    def test_wrong_shape_quarantined(self, dirs, caplog):
        original = '{"timestamp": "2026-10-13T21:00:00"}'  # parses, not a list
        data_dir, _, _ = dirs
        _decisions_file(data_dir, original)

        with caplog.at_level(logging.ERROR, logger="utils"):
            _run_main()

        quarantines = list(data_dir.glob("decision_history.json.corrupt-*"))
        assert len(quarantines) == 1
        assert quarantines[0].read_text() == original
        assert any(
            "wrong shape" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.ERROR
        )


class TestCorruptTradeLedgerQuarantine:
    def test_corrupt_trades_quarantined_with_artifact_warning(self, dirs, caplog, capsys):
        data_dir, _, out_dir = dirs
        original = "not json either"
        _trades_file(data_dir, original)
        _decisions_file(data_dir, json.dumps([GOOD_DECISION]))

        with caplog.at_level(logging.ERROR, logger="utils"):
            _run_main()

        quarantines = sorted(data_dir.glob("trades_history.json.corrupt-*"))
        assert len(quarantines) == 1
        assert quarantines[0].read_text() == original
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "trade ledger" in errors[0].getMessage()

        report = _artifact(out_dir)
        assert "quarantined" in report
        assert "Round trips: 0" in report


class TestMalformedDecisionDrops:
    """Element-level malformation on a parseable file: drop loud, keep siblings."""

    def test_non_dict_element_dropped_loudly_sibling_survives(self, dirs, capsys):
        data_dir, _, out_dir = dirs
        _decisions_file(data_dir, json.dumps(["garbage", 12345, GOOD_DECISION]))

        _run_main()

        report = _artifact(out_dir)
        assert "Total decisions: 1" in report  # sibling survived
        out = capsys.readouterr().out
        assert out.count("Dropping malformed decision record") == 2
        assert "Dropped 2 malformed record(s) from decision history" in report
        # Parseable file: never quarantined.
        assert (data_dir / "decision_history.json").exists()
        assert not list(data_dir.glob("decision_history.json.corrupt-*"))

    def test_non_string_timestamp_dropped_loudly(self, dirs, capsys):
        data_dir, _, out_dir = dirs
        bad = {"timestamp": 20261014, "error": False}
        _decisions_file(data_dir, json.dumps([bad, GOOD_DECISION]))

        _run_main()

        report = _artifact(out_dir)
        assert "Total decisions: 1" in report
        out = capsys.readouterr().out
        assert "missing/non-string timestamp" in out

    def test_unparseable_timestamp_dropped_loudly(self, dirs, capsys):
        data_dir, _, out_dir = dirs
        bad = {"timestamp": "not-a-date", "error": False}
        _decisions_file(data_dir, json.dumps([bad, GOOD_DECISION]))

        _run_main()

        report = _artifact(out_dir)
        assert "Total decisions: 1" in report
        out = capsys.readouterr().out
        assert "unparseable timestamp" in out

    def test_tz_aware_timestamp_dropped_loudly(self, dirs, capsys):
        # Parses fine, detonates one operation downstream at the naive/aware
        # subtraction in the churn header (PR #86 rule: the guard must wrap
        # the whole element evaluation, not just its decoding).
        data_dir, _, out_dir = dirs
        bad = {"timestamp": "2026-10-13T21:00:00+00:00", "error": False}
        _decisions_file(data_dir, json.dumps([bad, GOOD_DECISION]))

        _run_main()

        report = _artifact(out_dir)
        assert "Total decisions: 1" in report
        out = capsys.readouterr().out
        assert "timezone-aware timestamp" in out


class TestMalformedTradeDrops:
    def test_non_dict_trade_dropped_sibling_round_trip_survives(self, dirs, capsys):
        data_dir, _, out_dir = dirs
        _trades_file(
            data_dir,
            json.dumps([42, GOOD_TRADE_BUY, GOOD_TRADE_SELL]),
        )
        _decisions_file(data_dir, json.dumps([GOOD_DECISION]))

        _run_main()

        report = _artifact(out_dir)
        assert "Round trips: 1" in report  # GLD buy->sell sibling survived
        out = capsys.readouterr().out
        assert out.count("Dropping malformed trade record") == 1
        assert (data_dir / "trades_history.json").exists()
        assert not list(data_dir.glob("trades_history.json.corrupt-*"))

    def test_missing_ticker_or_action_dropped_loudly(self, dirs, capsys):
        data_dir, _, out_dir = dirs
        no_ticker = {"timestamp": "2026-10-13T21:00:00", "action": "buy", "price": 1.0}
        no_action = {"timestamp": "2026-10-13T21:00:00", "ticker": "X", "price": 1.0}
        _trades_file(
            data_dir,
            json.dumps([no_ticker, no_action, GOOD_TRADE_BUY, GOOD_TRADE_SELL]),
        )
        _decisions_file(data_dir, json.dumps([GOOD_DECISION]))

        _run_main()

        report = _artifact(out_dir)
        assert "Round trips: 1" in report
        out = capsys.readouterr().out
        assert out.count("Dropping malformed trade record") == 2

    def test_tz_aware_trade_timestamp_dropped_sibling_survives(self, dirs, capsys):
        # Deferred from PR #88: a tz-aware trade timestamp parses fine and then
        # detonates inside match_round_trips' naive/aware comparisons,
        # voiding every round trip. The churn_analysis ingress validates the
        # same contract; both callers of match_round_trips must.
        data_dir, _, out_dir = dirs
        tz_trade = {**GOOD_TRADE_BUY, "timestamp": "2026-10-13T21:00:00+02:00"}
        _trades_file(
            data_dir,
            json.dumps([tz_trade, GOOD_TRADE_BUY, GOOD_TRADE_SELL]),
        )
        _decisions_file(data_dir, json.dumps([GOOD_DECISION]))

        _run_main()

        report = _artifact(out_dir)
        assert "Round trips: 1" in report  # GLD buy->sell sibling survived
        out = capsys.readouterr().out
        assert out.count("Dropping malformed trade record") == 1
        assert "timezone-aware timestamp" in out


class TestDegenerateStates:
    """Legitimate states that crashed pre-fix wholesale."""

    def test_empty_decision_history_renders_report(self, dirs):
        data_dir, _, out_dir = dirs
        _decisions_file(data_dir, "[]")

        _run_main()  # pre-fix: IndexError on decisions[0]

        report = _artifact(out_dir)
        assert "Total decisions: 0" in report
        assert "CHURN / ROUND-TRIP ANALYSIS" in report

    def test_missing_files_render_empty_report(self, dirs, caplog, capsys):
        _, _, out_dir = dirs

        with caplog.at_level(logging.WARNING):
            _run_main()  # pre-fix: FileNotFoundError

        report = _artifact(out_dir)
        assert "Total decisions: 0" in report
        # Expected absence stays quiet (bounded pin on the new code path).
        assert capsys.readouterr().out.count("⚠") == 0
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestHealthyPathsStayQuiet:
    """Bounded pins: healthy and empty-but-valid files must not get noisy."""

    def test_valid_histories_render_quietly(self, dirs, caplog, capsys):
        data_dir, _, out_dir = dirs
        _decisions_file(data_dir, json.dumps([GOOD_DECISION]))
        _trades_file(data_dir, json.dumps([GOOD_TRADE_BUY, GOOD_TRADE_SELL]))

        with caplog.at_level(logging.WARNING):
            _run_main()

        report = _artifact(out_dir)
        assert "Total decisions: 1" in report
        assert "Round trips: 1" in report
        assert "⚠" not in report
        assert capsys.readouterr().out.count("⚠") == 0
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert (data_dir / "decision_history.json").exists()
        assert (data_dir / "trades_history.json").exists()
        assert not list(data_dir.glob("*.corrupt-*"))

    def test_empty_trade_ledger_quiet(self, dirs, caplog, capsys):
        data_dir, _, out_dir = dirs
        _decisions_file(data_dir, json.dumps([GOOD_DECISION]))
        _trades_file(data_dir, "[]")

        with caplog.at_level(logging.WARNING):
            _run_main()

        report = _artifact(out_dir)
        assert "Round trips: 0" in report
        assert "⚠" not in report
        assert capsys.readouterr().out.count("⚠") == 0
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

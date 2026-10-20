"""Ingress loudness pins for churn_analysis ledger loaders.

churn_analysis used to read both append-then-overwrite ledgers
(data/trades_history.json, data/decision_history.json) with raw ``json.load``:
an unparseable file crashed the whole analysis, a non-standard token no writer
can emit (``NaN``/``Infinity``, PR #84) was silently ingested, a wrong-shape
file crashed on the first list operation, and any single malformed record —
non-dict element, missing/unparseable/timezone-aware decision timestamp,
non-dict trade or missing ticker/action — voided every section via
attribute/item access or naive/aware datetime arithmetic outside any try.
Sub-element malformation (``actions`` items inside a decision record) crashed
the analysis the same way, one level deeper.

The loaders now route through the same quarantine ingress as the ledgers'
writers (``save_trade`` / ``save_decision``) and drop malformed records loudly
one-per-record; degradation notes are embedded in the report artifact itself
(PR #87 doctrine). ``match_round_trips`` stays unguarded by design (PR #88):
the caller validates, the helper's crash is the contract enforcement.

Discriminators (fail on pre-fix code) assert the mechanism per the PR #83
doctrine: the quarantine rename with bytes preserved, the ERROR log record
itself (count, level, content), the per-record drop print, the artifact
warning line, and sibling survival with blast-radius checks. Bounded healthy
pins (missing/unparseable trade timestamps keep the documented
``datetime.min`` fallback; all-valid files) are exercised through the public
``load_trades``/``load_decisions`` wrappers and must pass on BOTH sides.

The new symbols are lazy-imported inside test functions (PR #84 lesson) so a
fail-on-old run discriminates per-test instead of aborting at collection.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis import churn_analysis

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

GOOD_DECISION = {
    "timestamp": "2026-10-14T21:00:00",
    "actions": [{"ticker": "GLD", "action": "hold"}],
    "reasoning": "hold steady",
}

TZ_AWARE_TS = "2026-10-14T21:00:00+02:00"


def _load_trades_ingress(data_dir):
    # Lazy import (PR #84 lesson): lets a fail-on-old run fail per-test with
    # the correct symptom instead of aborting collection.
    from analysis.churn_analysis import _load_trades_ingress as fn
    return fn(data_dir)


def _load_decisions_ingress(data_dir):
    from analysis.churn_analysis import _load_decisions_ingress as fn
    return fn(data_dir)


def _trades_file(data_dir, content):
    p = data_dir / "trades_history.json"
    p.write_text(content)
    return p


def _decisions_file(data_dir, content):
    p = data_dir / "decision_history.json"
    p.write_text(content)
    return p


def _corrupt_names(data_dir, name):
    return sorted(data_dir.glob(f"{name}.corrupt-*"))


# ---------------------------------------------------------------------------
# File-level malformation: quarantine, bytes preserved, loud
# ---------------------------------------------------------------------------

class TestCorruptTradeLedgerQuarantine:
    def test_corrupt_bytes_quarantined_with_log_and_warning(self, tmp_path, caplog):
        original = "{not json"
        _trades_file(tmp_path, original)

        with caplog.at_level(logging.ERROR, logger="utils"):
            trades, warnings = _load_trades_ingress(str(tmp_path))

        quarantines = _corrupt_names(tmp_path, "trades_history.json")
        assert len(quarantines) == 1
        assert not (tmp_path / "trades_history.json").exists()
        assert quarantines[0].read_text() == original  # bytes preserved

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "trade ledger" in errors[0].getMessage()

        assert trades == []
        assert any("quarantined" in w for w in warnings)

    def test_nan_token_quarantined_by_strict_hook(self, tmp_path, caplog):
        # No writer in the repo can emit NaN (allow_nan=False everywhere); a
        # NaN token is a corrupted/hand-edited file (PR #84).
        original = json.dumps([{**GOOD_TRADE_BUY, "price": float("nan")}])
        assert "NaN" in original
        _trades_file(tmp_path, original)

        with caplog.at_level(logging.ERROR, logger="utils"):
            trades, warnings = _load_trades_ingress(str(tmp_path))

        quarantines = _corrupt_names(tmp_path, "trades_history.json")
        assert len(quarantines) == 1
        assert quarantines[0].read_text() == original
        assert trades == []
        assert any("quarantined" in w for w in warnings)

    def test_wrong_shape_quarantined(self, tmp_path, caplog):
        original = '{"ticker": "GLD"}'  # parses, not a list
        _trades_file(tmp_path, original)

        with caplog.at_level(logging.ERROR, logger="utils"):
            trades, warnings = _load_trades_ingress(str(tmp_path))

        quarantines = _corrupt_names(tmp_path, "trades_history.json")
        assert len(quarantines) == 1
        assert quarantines[0].read_text() == original
        assert any(
            "wrong shape" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.ERROR
        )
        assert trades == []


class TestCorruptDecisionHistoryQuarantine:
    def test_corrupt_bytes_quarantined_with_log_and_warning(self, tmp_path, caplog):
        original = "[not json"
        _decisions_file(tmp_path, original)

        with caplog.at_level(logging.ERROR, logger="utils"):
            decisions, warnings = _load_decisions_ingress(str(tmp_path))

        quarantines = _corrupt_names(tmp_path, "decision_history.json")
        assert len(quarantines) == 1
        assert not (tmp_path / "decision_history.json").exists()
        assert quarantines[0].read_text() == original

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "decision history" in errors[0].getMessage()

        assert decisions == []
        assert any("quarantined" in w for w in warnings)


class TestDegenerateFileStates:
    def test_missing_files_quiet_empty(self, tmp_path, caplog):
        with caplog.at_level(logging.ERROR, logger="utils"):
            trades, trade_warnings = _load_trades_ingress(str(tmp_path))
            decisions, decision_warnings = _load_decisions_ingress(str(tmp_path))
        assert trades == [] and trade_warnings == []
        assert decisions == [] and decision_warnings == []
        assert not [r for r in caplog.records if r.levelno == logging.ERROR]

    def test_empty_lists_quiet(self, tmp_path, caplog, capsys):
        _trades_file(tmp_path, "[]")
        _decisions_file(tmp_path, "[]")
        with caplog.at_level(logging.ERROR, logger="utils"):
            trades, trade_warnings = _load_trades_ingress(str(tmp_path))
            decisions, decision_warnings = _load_decisions_ingress(str(tmp_path))
        assert trades == [] and decisions == []
        assert trade_warnings == [] and decision_warnings == []
        assert capsys.readouterr().out == ""  # healthy absence stays quiet


# ---------------------------------------------------------------------------
# Trade record-level drops
# ---------------------------------------------------------------------------

class TestMalformedTradeDrops:
    def test_non_dict_trade_dropped_loudly_sibling_round_trip_survives(self, tmp_path, capsys):
        _trades_file(tmp_path, json.dumps([42, GOOD_TRADE_BUY, GOOD_TRADE_SELL]))

        trades, warnings = _load_trades_ingress(str(tmp_path))

        assert trades == [GOOD_TRADE_BUY, GOOD_TRADE_SELL]
        out = capsys.readouterr().out
        assert "index 0" in out and "expected dict" in out
        assert any("Dropped 1 malformed record" in w for w in warnings)
        # Blast radius: the surviving pair still matches end-to-end.
        rts = churn_analysis.match_round_trips(trades)
        assert len(rts) == 1 and rts[0].pnl == 5.0

    def test_missing_ticker_or_action_dropped_loudly(self, tmp_path, capsys):
        no_ticker = {"timestamp": "2026-10-13T21:00:00", "action": "buy", "price": 1.0}
        no_action = {"timestamp": "2026-10-13T21:00:00", "ticker": "GLD", "price": 1.0}
        _trades_file(tmp_path, json.dumps([no_ticker, no_action, GOOD_TRADE_BUY]))

        trades, warnings = _load_trades_ingress(str(tmp_path))

        assert trades == [GOOD_TRADE_BUY]
        out = capsys.readouterr().out
        assert "missing string 'ticker' or 'action' key" in out
        assert any("Dropped 2 malformed record(s)" in w for w in warnings)

    def test_tz_aware_timestamp_dropped_sibling_survives(self, tmp_path, capsys):
        # A tz-aware timestamp PARSES — and then detonates at the first
        # comparison/subtraction against naive datetime.min siblings inside
        # match_round_trips, voiding EVERY round trip (anti-Lipschitz).
        tz_trade = {**GOOD_TRADE_BUY, "timestamp": TZ_AWARE_TS}
        _trades_file(tmp_path, json.dumps([tz_trade, GOOD_TRADE_BUY, GOOD_TRADE_SELL]))

        trades, warnings = _load_trades_ingress(str(tmp_path))

        assert trades == [GOOD_TRADE_BUY, GOOD_TRADE_SELL]
        out = capsys.readouterr().out
        assert "timezone-aware timestamp" in out and TZ_AWARE_TS in out
        assert any("Dropped 1 malformed record" in w for w in warnings)
        assert len(churn_analysis.match_round_trips(trades)) == 1  # sibling intact

    def test_missing_timestamp_kept_documented_fallback(self, tmp_path, capsys):
        # Absence is not failure: _parse_trade_timestamp's datetime.min
        # fallback is test-pinned existing contract. Exercised through the
        # public wrapper — bounded healthy pin, passes on BOTH sides.
        no_ts = {"ticker": "GLD", "action": "buy", "price": 210.0}
        _trades_file(tmp_path, json.dumps([no_ts, GOOD_TRADE_SELL]))

        trades = churn_analysis.load_trades(str(tmp_path))

        assert trades == [no_ts, GOOD_TRADE_SELL]
        assert capsys.readouterr().out == ""

    def test_unparseable_timestamp_kept_documented_fallback(self, tmp_path, capsys):
        bad_ts = {**GOOD_TRADE_BUY, "timestamp": "not-a-date"}
        _trades_file(tmp_path, json.dumps([bad_ts, GOOD_TRADE_SELL]))

        trades = churn_analysis.load_trades(str(tmp_path))

        assert trades == [bad_ts, GOOD_TRADE_SELL]
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Decision record-level drops
# ---------------------------------------------------------------------------

class TestMalformedDecisionDrops:
    def test_non_dict_decision_dropped_sibling_survives(self, tmp_path, capsys):
        _decisions_file(tmp_path, json.dumps(["nope", GOOD_DECISION]))

        decisions, warnings = _load_decisions_ingress(str(tmp_path))

        assert decisions == [GOOD_DECISION]
        out = capsys.readouterr().out
        assert "index 0" in out and "expected dict" in out
        assert any("Dropped 1 malformed record" in w for w in warnings)

    def test_non_string_timestamp_dropped_loudly(self, tmp_path, capsys):
        bad = {"timestamp": 20261014, "actions": []}
        _decisions_file(tmp_path, json.dumps([bad, GOOD_DECISION]))

        decisions, _ = _load_decisions_ingress(str(tmp_path))

        assert decisions == [GOOD_DECISION]
        out = capsys.readouterr().out
        assert "missing/non-string timestamp" in out

    def test_unparseable_timestamp_dropped_loudly(self, tmp_path, capsys):
        bad = {"timestamp": "yesterday-ish", "actions": []}
        _decisions_file(tmp_path, json.dumps([bad, GOOD_DECISION]))

        decisions, _ = _load_decisions_ingress(str(tmp_path))

        assert decisions == [GOOD_DECISION]
        assert "unparseable timestamp" in capsys.readouterr().out

    def test_tz_aware_timestamp_dropped_loudly(self, tmp_path, capsys):
        # Parses fine; detonates one op downstream at
        # ``(last - first).days`` in the date-range block (naive/aware).
        bad = {"timestamp": TZ_AWARE_TS, "actions": []}
        _decisions_file(tmp_path, json.dumps([bad, GOOD_DECISION]))

        decisions, _ = _load_decisions_ingress(str(tmp_path))

        assert decisions == [GOOD_DECISION]
        out = capsys.readouterr().out
        assert "timezone-aware timestamp" in out and TZ_AWARE_TS in out


# ---------------------------------------------------------------------------
# Sub-element guards: actions items inside decision records
# ---------------------------------------------------------------------------

class TestMalformedActionItemDrops:
    def _decision(self, actions):
        return {"timestamp": "2026-10-14T21:00:00", "actions": actions}

    def test_non_dict_action_item_dropped_sibling_survives(self, capsys):
        d = self._decision([42, {"ticker": "GLD", "action": "buy"},
                            {"ticker": "GLD", "action": "sell"}])
        metrics = churn_analysis.analyze_churn([], [], [d])

        out = capsys.readouterr().out
        assert "malformed action item" in out
        # The two valid non-hold items are one buy→sell flip; the malformed
        # item must not void them.
        assert metrics["action_flips"] == 1

    def test_action_item_missing_keys_dropped(self, capsys):
        d = self._decision([{"action": "buy"}, {"ticker": "GLD"},
                            {"ticker": "GLD", "action": "hold"}])
        metrics = churn_analysis.analyze_churn([], [], [d])

        out = capsys.readouterr().out
        assert out.count("malformed action item") == 2
        # Only the valid 'hold' remains → no flips; the decision's own
        # timestamp-based metrics were still computed (single timestamp →
        # 0 active days, no crash).
        assert metrics["action_flips"] == 0
        assert metrics["days_active"] == 0

    def test_non_list_actions_payload_dropped_decision_kept(self, capsys):
        d = self._decision("buy")
        metrics = churn_analysis.analyze_churn([], [], [d])

        out = capsys.readouterr().out
        assert "malformed 'actions' payload" in out
        assert metrics["action_flips"] == 0
        assert metrics["days_active"] == 0  # decision survives the bad payload

    def test_absent_and_none_actions_quiet(self, capsys):
        d1 = {"timestamp": "2026-10-14T21:00:00"}
        d2 = {"timestamp": "2026-10-15T21:00:00", "actions": None}
        metrics = churn_analysis.analyze_churn([], [], [d1, d2])

        assert capsys.readouterr().out == ""  # legitimate absence stays quiet
        assert metrics["action_flips"] == 0
        assert metrics["days_active"] == 1


# ---------------------------------------------------------------------------
# Artifact embedding: warnings visible in the printed report
# ---------------------------------------------------------------------------

class TestArtifactEmbedding:
    def test_main_prints_quarantine_warning_before_report(self, tmp_path, monkeypatch, capsys):
        data = tmp_path / "data"
        data.mkdir()
        _trades_file(data, "{corrupt")
        _decisions_file(data, json.dumps([GOOD_DECISION]))
        monkeypatch.chdir(tmp_path)

        churn_analysis.main()

        out = capsys.readouterr().out
        warning_pos = out.index("quarantined")
        report_pos = out.index("PORTFOLIO CHURN ANALYSIS")
        assert warning_pos < report_pos  # degradation leads the artifact
        # The analysis still renders end-to-end on the surviving decision.
        assert "Ledger Reconciliation" in out

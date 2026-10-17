"""Per-element robustness contract for ``TradingAgent.load_recent_decisions``.

The decision-history file feeds the LLM prompt context on every daily run.
Two defects lived at this ingress:

1. **All-or-nothing filter swallow.** A single malformed record — a
   non-dict element, a missing ``timestamp`` key, an unparseable or
   timezone-aware timestamp — raised inside the list comprehension, was
   caught by the broad ``except Exception`` around the *whole* filter,
   and returned ``[]``: one bad element voided the n-1 valid siblings
   and the LLM lost its entire recent-decision context. That is
   anti-Lipschitz — the scope of the failure exceeded the scope of the
   malformation (PR #64 doctrine). The fix drops each malformed record
   individually with one loud warning naming the record and the reason;
   valid siblings survive unchanged.

2. **Lenient parse at a quarantine-class boundary.** The raw
   ``json.load`` accepted ``NaN``/``Infinity`` tokens no repo writer can
   emit (every persistence path is strict ``allow_nan=False``) and a
   corrupt file was left in place for the next ``save_decision()`` to
   silently overwrite — the PR #60 evidence-destruction class. The load
   now goes through the shared ``load_json_list_or_quarantine`` ingress
   (the same one ``save_decision`` uses): unparseable/token/wrong-shape
   files are quarantined whole (bytes preserved, one ERROR log), and the
   strict-reader hook rejects non-standard tokens at the parse boundary
   (PR #84 convention).

Discriminators assert the mechanism (quarantine rename, per-record
warning count and content, valid siblings surviving); bounded healthy
pins (missing file, all-valid round-trip, cutoff window, no warnings on
healthy loads) must pass on BOTH sides of a fail-on-old stash.
"""

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.trading_agent import TradingAgent

AGENT_LOGGER = "llm.trading_agent"


def _decision(hours_ago=1, **extra):
    # Naive on purpose: history timestamps are naive ISO strings compared
    # against a naive cutoff (house convention).
    ts = (datetime.now() - timedelta(hours=hours_ago)).isoformat()  # noqa: DTZ005
    d = {"timestamp": ts, "actions": [], "reasoning": f"decision {hours_ago}h ago"}
    d.update(extra)
    return d


def _make_agent(tmp_path):
    return TradingAgent(
        api_key="test", history_file=str(tmp_path / "decision_history.json")
    )


def _history_path(tmp_path):
    return tmp_path / "decision_history.json"


def _write(tmp_path, obj):
    path = _history_path(tmp_path)
    path.write_text(json.dumps(obj))
    return path


def _write_token_file(tmp_path, text):
    path = _history_path(tmp_path)
    path.write_text(text)
    return path


def _quarantine_files(path):
    return list(path.parent.glob(path.name + ".corrupt-*"))


def _agent_records(caplog, level):
    return [
        r for r in caplog.records
        if r.name == AGENT_LOGGER and r.levelno == level
    ]


def _error_records(caplog):
    """ERROR records from any logger (the quarantine helper logs under
    ``utils``, not the agent's logger — the mechanism is the rename and
    the loud error, whichever module emits it)."""
    return [r for r in caplog.records if r.levelno == logging.ERROR]


class TestQuarantineIngress:
    """File-level boundary: unparseable/token/wrong-shape files quarantine whole."""

    def test_token_file_quarantined_and_returns_empty(self, tmp_path, caplog):
        """A NaN token is data no repo writer can emit; the strict reader
        rejects it, the file quarantines whole (PR #84 branch 2), and the
        load honestly returns []. Pre-fix the token parsed silently and
        the NaN rode into the LLM context."""
        decision = _decision(1, pnl_pct=float("nan"))
        # json.dumps default allow_nan=True emits the literal NaN token.
        path = _write_token_file(tmp_path, json.dumps([decision]))
        agent = _make_agent(tmp_path)

        with caplog.at_level(logging.WARNING):
            result = agent.load_recent_decisions(days=5)

        assert result == []
        quarantined = _quarantine_files(path)
        assert len(quarantined) == 1
        assert "NaN" in quarantined[0].read_text()  # bytes preserved
        errors = _error_records(caplog)
        assert len(errors) == 1
        assert "decision history" in errors[0].getMessage()

    def test_garbage_bytes_quarantined(self, tmp_path, caplog):
        path = _write_token_file(tmp_path, "\x80\x81 not json at all")
        agent = _make_agent(tmp_path)

        with caplog.at_level(logging.WARNING):
            result = agent.load_recent_decisions(days=5)

        assert result == []
        assert len(_quarantine_files(path)) == 1
        assert len(_error_records(caplog)) == 1

    def test_wrong_shape_dict_quarantined(self, tmp_path, caplog):
        """A parseable non-list top level makes no claim usable as a
        decision list; quarantine owns it (no per-element handling of a
        file that never was a list)."""
        path = _write(tmp_path, {"history": []})
        agent = _make_agent(tmp_path)

        with caplog.at_level(logging.WARNING):
            result = agent.load_recent_decisions(days=5)

        assert result == []
        assert len(_quarantine_files(path)) == 1
        assert len(_error_records(caplog)) == 1

    def test_save_after_quarantine_starts_clean_ledger(self, tmp_path, caplog):
        """Composition pin: after the read quarantines a token file, the
        next save starts a fresh ledger at the original path (the PR #85
        honest no-claim path), and the corrupted bytes survive under the
        quarantine name for manual recovery."""
        decision = _decision(1, pnl_pct=float("inf"))
        path = _write_token_file(tmp_path, json.dumps([decision]))
        agent = _make_agent(tmp_path)

        with caplog.at_level(logging.WARNING):
            assert agent.load_recent_decisions(days=5) == []
            agent.save_decision(_decision(0))

        assert path.exists()  # original path free for the clean rewrite
        reloaded = json.loads(path.read_text())
        assert len(reloaded) == 1
        assert "Infinity" in _quarantine_files(path)[0].read_text()


class TestPerElementDrop:
    """Element-level boundary: one malformed record must not void the siblings."""

    def test_single_bad_element_does_not_void_siblings(self, tmp_path, caplog):
        """THE core discriminator. A record missing 'timestamp' among valid
        ones: pre-fix the KeyError voided the whole filter -> []; post-fix
        the bad record drops loudly and the valid tail survives."""
        good = [_decision(h) for h in (3, 2, 1)]
        bad = {"actions": [], "reasoning": "recorded without a timestamp"}
        path = _write(tmp_path, [good[0], bad, good[1], good[2]])
        agent = _make_agent(tmp_path)

        with caplog.at_level(logging.WARNING):
            result = agent.load_recent_decisions(days=5)

        assert [r["reasoning"] for r in result] == [
            "decision 3h ago", "decision 2h ago", "decision 1h ago",
        ]
        assert len(_quarantine_files(path)) == 0  # parseable file untouched
        warnings = _agent_records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert "missing 'timestamp'" in warnings[0].getMessage()

    def test_unparseable_timestamp_dropped_loudly(self, tmp_path, caplog):
        bad_ts = "2026-13-45T99:99:99"
        _write(tmp_path, [_decision(2), {"timestamp": bad_ts}, _decision(1)])
        agent = _make_agent(tmp_path)

        with caplog.at_level(logging.WARNING):
            result = agent.load_recent_decisions(days=5)

        assert len(result) == 2
        warnings = _agent_records(caplog, logging.WARNING)
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "unusable" in msg and bad_ts in msg

    def test_non_dict_element_dropped_with_type(self, tmp_path, caplog):
        _write(tmp_path, [_decision(2), "just a string", _decision(1)])
        agent = _make_agent(tmp_path)

        with caplog.at_level(logging.WARNING):
            result = agent.load_recent_decisions(days=5)

        assert len(result) == 2
        warnings = _agent_records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert "not a mapping" in warnings[0].getMessage()
        assert "str" in warnings[0].getMessage()

    def test_timezone_aware_timestamp_dropped_siblings_survive(self, tmp_path, caplog):
        """A tz-aware timestamp parses but cannot compare against the naive
        cutoff (TypeError). Pre-fix this voided the whole history; post-fix
        it drops just that record."""
        aware_ts = (datetime.now(timezone(timedelta(hours=2)))
                    - timedelta(hours=2)).isoformat()
        _write(tmp_path, [_decision(2), {"timestamp": aware_ts}, _decision(1)])
        agent = _make_agent(tmp_path)

        with caplog.at_level(logging.WARNING):
            result = agent.load_recent_decisions(days=5)

        assert [r["reasoning"] for r in result] == [
            "decision 2h ago", "decision 1h ago",
        ]
        assert len(_agent_records(caplog, logging.WARNING)) == 1

    def test_one_warning_per_dropped_element_no_batching(self, tmp_path, caplog):
        _write(
            tmp_path,
            [_decision(3), "a string", {"no": "timestamp"}, _decision(1)],
        )
        agent = _make_agent(tmp_path)

        with caplog.at_level(logging.WARNING):
            result = agent.load_recent_decisions(days=5)

        assert len(result) == 2
        assert len(_agent_records(caplog, logging.WARNING)) == 2

    def test_quarantine_never_fires_on_parseable_file(self, tmp_path, caplog):
        """Layering cross-check (PR #81 tree at two resolutions): per-element
        drops own parseable files with malformed members; quarantine never
        fires on a file that parses. The original bytes stay in place."""
        path = _write(tmp_path, [_decision(1), {"reasoning": "no ts"}])
        agent = _make_agent(tmp_path)

        with caplog.at_level(logging.WARNING):
            result = agent.load_recent_decisions(days=5)

        assert len(result) == 1
        assert path.exists()
        assert len(_quarantine_files(path)) == 0


class TestHealthyPathPins:
    """Bounded pins: expected absence stays quiet; behavior on healthy
    files is byte-identical to the pre-fix contract. Must pass on BOTH
    sides of a fail-on-old stash."""

    def test_missing_file_quiet_empty(self, tmp_path, caplog):
        agent = _make_agent(tmp_path)

        with caplog.at_level(logging.WARNING):
            result = agent.load_recent_decisions(days=5)

        assert result == []
        assert _agent_records(caplog, logging.WARNING) == []
        assert _agent_records(caplog, logging.ERROR) == []

    def test_all_valid_roundtrip_returns_last_five(self, tmp_path, caplog):
        oldest_first = [_decision(h) for h in (7, 6, 5, 4, 3, 2, 1)]
        _write(tmp_path, oldest_first)
        agent = _make_agent(tmp_path)

        with caplog.at_level(logging.WARNING):
            result = agent.load_recent_decisions(days=5)

        assert [r["reasoning"] for r in result] == [
            "decision 5h ago", "decision 4h ago", "decision 3h ago",
            "decision 2h ago", "decision 1h ago",
        ]
        assert _agent_records(caplog, logging.WARNING) == []

    def test_cutoff_window_excludes_old_decisions(self, tmp_path, caplog):
        decisions = [
            _decision(24 * 10),  # 10 days ago — outside the window
            _decision(24 * 6),   # 6 days ago — outside
            _decision(1),        # inside
        ]
        _write(tmp_path, decisions)
        agent = _make_agent(tmp_path)

        with caplog.at_level(logging.WARNING):
            result = agent.load_recent_decisions(days=5)

        assert [r["reasoning"] for r in result] == ["decision 1h ago"]
        assert _agent_records(caplog, logging.WARNING) == []

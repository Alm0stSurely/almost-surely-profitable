"""Reproduction gate: churn-analysis ledger ingress loudness.

Asserts the loudness contract at every fixed site of
``churn_analysis.main()`` (via ``_load_trades_ingress`` /
``_load_decisions_ingress`` and the ``analyze_churn`` action guards):

  1. Corrupt-bytes trade ledger    -> quarantined (.corrupt-*), ERROR logged,
     bytes preserved, warning leads the report, report still rendered (old
     code: JSONDecodeError crash, corrupt file left in place for the next
     save to overwrite — the PR #60 evidence-destruction loop).
  2. NaN-token trade ledger        -> quarantined (strict parse hook, PR #84)
     — no writer can emit NaN, so the token is corruption, silently ingested
     by the old raw json.load.
  3. Wrong-shape decision history  -> quarantined (parses to a dict, not a
     list; old code: TypeError on the first list operation).
  4. Non-dict trade element        -> dropped loudly (print), sibling round
     trip survives (old code: AttributeError inside match_round_trips, whole
     analysis lost).
  5. Tz-aware trade timestamp      -> dropped loudly, sibling round trip
     survives (old code: parses fine, then TypeError at the naive/aware
     comparison inside match_round_trips — PR #86 whole-element rule).
  6. Missing ticker/action trade   -> dropped loudly, sibling survives (old
     code: KeyError inside match_round_trips).
  7. Non-dict decision element     -> dropped loudly, sibling survives (old
     code: TypeError at d["timestamp"][:10], whole analysis lost).
  8. Unparseable decision ts       -> dropped loudly, sibling survives (old
     code: ValueError at datetime.fromisoformat(decisions[0]["timestamp"])).
  9. Tz-aware decision timestamp   -> dropped loudly, sibling survives (old
     code: parses fine, then TypeError at (last - first).days one op
     downstream of a successful parse).
 10. Malformed action items        -> dropped loudly one-per-item, valid
     siblings still counted as flips (old code: KeyError/TypeError on the
     subscript, whole analysis lost).
 11. Non-list actions payload      -> dropped loudly, decision's own
     timestamp metrics survive (old code: TypeError iterating the payload).

Healthy-path pins (must stay quiet on BOTH sides of the fix):
 12. Valid ledgers    -> report rendered, zero warnings, files intact.
 13. Missing ledgers  -> report rendered, zero warnings (old code: same).

Exit 1 if any check fails, 0 if all pass. Checks are written defensively and
the scenario harness wraps main() in try/except (PR #86/#87 gate rules) so
every pre-fix expectation lands as a failed check line instead of a crash —
a gate that limps through all scenarios with the full red set proves more
than one that dies early.

Run from the repo root:
    .venv/bin/python benchmarks/repro_benchmark_churn_analysis_ingress.py
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis import churn_analysis

FAILURES = []

GOOD_DECISION = {
    "timestamp": "2026-10-14T21:00:00",
    "actions": [{"ticker": "GLD", "action": "hold"}],
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


def check(name, cond, detail=""):
    status = "OK " if cond else "XX "
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


class _Recorder(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _run_main_in_tmp(decisions_content=None, trades_content=None):
    """Run churn_analysis.main() against a temp data dir (chdir-based, since
    main() reads the relative 'data/' path).

    Contents are written verbatim (lets the gate plant literal NaN tokens no
    strict writer can emit). Captures quarantine artifact CONTENTS inside the
    tempdir context (PR #87 gate rule) and records any main() exception as
    ``error`` so the harness limps through every scenario (PR #86 gate rule).
    """
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()
        if decisions_content is not None:
            (data_dir / "decision_history.json").write_text(decisions_content)
        if trades_content is not None:
            (data_dir / "trades_history.json").write_text(trades_content)

        recorder = _Recorder()
        root = logging.getLogger()
        root.addHandler(recorder)
        old_cwd = os.getcwd()
        error = None
        stdout = ""
        try:
            os.chdir(tmp)
            import contextlib
            import io as _io
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    churn_analysis.main()
                except Exception as exc:  # noqa: BLE001 — gate must limp
                    error = f"{type(exc).__name__}: {exc}"
            stdout = buf.getvalue()
        finally:
            os.chdir(old_cwd)
            root.removeHandler(recorder)

        quarantines = {}
        for name in ("trades_history.json", "decision_history.json"):
            files = sorted(data_dir.glob(f"{name}.corrupt-*"))
            quarantines[name] = [
                (f.name, f.read_text()) for f in files
            ]
        originals = {
            name: (data_dir / name).exists()
            for name in ("trades_history.json", "decision_history.json")
        }
        return {
            "error": error,
            "stdout": stdout,
            "quarantines": quarantines,
            "originals": originals,
            "errors": [r for r in recorder.records if r.levelno == logging.ERROR],
        }


def scenario_corrupt_trades_bytes():
    print("\n-- scenario 1: corrupt-bytes trade ledger")
    original = "{not json"
    r = _run_main_in_tmp(
        decisions_content=json.dumps([GOOD_DECISION]),
        trades_content=original,
    )
    q = r["quarantines"]["trades_history.json"]
    check("s1: no crash", r["error"] is None, r["error"])
    check("s1: trades file quarantined exactly once", len(q) == 1)
    check("s1: quarantine bytes preserved", bool(q) and q[0][1] == original)
    check("s1: original path cleared", not r["originals"]["trades_history.json"])
    check("s1: exactly one ERROR log",
          len(r["errors"]) == 1 and "trade ledger" in r["errors"][0].getMessage())
    check("s1: warning leads the report",
          "quarantined" in r["stdout"]
          and "PORTFOLIO CHURN ANALYSIS" in r["stdout"]
          and r["stdout"].index("quarantined") < r["stdout"].index("PORTFOLIO CHURN ANALYSIS"))
    check("s1: report still rendered end-to-end", "Ledger Reconciliation" in r["stdout"])


def scenario_nan_token_trades():
    print("\n-- scenario 2: NaN-token trade ledger (strict parse hook)")
    original = json.dumps([{**GOOD_TRADE_BUY, "price": float("nan")}])
    r = _run_main_in_tmp(trades_content=original)
    q = r["quarantines"]["trades_history.json"]
    check("s2: no crash", r["error"] is None, r["error"])
    check("s2: NaN file quarantined, bytes preserved",
          len(q) == 1 and q[0][1] == original)


def scenario_wrong_shape_decisions():
    print("\n-- scenario 3: wrong-shape decision history (dict, not list)")
    original = '{"timestamp": "2026-10-14T21:00:00"}'
    r = _run_main_in_tmp(
        decisions_content=original,
        trades_content=json.dumps([GOOD_TRADE_BUY, GOOD_TRADE_SELL]),
    )
    q = r["quarantines"]["decision_history.json"]
    check("s3: no crash", r["error"] is None, r["error"])
    check("s3: wrong-shape file quarantined, bytes preserved",
          len(q) == 1 and q[0][1] == original)
    check("s3: ERROR log mentions wrong shape",
          any("wrong shape" in e.getMessage() for e in r["errors"]))
    check("s3: report rendered", "PORTFOLIO CHURN ANALYSIS" in r["stdout"])


def scenario_non_dict_trade():
    print("\n-- scenario 4: non-dict trade element")
    r = _run_main_in_tmp(
        decisions_content=json.dumps([GOOD_DECISION]),
        trades_content=json.dumps([42, GOOD_TRADE_BUY, GOOD_TRADE_SELL]),
    )
    check("s4: no crash", r["error"] is None, r["error"])
    check("s4: loud per-record drop print",
          "Dropping malformed trade record at index 0" in r["stdout"])
    check("s4: file intact (element drop, not quarantine)",
          r["originals"]["trades_history.json"]
          and not r["quarantines"]["trades_history.json"])
    check("s4: drop count warning in artifact",
          "Dropped 1 malformed record" in r["stdout"])
    check("s4: sibling round trip survived",
          "Round Trips:          1" in r["stdout"])


def scenario_tz_aware_trade():
    print("\n-- scenario 5: tz-aware trade timestamp (parses, then detonates)")
    tz_trade = {**GOOD_TRADE_BUY, "timestamp": "2026-10-13T21:00:00+02:00"}
    r = _run_main_in_tmp(
        decisions_content=json.dumps([GOOD_DECISION]),
        trades_content=json.dumps([tz_trade, GOOD_TRADE_BUY, GOOD_TRADE_SELL]),
    )
    check("s5: no crash", r["error"] is None, r["error"])
    check("s5: loud tz-aware drop print", "timezone-aware timestamp" in r["stdout"])
    check("s5: sibling round trip survived",
          "Round Trips:          1" in r["stdout"])


def scenario_missing_ticker_or_action():
    print("\n-- scenario 6: missing ticker/action trade")
    no_ticker = {"timestamp": "2026-10-13T21:00:00", "action": "buy", "price": 1.0}
    r = _run_main_in_tmp(
        decisions_content=json.dumps([GOOD_DECISION]),
        trades_content=json.dumps([no_ticker, GOOD_TRADE_BUY, GOOD_TRADE_SELL]),
    )
    check("s6: no crash", r["error"] is None, r["error"])
    check("s6: loud drop print", "missing string 'ticker' or 'action' key" in r["stdout"])
    check("s6: sibling round trip survived",
          "Round Trips:          1" in r["stdout"])


def scenario_non_dict_decision():
    print("\n-- scenario 7: non-dict decision element")
    r = _run_main_in_tmp(
        decisions_content=json.dumps(["nope", GOOD_DECISION]),
        trades_content=json.dumps([GOOD_TRADE_BUY, GOOD_TRADE_SELL]),
    )
    check("s7: no crash", r["error"] is None, r["error"])
    check("s7: loud drop print", "Dropping malformed decision record at index 0" in r["stdout"])
    check("s7: sibling decision survived (days rendered)", "Trades/Week:" in r["stdout"])


def scenario_unparseable_decision_ts():
    print("\n-- scenario 8: unparseable decision timestamp")
    bad = {"timestamp": "yesterday-ish", "actions": []}
    r = _run_main_in_tmp(
        decisions_content=json.dumps([bad, GOOD_DECISION]),
        trades_content="[]",
    )
    check("s8: no crash", r["error"] is None, r["error"])
    check("s8: loud drop print", "unparseable timestamp" in r["stdout"])
    check("s8: sibling survived", "PORTFOLIO CHURN ANALYSIS" in r["stdout"])


def scenario_tz_aware_decision():
    print("\n-- scenario 9: tz-aware decision timestamp (parses, then detonates)")
    bad = {"timestamp": "2026-10-14T21:00:00+02:00", "actions": []}
    r = _run_main_in_tmp(
        decisions_content=json.dumps([bad, GOOD_DECISION]),
        trades_content="[]",
    )
    check("s9: no crash", r["error"] is None, r["error"])
    check("s9: loud drop print", "timezone-aware timestamp" in r["stdout"])
    check("s9: sibling survived", "PORTFOLIO CHURN ANALYSIS" in r["stdout"])


def scenario_malformed_action_items():
    print("\n-- scenario 10: malformed action items inside a decision")
    decision = {
        "timestamp": "2026-10-14T21:00:00",
        "actions": [
            42,
            {"action": "buy"},
            {"ticker": "GLD", "action": "buy"},
            {"ticker": "GLD", "action": "sell"},
        ],
    }
    r = _run_main_in_tmp(
        decisions_content=json.dumps([decision]),
        trades_content="[]",
    )
    check("s10: no crash", r["error"] is None, r["error"])
    check("s10: two loud item-drop prints", r["stdout"].count("malformed action item") == 2)
    check("s10: valid buy→sell pair still counted", "Action Flips:         1" in r["stdout"])


def scenario_non_list_actions():
    print("\n-- scenario 11: non-list actions payload")
    decision = {"timestamp": "2026-10-14T21:00:00", "actions": "buy"}
    r = _run_main_in_tmp(
        decisions_content=json.dumps([decision]),
        trades_content="[]",
    )
    check("s11: no crash", r["error"] is None, r["error"])
    check("s11: loud payload-drop print", "malformed 'actions' payload" in r["stdout"])
    check("s11: decision survived (report rendered)", "PORTFOLIO CHURN ANALYSIS" in r["stdout"])


def scenario_healthy_valid_ledgers():
    print("\n-- scenario 12: healthy valid ledgers stay quiet (both-sides pin)")
    r = _run_main_in_tmp(
        decisions_content=json.dumps([GOOD_DECISION]),
        trades_content=json.dumps([GOOD_TRADE_BUY, GOOD_TRADE_SELL]),
    )
    check("s12: no crash", r["error"] is None, r["error"])
    check("s12: zero ERROR logs", not r["errors"])
    check("s12: zero drop prints / warnings",
          "Dropping malformed" not in r["stdout"] and "⚠" not in r["stdout"])
    check("s12: files intact",
          r["originals"]["trades_history.json"] and r["originals"]["decision_history.json"])
    check("s12: round trip counted", "Round Trips:          1" in r["stdout"])


def scenario_healthy_missing_files():
    print("\n-- scenario 13: missing ledger files stay quiet (both-sides pin)")
    r = _run_main_in_tmp()
    check("s13: no crash", r["error"] is None, r["error"])
    check("s13: zero ERROR logs", not r["errors"])
    check("s13: report rendered", "PORTFOLIO CHURN ANALYSIS" in r["stdout"])


def main():
    scenarios = [
        scenario_corrupt_trades_bytes,
        scenario_nan_token_trades,
        scenario_wrong_shape_decisions,
        scenario_non_dict_trade,
        scenario_tz_aware_trade,
        scenario_missing_ticker_or_action,
        scenario_non_dict_decision,
        scenario_unparseable_decision_ts,
        scenario_tz_aware_decision,
        scenario_malformed_action_items,
        scenario_non_list_actions,
        scenario_healthy_valid_ledgers,
        scenario_healthy_missing_files,
    ]
    for s in scenarios:
        s()
    print(f"\n{'=' * 60}")
    if FAILURES:
        print(f"GATE FAILED: {len(FAILURES)} check(s) red:")
        for f in FAILURES:
            print(f"  XX {f}")
        return 1
    print("GATE PASSED: all checks green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

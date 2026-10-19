"""Reproduction gate: behavioral-analysis ledger ingress loudness.

Asserts the loudness contract at every fixed site of
``behavioral_analysis._load_analysis_histories`` (called by ``main()``):

  1. Corrupt-bytes decision history -> quarantined (.corrupt-*), ERROR logged,
     bytes preserved, report still rendered (old code: JSONDecodeError crash,
     corrupt file left in place for the next save to overwrite).
  2. NaN-token decision history   -> quarantined (strict parse hook, PR #84) —
     no writer can emit NaN, so the token is corruption, silently ingested
     by the old raw json.load.
  3. Wrong-shape decision history -> quarantined (parses to a dict, not a list;
     old code: iterates zero keys then IndexError on decisions[0]).
  4. Non-dict decision element    -> dropped loudly (print), siblings survive,
     artifact notes the drop (old code: AttributeError, whole report lost).
  5. Bad timestamp decision       -> int / unparseable / tz-aware strings all
     dropped loudly, sibling survives (old code: TypeError/ValueError at the
     slice/parse, TypeError at the naive/aware subtraction one op downstream
     of a successful parse — PR #86 whole-element-evaluation rule).
  6. Corrupt-bytes trade ledger   -> quarantined, ERROR logged, artifact notes it.
  7. Non-dict trade element       -> dropped loudly, sibling round trip survives
     (old code: AttributeError inside match_round_trips, whole report lost).
  8. Missing ticker/action trade  -> dropped loudly, sibling survives (old code:
     KeyError inside match_round_trips).
  9. Empty decision history []    -> report rendered (old code: IndexError on
     decisions[0] — a legitimate fresh-account state).
 10. Missing ledger files         -> quiet empty report (old code:
     FileNotFoundError).

Healthy-path pins (must stay quiet on BOTH sides of the fix):
 11. Valid histories   -> report rendered, zero warnings, files intact.
 12. Empty trade ledger -> report rendered, zero warnings, file intact.

Exit 1 if any check fails, 0 if all pass. Checks are written defensively
and the scenario harness wraps main() in try/except (PR #86/#87 gate rules)
so every pre-fix expectation lands as a failed check line instead of a
crash — a gate that limps through all scenarios with the full red set proves
more than one that dies early.

Run from the repo root:
    .venv/bin/python benchmarks/repro_benchmark_behavioral_analysis_ingress.py
"""

import contextlib
import io
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis import behavioral_analysis

FAILURES = []

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
    """Run behavioral_analysis.main() against a temp data dir.

    Contents are written verbatim (lets the gate plant literal NaN tokens no
    strict writer can emit). Captures the report artifact CONTENTS inside the
    tempdir context (PR #87 gate rule) and records any main() exception as
    ``error`` so the harness limps through every scenario (PR #86 gate rule).
    """
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        results_dir = Path(tmp) / "results"
        out_dir = Path(tmp) / "out"
        data_dir.mkdir()
        if decisions_content is not None:
            (data_dir / "decision_history.json").write_text(decisions_content)
        if trades_content is not None:
            (data_dir / "trades_history.json").write_text(trades_content)

        old = (
            behavioral_analysis.DATA_DIR,
            behavioral_analysis.RESULTS_DIR,
            behavioral_analysis.OUTPUT_DIR,
        )
        behavioral_analysis.DATA_DIR = data_dir
        behavioral_analysis.RESULTS_DIR = results_dir
        behavioral_analysis.OUTPUT_DIR = out_dir

        handler = _Recorder()
        root = logging.getLogger()
        old_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)

        buf = io.StringIO()
        error = None
        try:
            with contextlib.redirect_stdout(buf):
                behavioral_analysis.main()
        except Exception as exc:  # noqa: BLE001 - the gate must record, not die
            error = f"{type(exc).__name__}: {exc}"
        finally:
            root.removeHandler(handler)
            root.setLevel(old_level)
            (
                behavioral_analysis.DATA_DIR,
                behavioral_analysis.RESULTS_DIR,
                behavioral_analysis.OUTPUT_DIR,
            ) = old

        # Capture artifact contents INSIDE the TemporaryDirectory context.
        artifacts = {}
        if out_dir.exists():
            for p in out_dir.glob("behavioral_analysis_*.txt"):
                artifacts[p.name] = p.read_text()
        quarantines = {}
        for name in ("decision_history.json", "trades_history.json"):
            files = sorted(data_dir.glob(f"{name}.corrupt-*"))
            quarantines[name] = [
                (f.name, f.read_text()) for f in files
            ]
        originals = {
            name: (data_dir / name).read_text()
            if (data_dir / name).exists()
            else None
            for name in ("decision_history.json", "trades_history.json")
        }

        return {
            "error": error,
            "stdout": buf.getvalue(),
            "artifacts": artifacts,
            "quarantines": quarantines,
            "originals": originals,
            "records": handler.records,
        }


def _errors_above(records, level):
    return [r for r in records if r.levelno >= level]


def _report(res):
    # Defensive: pre-fix code never creates the artifact (PR #86 gate rule).
    return next(iter(res["artifacts"].values()), "")


def scenario_corrupt_decisions():
    res = _run_main_in_tmp(decisions_content="{not json")
    check("S1 corrupt decisions: main() completes", res["error"] is None,
          res["error"] or "")
    qs = res["quarantines"]["decision_history.json"]
    check("S1 quarantined exactly once", len(qs) == 1, f"got {len(qs)}")
    check("S1 original file removed",
          res["originals"]["decision_history.json"] is None)
    check("S1 bytes preserved", bool(qs) and qs[0][1] == "{not json")
    errors = _errors_above(res["records"], logging.ERROR)
    check("S1 exactly one ERROR log", len(errors) == 1, f"got {len(errors)}")
    check("S1 ERROR names decision history",
          any("decision history" in r.getMessage() for r in errors))
    report = _report(res)
    check("S1 artifact notes quarantine",
          "quarantined" in report and "decision_history" in report)
    check("S1 report still rendered", "Total decisions: 0" in report)


def scenario_nan_token():
    content = json.dumps([{**GOOD_DECISION, "cash_pct": float("nan")}])
    assert "NaN" in content
    res = _run_main_in_tmp(decisions_content=content)
    check("S2 NaN token: main() completes", res["error"] is None, res["error"] or "")
    qs = res["quarantines"]["decision_history.json"]
    check("S2 quarantined", len(qs) == 1, f"got {len(qs)}")
    check("S2 bytes preserved", bool(qs) and qs[0][1] == content)


def scenario_wrong_shape():
    content = '{"timestamp": "2026-10-13T21:00:00"}'
    res = _run_main_in_tmp(decisions_content=content)
    check("S3 wrong shape: main() completes", res["error"] is None,
          res["error"] or "")
    qs = res["quarantines"]["decision_history.json"]
    check("S3 quarantined", len(qs) == 1, f"got {len(qs)}")
    errors = _errors_above(res["records"], logging.ERROR)
    check("S3 ERROR mentions wrong shape",
          any("wrong shape" in r.getMessage() for r in errors))


def scenario_non_dict_decision():
    content = json.dumps(["garbage", 12345, GOOD_DECISION])
    res = _run_main_in_tmp(decisions_content=content)
    check("S4 non-dict decisions: main() completes", res["error"] is None,
          res["error"] or "")
    report = _report(res)
    check("S4 sibling survived", "Total decisions: 1" in report)
    check("S4 one print per dropped record",
          res["stdout"].count("Dropping malformed decision record") == 2,
          f"got {res['stdout'].count('Dropping malformed decision record')}")
    check("S4 artifact notes the drops",
          "Dropped 2 malformed record(s) from decision history" in report)
    check("S4 parseable file not quarantined",
          len(res["quarantines"]["decision_history.json"]) == 0
          and res["originals"]["decision_history.json"] is not None)


def scenario_bad_timestamps():
    bad_int = {"timestamp": 20261014, "error": False}
    bad_str = {"timestamp": "not-a-date", "error": False}
    bad_tz = {"timestamp": "2026-10-13T21:00:00+00:00", "error": False}
    content = json.dumps([bad_int, bad_str, bad_tz, GOOD_DECISION])
    res = _run_main_in_tmp(decisions_content=content)
    check("S5 bad timestamps: main() completes", res["error"] is None,
          res["error"] or "")
    report = _report(res)
    check("S5 sibling survived", "Total decisions: 1" in report)
    out = res["stdout"]
    check("S5 int timestamp dropped loudly",
          "missing/non-string timestamp" in out)
    check("S5 unparseable timestamp dropped loudly",
          "unparseable timestamp" in out)
    check("S5 tz-aware timestamp dropped loudly",
          "timezone-aware timestamp" in out)
    check("S5 no quarantine on parseable file",
          len(res["quarantines"]["decision_history.json"]) == 0)


def scenario_corrupt_trades():
    res = _run_main_in_tmp(
        decisions_content=json.dumps([GOOD_DECISION]),
        trades_content="not json either",
    )
    check("S6 corrupt trades: main() completes", res["error"] is None,
          res["error"] or "")
    qs = res["quarantines"]["trades_history.json"]
    check("S6 quarantined", len(qs) == 1, f"got {len(qs)}")
    errors = _errors_above(res["records"], logging.ERROR)
    check("S6 ERROR names trade ledger",
          any("trade ledger" in r.getMessage() for r in errors))
    report = _report(res)
    check("S6 artifact notes quarantine",
          "quarantined" in report and "trades_history" in report)
    check("S6 report rendered", "Round trips: 0" in report)


def scenario_non_dict_trade():
    content = json.dumps([42, GOOD_TRADE_BUY, GOOD_TRADE_SELL])
    res = _run_main_in_tmp(
        decisions_content=json.dumps([GOOD_DECISION]),
        trades_content=content,
    )
    check("S7 non-dict trade: main() completes", res["error"] is None,
          res["error"] or "")
    report = _report(res)
    check("S7 sibling round trip survived", "Round trips: 1" in report)
    check("S7 one print per dropped record",
          res["stdout"].count("Dropping malformed trade record") == 1)
    check("S7 parseable file not quarantined",
          len(res["quarantines"]["trades_history.json"]) == 0)


def scenario_missing_trade_keys():
    no_ticker = {"timestamp": "2026-10-13T21:00:00", "action": "buy", "price": 1.0}
    no_action = {"timestamp": "2026-10-13T21:00:00", "ticker": "X", "price": 1.0}
    content = json.dumps([no_ticker, no_action, GOOD_TRADE_BUY, GOOD_TRADE_SELL])
    res = _run_main_in_tmp(
        decisions_content=json.dumps([GOOD_DECISION]),
        trades_content=content,
    )
    check("S8 missing keys: main() completes", res["error"] is None,
          res["error"] or "")
    report = _report(res)
    check("S8 sibling round trip survived", "Round trips: 1" in report)
    check("S8 two loud drops",
          res["stdout"].count("Dropping malformed trade record") == 2)


def scenario_empty_decisions():
    res = _run_main_in_tmp(decisions_content="[]")
    check("S9 empty decisions: main() completes", res["error"] is None,
          res["error"] or "")
    report = _report(res)
    check("S9 report rendered", "Total decisions: 0" in report)
    check("S9 churn section rendered", "CHURN / ROUND-TRIP ANALYSIS" in report)
    check("S9 no spurious quarantine",
          len(res["quarantines"]["decision_history.json"]) == 0)


def scenario_missing_files():
    res = _run_main_in_tmp()
    check("S10 missing files: main() completes", res["error"] is None,
          res["error"] or "")
    report = _report(res)
    check("S10 report rendered", "Total decisions: 0" in report)
    warnings = _errors_above(res["records"], logging.WARNING)
    check("S10 stays quiet", res["stdout"].count("⚠") == 0 and len(warnings) == 0,
          f"prints={res['stdout'].count('⚠')} warnings={len(warnings)}")


def scenario_healthy():
    res = _run_main_in_tmp(
        decisions_content=json.dumps([GOOD_DECISION]),
        trades_content=json.dumps([GOOD_TRADE_BUY, GOOD_TRADE_SELL]),
    )
    check("S11 healthy: main() completes", res["error"] is None, res["error"] or "")
    report = _report(res)
    check("S11 decisions rendered", "Total decisions: 1" in report)
    check("S11 round trip rendered", "Round trips: 1" in report)
    check("S11 zero warnings in artifact", "⚠" not in report)
    check("S11 zero warning prints", res["stdout"].count("⚠") == 0)
    warnings = _errors_above(res["records"], logging.WARNING)
    check("S11 zero warning logs", len(warnings) == 0)
    check("S11 files intact",
          res["originals"]["decision_history.json"] is not None
          and res["originals"]["trades_history.json"] is not None)
    check("S11 no quarantine",
          len(res["quarantines"]["decision_history.json"]) == 0
          and len(res["quarantines"]["trades_history.json"]) == 0)


def scenario_empty_trades():
    res = _run_main_in_tmp(
        decisions_content=json.dumps([GOOD_DECISION]),
        trades_content="[]",
    )
    check("S12 empty trades: main() completes", res["error"] is None,
          res["error"] or "")
    report = _report(res)
    check("S12 report rendered", "Round trips: 0" in report)
    check("S12 zero warnings",
          "⚠" not in report and res["stdout"].count("⚠") == 0)


def main():
    print("=" * 60)
    print("behavioral_analysis ingress gate")
    print("=" * 60)
    for scenario in (
        scenario_corrupt_decisions,
        scenario_nan_token,
        scenario_wrong_shape,
        scenario_non_dict_decision,
        scenario_bad_timestamps,
        scenario_corrupt_trades,
        scenario_non_dict_trade,
        scenario_missing_trade_keys,
        scenario_empty_decisions,
        scenario_missing_files,
        scenario_healthy,
        scenario_empty_trades,
    ):
        scenario()
    print("=" * 60)
    if FAILURES:
        print(f"GATE FAILED — {len(FAILURES)} check(s):")
        for name in FAILURES:
            print(f"  XX {name}")
        sys.exit(1)
    print("GATE PASSED — all checks green.")
    sys.exit(0)


if __name__ == "__main__":
    main()

"""Reproduction gate: weekly-report trade-ledger ingress loudness.

Asserts the loudness contract at every fixed site of
``weekly_report._load_week_trades_from_history``:

  1. Corrupt-bytes ledger  -> quarantined (.corrupt-*), ERROR logged, bytes
     preserved, [] returned (old code: silent [], file left in place).
  2. NaN-token ledger      -> quarantined (strict parse hook), same class.
  3. Wrong-shape ledger    -> quarantined (parses to a dict, not a list).
  4. Non-dict element      -> dropped loudly (print), siblings survive, file
     untouched (old code: AttributeError crash of the whole report).
  5. Non-string timestamp  -> dropped loudly, siblings survive.

Healthy-path pins (must stay quiet on BOTH sides of the fix):
  6. Missing ledger  -> [], no print, no ERROR log.
  7. Valid ledger    -> trades merged, no print, no ERROR log, file intact.
  8. Empty ledger [] -> [], no print, no ERROR log, file intact.

Exit 1 if any check fails, 0 if all pass. Checks are written defensively
(PR #86 gate rule) so every pre-fix expectation lands as a failed check line
instead of crashing on a missing artifact.

Run from the repo root:
    .venv/bin/python benchmarks/repro_benchmark_weekly_report_trades_ingress.py
"""

import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weekly_report import _load_week_trades_from_history

FAILURES = []


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


def _run_scenario(body, file_content=None, ledger_name="trades_history.json"):
    """Run body(history_file) in a temp data dir; return captured artifacts.

    file_content=None -> ledger file absent; a string -> written verbatim
    (lets the gate plant literal NaN tokens no strict writer can emit).
    """
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()
        ledger = data_dir / ledger_name
        if file_content is not None:
            ledger.write_text(file_content)

        import weekly_report
        old_data_dir = weekly_report.DATA_DIR
        handler = _Recorder()
        root = logging.getLogger()
        old_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        weekly_report.DATA_DIR = data_dir
        import contextlib
        import io
        buf = io.StringIO()
        error = None
        try:
            with contextlib.redirect_stdout(buf):
                result = body(ledger)
        except Exception as exc:  # noqa: BLE001 — the gate must record a
            result = None          # pre-fix crash as a check failure, not
            error = exc            # die with a truncated failure set (PR #86
        finally:                   # defensive-gate rule).
            weekly_report.DATA_DIR = old_data_dir
            root.removeHandler(handler)
            root.setLevel(old_level)

        quarantine_files = sorted(data_dir.glob(f"{ledger_name}.corrupt-*"))
        return {
            "result": result,
            "error": error,
            "prints": buf.getvalue(),
            "records": handler.records,
            "ledger_exists": ledger.exists(),
            "quarantine_files": quarantine_files,
            "quarantine_contents": [p.read_text() for p in quarantine_files],
        }


def main():
    week = ("2026-10-12", "2026-10-16")

    # --- 1. corrupt bytes ---------------------------------------------------
    s = _run_scenario(
        lambda led: _load_week_trades_from_history(*week, []),
        file_content="{not json",
    )
    q = s["quarantine_files"]
    check("corrupt-bytes: returns []", s["result"] == [])
    check("corrupt-bytes: ledger quarantined",
          bool(q) and not s["ledger_exists"],
          f"quarantines={len(q)} ledger_exists={s['ledger_exists']}")
    check("corrupt-bytes: bytes preserved",
          bool(q) and s["quarantine_contents"]
          and s["quarantine_contents"][0] == "{not json")
    check("corrupt-bytes: ERROR logged on utils logger",
          any(r.levelno == logging.ERROR and r.name == "utils"
              and "trade ledger" in r.getMessage() for r in s["records"]))
    check("corrupt-bytes: warning printed in report",
          "quarantined" in s["prints"] and "trades_history" in s["prints"])

    # --- 2. NaN token -------------------------------------------------------
    s = _run_scenario(
        lambda led: _load_week_trades_from_history(*week, []),
        file_content='[{"timestamp": "2026-10-13T09:00:00", '
                     '"ticker": "SPY", "action": "buy", '
                     '"price": NaN, "realized_pnl": 0}]',
    )
    q = s["quarantine_files"]
    check("nan-token: ledger quarantined",
          bool(q) and not s["ledger_exists"])
    check("nan-token: ERROR logged",
          any(r.levelno == logging.ERROR and r.name == "utils"
              for r in s["records"]))
    check("nan-token: returns []", s["result"] == [])

    # --- 3. wrong shape (parses to dict) ------------------------------------
    s = _run_scenario(
        lambda led: _load_week_trades_from_history(*week, []),
        file_content='{"timestamp": "2026-10-13T09:00:00"}',
    )
    q = s["quarantine_files"]
    check("wrong-shape: ledger quarantined",
          bool(q) and not s["ledger_exists"])
    check("wrong-shape: ERROR names the shape problem",
          any(r.levelno == logging.ERROR and "wrong shape" in r.getMessage()
              for r in s["records"]))
    check("wrong-shape: returns []", s["result"] == [])

    # --- 4. non-dict element (parseable file) -------------------------------
    s = _run_scenario(
        lambda led: _load_week_trades_from_history(*week, []),
        file_content='["garbage-string", 12345, '
                     + '{"timestamp": "2026-10-14T21:00:00", "ticker": "GLD", '
                       '"action": "buy", "price": 210.0}]',
    )
    tickers = [t["ticker"] for t in s["result"]] if s["result"] else []
    check("non-dict element: report does not crash",
          s["error"] is None, f"raised {s['error']!r}")
    check("non-dict element: sibling survives (Lipschitz)",
          tickers == ["GLD"], f"got {tickers}")
    check("non-dict element: dropped loudly (print per record)",
          s["prints"].count("Skipping malformed trade record") >= 1,
          f"prints={s['prints']!r}")
    check("non-dict element: file untouched (parseable => no quarantine)",
          s["ledger_exists"] and not s["quarantine_files"])

    # --- 5. non-string timestamp --------------------------------------------
    s = _run_scenario(
        lambda led: _load_week_trades_from_history(*week, []),
        file_content='[{"timestamp": 20261014, "ticker": "X", "action": "buy", '
                     '"price": 1.0}, '
                     '{"timestamp": "2026-10-14T21:00:00", "ticker": "GLD", '
                     '"action": "buy", "price": 210.0}]',
    )
    tickers = [t["ticker"] for t in s["result"]] if s["result"] else []
    check("non-string timestamp: report does not crash",
          s["error"] is None, f"raised {s['error']!r}")
    check("non-string timestamp: sibling survives",
          tickers == ["GLD"], f"got {tickers}")
    check("non-string timestamp: dropped loudly",
          "non-string timestamp" in s["prints"])

    # --- 6. healthy pin: missing ledger -> quiet ----------------------------
    s = _run_scenario(lambda led: _load_week_trades_from_history(*week, []),
                      file_content=None)
    check("healthy/missing: returns [] quietly",
          s["result"] == [] and s["prints"] == ""
          and not any(r.levelno >= logging.WARNING for r in s["records"]))

    # --- 7. healthy pin: valid ledger -> merged, quiet ----------------------
    s = _run_scenario(
        lambda led: _load_week_trades_from_history(*week, []),
        file_content='[{"timestamp": "2026-10-14T21:00:00", "ticker": "GLD", '
                     '"action": "buy", "price": 210.0}]',
    )
    check("healthy/valid: trade merged, file intact, quiet",
          len(s["result"]) == 1 and s["result"][0]["ticker"] == "GLD"
          and s["ledger_exists"] and not s["quarantine_files"]
          and s["prints"] == ""
          and not any(r.levelno >= logging.WARNING for r in s["records"]))

    # --- 8. healthy pin: empty ledger -> quiet, no spurious quarantine ------
    s = _run_scenario(
        lambda led: _load_week_trades_from_history(*week, []),
        file_content="[]",
    )
    check("healthy/empty: returns [] quietly, file intact",
          s["result"] == [] and s["ledger_exists"]
          and not s["quarantine_files"] and s["prints"] == "")

    print()
    if FAILURES:
        print(f"GATE FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("GATE PASSED: all loudness checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

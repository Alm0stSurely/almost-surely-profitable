"""Reproduction gate for the keyword-trends ingress hardening.

Pre-fix, ``keyword_trends.load_decisions`` parsed ``decision_history.json``
with a bare ``json.load``: non-standard ``NaN``/``Infinity`` tokens (data no
strict writer in the repo can emit, PR #84) were silently ingested into the
trend aggregates, garbage bytes crashed the run with a ``JSONDecodeError``,
and a wrong-shaped top level crashed one call frame downstream (an
``AttributeError`` on a dict's keys, or a ``TypeError`` on a scalar) —
maximally far from the cause. The element loop in ``group_by_iso_week``
then iterated the list with no validation: one malformed element (a bare
string, a missing/unparseable ``timestamp``) crashed the whole trend
report, voiding the n-1 valid records with it (anti-Lipschitz violation,
PR #64).

Post-fix, the load routes through the shared quarantine ingress
(``load_json_list_or_quarantine``): unparseable / token / wrong-shape files
are renamed aside (bytes preserved, one ERROR log) and the analysis starts
honestly empty. ``group_by_iso_week`` validates each element against the
codomain its consumers actually compute in (PR #88): mapping + ISO-parseable
string ``timestamp`` — the only field touched outside a guard. One malformed
element drops loudly with its index and reason; valid siblings survive.
``error``-flagged records keep their documented quiet skip, and tz-aware
timestamps are accepted (the module only calls ``isocalendar()`` per record;
no naive/aware arithmetic exists here — contrast ``churn_analysis``).

Gate convention (PR #86/#87): every scenario runs inside a
``tempfile.TemporaryDirectory`` with artifact *contents* captured before the
directory vanishes; the functions under test are wrapped so pre-fix
exceptions are recorded as scenario errors instead of killing the harness;
a pre-fix run must exit 1 with the FULL named failure set visible (zero
tracebacks), a post-fix run exits 0. No timing benchmark ships with this
gate: loudness/ingress fixes are validated by the discrimination scenarios
below, not by wall-clock (PR #75 doctrine).

Usage:
    .venv/bin/python benchmarks/repro_benchmark_keyword_trends_ingress.py
"""

import io
import json
import logging
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.keyword_trends import group_by_iso_week, load_decisions

FAILURES = []


def _check(scenario: str, ok: bool, label: str) -> None:
    if not ok:
        FAILURES.append(f"{scenario}: {label}")


def _record_errors() -> list:
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    return records


def _load_scenario(tmp: Path, payload) -> dict:
    """Write decision_history.json with *payload*, call load_decisions."""
    path = tmp / "decision_history.json"
    result = {"error": None, "records": None,
              "quarantine_names": [], "quarantine_bytes": []}
    if payload is not None:
        if isinstance(payload, bytes):
            path.write_bytes(payload)
        else:
            path.write_text(payload)
    try:
        result["records"] = load_decisions(path)
    except Exception as exc:  # noqa: BLE001 - gate must limp, not crash
        result["error"] = f"{type(exc).__name__}: {exc}"
    quarantines = sorted(tmp.glob("decision_history.json.corrupt-*"))
    result["quarantine_names"] = [q.name for q in quarantines]
    result["quarantine_bytes"] = [q.read_bytes() for q in quarantines]
    return result


def _group_scenario(decisions: list) -> dict:
    """Call group_by_iso_week, capturing stdout and any crash."""
    result = {"error": None, "stdout": "", "weeks": None}
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            result["weeks"] = group_by_iso_week(decisions)
    except Exception as exc:  # noqa: BLE001 - gate must limp, not crash
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["stdout"] = buf.getvalue()
    return result


def run() -> int:
    log_records = _record_errors()

    HEALTHY = [
        {"timestamp": "2026-10-19T09:00:00", "reasoning": "hold"},
        {"timestamp": "2026-10-20T09:00:00", "reasoning": "buy SPY"},
        {"timestamp": "2026-10-26T09:00:00", "reasoning": "sell GLD"},
    ]

    # --- File-level scenarios (load_decisions) ---

    with tempfile.TemporaryDirectory() as td:
        r = _load_scenario(Path(td), json.dumps(HEALTHY))
        _check("load/healthy", r["error"] is None, f"unexpected error: {r['error']}")
        _check("load/healthy", r["records"] == HEALTHY,
               "healthy records must round-trip unchanged")
        _check("load/healthy", not r["quarantine_names"],
               "healthy file must not be quarantined")

    with tempfile.TemporaryDirectory() as td:
        r = _load_scenario(
            Path(td),
            '[{"timestamp": "2026-10-20T09:00:00", "reasoning": NaN}]')
        _check("load/nan-token", r["error"] is None,
               f"NaN token file must not crash: {r['error']}")
        _check("load/nan-token", r["records"] == [],
               "NaN token file must yield an honest empty start")
        _check("load/nan-token", bool(r["quarantine_names"]),
               "NaN token file must be quarantined")
        _check("load/nan-token",
               bool(r["quarantine_bytes"]) and b"NaN" in r["quarantine_bytes"][0],
               "quarantine must preserve the original bytes")
        _check("load/nan-token",
               any("decision history" in rec.getMessage()
                   for rec in log_records if rec.levelno >= logging.ERROR),
               "quarantine must emit one ERROR log naming the context")

    with tempfile.TemporaryDirectory() as td:
        r = _load_scenario(
            Path(td),
            '[{"timestamp": "2026-10-20T09:00:00", "score": -Infinity}]')
        _check("load/infinity-token", r["error"] is None,
               f"Infinity token file must not crash: {r['error']}")
        _check("load/infinity-token", r["records"] == [] and
               bool(r["quarantine_names"]),
               "Infinity token file must quarantine to an empty start")

    with tempfile.TemporaryDirectory() as td:
        r = _load_scenario(Path(td), b"\xff\xfe{not json at all")
        _check("load/garbage-bytes", r["error"] is None,
               f"garbage bytes must not crash: {r['error']}")
        _check("load/garbage-bytes", r["records"] == [] and
               bool(r["quarantine_names"]),
               "garbage bytes must quarantine to an empty start")

    with tempfile.TemporaryDirectory() as td:
        r = _load_scenario(Path(td), '{"timestamp": "2026-10-20T09:00:00"}')
        _check("load/wrong-shape-dict", r["error"] is None,
               f"dict top level must not crash downstream: {r['error']}")
        _check("load/wrong-shape-dict", r["records"] == [] and
               bool(r["quarantine_names"]),
               "dict top level must quarantine to an empty start")

    with tempfile.TemporaryDirectory() as td:
        r = _load_scenario(Path(td), "42")
        _check("load/wrong-shape-scalar", r["error"] is None,
               f"scalar top level must not crash downstream: {r['error']}")
        _check("load/wrong-shape-scalar", r["records"] == [] and
               bool(r["quarantine_names"]),
               "scalar top level must quarantine to an empty start")

    with tempfile.TemporaryDirectory() as td:
        r = _load_scenario(Path(td), None)  # file absent
        _check("load/missing-file", r["error"] is None,
               f"missing file must not crash: {r['error']}")
        _check("load/missing-file", r["records"] == [],
               "missing file must yield an empty start")
        _check("load/missing-file", not r["quarantine_names"],
               "missing file must not be quarantined")

    # --- Element-level scenarios (group_by_iso_week) ---

    r = _group_scenario(HEALTHY)
    _check("group/healthy", r["error"] is None, f"unexpected error: {r['error']}")
    _check("group/healthy", r["weeks"] is not None
           and len(r["weeks"].get("2026-W43", [])) == 2
           and len(r["weeks"].get("2026-W44", [])) == 1,
           "healthy records must group by ISO week")
    _check("group/healthy", "Dropping malformed" not in r["stdout"],
           "healthy records must produce no drop warnings")

    r = _group_scenario([HEALTHY[0], "not-a-dict", HEALTHY[1]])
    _check("group/non-dict", r["error"] is None,
           f"non-dict element must not crash the report: {r['error']}")
    _check("group/non-dict", "index 1" in r["stdout"]
           and "expected dict, got str" in r["stdout"],
           "non-dict element must drop loudly with index and reason")
    _check("group/non-dict", r["weeks"] is not None
           and sum(len(v) for v in r["weeks"].values()) == 2,
           "valid siblings of a non-dict element must survive")

    r = _group_scenario([HEALTHY[0], {"reasoning": "no timestamp"}])
    _check("group/missing-timestamp", r["error"] is None,
           f"missing timestamp must not crash the report: {r['error']}")
    _check("group/missing-timestamp", "index 1" in r["stdout"]
           and "missing/empty timestamp" in r["stdout"],
           "missing timestamp must drop loudly with index and reason")
    _check("group/missing-timestamp", r["weeks"] is not None
           and sum(len(v) for v in r["weeks"].values()) == 1,
           "valid siblings of a timestamp-less record must survive")

    r = _group_scenario([HEALTHY[0], {"timestamp": "not-a-date"}])
    _check("group/unparseable-timestamp", r["error"] is None,
           f"unparseable timestamp must not crash the report: {r['error']}")
    _check("group/unparseable-timestamp", "index 1" in r["stdout"]
           and "unparseable timestamp" in r["stdout"],
           "unparseable timestamp must drop loudly with index and reason")

    r = _group_scenario([HEALTHY[0], {"timestamp": 1234567890}])
    _check("group/non-string-timestamp", r["error"] is None,
           f"non-string timestamp must not crash the report: {r['error']}")
    _check("group/non-string-timestamp", "index 1" in r["stdout"],
           "non-string timestamp must drop loudly with index and reason")

    r = _group_scenario([HEALTHY[0], {"error": True, "reasoning": "LLM call failed"}])
    _check("group/error-flagged", r["error"] is None,
           f"error-flagged record must not crash: {r['error']}")
    _check("group/error-flagged", "Dropping malformed" not in r["stdout"],
           "error-flagged record keeps its documented quiet skip")
    _check("group/error-flagged", r["weeks"] is not None
           and sum(len(v) for v in r["weeks"].values()) == 1,
           "error-flagged record must be excluded from the grouping")

    r = _group_scenario([{"timestamp": "2026-10-19T09:00:00+02:00"}])
    _check("group/tz-aware", r["error"] is None,
           f"tz-aware timestamp must not crash: {r['error']}")
    _check("group/tz-aware", "Dropping malformed" not in r["stdout"],
           "tz-aware timestamp is accepted: no naive/aware arithmetic in this codomain")
    _check("group/tz-aware", r["weeks"] is not None
           and sum(len(v) for v in r["weeks"].values()) == 1,
           "tz-aware record must group normally")

    # --- Verdict ---

    if FAILURES:
        print(f"GATE FAILED — {len(FAILURES)} named failure(s), zero tracebacks:")
        for f in FAILURES:
            print(f"  XX {f}")
        return 1
    print(f"GATE PASSED — all scenarios green "
          f"({len(log_records)} log record(s) captured).")
    return 0


if __name__ == "__main__":
    sys.exit(run())

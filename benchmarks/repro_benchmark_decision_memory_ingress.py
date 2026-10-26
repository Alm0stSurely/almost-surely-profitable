"""Reproduction gate for the decision-memory ingress hardening.

Pre-fix, ``DecisionMemory._load_memory`` parsed ``decision_memory.json``
with a bare ``json.load`` inside ``except Exception`` and, on ANY failure,
fabricated an empty decision list. Because ``save_memory()`` rewrites the
whole file after the session, the wipe flowed into an overwrite (PR #60
evidence-destruction class): one malformed record among n valid ones made
``DecisionRecord.from_dict`` raise, the blanket except erased every valid
record, and the next save persisted the erasure. Files carrying NaN /
Infinity tokens (unwritable by any strict writer in the repo, PR #84) or a
wrong-shaped top level took the same silent path.

Post-fix, the load routes through the shared quarantine ingress
(``load_json_list_or_quarantine``): unparseable / token / wrong-shape files
are renamed aside (bytes preserved, one ERROR log, honest empty start),
and parseable lists are validated per record — one malformed element drops
loudly with its index and reason, valid siblings survive (anti-Lipschitz,
PR #64), and the ``date`` field is checked against the ``%Y-%m-%d``
codomain its consumers compute in (PR #88).

Gate convention (PR #86/#87): every scenario runs inside a
``tempfile.TemporaryDirectory`` with artifact *contents* captured before
the directory vanishes; the function under test is wrapped so pre-fix
exceptions are recorded as scenario errors instead of killing the harness;
a pre-fix run must exit 1 with the FULL named failure set visible (zero
tracebacks), a post-fix run exits 0.

Usage:
    .venv/bin/python benchmarks/repro_benchmark_decision_memory_ingress.py
"""

import io
import json
import logging
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.decision_memory import DecisionMemory

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


def _valid_record(**overrides):
    rec = {
        "date": "2026-10-20",
        "ticker": "SPY",
        "action": "buy",
        "quantity": 10.0,
        "price": 100.0,
        "portfolio_value_before": 10000.0,
        "portfolio_value_after": 9900.0,
    }
    rec.update(overrides)
    return rec


def scenario(tmp: Path, name: str, files: dict):
    """Write *files* into *tmp*, construct DecisionMemory, return artifacts."""
    result = {"error": None, "stdout": "", "decisions": None,
              "quarantine_names": [], "quarantine_bytes": [],
              "saved_bytes": None}
    for fname, payload in files.items():
        p = tmp / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, bytes):
            p.write_bytes(payload)
        else:
            p.write_text(payload)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            mem = DecisionMemory(str(tmp / "decision_memory.json"))
            result["decisions"] = mem.decisions
    except Exception as exc:  # noqa: BLE001 - gate must limp, not crash
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["stdout"] = buf.getvalue()
    quarantines = sorted(tmp.glob("decision_memory.json.corrupt-*"))
    result["quarantine_names"] = [q.name for q in quarantines]
    result["quarantine_bytes"] = [q.read_bytes() for q in quarantines]
    target = tmp / "decision_memory.json"
    if target.exists():
        result["saved_bytes"] = target.read_bytes()
    return result


def run() -> int:
    log_records = _record_errors()

    # 1. Healthy file: both records load, no drops, no quarantine.
    with tempfile.TemporaryDirectory() as td:
        r = scenario(Path(td), "healthy", {
            "decision_memory.json": json.dumps(
                [_valid_record(), _valid_record(ticker="GLD", action="hold")])
        })
        _check("healthy", r["error"] is None, f"unexpected error: {r['error']}")
        _check("healthy", r["decisions"] is not None
               and len(r["decisions"]) == 2, "expected 2 decisions loaded")
        _check("healthy", not r["quarantine_names"],
               "healthy file must not be quarantined")
        _check("healthy", "Dropping malformed" not in r["stdout"],
               "healthy file must produce no drop warnings")

    # 2. Garbage bytes: quarantined, empty memory, ERROR logged.
    with tempfile.TemporaryDirectory() as td:
        r = scenario(Path(td), "garbage", {
            "decision_memory.json": b"not json at all\x00\xff"})
        _check("garbage", r["error"] is None, f"unexpected error: {r['error']}")
        _check("garbage", r["decisions"] == [], "expected empty memory")
        _check("garbage", bool(r["quarantine_names"]),
               "corrupt file must be quarantined (bytes preserved)")
        _check("garbage", bool(r["quarantine_bytes"])
               and b"not json" in (r["quarantine_bytes"][0] if r["quarantine_bytes"] else b""),
               "quarantined bytes must contain the original garbage")
        _check("garbage", not (Path(td) / "decision_memory.json").exists(),
               "original path must be clear after quarantine")

    # 3. NaN token: strict-read rejection, quarantined.
    with tempfile.TemporaryDirectory() as td:
        payload = json.dumps([_valid_record()])
        payload = payload.replace("100.0", "NaN", 1)
        r = scenario(Path(td), "nan_token", {"decision_memory.json": payload})
        _check("nan_token", r["error"] is None, f"unexpected error: {r['error']}")
        _check("nan_token", r["decisions"] == [], "expected empty memory")
        _check("nan_token", bool(r["quarantine_names"]),
               "NaN-token file must be quarantined")

    # 4. Top-level dict: wrong shape, quarantined.
    with tempfile.TemporaryDirectory() as td:
        r = scenario(Path(td), "top_dict", {
            "decision_memory.json": json.dumps({"decisions": [_valid_record()]})})
        _check("top_dict", r["error"] is None, f"unexpected error: {r['error']}")
        _check("top_dict", r["decisions"] == [], "expected empty memory")
        _check("top_dict", bool(r["quarantine_names"]),
               "dict-shaped file must be quarantined")

    # 5. Top-level null: wrong shape, quarantined.
    with tempfile.TemporaryDirectory() as td:
        r = scenario(Path(td), "top_null", {"decision_memory.json": "null"})
        _check("top_null", r["error"] is None, f"unexpected error: {r['error']}")
        _check("top_null", r["decisions"] == [], "expected empty memory")
        _check("top_null", bool(r["quarantine_names"]),
               "null file must be quarantined")

    # 6. One non-dict element among valid ones: drop loudly, siblings survive.
    with tempfile.TemporaryDirectory() as td:
        r = scenario(Path(td), "non_dict_element", {
            "decision_memory.json": json.dumps(
                [_valid_record(), "a string record", _valid_record(ticker="GLD")])})
        _check("non_dict_element", r["error"] is None,
               f"unexpected error: {r['error']}")
        _check("non_dict_element", r["decisions"] is not None
               and len(r["decisions"]) == 2,
               "one malformed element must not void the 2 valid siblings")
        _check("non_dict_element", "Dropping malformed decision record at index 1"
               in r["stdout"], "drop must be loud with the element index")
        _check("non_dict_element", not r["quarantine_names"],
               "partially-valid list must NOT be quarantined")

    # 7. Missing required field (price): drop loudly, sibling survives.
    with tempfile.TemporaryDirectory() as td:
        bad = _valid_record()
        del bad["price"]
        r = scenario(Path(td), "missing_field", {
            "decision_memory.json": json.dumps([_valid_record(), bad])})
        _check("missing_field", r["error"] is None,
               f"unexpected error: {r['error']}")
        _check("missing_field", r["decisions"] is not None
               and len(r["decisions"]) == 1
               and r["decisions"][0].ticker == "SPY",
               "missing-required-field record dropped, valid sibling survives")
        _check("missing_field", "price" in r["stdout"],
               "drop reason must name the missing field")

    # 8. Unknown key: from_dict raises TypeError, drop loudly, sibling survives.
    with tempfile.TemporaryDirectory() as td:
        bad = _valid_record(schema_version=2)
        r = scenario(Path(td), "unknown_key", {
            "decision_memory.json": json.dumps([_valid_record(), bad])})
        _check("unknown_key", r["error"] is None,
               f"unexpected error: {r['error']}")
        _check("unknown_key", r["decisions"] is not None
               and len(r["decisions"]) == 1,
               "unknown-key record dropped, valid sibling survives")
        _check("unknown_key", "schema_version" in r["stdout"],
               "drop reason must name the unexpected keyword")

    # 9. Date outside %Y-%m-%d codomain (parse-but-detonate, PR #88 class):
    #    a full ISO datetime string is a valid JSON string but crashes
    #    get_decision_summary's strptime on the first record it touches.
    with tempfile.TemporaryDirectory() as td:
        bad = _valid_record(date="2026-10-20T21:30:00")
        r = scenario(Path(td), "bad_date", {
            "decision_memory.json": json.dumps([_valid_record(), bad])})
        _check("bad_date", r["error"] is None,
               f"unexpected error: {r['error']}")
        _check("bad_date", r["decisions"] is not None
               and len(r["decisions"]) == 1,
               "wrong-codomain date dropped, valid sibling survives")
        _check("bad_date", "2026-10-20T21:30:00" in r["stdout"],
               "drop reason must name the offending date")

    # 10. Non-string date: drop loudly, sibling survives.
    with tempfile.TemporaryDirectory() as td:
        bad = _valid_record(date=20261020)
        r = scenario(Path(td), "non_string_date", {
            "decision_memory.json": json.dumps([_valid_record(), bad])})
        _check("non_string_date", r["error"] is None,
               f"unexpected error: {r['error']}")
        _check("non_string_date", r["decisions"] is not None
               and len(r["decisions"]) == 1,
               "non-string date dropped, valid sibling survives")

    # 11. Persist-after-drop: the next save writes only valid records and
    #     the quarantined bytes survive alongside (PR #60 evidence class).
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        bad = _valid_record()
        del bad["price"]
        files = {"decision_memory.json": json.dumps([_valid_record(), bad])}
        for fname, payload in files.items():
            (tmp / fname).write_text(payload)
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                mem = DecisionMemory(str(tmp / "decision_memory.json"))
                mem.save_memory()
                reloaded = DecisionMemory(str(tmp / "decision_memory.json"))
        except Exception as exc:  # noqa: BLE001
            _check("persist_after_drop", False,
                   f"unexpected error: {type(exc).__name__}: {exc}")
        else:
            _check("persist_after_drop", len(reloaded.decisions) == 1
                   and reloaded.decisions[0].ticker == "SPY",
                   "post-save reload must contain exactly the 1 valid record")

    # 12. Quarantine path is loud at ERROR level (helper logger).
    _check("loudness", any(rec.levelno >= logging.ERROR for rec in log_records),
           "at least one ERROR log expected from the quarantine scenarios")

    if FAILURES:
        print(f"XX {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"XX {f}")
        return 1
    print("OK all decision-memory ingress scenarios passed (12 scenarios)")
    return 0


if __name__ == "__main__":
    sys.exit(run())

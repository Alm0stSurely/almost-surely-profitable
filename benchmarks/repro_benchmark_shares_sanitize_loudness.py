"""Functional gate: benchmark shares sanitizer must drop loudly.

Exits 0 when every malformed entry produces a loud warning (and healthy
paths stay silent); exits 1 on pre-fix code, where the sanitizer drops
through a silent dict comprehension and nothing is logged.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark import LiveEqualWeightBenchmark

failures = []


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    if not ok:
        failures.append(name)


def load_with_records(state_dir, state):
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / "equalweight_benchmark_state.json", "w") as f:
        json.dump(state, f)
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = _Capture()
    root.addHandler(handler)
    try:
        bench = LiveEqualWeightBenchmark(
            initial_capital=10000.0, data_dir=str(state_dir)
        )
    finally:
        root.removeHandler(handler)
    return bench, [r.getMessage() for r in records]


BASE = {
    "cash": 2000.0,
    "start_date": "2026-07-31",
    "last_rebalanced": "2026-07-31T10:00:00",
}

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # 1. Non-finite quantity: dropped loudly, healthy sibling survives.
    bench, msgs = load_with_records(
        tmp / "a", {**BASE, "shares": {"SPY": float("nan"), "GLD": 5.0}}
    )
    check("NaN quantity dropped", "SPY" not in bench.shares)
    check("healthy sibling survives", bench.shares == {"GLD": 5.0})
    check("NaN drop is loud", any("SPY" in m and "finite" in m for m in msgs))

    # 2. Bool launders through arithmetic — must be dropped loudly.
    bench, msgs = load_with_records(
        tmp / "b", {**BASE, "shares": {"SPY": True, "GLD": 5.0}}
    )
    check("bool quantity dropped loudly", "SPY" not in bench.shares
          and any("SPY" in m for m in msgs))

    # 3. String quantity dropped loudly.
    bench, msgs = load_with_records(
        tmp / "c", {**BASE, "shares": {"SPY": "10", "GLD": 5.0}}
    )
    check("string quantity dropped loudly", "SPY" not in bench.shares
          and any("SPY" in m for m in msgs))

    # 4. Negative quantity dropped loudly.
    bench, msgs = load_with_records(
        tmp / "d", {**BASE, "shares": {"SPY": -5.0, "GLD": 5.0}}
    )
    check("negative quantity dropped loudly", "SPY" not in bench.shares
          and any("SPY" in m and "negative" in m for m in msgs))

    # 5. Non-mapping shares: loud, treated as no positions.
    bench, msgs = load_with_records(tmp / "e", {**BASE, "shares": ["SPY", 10.0]})
    check("non-mapping shares loud + empty", bench.shares == {}
          and any("not a mapping" in m for m in msgs))

    # 6. Every dropped entry gets its own warning (no batch silence).
    bench, msgs = load_with_records(
        tmp / "f",
        {**BASE, "shares": {"AAA": float("inf"), "BBB": False, "GLD": 5.0}},
    )
    dropped = {"AAA", "BBB"}
    warned = {
        m.split("entry ")[1].split(":")[0].strip("'\"")
        for m in msgs
        if "Dropping benchmark share entry" in m
    }
    check("one warning per dropped entry", dropped <= warned)

    # 7. Healthy path: byte-identical load, zero warnings.
    bench, msgs = load_with_records(
        tmp / "g", {**BASE, "shares": {"SPY": 10.0, "GLD": 5}}
    )
    check("healthy path silent + exact", bench.shares == {"SPY": 10.0, "GLD": 5.0}
          and msgs == [])

    # 8. Absent shares key: no-claim quiet fresh start.
    bench, msgs = load_with_records(tmp / "h", dict(BASE))
    check("absent shares key quiet", bench.shares == {} and msgs == [])

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)

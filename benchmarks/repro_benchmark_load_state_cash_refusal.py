"""Functional gate for the benchmark load_state cash-refusal guard (PR #82).

Exits 0 when invalid benchmark cash is REFUSED at ingress (ValueError, file
byte-untouched), exits 1 when the pre-fix fabrication path is live (invalid
cash silently substituted by initial_capital, which rebalance() then
persists via save_state()).

Run: python benchmarks/repro_benchmark_load_state_cash_refusal.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark import LiveEqualWeightBenchmark

FAILURES = []

CASES = {
    "nan": float("nan"),
    "infinity": float("inf"),
    "bool_true": True,
    "string": "10000.0",
    "negative": -500.0,
    "missing": None,  # sentinel: key deleted from payload
}


def check(name, cash):
    tmp = Path(tempfile.mkdtemp(prefix=f"repro-bench-cash-{name}-"))
    try:
        data_dir = tmp / "data"
        data_dir.mkdir()
        state_path = data_dir / "equalweight_benchmark_state.json"
        payload = {
            "shares": {"SPY": 10.0},
            "cash": 2000.0,
            "start_date": "2026-07-31",
            "last_rebalanced": "2026-07-31T10:00:00",
        }
        if name == "missing":
            del payload["cash"]
        else:
            payload["cash"] = cash
        with open(state_path, "w") as f:
            json.dump(payload, f)
        before = state_path.read_bytes()

        try:
            LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(data_dir))
        except ValueError as exc:
            if "cash" not in str(exc):
                FAILURES.append(f"{name}: ValueError raised but does not name 'cash'")
                return
            if state_path.read_bytes() != before:
                FAILURES.append(f"{name}: refusal raised but file was mutated")
                return
            print(f"  PASS {name}: refused, file untouched")
            return

        FAILURES.append(
            f"{name}: invalid cash loaded silently (fabrication path live)"
        )
        print(f"  FAIL {name}: no refusal raised")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    for name, cash in CASES.items():
        check(name, cash)

    # Healthy-path pin: a valid file must still load.
    tmp = Path(tempfile.mkdtemp(prefix="repro-bench-cash-healthy-"))
    try:
        data_dir = tmp / "data"
        data_dir.mkdir()
        with open(data_dir / "equalweight_benchmark_state.json", "w") as f:
            json.dump({"shares": {"SPY": 10.0}, "cash": 2000.0}, f)
        bm = LiveEqualWeightBenchmark(
            initial_capital=10000.0, data_dir=str(data_dir)
        )
        if bm.cash != 2000.0:
            FAILURES.append(f"healthy: expected cash 2000.0, got {bm.cash!r}")
        else:
            print("  PASS healthy: valid cash loads")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if FAILURES:
        print(f"\nGATE FAILED ({len(FAILURES)} failures):")
        for failure in FAILURES:
            print(f"  - {failure}")
        sys.exit(1)
    print("\nGATE PASSED: invalid benchmark cash refused at ingress.")


if __name__ == "__main__":
    main()

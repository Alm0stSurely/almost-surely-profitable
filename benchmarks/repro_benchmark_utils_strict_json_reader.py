"""Repro + benchmark for the strict-reader convention on the shared utils ingress.

Repro gate (the functional proof):
    1. A ledger/state file containing a literal ``NaN`` token — data no
       repo writer can emit (every writer uses ``allow_nan=False``) — is
       silently ingested by the pre-fix lenient parser: the non-finite
       float flows straight to the caller and later crashes the next
       strict write or poisons an aggregate maximally far from the cause.
    2. Post-fix, the shared helpers reject the token at the read
       boundary: quarantine helpers rename the file aside (bytes
       preserved) with a loud error; load_valid_daily_results skips the
       file with a loud warning.

    Exit code 0 iff every post-fix expectation holds. Pre-fix the
    gate fails (silent ingestion observed).

Benchmark (honest no-claim): the hook fires once per load on a cold
path; a timing delta is not the point. We still time the guarded load
vs a plain json.load over healthy files so the PR can state observed
numbers instead of asserting "no regression" from assumption.
"""

import json
import logging
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import (
    load_json_dict_or_quarantine,
    load_json_list_or_quarantine,
    load_valid_daily_results,
)

logging.basicConfig(level=logging.CRITICAL)  # keep the benchmark table readable


def _write(path: Path, text: str) -> None:
    path.write_text(text)


def repro() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # 1. NaN-token ledger through the list quarantine helper
        ledger = tmp / "trades_history.json"
        _write(ledger, '[{"ticker": "SPY", "price": NaN}]')
        result = load_json_list_or_quarantine(ledger, context="ledger")
        if result != []:
            failures.append(
                f"list helper ingested NaN token silently: returned {result!r}"
            )
        if ledger.exists():
            failures.append("list helper left the NaN-token file in place")
        if not list(tmp.glob("trades_history.json.corrupt-*")):
            failures.append("list helper did not quarantine the NaN-token file")

        # 2. Infinity-token state through the dict quarantine helper
        state = tmp / "market_state.json"
        _write(state, '{"SPY": {"price": Infinity}}')
        result = load_json_dict_or_quarantine(state, context="state")
        if result != {}:
            failures.append(
                f"dict helper ingested Infinity token silently: returned {result!r}"
            )
        if state.exists():
            failures.append("dict helper left the Infinity-token file in place")
        if not list(tmp.glob("market_state.json.corrupt-*")):
            failures.append("dict helper did not quarantine the Infinity-token file")

        # 3. NaN-token daily result skipped loudly
        daily = tmp / "daily"
        daily.mkdir()
        good = daily / "2026-10-14.json"
        with open(good, "w") as f:
            json.dump(
                {
                    "date": "2026-10-14",
                    "portfolio_summary": {"cash": 100.0, "total_value": 1000.0},
                    "llm_response": {},
                    "actions": [],
                },
                f,
                allow_nan=False,
            )
        _write(daily / "2026-10-15.json", '{"date": "2026-10-15", "cash": NaN}')
        results = load_valid_daily_results(str(daily))
        if [r["date"] for r in results] != ["2026-10-14"]:
            failures.append(
                f"daily-results ingested the NaN-token file: {[r['date'] for r in results]}"
            )

        # 4. Healthy files parse identically (no regression on the happy path)
        healthy_ledger = tmp / "healthy_ledger.json"
        records = [{"ticker": "SPY", "price": 412.5}]
        with open(healthy_ledger, "w") as f:
            json.dump(records, f, allow_nan=False)
        if load_json_list_or_quarantine(healthy_ledger, context="ledger") != records:
            failures.append("healthy ledger no longer round-trips")

    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        print(f"\nREPRO: {len(failures)} check(s) failed — pre-fix behaviour observed.")
        return 1
    print("REPRO: all 4 checks passed — strict-reader behaviour confirmed.")
    return 0


def benchmark() -> None:
    n_iter, rounds = 2000, 5
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ledger = tmp / "ledger.json"
        records = [{"ticker": f"T{i}", "price": 100.0 + i} for i in range(100)]
        with open(ledger, "w") as f:
            json.dump(records, f, allow_nan=False)

        # Warmup
        for _ in range(200):
            load_json_list_or_quarantine(ledger, context="ledger")
            with open(ledger) as f:
                json.load(f)

        guarded, plain = [], []
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(n_iter):
                load_json_list_or_quarantine(ledger, context="ledger")
            guarded.append((time.perf_counter() - t0) / n_iter * 1e3)
            t0 = time.perf_counter()
            for _ in range(n_iter):
                with open(ledger) as f:
                    json.load(f)
            plain.append((time.perf_counter() - t0) / n_iter * 1e3)

        print(
            f"guarded load (helper + hook): {min(guarded):.4f} ms "
            f"(median {statistics.median(guarded):.4f})"
        )
        print(
            f"plain json.load (no helper):  {min(plain):.4f} ms "
            f"(median {statistics.median(plain):.4f})"
        )


if __name__ == "__main__":
    rc = repro()
    if rc == 0:
        benchmark()
    sys.exit(rc)

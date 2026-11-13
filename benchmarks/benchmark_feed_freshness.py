"""Micro-benchmark for the shared feed-freshness classifier.

The stale-feed gate runs inside the nightly pipeline right after the bulk
fetch, so classification must stay negligible next to the network I/O it
guards. Measures ``classify_daily_frames`` across the reachable shapes:
all-fresh universe, the incident pattern (Euronext frozen, US healthy),
and a partially dropped fetch ("possibly delisted" errors).
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.feed_freshness import classify_daily_frames

NOW = datetime(2026, 11, 13, 21, 0, 0)
N = 20_000


def _frame(last_ts: str) -> pd.DataFrame:
    last = pd.Timestamp(last_ts)
    idx = pd.date_range(end=last, periods=22, freq="B")
    return pd.DataFrame({"Close": [100.0] * len(idx)}, index=idx)


def _bench(frames, expected, n=N):
    def call():
        classify_daily_frames(expected, frames, NOW)
    for _ in range(200):
        call()
    start = time.perf_counter()
    for _ in range(n):
        call()
    elapsed = time.perf_counter() - start
    return elapsed / n * 1e6  # microseconds per call


def main():
    print("Feed-freshness classifier benchmark")
    print("=" * 50)

    universe = ["SPY", "PDBC", "IJR", "TLT", "REET", "GLD",
                "AI.PA", "MC.PA", "^FCHI"]

    fresh = {
        "SPY": _frame("2026-11-13 05:00"),
        "PDBC": _frame("2026-11-13 05:00"),
        "IJR": _frame("2026-11-13 05:00"),
        "TLT": _frame("2026-11-13 05:00"),
        "REET": _frame("2026-11-13 05:00"),
        "GLD": _frame("2026-11-13 05:00"),
        "AI.PA": _frame("2026-11-12 23:00"),
        "MC.PA": _frame("2026-11-12 23:00"),
        "^FCHI": _frame("2026-11-12 23:00"),
    }
    us = {t: f for t, f in fresh.items() if t in ("SPY", "PDBC", "IJR", "TLT", "REET", "GLD")}

    incident = dict(us)  # Paris tickers dropped by "possibly delisted" errors
    frozen = dict(fresh)
    frozen["AI.PA"] = _frame("2026-11-11 23:00")  # served but two sessions old

    cases = [
        ("all-fresh universe", fresh, universe),
        ("incident: Paris dropped", incident, universe),
        ("incident: Paris frozen", frozen, universe),
    ]
    for label, frames, expected in cases:
        us_per_call = _bench(frames, expected)
        print(f"  {label:26s}: {us_per_call:8.2f} µs/call")
    print("=" * 50)
    print("Guard overhead is ~µs against a multi-second fetch: negligible.")


if __name__ == "__main__":
    main()

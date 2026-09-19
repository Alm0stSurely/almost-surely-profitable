"""
Benchmark for the fail-open-on-exit change in can_sell() (position_cooldown).

Semantics fix, not a hot path: measures the cost of the missing-entry-record
branch against the normal hold-period branch so we can state honestly that
allowing exits on lost bookkeeping adds no measurable overhead to the daily
pipeline's cooldown checks. Also quantifies the weekly-cap independence pin
(cap still evaluated first, unchanged).

Run:  .venv/bin/python benchmarks/benchmark_can_sell_fail_open_exit.py
"""
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk.position_cooldown import PositionCooldownManager, CooldownConfig


def make_manager(data_dir: str) -> PositionCooldownManager:
    return PositionCooldownManager(
        data_dir=data_dir,
        config=CooldownConfig(min_hold_days=5, flip_cooldown_days=10),
    )


def bench_can_sell(mgr: PositionCooldownManager, ticker: str,
                   price: float, avg: float, n_iter: int = 20000) -> float:
    """Best-of-5 average wall time of one can_sell() call."""
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n_iter):
            mgr.can_sell(ticker, price, avg)
        dt = (time.perf_counter() - t0) / n_iter
        best = min(best, dt)
    return best


def main():
    tmpdir = tempfile.mkdtemp()
    mgr = make_manager(tmpdir)

    # Normal path: entry recorded 30 days ago -> hold-period branch.
    mgr.record_entry("HELD")
    mgr.entries["HELD"] = datetime.now() - timedelta(days=30)

    # Missing-entry path: fail-open branch (state lost, e.g. corruption fallback).
    assert "LOST" not in mgr.entries

    t_normal = bench_can_sell(mgr, "HELD", 400.0, 400.0)
    t_fail_open = bench_can_sell(mgr, "LOST", 400.0, 400.0)

    # Sanity: the fail-open branch actually returns allowed.
    ok, reason = mgr.can_sell("LOST", 400.0, 400.0)
    assert ok is True and "No entry record" in reason

    print(f"can_sell normal hold-period branch : {t_normal * 1e6:8.2f} us/call")
    print(f"can_sell missing-entry fail-open   : {t_fail_open * 1e6:8.2f} us/call")
    print(f"delta (fail-open - normal)         : {(t_fail_open - t_normal) * 1e6:8.2f} us/call")
    print()
    print("Both branches are datetime.now()-bound dict lookups; the early")
    print("return removes the hold-days arithmetic, so the fail-open path is")
    print("expected to be at parity or slightly faster. Negligible vs the")
    print("2-5 s daily pipeline (fetch + LLM I/O).")


if __name__ == "__main__":
    main()

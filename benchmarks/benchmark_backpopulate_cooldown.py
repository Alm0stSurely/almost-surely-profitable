"""
Benchmark for backpopulate_cooldown_entries failure-direction fix (PR #62).

Cold path (one call per daily pipeline run): measures the healthy backpopulation
cost and the degraded paths (corrupt ledger, wrong-shape ledger, walk-back over a
damaged latest buy, all-unparseable fallback) so we can state honestly that
failing loud + preserving parseable evidence adds no measurable overhead to the
daily run.

Run:  .venv/bin/python benchmarks/benchmark_backpopulate_cooldown.py
"""
import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.getLogger("daily_run").setLevel(logging.CRITICAL)
logging.getLogger("risk.position_cooldown").setLevel(logging.CRITICAL)

from daily_run import backpopulate_cooldown_entries
from risk.position_cooldown import PositionCooldownManager

BASE = datetime(2026, 5, 1, 10, 0, 0)


def _buys(n: int, ticker: str = "SPY", broken_last: bool = False, broken_all: bool = False):
    trades = []
    for i in range(n):
        ts = f"garbage-{i}" if (broken_all or (broken_last and i == n - 1)) else (BASE + timedelta(days=i)).isoformat()
        trades.append({
            "timestamp": ts,
            "ticker": ticker,
            "action": "buy",
            "price": 400.0 + i,
            "quantity": 1.0,
            "total_value": 400.0 + i,
        })
    return trades


def _portfolio(trades_file: Path) -> Mock:
    portfolio = Mock()
    portfolio.positions = {"SPY": Mock(avg_price=100.0, quantity=1.0, current_price=100.0)}
    portfolio.trades_file = trades_file
    return portfolio


def bench_case(name: str, payload, n_iter: int, expect_entry: bool = True,
               expect_time: datetime = None, rounds: int = 3) -> float:
    """Best-of-*rounds* average wall time of one backpopulate call.

    Paths are computed inside the temp dir (never touches the real data dir).
    Each iteration resets the manager's in-memory entries so every timed call
    performs the full path under test.
    """
    best = float("inf")
    for _ in range(rounds):
        with tempfile.TemporaryDirectory() as tmp:
            trades_file = Path(tmp) / "trades_history.json"
            if payload is not None:
                if isinstance(payload, str):
                    trades_file.write_text(payload)
                else:
                    trades_file.write_text(json.dumps(payload))
            mgr = PositionCooldownManager(data_dir=tmp)
            portfolio = _portfolio(trades_file)

            # Integrity check on the first call, outside timing.
            backpopulate_cooldown_entries(mgr, portfolio)
            if expect_entry:
                assert "SPY" in mgr.entries, f"{name}: expected entry"
                if expect_time is not None:
                    assert mgr.entries["SPY"] == expect_time, f"{name}: wrong entry time"
            else:
                assert "SPY" not in mgr.entries, f"{name}: expected no entry"

            t0 = time.perf_counter()
            for _ in range(n_iter):
                mgr.entries.clear()
                backpopulate_cooldown_entries(mgr, portfolio)
            dt = (time.perf_counter() - t0) / n_iter
        best = min(best, dt)
    return best


def main() -> None:
    results = {}

    results["healthy (50 buys)"] = bench_case(
        "healthy50", _buys(50), n_iter=2000,
        expect_time=BASE + timedelta(days=49),
    )
    results["healthy (1000 buys)"] = bench_case(
        "healthy1000", _buys(1000), n_iter=500,
        expect_time=BASE + timedelta(days=999),
    )
    results["corrupt ledger"] = bench_case(
        "corrupt", "{corrupt", n_iter=2000, expect_entry=False,
    )
    results["wrong-shape ledger"] = bench_case(
        "wrongshape", '{"not": "a list"}', n_iter=2000, expect_entry=False,
    )
    results["walk-back (damaged latest)"] = bench_case(
        "walkback", _buys(50, broken_last=True), n_iter=2000,
        expect_time=BASE + timedelta(days=48),
    )
    results["all-unparseable (5 buys)"] = bench_case(
        "allunparseable", _buys(5, broken_all=True), n_iter=2000,
    )

    print(f"\n{'case':<32} {'per call':>12}")
    print("-" * 46)
    for name, dt in results.items():
        print(f"{name:<32} {dt * 1e6:>10.1f} µs")
    print("-" * 46)
    print("best of 3 rounds; module loggers at CRITICAL (error paths are loud")
    print("in production — timing here excludes the log-write cost by design).")


if __name__ == "__main__":
    main()

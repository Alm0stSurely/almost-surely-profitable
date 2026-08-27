"""
Benchmark churn-analysis guard paths.

Compares FIFO matching and bucketed aggregation with 0%, 10%, and 50% of the
records corrupted with non-finite values, measuring that filtering invalid
round trips does not dominate runtime.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.churn_analysis import _bucket_metrics, match_round_trips


def _make_trades(n, corrupt_ratio=0.0):
    trades = []
    for i in range(n):
        date = f"2026-01-{1 + (i % 30):02d}T10:00:00"
        price = 100.0 + i
        realized_pnl = (i % 11) - 5.0
        if corrupt_ratio > 0 and i % int(1 / corrupt_ratio) == 0:
            price = float("nan") if i % 2 == 0 else float("inf")
            realized_pnl = float("nan")
        trades.append({
            "ticker": "SPY",
            "action": "buy" if i % 2 == 0 else "sell",
            "timestamp": date,
            "price": price,
            "realized_pnl": realized_pnl,
        })
    return trades


def _run(func):
    best = float("inf")
    for _ in range(5):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    return best


def main():
    sizes = [100, 500, 1000]
    ratios = [0.0, 0.1, 0.5]

    print("| Operation | Trades | Invalid ratio | Avg ms/run |")
    print("|---|---|---|---|")
    for n in sizes:
        trades_by_ratio = {r: _make_trades(n, r) for r in ratios}
        for r in ratios:
            trades = trades_by_ratio[r]
            match_time = _run(lambda trades=trades: match_round_trips(trades))
            rts = match_round_trips(trades)
            bucket_time = _run(lambda rts=rts: _bucket_metrics(rts))
            total_time = match_time + bucket_time
            print(
                f"| match + bucket | {n} | {r*100:.0f}% | {total_time*1000:.3f} |"
            )


if __name__ == "__main__":
    main()

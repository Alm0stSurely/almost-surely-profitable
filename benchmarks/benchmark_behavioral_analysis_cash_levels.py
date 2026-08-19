"""Benchmark for behavioral_analysis cash-level formatting.

Measures the _format_cash_levels helper under normal and degenerate
portfolio conditions. The helper must stay cheap because it runs for every
row of the daily-results table in the behavioral analysis report.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analysis.behavioral_analysis import _format_cash_levels


def _result(date, cash, total, num_positions=5, total_return_pct=0.0, trades=None):
    return {
        "date": date,
        "portfolio_after": {
            "cash": cash,
            "total_value": total,
            "num_positions": num_positions,
            "total_return_pct": total_return_pct,
        },
        "executed_trades": trades or [],
    }


def _make_dataset(n, invalid_ratio=0.0):
    results = []
    for i in range(n):
        if i / max(n, 1) < invalid_ratio:
            results.append(_result(f"2026-08-{i + 1:02d}", 2500.0, 0.0))
        else:
            results.append(_result(f"2026-08-{i + 1:02d}", 2500.0, 10000.0))
    return results


def _bench_one(dataset, iterations=100):
    start = time.perf_counter()
    for _ in range(iterations):
        _format_cash_levels(dataset)
    elapsed = time.perf_counter() - start
    return elapsed / iterations


def main():
    sizes = [20, 100, 1000]
    invalid_ratios = [0.0, 0.1]

    print("=" * 70)
    print("Benchmark: behavioral_analysis._format_cash_levels")
    print("=" * 70)
    print(f"{'Rows':>10} {'Invalid %':>12} {'Iter':>8} {'Time/iter (s)':>16} {'Rows/s':>14}")
    print("-" * 70)

    for n in sizes:
        for ratio in invalid_ratios:
            dataset = _make_dataset(n, invalid_ratio=ratio)
            # Scale iterations so the benchmark does not take too long for large n.
            iterations = max(10, 10000 // max(n, 1))
            t = _bench_one(dataset, iterations=iterations)
            rows_per_sec = n / t
            print(
                f"{n:>10} {ratio * 100:>11.0f}% {iterations:>8} {t:>16.6f} {rows_per_sec:>14,.0f}"
            )


if __name__ == "__main__":
    main()

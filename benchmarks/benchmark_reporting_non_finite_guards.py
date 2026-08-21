"""Benchmark for reporting non-finite guards.

Measures weekly/monthly report generation when daily portfolio values contain
non-finite entries. The guarded path must keep aggregate metrics finite and must
not add measurable overhead compared to the all-finite baseline.
"""

import json
import logging
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from reporting import ReportGenerator

logging.getLogger("reporting").setLevel(logging.CRITICAL)


BASE_DATE = "2026-01-05"
BASE_YEAR = 2026
BASE_WEEK = 2


def _make_daily_result(total_value, total_return_pct, positions, trades):
    return {
        "date": BASE_DATE,
        "portfolio_after": {
            "total_value": total_value,
            "total_return_pct": total_return_pct,
            "positions": positions,
        },
        "executed_trades": trades,
    }


def _build_results_dir(n, invalid_ratio=0.0):
    """Create a temporary directory with *n* daily result files inside one ISO week."""
    tmpdir = tempfile.mkdtemp()
    start_value = 10000.0
    for i in range(n):
        value = start_value * (1.0 + i * 0.001)
        ret_pct = i * 0.1
        if i / max(n, 1) < invalid_ratio:
            value = float("nan") if i % 2 == 0 else float("inf")
        result = _make_daily_result(
            value,
            ret_pct,
            positions=[{"ticker": "SPY", "unrealized_pnl_pct": ret_pct}],
            trades=[{"ticker": "SPY", "action": "buy"}] if i % 5 == 0 else [],
        )
        with open(Path(tmpdir) / f"{BASE_DATE}_{i:05d}.json", "w") as f:
            json.dump(result, f)
    return tmpdir


def _bench(generator, results_dir, iterations=100):
    start = time.perf_counter()
    for _ in range(iterations):
        generator.generate_weekly_report(BASE_YEAR, BASE_WEEK)
    elapsed = time.perf_counter() - start
    return elapsed / iterations


def main():
    sizes = [20, 100, 500]
    invalid_ratios = [0.0, 0.1, 0.5]

    print("=" * 90)
    print("Benchmark: ReportGenerator.generate_weekly_report non-finite guards")
    print("=" * 90)
    print(f"{'Rows':>10} {'Invalid %':>12} {'Iter':>8} {'Time/iter (s)':>16} {'Rows/s':>14}")
    print("-" * 90)

    generator = ReportGenerator()
    for n in sizes:
        for ratio in invalid_ratios:
            results_dir = _build_results_dir(n, invalid_ratio=ratio)
            generator.results_dir = Path(results_dir)
            iterations = max(10, 50000 // max(n, 1))
            t = _bench(generator, results_dir, iterations=iterations)
            rows_per_sec = n / t

            report = generator.generate_weekly_report(BASE_YEAR, BASE_WEEK)
            assert report
            assert np.isfinite(report["start_value"])
            assert report["weekly_return_pct"] is None or np.isfinite(
                report["weekly_return_pct"]
            )
            assert np.isfinite(report["volatility"])

            print(
                f"{n:>10} {ratio * 100:>11.0f}% {iterations:>8} {t:>16.8f} {rows_per_sec:>14,.0f}"
            )


if __name__ == "__main__":
    main()

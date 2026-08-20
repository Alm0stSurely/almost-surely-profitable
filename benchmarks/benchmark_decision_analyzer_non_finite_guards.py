"""Benchmark for decision analyzer non-finite guards.

Measures `_calculate_metrics` when the outcome list contains a mix of
finite and non-finite forward returns. The guard path must not add
measurable overhead compared to the baseline NaN-only filter, and it must
keep aggregate metrics finite even under adversarial inputs.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from analysis.decision_analyzer import DecisionAnalyzer


def _make_outcomes(n, invalid_ratio=0.0):
    buys = []
    sells = []
    for i in range(n):
        if i % 2 == 0:
            ret = float("nan") if i / max(n, 1) < invalid_ratio else 0.02
            buys.append({"forward_return": ret, "success": ret > 0})
        else:
            ret = float("-inf") if i / max(n, 1) < invalid_ratio else -0.01
            sells.append({"forward_return": ret, "success": ret < 0})
    return {"buys": buys, "sells": sells}


def _bench_one(outcomes, iterations=1000):
    analyzer = DecisionAnalyzer()
    start = time.perf_counter()
    for _ in range(iterations):
        analyzer._calculate_metrics(outcomes)
    elapsed = time.perf_counter() - start
    return elapsed / iterations


def main():
    sizes = [100, 1000, 10000]
    invalid_ratios = [0.0, 0.1, 0.5]

    print("=" * 80)
    print("Benchmark: decision_analyzer._calculate_metrics non-finite guards")
    print("=" * 80)
    print(
        f"{'Rows':>10} {'Invalid %':>12} {'Iter':>8} {'Time/iter (s)':>16} {'Rows/s':>14}"
    )
    print("-" * 80)

    analyzer = DecisionAnalyzer()
    for n in sizes:
        for ratio in invalid_ratios:
            outcomes = _make_outcomes(n, invalid_ratio=ratio)
            iterations = max(10, 100000 // max(n, 1))
            t = _bench_one(outcomes, iterations=iterations)
            rows_per_sec = n / t
            # Verify correctness on the last run.
            metrics = analyzer._calculate_metrics(outcomes)
            assert np.isfinite(metrics["win_rate"])
            assert np.isfinite(metrics["avg_forward_return_buy"])
            assert np.isfinite(metrics["avg_forward_return_sell"])
            print(
                f"{n:>10} {ratio * 100:>11.0f}% {iterations:>8} {t:>16.8f} {rows_per_sec:>14,.0f}"
            )


if __name__ == "__main__":
    main()

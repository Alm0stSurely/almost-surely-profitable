"""Micro-benchmark for the chart non-finite guards in visualize.py.

The chart path renders on the backtest comparison and single-run flows, so
the defensive NaN coercion must stay on the same order of magnitude as the
raw float scaling it replaces. Measures the ``_finite_or_nan`` helper
directly across the reachable input distributions, plus the full metrics
extraction pass used by ``plot_metrics_comparison``.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest.visualize import _finite_or_nan


def _bench(func, n=50_000):
    for _ in range(1_000):
        func()
    start = time.perf_counter()
    for _ in range(n):
        func()
    elapsed = time.perf_counter() - start
    return elapsed / n * 1e6  # microseconds per call


def main():
    print("Chart non-finite guards benchmark")
    print("=" * 50)

    finite = _bench(lambda: _finite_or_nan(0.0428))
    nan_in = _bench(lambda: _finite_or_nan(float("nan")))
    none_in = _bench(lambda: _finite_or_nan(None))
    int_in = _bench(lambda: _finite_or_nan(7))
    raw_baseline = _bench(lambda: float(0.0428))

    print(f"_finite_or_nan(finite float) : {finite:8.3f} us/call")
    print(f"_finite_or_nan(int)          : {int_in:8.3f} us/call")
    print(f"_finite_or_nan(NaN)          : {nan_in:8.3f} us/call")
    print(f"_finite_or_nan(None)         : {none_in:8.3f} us/call")
    print(f"raw float() baseline         : {raw_baseline:8.3f} us/call")
    print()

    rows = {
        "buy_and_hold": {"total_return": 0.15, "sharpe_ratio": 1.2, "max_drawdown": -0.10},
        "equal_weight": {"total_return": 0.12, "sharpe_ratio": 0.9, "max_drawdown": -0.12},
        "llm": {"total_return": 0.20, "sharpe_ratio": 1.5, "max_drawdown": -0.08},
    }

    def extraction_guarded():
        strategies = list(rows.keys())
        [_finite_or_nan(rows[s]["total_return"]) * 100 for s in strategies]
        [_finite_or_nan(rows[s]["sharpe_ratio"]) for s in strategies]
        [_finite_or_nan(rows[s]["max_drawdown"]) * 100 for s in strategies]

    def extraction_raw():
        strategies = list(rows.keys())
        [rows[s]["total_return"] * 100 for s in strategies]
        [rows[s]["sharpe_ratio"] for s in strategies]
        [rows[s]["max_drawdown"] * 100 for s in strategies]

    guarded = _bench(extraction_guarded, n=20_000)
    raw = _bench(extraction_raw, n=20_000)
    print(f"metrics extraction, guarded (3 strategies) : {guarded:8.3f} us/pass")
    print(f"metrics extraction, raw baseline           : {raw:8.3f} us/pass")
    print(f"overhead factor                            : {guarded / raw:8.2f}x")


if __name__ == "__main__":
    main()

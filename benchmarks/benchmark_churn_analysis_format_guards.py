"""
Micro-benchmark for the safe formatting helpers in churn_analysis.py.

These helpers guard the churn report against non-finite metrics. The benchmark
verifies that the defensive ``n/a`` fallback is as fast as the happy path.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.churn_analysis import (
    RoundTrip,
    _safe_pct_str,
    _safe_value_str,
    analyze_churn,
    print_report,
)


class _NullIO:
    def write(self, _):
        pass

    def flush(self):
        pass


class _SuppressOutput:
    def __enter__(self):
        import sys as _sys

        self._old_stdout = _sys.stdout
        self._old_stderr = _sys.stderr
        _sys.stdout = _NullIO()
        _sys.stderr = _NullIO()
        return self

    def __exit__(self, *args):
        import sys as _sys

        _sys.stdout = self._old_stdout
        _sys.stderr = self._old_stderr


def _bench(func, value, n=100_000):
    # Warmup
    for _ in range(1000):
        func(value)

    with _SuppressOutput():
        start = time.perf_counter()
        for _ in range(n):
            func(value)
        elapsed = time.perf_counter() - start

    return elapsed / n * 1e6  # microseconds per call


def main():
    rt = RoundTrip("AAPL", datetime(2026, 1, 1), datetime(2026, 1, 5), 4.0, 100.0, 150.0, 160.0)
    metrics = analyze_churn([rt], [], [])

    with _SuppressOutput():
        start = time.perf_counter()
        for _ in range(10_000):
            print_report(metrics)
        full_report_us = (time.perf_counter() - start) / 10_000 * 1e6

    results = [
        ("print_report", "finite 1 RT", full_report_us),
        ("_safe_value_str", "finite", _bench(_safe_value_str, 1234.56)),
        ("_safe_value_str", "NaN", _bench(_safe_value_str, float("nan"))),
        ("_safe_pct_str", "finite", _bench(_safe_pct_str, 50.0)),
        ("_safe_pct_str", "NaN", _bench(_safe_pct_str, float("nan"))),
    ]

    print("| Helper | Input | µs/call |")
    print("|---|---|---|")
    for helper, inp, us in results:
        print(f"| {helper} | {inp} | {us:.3f} |")


if __name__ == "__main__":
    main()

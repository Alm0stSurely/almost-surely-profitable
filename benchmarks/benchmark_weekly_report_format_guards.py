"""
Micro-benchmark for the safe formatting helpers in weekly_report.py.

These helpers are on the report-printing hot path; when the weekly report is run
after market close, every portfolio field, position, and trade is formatted.
The benchmark verifies that the defensive ``n/a`` fallback does not add
noticeable overhead compared to the happy path.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weekly_report import _safe_pct_str, _safe_position_field, _safe_value_str


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
    results = [
        ("_safe_value_str", "finite", _bench(_safe_value_str, 1234.56)),
        ("_safe_value_str", "NaN", _bench(_safe_value_str, float("nan"))),
        ("_safe_pct_str", "finite", _bench(_safe_pct_str, 5.123)),
        ("_safe_pct_str", "NaN", _bench(_safe_pct_str, float("nan"))),
        ("_safe_position_field", "finite", _bench(_safe_position_field, 12.345)),
        ("_safe_position_field", "NaN", _bench(_safe_position_field, float("nan"))),
    ]

    print("| Helper | Input | µs/call |")
    print("|---|---|---|")
    for helper, inp, us in results:
        print(f"| {helper} | {inp} | {us:.3f} |")


if __name__ == "__main__":
    main()

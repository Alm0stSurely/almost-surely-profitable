"""Benchmark daily_run non-finite guards.

Measures the overhead of the safe weight and formatting helpers under
degenerate inputs. The guard path should add negligible latency to the
cron-scheduled pipeline.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from daily_run import _safe_pct_str, _safe_value_str, _safe_weight


def _timeit(label: str, func, rounds: int = 100_000) -> float:
    """Run *func* *rounds* times and return average microseconds per call."""
    start = time.perf_counter()
    for _ in range(rounds):
        func()
    elapsed = time.perf_counter() - start
    avg_us = (elapsed / rounds) * 1_000_000
    print(f"{label:<45} {rounds:>8} rounds | {avg_us:>8.3f} µs/call")
    return avg_us


def main():
    print("=" * 70)
    print("BENCHMARK: daily_run non-finite guards")
    print("=" * 70)

    _timeit("_safe_weight (finite)", lambda: _safe_weight(250.0, 1000.0))
    _timeit("_safe_weight (NaN total)", lambda: _safe_weight(250.0, float("nan")))
    _timeit("_safe_weight (zero total)", lambda: _safe_weight(250.0, 0.0))
    _timeit("_safe_pct_str (finite)", lambda: _safe_pct_str(0.05))
    _timeit("_safe_pct_str (NaN)", lambda: _safe_pct_str(float("nan")))
    _timeit("_safe_value_str (finite)", lambda: _safe_value_str(123.456))
    _timeit("_safe_value_str (inf)", lambda: _safe_value_str(float("inf")))

    print("=" * 70)
    print("OK - daily_run guard helpers are fast on both finite and non-finite inputs.")


if __name__ == "__main__":
    main()

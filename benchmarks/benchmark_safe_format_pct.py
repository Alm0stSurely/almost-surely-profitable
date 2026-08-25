"""Benchmark safe_format_pct versus raw f-string formatting.

The helper is intentionally a thin guard, so the expectation is that the
overhead on valid inputs is negligible while it eliminates the risk of
non-finite tokens leaking into cron output.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import safe_format_pct

VALUES = [0.1234, -0.05, 0.0, 0.0042, -0.0042]
N = 100_000


def _bench(func, label):
    start = time.perf_counter()
    for _ in range(N):
        for v in VALUES:
            func(v)
    elapsed = (time.perf_counter() - start) / (N * len(VALUES)) * 1e6
    print(f"  {label}: {elapsed:.3f} µs/call")


def raw_format(v):
    return f"{v:.2%}"


def safe_format(v):
    return safe_format_pct(v)


def main():
    print("Benchmark: percentage formatting (5 values × 100,000 iterations)")
    _bench(raw_format, "raw f-string")
    _bench(safe_format, "safe_format_pct")


if __name__ == "__main__":
    main()

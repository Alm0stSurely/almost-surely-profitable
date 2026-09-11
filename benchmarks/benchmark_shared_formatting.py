"""
Micro-benchmark for the shared finite-safe formatting helpers in
utils.formatting.

The helpers sit on every console/LLM formatting path, so the defensive
``n/a`` fallback must stay on the same order of magnitude as the happy
path, and the extra isinstance dispatch versus raw format() must remain
negligible.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.formatting import _fmt_finite, _fmt_pct


def _bench(func, n=100_000):
    for _ in range(1000):
        func()

    start = time.perf_counter()
    for _ in range(n):
        func()
    elapsed = time.perf_counter() - start
    return elapsed / n * 1e6  # microseconds per call


def main():
    print("Shared formatting helpers benchmark")
    print("=" * 50)

    cases = [
        ("_fmt_finite float", lambda: _fmt_finite(82.5, ".0f")),
        ("_fmt_finite int", lambda: _fmt_finite(42, ">3")),
        ("_fmt_finite np.int64", lambda: _fmt_finite(np.int64(7), ">3")),
        ("_fmt_finite np.float64", lambda: _fmt_finite(np.float64(2.5), ".1f")),
        ("_fmt_finite NaN", lambda: _fmt_finite(float("nan"), ".0f")),
        ("_fmt_finite None", lambda: _fmt_finite(None, ".2f")),
        ("_fmt_finite bool", lambda: _fmt_finite(True, ">3")),
        ("_fmt_pct finite", lambda: _fmt_pct(0.0521, ">6.2f")),
        ("_fmt_pct NaN", lambda: _fmt_pct(float("nan"), ">6.2f")),
        ("raw format() (baseline)", lambda: format(82.5, ".0f")),
    ]

    for label, func in cases:
        us = _bench(func)
        print(f"{label:<28} {us:>10.3f} µs/call")


if __name__ == "__main__":
    main()

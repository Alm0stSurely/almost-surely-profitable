#!/usr/bin/env python3
"""
Benchmark: tz-aware index normalization overhead.

The convention fix (tz_convert("UTC") before tz_localize(None)) adds a
timezone conversion step to defensive normalization branches. This measures
whether that step is material relative to the downstream pandas work.
"""

import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def make_frame(n: int) -> pd.DataFrame:
    tz_index = pd.date_range("2026-01-01", periods=n, freq="D", tz="Europe/Paris")
    return pd.DataFrame({"Close": range(n)}, index=tz_index)


def bench(fn, frame, repeats: int = 200) -> float:
    # Warmup
    fn(frame.copy())
    best = float("inf")
    for _ in range(repeats):
        df = frame.copy()
        t0 = time.perf_counter_ns()
        fn(df)
        best = min(best, time.perf_counter_ns() - t0)
    return best / 1000.0  # µs


def old_convention(df: pd.DataFrame) -> None:
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)


def new_convention(df: pd.DataFrame) -> None:
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)


def main() -> None:
    for n in (30, 250):
        frame = make_frame(n)
        old_us = bench(old_convention, frame)
        new_us = bench(new_convention, frame)
        print(f"n={n:>4} rows  old (localize only): {old_us:8.2f} µs  "
              f"new (convert+localize): {new_us:8.2f} µs  "
              f"delta: {new_us - old_us:+.2f} µs "
              f"({(new_us / old_us - 1) * 100:+.1f}%)")

    print("\nTypical workload context: a daily pipeline touches 21 tickers x "
          "~250 bars, and the branch fires only when upstream data arrives "
          "tz-aware (the canonical fetch path already normalizes).")


if __name__ == "__main__":
    main()

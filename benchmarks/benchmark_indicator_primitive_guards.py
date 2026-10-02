"""Benchmark: primitive-level non-finite drop vs pre-fix baseline.

Methodology: N_ITER iterations, best of 5 rounds, A/B via targeted stash of
src/data/indicators.py. No perf claim is made: the guard adds one O(n) pass
over the series and is expected to be within noise on clean data. The
benchmark exists because a benchmark that never triggers the guarded path
verifies nothing: the degraded rows (dirty input) exercise the new branch,
and the embedded functional asserts below fail on old code (in-benchmark
fail-on-old).

Run from the repo root:
    python3 benchmarks/benchmark_indicator_primitive_guards.py
"""

import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.indicators import (  # noqa: E402
    calculate_bollinger_bands,
    calculate_drawdown,
    calculate_rsi,
    calculate_sma,
    calculate_volatility,
)

N_ITER = 2000
ROUNDS = 5


def _clean_series(n=120):
    rng = np.linspace(100.0, 130.0, n) + np.sin(np.arange(n) / 5.0)
    return pd.Series(rng)


def _dirty_series(n=120):
    s = _clean_series(n)
    s.iloc[10] = np.nan
    s.iloc[40] = np.inf
    s.iloc[-1] = np.nan
    return s


def _bench(fn, series, window=20):
    best = float("inf")
    for _ in range(ROUNDS):
        devnull = io.StringIO()
        with contextlib.redirect_stderr(devnull):  # loud-by-design logging timed, not printed
            t0 = time.perf_counter()
            for _ in range(N_ITER):
                fn(series.copy(), window) if fn is calculate_sma else fn(series.copy())
            best = min(best, time.perf_counter() - t0)
    return best / N_ITER * 1e6  # us per call


def _fn(fn, series, window=20):
    return fn(series.copy(), window) if fn is calculate_sma else fn(series.copy())


# Signature unification without per-iteration branching cost inside timing:
def bench_one(fn, series, window=20):
    best = float("inf")
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            _fn(fn, series, window)
        best = min(best, time.perf_counter() - t0)
    return best / N_ITER * 1e6


def functional_gate():
    """Fail loudly if the guarded contract is violated (fail-on-old)."""
    dirty = _dirty_series()
    clean = dirty[np.isfinite(dirty.to_numpy(dtype=float))]
    sma_d = calculate_sma(dirty.copy(), 20)
    sma_c = calculate_sma(clean.copy(), 20)
    assert len(sma_d) == len(sma_c), "sma did not drop non-finite ticks"
    dd = calculate_drawdown(dirty.copy())
    assert np.isfinite(dd.iloc[-1]), "drawdown propagated trailing NaN"
    rsi = calculate_rsi(dirty.copy(), 14)
    assert np.isfinite(rsi.iloc[-1]), "rsi non-finite final value"


def main():
    functional_gate()

    clean = _clean_series()
    dirty = _dirty_series()

    rows = [
        ("sma(20) clean ", calculate_sma, clean),
        ("sma(20) dirty ", calculate_sma, dirty),
        ("rsi(14) clean ", calculate_rsi, clean),
        ("rsi(14) dirty ", calculate_rsi, dirty),
        ("boll(20) clean", calculate_bollinger_bands, clean),
        ("boll(20) dirty", calculate_bollinger_bands, dirty),
        ("vol(20) clean ", calculate_volatility, clean),
        ("vol(20) dirty ", calculate_volatility, dirty),
        ("dd clean      ", calculate_drawdown, clean),
        ("dd dirty      ", calculate_drawdown, dirty),
    ]

    print(f"{'primitive':<16} {'us/call':>10}  (N_ITER={N_ITER}, best of {ROUNDS})")
    for label, fn, series in rows:
        us = bench_one(fn, series)
        print(f"{label:<16} {us:>9.2f}us")


if __name__ == "__main__":
    main()

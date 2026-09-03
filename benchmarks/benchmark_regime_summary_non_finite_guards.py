"""
Micro-benchmark for the non-finite guards in RegimeState.summary().

The summary feeds both the console and the LLM prompt, so the defensive
``n/a`` fallback must stay on the same order of magnitude as the happy path.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.regime_detector import RegimeState, _fmt_finite


def _make_state(vol_pct=82.5, adx=30.2, corr=0.45):
    return RegimeState(
        volatility_regime="high",
        trend_regime="trending_up",
        correlation_regime="normal",
        volatility_percentile=vol_pct,
        adx_value=adx,
        avg_correlation=corr,
    )


def _bench(func, n=100_000):
    for _ in range(1000):
        func()

    start = time.perf_counter()
    for _ in range(n):
        func()
    elapsed = time.perf_counter() - start
    return elapsed / n * 1e6  # microseconds per call


def main():
    print("Regime summary non-finite guards benchmark")
    print("=" * 50)

    finite_state = _make_state()
    nan_state = _make_state(vol_pct=float("nan"), adx=float("inf"), corr=float("-inf"))

    cases = [
        ("summary() finite", lambda: finite_state.summary()),
        ("summary() non-finite", lambda: nan_state.summary()),
        ("_fmt_finite finite", lambda: _fmt_finite(82.5, ".0f")),
        ("_fmt_finite NaN", lambda: _fmt_finite(float("nan"), ".0f")),
        ("_fmt_finite inf", lambda: _fmt_finite(float("inf"), ".1f")),
        ("_fmt_finite None", lambda: _fmt_finite(None, ".2f")),
    ]

    for label, func in cases:
        us = _bench(func)
        print(f"{label:<28} {us:>10.3f} µs/call")


if __name__ == "__main__":
    main()

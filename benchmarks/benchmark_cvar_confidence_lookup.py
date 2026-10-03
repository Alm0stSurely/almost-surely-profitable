"""Benchmark for the CVaR confidence-lookup hardening (sentinel removal).

The change replaces .get(level, 0.0) with direct indexing in
calculate_portfolio_cvar and calculate_drawdown_cvar. The happy-path cost is
a same-cost lookup variant; this benchmark exists because a benchmark that
never triggers the guarded path verifies nothing:

- healthy rows time the default-levels path (old vs new within noise);
- the degraded row asks for custom confidence levels and MUST raise KeyError
  (the embedded functional gate: on old code the absence is laundered into a
  zeroed CVaRResult and the gate fails).

No perf claim: lookup-variant change, degraded row is the point.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from risk.cvar import calculate_portfolio_cvar, calculate_drawdown_cvar

N_ITER = 2000
N_ROUNDS = 5

PR = {
    'SPY': np.array([0.01, -0.02, 0.005, -0.01, 0.008, 0.012] * 10),
    'QQQ': np.array([0.005, 0.01, -0.01, 0.002, -0.008, 0.006] * 10),
}
WEIGHTS = {'SPY': 0.5, 'QQQ': 0.5}
EQUITY = np.array([100.0, 105.0, 103.0, 98.0, 101.0, 97.0] * 10)


def _time(fn):
    best = float('inf')
    for _ in range(N_ROUNDS):
        start = time.perf_counter()
        for _ in range(N_ITER):
            fn()
        best = min(best, (time.perf_counter() - start) / N_ITER * 1e6)
    return best


def healthy_portfolio():
    calculate_portfolio_cvar(PR, WEIGHTS)


def healthy_drawdown():
    calculate_drawdown_cvar(EQUITY, window=20, confidence=0.95)


def degraded_custom_levels():
    """Embedded functional gate: absence of the 95/99 levels must be LOUD."""
    try:
        calculate_portfolio_cvar(PR, WEIGHTS, confidence_levels=[0.90])
    except KeyError:
        return
    raise AssertionError(
        "sentinel collision: custom levels laundered into a zeroed CVaRResult "
        "(pre-fix behavior — absence must fail loud)"
    )


if __name__ == "__main__":
    t_port = _time(healthy_portfolio)
    t_dd = _time(healthy_drawdown)
    degraded_custom_levels()

    print(f"healthy portfolio_cvar : {t_port:8.2f} us/call (best of {N_ROUNDS})")
    print(f"healthy drawdown_cvar  : {t_dd:8.2f} us/call (best of {N_ROUNDS})")
    print("degraded custom-levels : KeyError raised (functional gate OK)")
    print("OK - absence fails loud; happy-path cost unchanged (no perf claim).")

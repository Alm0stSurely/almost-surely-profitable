"""Benchmark: get_latest_indicators omission contract (PR #74).

Not a performance claim — the change replaces a dict literal of .get
lookups with a loop of the same lookups; the timing rows exist to prove the
cost is unchanged (within shared-runner noise) and to pin the embedded
functional gate, which FAILS on pre-fix code: any sentinel default (rsi 50,
bb_position 0.5, 0.0 laundering) raises AssertionError.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.indicators import calculate_all_indicators, get_latest_indicators

N_ITER = 2000
ROUNDS = 5


def _healthy_frame():
    df = pd.DataFrame({"Close": np.linspace(100.0, 129.0, 60)})
    return calculate_all_indicators(df)


def _degraded_frame():
    """1-tick frame through the real producer: BB bands / vol / daily
    return are NaN by pandas ddof=1 semantics on a single observation."""
    df = pd.DataFrame({"Close": [100.0]}, index=pd.date_range("2026-10-03", periods=1))
    return calculate_all_indicators(df)


def functional_gate():
    degraded = get_latest_indicators(_degraded_frame())
    missing = get_latest_indicators(pd.DataFrame({"Close": [100.0]}))
    mixed = get_latest_indicators(pd.DataFrame({
        "Close": [100.0], "SMA_20": [105.0], "RSI_14": [np.nan],
    }))

    # No sentinel laundering anywhere: keys absent, not defaulted.
    for absent_key in ("bb_upper", "volatility_annual", "daily_return"):
        assert absent_key not in degraded, f"degraded frame laundered {absent_key}"
        assert degraded.get(absent_key) != 0.0, f"{absent_key} defaulted to 0.0"
    assert "rsi_14" not in missing, "missing-columns frame fabricated rsi_14"
    assert missing.get("rsi_14") != 50.0, "rsi_14 defaulted to 50.0"
    assert "bb_position" not in missing, "missing-columns frame fabricated bb_position"
    assert missing.get("bb_position") != 0.5, "bb_position defaulted to 0.5"
    # Genuine readings survive on every frame (per-element, not all-or-nothing).
    assert mixed == {"price": 100.0, "sma_20": 105.0}, f"mixed frame: {mixed}"
    assert degraded["price"] == 100.0
    assert degraded["rsi_14"] == 50.0  # genuine flat-tick reading via producer
    healthy = get_latest_indicators(_healthy_frame())
    assert len(healthy) == 12, f"healthy frame lost keys: {sorted(healthy)}"


def _time(func, frame):
    best = float("inf")
    for _ in range(ROUNDS):
        start = time.perf_counter()
        for _ in range(N_ITER):
            func(frame)
        best = min(best, time.perf_counter() - start)
    return best / N_ITER * 1e6


if __name__ == "__main__":
    functional_gate()
    print("functional gate: PASS (no sentinel laundering; genuine readings survive)")

    healthy_us = _time(get_latest_indicators, _healthy_frame())
    degraded_us = _time(get_latest_indicators, _degraded_frame())
    print(f"get_latest_indicators  N_ITER={N_ITER}  best of {ROUNDS}")
    print(f"  healthy  frame (12 keys): {healthy_us:8.2f} us/call")
    print(f"  degraded frame (7 keys):  {degraded_us:8.2f} us/call")
    print("same-cost lookup variant — no perf claim; gate fails on pre-fix code")

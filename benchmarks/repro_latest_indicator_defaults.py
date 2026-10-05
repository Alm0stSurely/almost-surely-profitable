"""Reproduction: get_latest_indicators launders absence/non-finite into
sentinel readings that collide with real measurements.

Doctrine (PR #65 sentinel-collision): a sentinel is only safe when real data
can never produce it. Here real data CAN produce every one of these:
  rsi_14 = 50.0       -> a genuine neutral-momentum reading
  bb_position = 0.5   -> a genuine mid-band reading
  drawdown = 0.0      -> a genuine at-peak reading
  daily_return = 0.0  -> a genuine flat day
  sma_20 = 0.0 etc.   -> €0.00 (self-announcing, but still a lie in JSON)

Routes of malformation exercised:
  R1: missing indicator columns (hand-built frame, short-history fetch)
  R2: non-finite values in populated columns (1-tick frame -> NaN BB bands,
      NaN volatility, NaN daily return via pandas ddof=1 semantics)
  R3: mixed frame — some finite, some non-finite (per-element drop must
      NOT void the whole dict: scope of failure == scope of malformation)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.indicators import get_latest_indicators


def check(label, condition):
    status = "OK " if condition else "LIE"
    print(f"[{status}] {label}")
    return condition


lies = 0

# R1: missing columns — every indicator fabricated
df_missing = pd.DataFrame({"Close": [100.0]})
latest = get_latest_indicators(df_missing)
print(f"\nR1 missing-columns frame -> {latest}")
lies += not check("R1 rsi_14 fabricated (50.0 neutral lie)", latest.get("rsi_14") == 50.0)
lies += not check("R1 bb_position fabricated (0.5 mid-band lie)", latest.get("bb_position") == 0.5)
lies += not check("R1 sma_20 fabricated (0.0)", latest.get("sma_20") == 0.0)

# R2: 1-tick frame through the real producer — NaN surfaces as sentinels
from data.indicators import calculate_all_indicators

df_one = pd.DataFrame({"Close": [100.0]}, index=pd.date_range("2026-10-03", periods=1))
with_ind = calculate_all_indicators(df_one)
print(f"\nR2 1-tick producer frame, raw columns:\n{with_ind[['Close','BB_upper','Volatility_20','Daily_Return']]}")
latest2 = get_latest_indicators(with_ind)
print(f"R2 1-tick frame -> {latest2}")
lies += not check("R2 NaN BB_upper laundered to 0.0", latest2.get("bb_upper") == 0.0)
lies += not check("R2 NaN volatility laundered to 0.0", latest2.get("volatility_annual") == 0.0)
lies += not check("R2 NaN daily_return laundered to 0.0", latest2.get("daily_return") == 0.0)

# R3: mixed frame — one finite sibling must survive (and currently does, but
# must KEEP doing so after the fix; this pins the per-element property)
df_mixed = pd.DataFrame({
    "Close": [100.0],
    "SMA_20": [105.0],
    "RSI_14": [np.nan],
})
latest3 = get_latest_indicators(df_mixed)
print(f"\nR3 mixed frame -> {latest3}")
lies += not check("R3 finite sma_20 present alongside fabricated rsi", latest3.get("sma_20") == 105.0 and latest3.get("rsi_14") == 50.0)

# Where the lie travels: the LLM prompt consumer renders absent keys as n/a
# (PR #70) — but the producer never lets absence through.
print(f"\n=== TOTAL LAUNDRED READINGS: {lies} ===")
print("Expected: >0 on pre-fix code (reproduction), 0 after the fix.")

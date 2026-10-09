"""Benchmark: cost of the finite-guard in MetaLabeler._extract_features.

This is NOT a performance PR — the guard's cost is reported honestly so the
PR makes no unverified claim. Timing covers the healthy path only (the
degenerate path logs warnings, which dominates by orders of magnitude and
is not the metric of interest); both the A (pre-fix) and B (post-fix)
states are measured via git stash A/B in the session, best of 5.

Methodology note: sklearn fit/predict dominates any real meta-labeling run
by orders of magnitude; _extract_features is the only function this PR
touches on the hot path.
"""

import time

import numpy as np
import pandas as pd

from src.backtest.meta_labeling import (
    MetaLabeler,
    MetaLabelingConfig,
    PrimarySignal,
    SignalType,
)

N_ITER = 2000
ROUNDS = 5

dates = pd.date_range("2024-01-01", periods=60, freq="B")
np.random.seed(42)
data = pd.DataFrame(
    {
        "close": 100 + np.cumsum(np.random.randn(60) * 0.5),
        "volume": np.random.randint(1_000_000, 5_000_000, 60),
        "rsi": 30 + np.random.rand(60) * 40,
        "bb_position": np.random.rand(60),
    },
    index=dates,
)
labeler = MetaLabeler(MetaLabelingConfig())
signal = PrimarySignal(timestamp=dates[30], ticker="T",
                       signal=SignalType.BUY, confidence=0.8)

# warmup
for _ in range(50):
    labeler._extract_features(signal, data)

best = float("inf")
for _ in range(ROUNDS):
    start = time.perf_counter()
    for _ in range(N_ITER):
        labeler._extract_features(signal, data)
    elapsed = (time.perf_counter() - start) / N_ITER
    best = min(best, elapsed)

print(f"_extract_features healthy path: {best * 1e6:.2f} us/call "
      f"(N_ITER={N_ITER}, best of {ROUNDS})")

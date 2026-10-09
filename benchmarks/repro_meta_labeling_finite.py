"""Functional gate + pre-fix record: meta-labeling finiteness guards.

Embedded gate for PR (fails on pre-fix code, passes post-fix).

PRE-FIX SYMPTOMS (sklearn 1.8 boundary behavior, verified 2026-10-09):
  1. _extract_features emitted NaN/inf features from degenerate windows
     (e.g. NaN tail close -> NaN cumulative_return / price_vs_sma20).
  2. fit() did NOT crash on NaN features — sklearn >= 1.4 RandomForest
     silently learns NaN as a distinct split direction. Garbage in,
     zero trace. (+inf DID crash the whole call: all-or-nothing ValueError
     over one bad window.)
  3. predict() same: NaN silently scored, inf crashed all signals.
  4. size_positions: NaN proba -> 0.0 only by accident of max() comparison
     semantics (undesigned); None proba -> TypeError crash;
     avg_win_loss_ratio=0 -> ZeroDivisionError; negative ratio silently
     inflated Kelly.
  5. signal.confidence == 0.0 laundered to 0.5 (`or` falsy sentinel),
     colliding with the absent-confidence default.

POST-FIX CONTRACT (asserted below):
  - features: only finite values emitted, per-element drop, loud log,
    n-1 healthy siblings survive.
  - fit: non-finite rows dropped loudly (one bad sample voids only itself).
  - predict: signals missing any trained feature are skipped (proba 0.0),
    never zero-filled.
  - size_positions: None/NaN proba refused loudly with position 0.0;
    avg_win_loss_ratio validated positive-finite at the boundary.
  - confidence 0.0 survives as 0.0; None defaults to 0.5.

Run: PYTHONPATH=. .venv/bin/python benchmarks/repro_meta_labeling_finite.py
"""

import logging

import numpy as np
import pandas as pd

from src.backtest.meta_labeling import (
    MetaLabel,
    MetaLabeler,
    MetaLabelingConfig,
    PrimarySignal,
    SignalType,
)

logging.basicConfig(level=logging.WARNING)

dates = pd.date_range("2024-01-01", periods=60, freq="B")
np.random.seed(42)
healthy = pd.DataFrame(
    {
        "close": 100 + np.cumsum(np.random.randn(60) * 0.5),
        "volume": np.random.randint(1_000_000, 5_000_000, 60),
        "rsi": 30 + np.random.rand(60) * 40,
        "bb_position": np.random.rand(60),
    },
    index=dates,
)

# NaN tail close at the last timestamp
dirty = healthy.copy()
dirty.loc[dates[59], "close"] = np.nan
sig_dirty = PrimarySignal(timestamp=dates[59], ticker="T", signal=SignalType.BUY, confidence=0.8)
# inf first close -> cumulative_return = x / 0 = +inf candidate
inf_dirty = healthy.copy()
inf_dirty.loc[dates[40], "close"] = 0.0
sig_inf = PrimarySignal(timestamp=dates[59], ticker="T", signal=SignalType.BUY, confidence=0.8)


class LogTrap(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record)


trap = LogTrap()
logging.getLogger("src.backtest.meta_labeling").addHandler(trap)


def warnings_containing(sub):
    return [r for r in trap.records if sub in r.getMessage()]


labeler = MetaLabeler(MetaLabelingConfig())

# --- 1. per-element finite drop, siblings survive ----------------------------
trap.records.clear()
feats = labeler._extract_features(sig_dirty, dirty)
non_finite = {k: v for k, v in feats.items() if not np.isfinite(v)}
assert not non_finite, f"non-finite features leaked: {non_finite}"
assert "cumulative_return" not in feats, "NaN cumulative_return should be omitted"
assert "price_vs_sma20" not in feats, "NaN price_vs_sma20 should be omitted"
assert "returns_mean" in feats and "rsi" in feats, "healthy siblings must survive"
assert warnings_containing("Dropped non-finite features"), "drop must be loud"
print("[ok] 1. per-element finite drop + loud log + siblings survive")

# --- 2. +inf candidate dropped at extraction ---------------------------------
# A zero close at the window start makes pct_change divide by zero on the
# next tick, so returns_* are poisoned alongside cumulative_return — all
# correctly dropped; the volume/indicator families must survive.
trap.records.clear()
feats_inf = labeler._extract_features(sig_inf, inf_dirty)
assert "cumulative_return" not in feats_inf, "+inf cumulative_return should be omitted"
assert "volume_vs_mean" in feats_inf and "rsi" in feats_inf and "bb_position" in feats_inf, \
    "unrelated feature families must survive the inf window"
print("[ok] 2. +inf window candidates dropped, unrelated families survive")

# --- 3. fit: one degenerate sample drops loudly, model still trains ----------
signals = [
    PrimarySignal(timestamp=dates[min(20 + (i % 35), 59)], ticker="T",
                  signal=SignalType.BUY, confidence=0.6)
    for i in range(120)
]
outcomes = [i % 2 for i in range(120)]
signals[7] = sig_dirty

mixed = healthy.copy()
mixed.loc[dates[59], "close"] = np.nan

trap.records.clear()
fitted = MetaLabeler(MetaLabelingConfig()).fit(signals, mixed, outcomes)
assert fitted.is_fitted, "fit must succeed with 119/120 healthy samples"
assert warnings_containing("Dropping 1/120"), "sample drop must be loud with count"
print("[ok] 3. fit drops 1/120 loudly and trains on the rest")

# --- 4. predict: degenerate signal skipped, healthy sibling scored -----------
trap.records.clear()
healthy_sig = PrimarySignal(timestamp=dates[30], ticker="T",
                            signal=SignalType.BUY, confidence=0.6)
preds = fitted.predict([sig_dirty, healthy_sig], mixed)
assert preds[0].predicted_proba == 0.0, "degenerate signal must be skipped (0.0)"
assert 0.0 < preds[1].predicted_proba < 1.0, "healthy signal must be scored"
assert warnings_containing("missing features") or warnings_containing("Insufficient"), \
    "skip must be loud"
print("[ok] 4. predict skips degenerate signal, scores healthy sibling")

# --- 5. size_positions: proba guards + ratio validation -----------------------
trap.records.clear()
sized = fitted.size_positions(
    [MetaLabel(signal=sig_dirty, features={}, actual_outcome=0, predicted_proba=float("nan")),
     MetaLabel(signal=sig_dirty, features={}, actual_outcome=0, predicted_proba=None),
     MetaLabel(signal=healthy_sig, features={}, actual_outcome=0, predicted_proba=0.9)]
)
assert sized[0].position_size == 0.0 and sized[1].position_size == 0.0
assert sized[2].position_size > 0.0, "healthy proba must size normally"
assert len(warnings_containing("Refusing to size")) == 2, "each refusal must be loud"
for bad_b in (0.0, -1.0, float("nan"), float("inf")):
    try:
        fitted.size_positions(
            [MetaLabel(signal=healthy_sig, features={}, actual_outcome=0, predicted_proba=0.9)],
            avg_win_loss_ratio=bad_b,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(f"avg_win_loss_ratio={bad_b!r} must raise ValueError")
print("[ok] 5. size_positions refuses None/NaN loudly, validates Kelly ratio")

# --- 6. confidence 0.0 survives; None defaults to 0.5 -------------------------
feats_c0 = labeler._extract_features(
    PrimarySignal(timestamp=dates[30], ticker="T", signal=SignalType.BUY, confidence=0.0),
    healthy,
)
assert feats_c0["signal_confidence"] == 0.0, "explicit 0.0 confidence must survive"
feats_cn = labeler._extract_features(
    PrimarySignal(timestamp=dates[30], ticker="T", signal=SignalType.BUY, confidence=None),
    healthy,
)
assert feats_cn["signal_confidence"] == 0.5, "absent confidence defaults to 0.5"
print("[ok] 6. confidence 0.0 survives, None defaults to 0.5")

# --- 7. healthy-path parity: extraction key set unchanged ---------------------
feats_h = labeler._extract_features(healthy_sig, healthy)
expected_keys = {
    "returns_mean", "returns_std", "returns_skew", "cumulative_return",
    "price_vs_sma20", "volatility_trend", "volume_vs_mean", "volume_trend",
    "rsi", "rsi_trend", "bb_position", "primary_signal", "signal_confidence",
    "hour", "day_of_week", "is_month_start", "is_month_end",
}
assert set(feats_h.keys()) == expected_keys, f"healthy key set drifted: {set(feats_h) ^ expected_keys}"
print("[ok] 7. healthy-path feature key set byte-parity (17 keys)")

print()
print("ALL 7 GATES PASS")

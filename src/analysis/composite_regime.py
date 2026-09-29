"""
Composite regime score for regime-conditioned cash bands (experiment H3).

Buckets three regime legs — volatility, trend, correlation — each into
{-1, 0, +1} (unfavorable / neutral / favorable) and sums them into a
composite score in {-3, ..., +3}. Each candidate arm of the experiment
maps the composite to a (lower, upper) cash band; the control arm is the
current 1-D volatility mapping (HIGH 30-50 %, NORMAL 15-30 %, LOW 10-20 %).

Design reference: docs/experiments/2026-09-28-regime-conditioned-cash-band.md

Degenerate inputs follow the house rule: non-finite values coerce to
neutral defaults instead of propagating NaN/inf into the band logic.
A degenerate ADX (0.0 or non-finite) buckets the trend leg as neutral —
it never counts as information.
"""

import math
import numbers
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

# Default thresholds — mirror RegimeDetector defaults so the composite and
# the 1-D classifier agree on what "high" / "low" mean.
VOL_PCT_HIGH = 75.0
VOL_PCT_LOW = 25.0
ADX_TRENDING = 25.0
ADX_MEAN_REVERTING = 20.0
TREND_UP_FRACTION = 0.6
TREND_DOWN_FRACTION = 0.4
CORR_PCT_HIGH = 75.0
CORR_PCT_LOW = 25.0

# Minimum composite shift required to change the band intra-week (hysteresis).
HYSTERESIS_MIN_SHIFT = 2

# Candidate cash-band tables, keyed by composite score. Percent of portfolio.
_COMPOSITE_BANDS: Dict[str, Dict[int, Tuple[float, float]]] = {
    "A": {3: (8.0, 18.0), 2: (10.0, 22.0), 1: (12.0, 26.0), 0: (15.0, 30.0),
          -1: (18.0, 35.0), -2: (25.0, 45.0), -3: (30.0, 55.0)},
    "B": {3: (5.0, 15.0), 2: (8.0, 20.0), 1: (12.0, 26.0), 0: (15.0, 30.0),
          -1: (20.0, 40.0), -2: (28.0, 50.0), -3: (35.0, 60.0)},
    "C": {3: (2.0, 10.0), 2: (5.0, 15.0), 1: (10.0, 25.0), 0: (15.0, 30.0),
          -1: (22.0, 45.0), -2: (30.0, 55.0), -3: (40.0, 65.0)},
}

# Control arm: current 1-D volatility mapping (production prompt, 2026-09).
_CONTROL_BANDS: Dict[str, Tuple[float, float]] = {
    "high": (30.0, 50.0),
    "normal": (15.0, 30.0),
    "low": (10.0, 20.0),
}


def _finite(value, default: float) -> float:
    """Coerce numeric input to a finite float, else return the default.

    Booleans are rejected (bool is a numbers.Real subclass) to avoid
    silently interpreting flags as measurements.
    """
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        v = float(value)
        if math.isfinite(v):
            return v
    return default


def _clamp_composite(score: int) -> int:
    return max(-3, min(3, int(score)))


def bucket_volatility(percentile: float,
                      high: float = VOL_PCT_HIGH,
                      low: float = VOL_PCT_LOW) -> int:
    """Volatility leg: high percentile is unfavorable (-1), low is favorable (+1).

    Boundary semantics match RegimeDetector: exactly `high` → -1,
    exactly `low` → +1. Non-finite → neutral 0.
    """
    p = _finite(percentile, 50.0)
    if p >= high:
        return -1
    if p <= low:
        return 1
    return 0


def bucket_trend(adx: float,
                 trend_fraction: float,
                 trending_threshold: float = ADX_TRENDING,
                 up_fraction: float = TREND_UP_FRACTION,
                 down_fraction: float = TREND_DOWN_FRACTION) -> int:
    """Trend leg from ADX strength and cross-asset trend direction.

    trend_fraction = fraction of the universe with SMA20 > SMA50
    (mirrors RegimeDetector.detect_trend_regime). Only a strong ADX
    (>= trending_threshold) carries directional information; below it —
    including the degenerate ADX 0.0 or non-finite values — the leg is
    neutral, never "missing".
    """
    a = _finite(adx, 0.0)
    f = _finite(trend_fraction, 0.5)
    if a >= trending_threshold:
        if f > up_fraction:
            return 1
        if f < down_fraction:
            return -1
    return 0


def bucket_correlation(percentile: float,
                       high: float = CORR_PCT_HIGH,
                       low: float = CORR_PCT_LOW) -> int:
    """Correlation leg: high percentile (crowded market) is unfavorable (-1),
    low percentile (diversification available) is favorable (+1)."""
    p = _finite(percentile, 50.0)
    if p >= high:
        return -1
    if p <= low:
        return 1
    return 0


@dataclass(frozen=True)
class CompositeRegime:
    """Three-leg regime classification and its composite score."""
    vol_leg: int
    trend_leg: int
    corr_leg: int
    composite: int  # sum of legs, in {-3, ..., +3}

    def cash_band(self, candidate: str = "A") -> Tuple[float, float]:
        return cash_band(candidate, self.composite)


def composite_score(volatility_percentile: float,
                    adx: float,
                    trend_fraction: float,
                    correlation_percentile: float) -> CompositeRegime:
    """Classify all three legs and return the composite regime."""
    v = bucket_volatility(volatility_percentile)
    t = bucket_trend(adx, trend_fraction)
    c = bucket_correlation(correlation_percentile)
    return CompositeRegime(vol_leg=v, trend_leg=t, corr_leg=c, composite=v + t + c)


def cash_band(candidate: str, composite: int) -> Tuple[float, float]:
    """Map a composite score to the (lower, upper) cash band for a candidate arm.

    Raises ValueError for unknown candidates. Out-of-range composites are
    clamped to [-3, 3] rather than raising — the sum of three legs is
    already bounded, so a wild value indicates caller misuse, not physics,
    and the clamp keeps replays deterministic.
    """
    table = _COMPOSITE_BANDS.get(candidate)
    if table is None:
        raise ValueError(
            f"Unknown candidate {candidate!r}; expected one of {sorted(_COMPOSITE_BANDS)}"
        )
    return table[_clamp_composite(composite)]


def control_band(volatility_regime: str) -> Tuple[float, float]:
    """Control arm: the current 1-D vol→band mapping. Raises ValueError for
    unknown regime labels (fail loud in replay, never silently default)."""
    key = str(volatility_regime).strip().lower()
    if key not in _CONTROL_BANDS:
        raise ValueError(
            f"Unknown volatility regime {volatility_regime!r}; "
            f"expected one of {sorted(_CONTROL_BANDS)}"
        )
    return _CONTROL_BANDS[key]


def apply_hysteresis(previous: Optional[int],
                     new: int,
                     min_shift: int = HYSTERESIS_MIN_SHIFT) -> int:
    """Return the band-holding composite under intra-week hysteresis.

    The composite must move at least `min_shift` levels from the previously
    applied value before the band shifts; smaller oscillations keep the
    previous band (anti flip-flop). `previous=None` means "no history yet"
    and passes `new` through. Both directions are symmetric.
    """
    n = _clamp_composite(_finite(new, 0))
    if previous is None:
        return n
    p = _clamp_composite(_finite(previous, 0))
    if abs(n - p) >= max(1, int(min_shift)):
        return n
    return p


def correlation_percentile(current_avg: float,
                           historical_avgs: Sequence[float]) -> float:
    """Percentile rank of today's average pairwise correlation within its
    own history (mirrors the vol-percentile computation in RegimeDetector).

    Empty or all-invalid history, and non-finite current values, degrade to
    50.0 (neutral) rather than propagating NaN.
    """
    cur = _finite(current_avg, math.nan)
    if not math.isfinite(cur):
        return 50.0
    hist = [
        float(v) for v in (historical_avgs or [])
        if isinstance(v, numbers.Real) and not isinstance(v, bool)
        and math.isfinite(float(v))
    ]
    if not hist:
        return 50.0
    return sum(1 for v in hist if v < cur) / len(hist) * 100.0

"""
Tests for the composite regime module (experiment H3, step 1).

Covers:
- Volatility leg bucket boundaries (25/75 percentiles)
- Trend leg: degenerate ADX (0.0 / NaN) → neutral, ADX thresholds,
  direction from cross-asset trend fraction
- Correlation leg bucket boundaries
- Composite scoring across legs
- Candidate cash-band tables (A/B/C × 7 composite levels) + control arm
- Intra-week hysteresis (min shift, both directions, first-call pass-through)
- Correlation percentile helper (empty history, NaN filtering)
- Non-finite input guards (NaN / inf → neutral, bool rejection)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from analysis.composite_regime import (
    CompositeRegime,
    apply_hysteresis,
    bucket_correlation,
    bucket_trend,
    bucket_volatility,
    cash_band,
    composite_score,
    control_band,
    correlation_percentile,
)

# ---------------------------------------------------------------------------
# Volatility leg
# ---------------------------------------------------------------------------


class TestBucketVolatility:
    def test_high_boundary_inclusive(self):
        assert bucket_volatility(75.0) == -1

    def test_just_below_high_is_neutral(self):
        assert bucket_volatility(74.99) == 0

    def test_low_boundary_inclusive(self):
        assert bucket_volatility(25.0) == 1

    def test_just_above_low_is_neutral(self):
        assert bucket_volatility(25.01) == 0

    def test_mid_is_neutral(self):
        assert bucket_volatility(50.0) == 0

    def test_custom_thresholds(self):
        assert bucket_volatility(80.0, high=80.0, low=20.0) == -1
        assert bucket_volatility(20.0, high=80.0, low=20.0) == 1

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None, "75"])
    def test_non_finite_is_neutral(self, bad):
        assert bucket_volatility(bad) == 0

    def test_bool_rejected(self):
        assert bucket_volatility(True) == 0


# ---------------------------------------------------------------------------
# Trend leg (incl. degenerate ADX)
# ---------------------------------------------------------------------------


class TestBucketTrend:
    def test_degenerate_adx_is_neutral_regardless_of_direction(self):
        # ADX 0.0 (observed in production, e.g. 2026-09-29) must never
        # carry directional information.
        assert bucket_trend(0.0, 0.9) == 0
        assert bucket_trend(0.0, 0.1) == 0

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None])
    def test_non_finite_adx_is_neutral(self, bad):
        assert bucket_trend(bad, 0.9) == 0
        assert bucket_trend(bad, 0.1) == 0

    def test_strong_adx_majority_up_is_favorable(self):
        assert bucket_trend(30.0, 0.7) == 1

    def test_strong_adx_majority_down_is_unfavorable(self):
        assert bucket_trend(30.0, 0.3) == -1

    def test_strong_adx_mixed_direction_is_neutral(self):
        assert bucket_trend(30.0, 0.5) == 0

    def test_up_boundary_is_strict(self):
        # fraction exactly 0.6 is not > 0.6 → neutral
        assert bucket_trend(30.0, 0.6) == 0

    def test_down_boundary_is_strict(self):
        # fraction exactly 0.4 is not < 0.4 → neutral
        assert bucket_trend(30.0, 0.4) == 0

    def test_moderate_adx_is_neutral(self):
        # Between mean-reverting and trending strength: no information.
        assert bucket_trend(22.5, 0.9) == 0
        assert bucket_trend(22.5, 0.1) == 0

    def test_adx_threshold_boundary_inclusive(self):
        assert bucket_trend(25.0, 0.9) == 1
        assert bucket_trend(25.0, 0.1) == -1


# ---------------------------------------------------------------------------
# Correlation leg
# ---------------------------------------------------------------------------


class TestBucketCorrelation:
    def test_high_boundary_inclusive(self):
        assert bucket_correlation(75.0) == -1

    def test_low_boundary_inclusive(self):
        assert bucket_correlation(25.0) == 1

    def test_mid_is_neutral(self):
        assert bucket_correlation(50.0) == 0

    @pytest.mark.parametrize("bad", [math.nan, math.inf, None])
    def test_non_finite_is_neutral(self, bad):
        assert bucket_correlation(bad) == 0


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


class TestCompositeScore:
    def test_all_favorable_is_plus_three(self):
        r = composite_score(10.0, 30.0, 0.7, 10.0)
        assert (r.vol_leg, r.trend_leg, r.corr_leg) == (1, 1, 1)
        assert r.composite == 3

    def test_all_unfavorable_is_minus_three(self):
        r = composite_score(90.0, 30.0, 0.2, 90.0)
        assert (r.vol_leg, r.trend_leg, r.corr_leg) == (-1, -1, -1)
        assert r.composite == -3

    def test_all_neutral_is_zero(self):
        r = composite_score(50.0, 0.0, 0.5, 50.0)
        assert r.composite == 0

    def test_degenerate_adx_reduces_to_two_legs(self):
        # Today-like state: vol normal, ADX 0.0, corr normal → composite 0,
        # but the score is carried by exactly the two informative legs.
        r = composite_score(10.0, 0.0, 0.9, 50.0)
        assert (r.vol_leg, r.trend_leg, r.corr_leg) == (1, 0, 0)
        assert r.composite == 1

    def test_mixed_signs_sum(self):
        r = composite_score(90.0, 30.0, 0.7, 10.0)
        assert (r.vol_leg, r.trend_leg, r.corr_leg) == (-1, 1, 1)
        assert r.composite == 1


# ---------------------------------------------------------------------------
# Cash-band tables
# ---------------------------------------------------------------------------


class TestCashBand:
    @pytest.mark.parametrize("candidate,composite,expected", [
        ("A", 3, (8.0, 18.0)), ("A", 0, (15.0, 30.0)), ("A", -3, (30.0, 55.0)),
        ("B", 3, (5.0, 15.0)), ("B", 1, (12.0, 26.0)), ("B", -2, (28.0, 50.0)),
        ("C", 3, (2.0, 10.0)), ("C", -1, (22.0, 45.0)), ("C", -3, (40.0, 65.0)),
    ])
    def test_table_values(self, candidate, composite, expected):
        assert cash_band(candidate, composite) == expected

    def test_all_candidates_agree_at_zero(self):
        # Composite 0 is the control-equivalent anchor for every candidate.
        assert cash_band("A", 0) == cash_band("B", 0) == cash_band("C", 0) == (15.0, 30.0)

    def test_unknown_candidate_raises(self):
        with pytest.raises(ValueError):
            cash_band("Z", 0)

    def test_composite_clamped_not_raised(self):
        assert cash_band("A", 5) == cash_band("A", 3)
        assert cash_band("A", -99) == cash_band("A", -3)

    def test_control_arm_mapping(self):
        assert control_band("high") == (30.0, 50.0)
        assert control_band("normal") == (15.0, 30.0)
        assert control_band("low") == (10.0, 20.0)

    def test_control_arm_case_insensitive(self):
        assert control_band("HIGH") == control_band("high")

    def test_control_arm_unknown_raises(self):
        with pytest.raises(ValueError):
            control_band("sideways")

    def test_composite_regime_dataclass_band_delegation(self):
        r = CompositeRegime(vol_leg=1, trend_leg=1, corr_leg=1, composite=3)
        assert r.cash_band("B") == (5.0, 15.0)


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------


class TestHysteresis:
    def test_first_call_passes_through(self):
        assert apply_hysteresis(None, 2) == 2

    def test_small_shift_keeps_previous(self):
        assert apply_hysteresis(0, 1) == 0
        assert apply_hysteresis(0, -1) == 0

    def test_min_shift_moves(self):
        assert apply_hysteresis(0, 2) == 2
        assert apply_hysteresis(0, -2) == -2

    def test_larger_shift_moves(self):
        assert apply_hysteresis(-1, 3) == 3

    def test_symmetric_downshift(self):
        assert apply_hysteresis(1, -1) == -1   # |−1 − 1| = 2 → moves
        assert apply_hysteresis(1, 0) == 1     # oscillation (Δ1) → holds

    def test_custom_min_shift(self):
        assert apply_hysteresis(0, 1, min_shift=1) == 1
        assert apply_hysteresis(0, 1, min_shift=3) == 0

    def test_non_finite_new_degrades_to_zero(self):
        # NaN new value → treated as 0; |0 − prev| decides, no exception.
        assert apply_hysteresis(0, math.nan) == 0

    def test_out_of_range_clamped(self):
        assert apply_hysteresis(None, 7) == 3
        assert apply_hysteresis(None, -7) == -3


# ---------------------------------------------------------------------------
# Correlation percentile helper
# ---------------------------------------------------------------------------


class TestCorrelationPercentile:
    def test_empty_history_is_neutral(self):
        assert correlation_percentile(0.5, []) == 50.0
        assert correlation_percentile(0.5, None) == 50.0

    def test_non_finite_current_is_neutral(self):
        assert correlation_percentile(math.nan, [0.1, 0.2]) == 50.0
        assert correlation_percentile(None, [0.1, 0.2]) == 50.0

    def test_history_nan_filtered(self):
        # Mirrors the vol percentile: invalid entries excluded, not fatal.
        # Valid history = [0.1, 0.2, 0.6]; 2 of 3 are < 0.5 → 66.67th pct.
        p = correlation_percentile(0.5, [0.1, math.nan, 0.2, 0.6])
        assert p == pytest.approx(200.0 / 3.0)

    def test_rank_computation(self):
        hist = [0.10, 0.20, 0.30, 0.40]
        assert correlation_percentile(0.35, hist) == pytest.approx(75.0)
        assert correlation_percentile(0.05, hist) == pytest.approx(0.0)
        assert correlation_percentile(0.50, hist) == pytest.approx(100.0)

    def test_bool_entries_rejected(self):
        assert correlation_percentile(0.5, [True, 0.1, 0.2]) == pytest.approx(
            sum(1 for v in [0.1, 0.2] if v < 0.5) / 2 * 100.0
        )

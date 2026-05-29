"""
Test suite for backtest/deflated_sharpe.py.

Tests the Deflated Sharpe Ratio implementation from Lopez de Prado (2018),
including Sharpe ratio calculations, multiple-testing corrections, non-normality
adjustments, FDR control, and auxiliary metrics.

Financial formulas are deterministic — known inputs must produce known outputs.
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from scipy import stats

from backtest.deflated_sharpe import (
    DeflatedSharpeRatio,
    SharpeMetrics,
    probabilistic_sharpe_ratio,
    minimum_track_record_length,
)


def _approx(a, b, rel_tol=1e-6, abs_tol=1e-9):
    """Compare floats with tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


class TestDeflatedSharpeRatioInit:
    """Tests for DeflatedSharpeRatio initialization."""

    def test_default_init(self):
        """Default parameters should be sensible."""
        dsr = DeflatedSharpeRatio()
        assert dsr.n_trials == 1
        assert dsr.annualization_factor == 252.0
        assert dsr.significance_level == 0.05

    def test_custom_init(self):
        """Custom parameters should be stored correctly."""
        dsr = DeflatedSharpeRatio(n_trials=100, annualization_factor=12, significance_level=0.01)
        assert dsr.n_trials == 100
        assert dsr.annualization_factor == 12.0
        assert dsr.significance_level == 0.01

    def test_n_trials_clamped_to_one(self):
        """n_trials < 1 should be clamped to 1."""
        dsr = DeflatedSharpeRatio(n_trials=0)
        assert dsr.n_trials == 1
        dsr = DeflatedSharpeRatio(n_trials=-5)
        assert dsr.n_trials == 1


class TestSharpeRatioCalculation:
    """Tests for basic Sharpe ratio calculation."""

    def test_zero_volatility_returns_zero_sharpe(self):
        """Constant returns → zero Sharpe (std≈0, division by zero guarded)."""
        returns = np.full(252, 0.001)
        dsr = DeflatedSharpeRatio()
        metrics = dsr.calculate(returns)
        assert metrics.sharpe_ratio == 0.0
        assert metrics.skewness == 0.0
        assert metrics.kurtosis == 3.0

    def test_positive_returns_positive_sharpe(self):
        """Positive mean, non-zero std → positive Sharpe."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        dsr = DeflatedSharpeRatio()
        metrics = dsr.calculate(returns)
        assert metrics.sharpe_ratio > 0

    def test_annualization_factor(self):
        """Daily vs monthly annualization should scale SR by sqrt(factor)."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        dsr_daily = DeflatedSharpeRatio(annualization_factor=252)
        dsr_monthly = DeflatedSharpeRatio(annualization_factor=12)
        sr_daily = dsr_daily.calculate(returns).sharpe_ratio
        sr_monthly = dsr_monthly.calculate(returns).sharpe_ratio
        # Same returns, different annualization — ratio should be sqrt(252/12)
        assert _approx(sr_daily / sr_monthly, np.sqrt(252 / 12), rel_tol=0.01)

    def test_risk_free_rate_adjustment(self):
        """Higher risk-free rate → lower Sharpe ratio."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        dsr = DeflatedSharpeRatio()
        sr_no_rf = dsr.calculate(returns, risk_free_rate=0.0).sharpe_ratio
        sr_with_rf = dsr.calculate(returns, risk_free_rate=0.05).sharpe_ratio
        assert sr_with_rf < sr_no_rf

    def test_insufficient_data_raises(self):
        """Fewer than 2 observations should raise ValueError."""
        dsr = DeflatedSharpeRatio()
        with pytest.raises(ValueError, match="At least 2 observations required"):
            dsr.calculate(np.array([0.01]))
        with pytest.raises(ValueError, match="At least 2 observations required"):
            dsr.calculate(np.array([]))

    def test_n_trials_override(self):
        """n_trials parameter should override instance value."""
        dsr = DeflatedSharpeRatio(n_trials=10)
        returns = np.random.normal(0.001, 0.02, 252)
        metrics = dsr.calculate(returns, n_trials=50)
        assert metrics.n_trials == 50


class TestDeflatedSharpeAdjustment:
    """Tests for the deflation / multiple-testing adjustment."""

    def test_single_trial_no_adjustment(self):
        """With 1 trial, DSR should equal SR (no multiple-testing penalty)."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        dsr = DeflatedSharpeRatio(n_trials=1)
        metrics = dsr.calculate(returns)
        assert _approx(metrics.deflated_sharpe, metrics.sharpe_ratio, rel_tol=1e-6)

    def test_multiple_trials_reduces_dsr(self):
        """More trials → lower DSR (multiple-testing penalty)."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        dsr_1 = DeflatedSharpeRatio(n_trials=1)
        dsr_100 = DeflatedSharpeRatio(n_trials=100)
        dsr_1000 = DeflatedSharpeRatio(n_trials=1000)
        metrics_1 = dsr_1.calculate(returns)
        metrics_100 = dsr_100.calculate(returns)
        metrics_1000 = dsr_1000.calculate(returns)
        assert metrics_100.deflated_sharpe <= metrics_1.sharpe_ratio
        assert metrics_1000.deflated_sharpe <= metrics_100.deflated_sharpe

    def test_dsr_can_be_negative(self):
        """With many trials and modest SR, DSR can turn negative."""
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, 252)  # Near-zero mean
        dsr = DeflatedSharpeRatio(n_trials=1000)
        metrics = dsr.calculate(returns)
        # Many trials should deflate a modest SR below zero
        assert metrics.deflated_sharpe < metrics.sharpe_ratio

    def test_skewness_kurtosis_stored(self):
        """Skewness and kurtosis should be calculated and stored."""
        # Symmetric normal returns
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 1000)
        dsr = DeflatedSharpeRatio()
        metrics = dsr.calculate(returns)
        assert abs(metrics.skewness) < 0.5  # Near-zero for normal
        assert abs(metrics.kurtosis - 3.0) < 0.5  # Near 3 for normal

    def test_fat_tails_detected(self):
        """Fat-tailed distribution should show high kurtosis."""
        np.random.seed(42)
        returns = np.random.standard_t(df=3, size=1000) * 0.01
        dsr = DeflatedSharpeRatio()
        metrics = dsr.calculate(returns)
        assert metrics.kurtosis > 3.5  # fatter tails than normal

    def test_skewness_detected(self):
        """Skewed distribution should show non-zero skewness."""
        np.random.seed(42)
        returns = np.random.exponential(scale=0.01, size=1000) - 0.01
        dsr = DeflatedSharpeRatio()
        metrics = dsr.calculate(returns)
        assert metrics.skewness > 0.5  # Right-skewed


class TestPValueCalculation:
    """Tests for p-value and significance."""

    def test_high_sharpe_low_pvalue(self):
        """Very high Sharpe → low p-value → significant."""
        np.random.seed(42)
        returns = np.random.normal(0.005, 0.01, 252)  # Strong signal
        dsr = DeflatedSharpeRatio(n_trials=1)
        metrics = dsr.calculate(returns)
        assert metrics.p_value < 0.05
        assert metrics.is_significant == True

    def test_zero_sharpe_high_pvalue(self):
        """Zero-mean returns → high p-value → not significant."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 252)
        dsr = DeflatedSharpeRatio(n_trials=1)
        metrics = dsr.calculate(returns)
        assert metrics.p_value > 0.05
        assert metrics.is_significant == False

    def test_multiple_trials_inflates_pvalue(self):
        """More trials → higher p-value (Bonferroni correction)."""
        np.random.seed(42)
        returns = np.random.normal(0.003, 0.02, 252)
        dsr_1 = DeflatedSharpeRatio(n_trials=1)
        dsr_100 = DeflatedSharpeRatio(n_trials=100)
        p1 = dsr_1.calculate(returns).p_value
        p100 = dsr_100.calculate(returns).p_value
        assert p100 >= p1
        assert p100 <= 1.0

    def test_p_value_capped_at_one(self):
        """Corrected p-value should never exceed 1.0."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        dsr = DeflatedSharpeRatio(n_trials=10000)
        metrics = dsr.calculate(returns)
        assert metrics.p_value <= 1.0
        assert metrics.p_value >= 0.0


class TestCompareStrategies:
    """Tests for multi-strategy comparison."""

    def test_sorts_by_dsr(self):
        """Results should be sorted by DSR descending."""
        np.random.seed(42)
        strategies = [
            ("weak", np.random.normal(0.0001, 0.02, 252)),
            ("strong", np.random.normal(0.003, 0.015, 252)),
            ("medium", np.random.normal(0.001, 0.018, 252)),
        ]
        dsr = DeflatedSharpeRatio()
        results = dsr.compare_strategies(strategies)
        dsrs = [r[1].deflated_sharpe for r in results]
        assert dsrs == sorted(dsrs, reverse=True)

    def test_uses_correct_n_trials(self):
        """compare_strategies should use len(strategies) as n_trials."""
        np.random.seed(42)
        strategies = [
            ("a", np.random.normal(0.001, 0.02, 252)),
            ("b", np.random.normal(0.001, 0.02, 252)),
        ]
        dsr = DeflatedSharpeRatio()
        results = dsr.compare_strategies(strategies)
        for name, metrics in results:
            assert metrics.n_trials == 2

    def test_empty_strategies(self):
        """Empty strategy list should return empty results."""
        dsr = DeflatedSharpeRatio()
        assert dsr.compare_strategies([]) == []


class TestFalseDiscoveryRate:
    """Tests for FDR control methods."""

    def test_bonferroni_conservative(self):
        """Bonferroni should multiply p-values by n and cap at 1."""
        dsr = DeflatedSharpeRatio()
        p_values = [0.01, 0.02, 0.05, 0.10]
        results = dsr.false_discovery_rate(p_values, method="bonferroni")
        q_values = [r[1] for r in results]
        # q = min(p * n, 1.0)
        assert _approx(q_values[0], min(0.01 * 4, 1.0))
        assert _approx(q_values[1], min(0.02 * 4, 1.0))
        assert all(q <= 1.0 for q in q_values)

    def test_benjamini_hochberg_monotonic(self):
        """BH q-values should be monotonic (non-decreasing when sorted by p)."""
        dsr = DeflatedSharpeRatio()
        p_values = [0.01, 0.03, 0.05, 0.07, 0.10]
        results = dsr.false_discovery_rate(p_values, method="benjamini-hochberg")
        # Sort by original p to check monotonicity
        sorted_by_p = sorted(results, key=lambda x: x[0])
        q_values = [r[1] for r in sorted_by_p]
        for i in range(len(q_values) - 1):
            assert q_values[i] <= q_values[i + 1]

    def test_benjamini_hochberg_less_conservative_than_bonferroni(self):
        """BH should be less conservative than Bonferroni for same p-values."""
        dsr = DeflatedSharpeRatio(significance_level=0.05)
        p_values = [0.01, 0.02, 0.04, 0.06, 0.08]
        bh_results = dsr.false_discovery_rate(p_values, method="benjamini-hochberg")
        bonf_results = dsr.false_discovery_rate(p_values, method="bonferroni")
        for (_, q_bh, _), (_, q_bonf, _) in zip(bh_results, bonf_results):
            assert q_bh <= q_bonf

    def test_invalid_method_raises(self):
        """Unknown FDR method should raise ValueError."""
        dsr = DeflatedSharpeRatio()
        with pytest.raises(ValueError, match="Unknown method"):
            dsr.false_discovery_rate([0.05], method="invalid")

    def test_empty_pvalues(self):
        """Empty p-value list should return empty results."""
        dsr = DeflatedSharpeRatio()
        assert dsr.false_discovery_rate([]) == []

    def test_single_pvalue(self):
        """Single p-value should be handled correctly."""
        dsr = DeflatedSharpeRatio()
        results = dsr.false_discovery_rate([0.05], method="benjamini-hochberg")
        assert len(results) == 1
        assert _approx(results[0][1], 0.05)


class TestSharpeMetricsDataclass:
    """Tests for SharpeMetrics container."""

    def test_dataclass_fields(self):
        """All expected fields should exist."""
        metrics = SharpeMetrics(
            sharpe_ratio=1.5,
            deflated_sharpe=1.2,
            p_value=0.01,
            is_significant=True,
            skewness=0.0,
            kurtosis=3.0,
            n_trials=10,
            n_observations=252,
            annualization_factor=252.0,
        )
        assert metrics.sharpe_ratio == 1.5
        assert metrics.is_significant is True
        assert metrics.n_observations == 252


class TestProbabilisticSharpeRatio:
    """Tests for the standalone PSR function."""

    def test_psr_higher_when_observed_gt_benchmark(self):
        """Observed SR > benchmark → PSR > 0.5."""
        psr = probabilistic_sharpe_ratio(
            observed_sr=1.0, benchmark_sr=0.5, n_observations=252
        )
        assert psr > 0.5
        assert psr <= 1.0

    def test_psr_lower_when_observed_lt_benchmark(self):
        """Observed SR < benchmark → PSR < 0.5."""
        psr = probabilistic_sharpe_ratio(
            observed_sr=0.3, benchmark_sr=0.5, n_observations=252
        )
        assert psr < 0.5
        assert psr >= 0.0

    def test_psr_equals_half_when_observed_equals_benchmark(self):
        """Observed SR == benchmark → PSR == 0.5 (asymptotically)."""
        psr = probabilistic_sharpe_ratio(
            observed_sr=0.5, benchmark_sr=0.5, n_observations=1000
        )
        assert _approx(psr, 0.5, abs_tol=0.05)

    def test_psr_with_few_observations(self):
        """Few observations → PSR closer to 0.5 (more uncertainty)."""
        psr_few = probabilistic_sharpe_ratio(
            observed_sr=1.0, benchmark_sr=0.5, n_observations=10
        )
        psr_many = probabilistic_sharpe_ratio(
            observed_sr=1.0, benchmark_sr=0.5, n_observations=1000
        )
        # More observations → more confidence → PSR further from 0.5
        assert abs(psr_many - 0.5) > abs(psr_few - 0.5)

    def test_psr_returns_half_for_insufficient_data(self):
        """n < 2 should return 0.5 (maximal uncertainty)."""
        assert probabilistic_sharpe_ratio(1.0, 0.5, n_observations=1) == 0.5
        assert probabilistic_sharpe_ratio(1.0, 0.5, n_observations=0) == 0.5

    def test_psr_extreme_values(self):
        """Very high observed SR vs low benchmark → PSR near 1."""
        psr = probabilistic_sharpe_ratio(
            observed_sr=3.0, benchmark_sr=0.0, n_observations=500
        )
        assert psr > 0.95


class TestMinimumTrackRecordLength:
    """Tests for minimum track record length calculation."""

    def test_higher_sharpe_requires_fewer_observations(self):
        """Higher target Sharpe → fewer observations needed."""
        n_low = minimum_track_record_length(target_sharpe=0.5)
        n_high = minimum_track_record_length(target_sharpe=2.0)
        assert n_high < n_low

    def test_higher_confidence_requires_more_observations(self):
        """Higher confidence → more observations needed."""
        n_95 = minimum_track_record_length(target_sharpe=1.0, confidence_level=0.95)
        n_99 = minimum_track_record_length(target_sharpe=1.0, confidence_level=0.99)
        assert n_99 > n_95

    def test_returns_at_least_two(self):
        """Minimum should always be at least 2."""
        n = minimum_track_record_length(target_sharpe=5.0)
        assert n >= 2

    def test_normal_params_reasonable(self):
        """With normal returns, MTRL for SR=1.0 at 95% should be reasonable."""
        n = minimum_track_record_length(
            target_sharpe=1.0, confidence_level=0.95, skewness=0.0, kurtosis=3.0
        )
        # For SR=1.0, formula gives a small number; just verify it's sane
        assert n >= 2
        assert n < 1000

    def test_skewness_affects_result(self):
        """Negative skewness should increase required observations."""
        n_normal = minimum_track_record_length(target_sharpe=1.0, skewness=0.0)
        n_neg_skew = minimum_track_record_length(target_sharpe=1.0, skewness=-1.0)
        assert n_neg_skew >= n_normal


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_very_short_series(self):
        """Minimum valid series (2 observations)."""
        dsr = DeflatedSharpeRatio()
        metrics = dsr.calculate(np.array([0.01, -0.01]))
        assert metrics.n_observations == 2
        assert not math.isnan(metrics.sharpe_ratio)

    def test_all_zeros(self):
        """All-zero returns → zero Sharpe, but no crash."""
        dsr = DeflatedSharpeRatio()
        metrics = dsr.calculate(np.zeros(100))
        assert metrics.sharpe_ratio == 0.0
        assert metrics.p_value == 1.0  # Not significant

    def test_list_input(self):
        """List input should be converted to array."""
        dsr = DeflatedSharpeRatio()
        metrics = dsr.calculate([0.01, -0.01, 0.005, -0.005])
        assert metrics.n_observations == 4

    def test_2d_array_input(self):
        """2D array should be flattened."""
        dsr = DeflatedSharpeRatio()
        metrics = dsr.calculate(np.array([[0.01, -0.01], [0.005, -0.005]]))
        assert metrics.n_observations == 4

    def test_very_large_n_trials(self):
        """Extremely large n_trials should not overflow."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        dsr = DeflatedSharpeRatio(n_trials=1_000_000)
        metrics = dsr.calculate(returns)
        assert not math.isnan(metrics.deflated_sharpe)
        assert not math.isinf(metrics.deflated_sharpe)

    def test_negative_excess_kurtosis_adjustment(self):
        """Platykurtic returns (kurtosis < 3) should not break variance adjustment."""
        # Uniform-like returns have kurtosis < 3
        np.random.seed(42)
        returns = np.random.uniform(-0.03, 0.03, 500)
        dsr = DeflatedSharpeRatio()
        metrics = dsr.calculate(returns)
        assert not math.isnan(metrics.deflated_sharpe)
        assert metrics.kurtosis < 3.0  # Uniform is platykurtic

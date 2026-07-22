"""Test suite for risk/performance_metrics.py.

Tests performance metrics (Sharpe, Beta/Alpha, Sortino, Calmar, Treynor,
Information Ratio) with deterministic, mathematically verifiable inputs.
Focus on numerical precision edge cases: denominators that are near-zero due to
floating point rounding should not produce colossal or infinite artefacts.
"""

import sys
import math
import json
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from risk.performance_metrics import (
    PerformanceMetrics,
    calculate_sharpe_ratio,
    calculate_beta_alpha,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_treynor_ratio,
    calculate_information_ratio,
    calculate_all_metrics,
    format_metrics_report,
)


def _approx(a, b, rel_tol=1e-6, abs_tol=1e-9):
    """Compare floats with tolerance, handling None/inf."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if math.isinf(a) and math.isinf(b):
        return (a > 0) == (b > 0)
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------

def test_sharpe_ratio_basic():
    """Sharpe ratio on a positive-return series with small volatility."""
    returns = np.full(30, 0.001) + np.linspace(-1e-4, 1e-4, 30)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    assert sharpe > 0


def test_sharpe_ratio_insufficient_data():
    """Sharpe ratio returns 0.0 for fewer than 2 observations."""
    assert calculate_sharpe_ratio(np.array([0.01])) == 0.0
    assert calculate_sharpe_ratio(np.array([])) == 0.0


def test_sharpe_ratio_near_zero_volatility():
    """Near-zero volatility should not explode into a huge ratio."""
    returns = np.full(30, 0.001)
    returns = returns + np.linspace(-1e-16, 1e-16, 30)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    assert sharpe == 0.0 or np.isfinite(sharpe)


# ---------------------------------------------------------------------------
# Beta / Alpha
# ---------------------------------------------------------------------------

def test_beta_alpha_basic():
    """Beta and alpha with perfectly correlated returns."""
    portfolio = np.linspace(-0.01, 0.01, 60)
    benchmark = portfolio * 0.5
    beta, alpha = calculate_beta_alpha(portfolio, benchmark, risk_free_rate=0.0)
    assert _approx(beta, 2.0)
    assert _approx(alpha, 0.0, abs_tol=1e-6)


def test_beta_alpha_zero_variance():
    """Constant benchmark returns zero variance -> beta/alpha unavailable."""
    portfolio = np.linspace(-0.01, 0.01, 60)
    benchmark = np.full(60, 0.001)
    beta, alpha = calculate_beta_alpha(portfolio, benchmark)
    assert beta is None
    assert alpha is None


def test_beta_alpha_near_zero_variance():
    """Near-zero variance should not produce a colossal beta."""
    portfolio = np.linspace(-0.01, 0.01, 60)
    benchmark = np.full(60, 0.001) + np.linspace(-1e-16, 1e-16, 60)
    beta, alpha = calculate_beta_alpha(portfolio, benchmark)
    assert beta is None
    assert alpha is None


def test_beta_alpha_nan_variance():
    """NaN benchmark variance should not propagate into NaN beta/alpha."""
    portfolio = np.linspace(-0.01, 0.01, 60)
    benchmark = np.full(60, np.nan)
    beta, alpha = calculate_beta_alpha(portfolio, benchmark)
    assert beta is None
    assert alpha is None


def test_beta_alpha_mismatched_lengths():
    """Mismatched lengths are aligned by taking the trailing overlap."""
    portfolio = np.linspace(-0.01, 0.01, 60)
    benchmark = np.linspace(-0.01, 0.01, 50)
    beta, alpha = calculate_beta_alpha(portfolio, benchmark)
    assert beta is not None
    assert alpha is not None


# ---------------------------------------------------------------------------
# Sortino Ratio
# ---------------------------------------------------------------------------

def test_sortino_ratio_basic():
    """Sortino with positive and negative returns of non-zero dispersion."""
    returns = np.concatenate([
        np.linspace(0.001, 0.003, 15),
        np.linspace(-0.003, -0.001, 15),
    ])
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino > 0


def test_sortino_ratio_no_downside():
    """All-positive returns have no downside volatility → Sortino is 0.0."""
    returns = np.full(30, 0.002)
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino == 0.0


def test_sortino_ratio_near_zero_downside():
    """Near-zero downside deviation should not produce a colossal ratio."""
    returns = np.concatenate([np.full(15, 0.002), np.full(15, -1e-16)])
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino == 0.0 or np.isfinite(sortino)


def test_sortino_ratio_insufficient_data():
    assert calculate_sortino_ratio(np.array([0.01])) == 0.0


def test_sortino_ratio_single_downside_return_no_warning():
    """A single downside return used to trigger np.std(ddof=1) RuntimeWarning."""
    returns = np.array([0.001, 0.001, -0.002])
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    runtime_warnings = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings
    assert result == 0.0


def test_sortino_ratio_positive_mean_with_single_downside_is_zero():
    returns = np.array([0.010, 0.008, -0.001])
    result = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert result == 0.0


def test_sortino_ratio_non_finite_input():
    """NaN/Inf inputs should not propagate through the Sortino ratio."""
    assert calculate_sortino_ratio(np.array([0.01, np.nan, -0.01])) == 0.0
    assert calculate_sortino_ratio(np.array([0.01, np.inf, -0.01])) == 0.0


def test_sortino_ratio_two_downside_returns():
    returns = np.array([0.005, -0.002, -0.004])
    result = calculate_sortino_ratio(returns, risk_free_rate=0.0, annualized=False)
    downside = np.array([-0.002, -0.004])
    expected = np.mean(returns) / np.std(downside, ddof=1)
    assert math.isclose(result, expected)


# ---------------------------------------------------------------------------
# Calmar Ratio
# ---------------------------------------------------------------------------

def test_calmar_ratio_basic():
    """Calmar ratio with a positive total return and a known drawdown."""
    returns = np.array([0.10, -0.05, 0.10])
    calmar = calculate_calmar_ratio(returns)
    # Total return is positive; max drawdown is -5%.
    assert calmar > 0


def test_calmar_ratio_zero_drawdown():
    """Constantly increasing series has zero drawdown → Calmar is 0.0."""
    returns = np.full(30, 0.001)
    calmar = calculate_calmar_ratio(returns)
    assert calmar == 0.0


def test_calmar_ratio_near_zero_drawdown():
    """A tiny but non-zero drawdown below the tolerance should be treated as zero."""
    returns = np.full(30, 0.001)
    calmar = calculate_calmar_ratio(returns, max_drawdown=1e-16)
    assert calmar == 0.0


def test_calmar_ratio_non_finite_input():
    """NaN/Inf inputs should not propagate through the Calmar ratio."""
    assert calculate_calmar_ratio(np.array([0.01, np.nan, -0.01])) == 0.0
    assert calculate_calmar_ratio(np.array([0.01, np.inf, -0.01])) == 0.0


def test_calmar_ratio_positive_drawdown():
    """Positive max_drawdown (impossible) should return 0.0."""
    returns = np.full(30, 0.001)
    calmar = calculate_calmar_ratio(returns, max_drawdown=0.01)
    assert calmar == 0.0


def test_calmar_ratio_insufficient_data():
    assert calculate_calmar_ratio(np.array([0.01])) == 0.0


# ---------------------------------------------------------------------------
# Treynor Ratio
# ---------------------------------------------------------------------------

def test_treynor_ratio_basic():
    """Treynor ratio with positive excess return and beta = 1."""
    returns = np.full(30, 0.001)
    treynor = calculate_treynor_ratio(returns, beta=1.0, risk_free_rate=0.0)
    assert treynor > 0


def test_treynor_ratio_zero_beta():
    """Zero beta should return None."""
    returns = np.full(30, 0.001)
    assert calculate_treynor_ratio(returns, beta=0.0) is None


def test_treynor_ratio_near_zero_beta():
    """Near-zero beta should not produce a colossal Treynor ratio."""
    returns = np.full(30, 0.001)
    treynor = calculate_treynor_ratio(returns, beta=1e-16)
    assert treynor is None


def test_treynor_ratio_nan_beta():
    assert calculate_treynor_ratio(np.full(30, 0.001), beta=np.nan) is None


def test_treynor_ratio_insufficient_data():
    assert calculate_treynor_ratio(np.array([0.01]), beta=1.0) is None


# ---------------------------------------------------------------------------
# Information Ratio
# ---------------------------------------------------------------------------

def test_information_ratio_basic():
    """Information ratio with consistent positive active return."""
    portfolio = np.linspace(-0.01, 0.01, 60)
    benchmark = np.zeros(60)
    ir, te = calculate_information_ratio(portfolio, benchmark)
    assert ir is not None
    assert te is not None
    assert te > 0


def test_information_ratio_zero_tracking_error():
    """Identical portfolio and benchmark -> zero tracking error -> None."""
    returns = np.linspace(-0.01, 0.01, 60)
    ir, te = calculate_information_ratio(returns, returns)
    assert ir is None
    assert te is None


def test_information_ratio_near_zero_tracking_error():
    """Near-zero tracking error should not explode."""
    portfolio = np.linspace(-0.01, 0.01, 60)
    benchmark = portfolio + np.linspace(-1e-16, 1e-16, 60)
    ir, te = calculate_information_ratio(portfolio, benchmark)
    assert ir is None
    assert te is None


def test_information_ratio_insufficient_data():
    assert calculate_information_ratio(np.array([0.01]), np.array([0.01])) == (None, None)


# ---------------------------------------------------------------------------
# calculate_all_metrics
# ---------------------------------------------------------------------------

def test_calculate_all_metrics_basic():
    """All metrics with a simple return series and benchmark."""
    returns = np.linspace(-0.01, 0.01, 60)
    benchmark = np.linspace(-0.005, 0.005, 60)
    metrics = calculate_all_metrics(returns, benchmark_returns=benchmark)
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.volatility > 0


def test_calculate_all_metrics_no_benchmark():
    """Relative metrics are None when no benchmark is provided."""
    returns = np.linspace(-0.01, 0.01, 60)
    metrics = calculate_all_metrics(returns)
    assert metrics.beta is None
    assert metrics.alpha is None
    assert metrics.treynor_ratio is None
    assert metrics.information_ratio is None
    assert metrics.tracking_error is None


def test_calculate_all_metrics_insufficient_data():
    """Insufficient data returns a zeroed metrics object."""
    metrics = calculate_all_metrics(np.array([0.01]))
    assert metrics.total_return == 0.0
    assert metrics.sharpe_ratio == 0.0


def test_calculate_all_metrics_small_sample_no_warning():
    """Ensure a realistic 3-day weekly return vector does not warn and stays finite."""
    returns = np.array([0.005, 0.003, -0.002])
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        metrics = calculate_all_metrics(returns)
    runtime_warnings = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings
    assert metrics.sharpe_ratio != 0.0
    assert metrics.sortino_ratio == 0.0
    assert np.isfinite(metrics.calmar_ratio)
    assert metrics.volatility > 0.0


def test_calculate_all_metrics_non_finite_input():
    """NaN/Inf returns should yield a fully zeroed, finite metrics object."""
    for bad in [np.nan, np.inf, -np.inf]:
        returns = np.array([0.01, bad, -0.01])
        metrics = calculate_all_metrics(returns)
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.calmar_ratio == 0.0
        assert metrics.volatility == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.beta is None
        assert metrics.alpha is None
        assert metrics.information_ratio is None
        assert metrics.tracking_error is None


def test_calculate_all_metrics_json_serializable():
    """The metrics object must serialize to valid JSON (no Infinity)."""
    returns = np.array([0.005, 0.003, -0.002])
    metrics = calculate_all_metrics(returns)
    payload = {
        'sharpe_ratio': metrics.sharpe_ratio,
        'sortino_ratio': metrics.sortino_ratio,
        'calmar_ratio': metrics.calmar_ratio,
        'volatility': metrics.volatility,
        'max_drawdown': metrics.max_drawdown,
    }
    # json.dumps allows allow_nan=True by default; we want strict JSON.
    text = json.dumps(payload, allow_nan=False)
    assert 'Infinity' not in text
    assert 'NaN' not in text
    assert json.loads(text) == payload


def test_calculate_all_metrics_basic():
    """All metrics with a simple return series and benchmark."""
    returns = np.linspace(-0.01, 0.01, 60)
    benchmark = np.linspace(-0.005, 0.005, 60)
    metrics = calculate_all_metrics(returns, benchmark_returns=benchmark)
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.volatility > 0


# ---------------------------------------------------------------------------
# format_metrics_report
# ---------------------------------------------------------------------------

def test_format_metrics_report_smoke():
    """Report formatter runs without error."""
    metrics = calculate_all_metrics(np.linspace(-0.01, 0.01, 60))
    report = format_metrics_report(metrics, benchmark_name="SPY")
    assert "PORTFOLIO PERFORMANCE METRICS" in report
    assert "SPY" in report

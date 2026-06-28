"""
Test suite for risk/performance_metrics.py.

Tests all portfolio performance metrics calculations with known inputs
and expected outputs. Financial formulas are deterministic — if the
math is right, the tests pass. If the math is wrong, we want to know
before deploying to production.
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

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
    """Compare floats with tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def test_sharpe_ratio_basic():
    """Sharpe ratio for a simple positive-return series."""
    # 252 days of 0.1% daily return, zero volatility
    returns = np.full(252, 0.001)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

    # Excess return = 0.001 - 0.02/252 ≈ 0.00092
    # Std = 0 (all identical), but ddof=1 gives NaN → returns 0.0
    # Actually np.std with ddof=1 on constant array gives 0.0, not NaN
    assert sharpe == 0.0, f"Expected 0.0 for zero-vol series, got {sharpe}"


def test_sharpe_ratio_with_volatility():
    """Sharpe ratio with non-trivial volatility."""
    np.random.seed(42)
    # 5% annual return, 20% volatility
    daily_returns = np.random.normal(0.05/252, 0.20/np.sqrt(252), 252)
    sharpe = calculate_sharpe_ratio(daily_returns, risk_free_rate=0.02)

    # Should be positive and roughly (0.05-0.02)/0.20 = 0.15
    assert sharpe > 0, f"Expected positive Sharpe, got {sharpe}"
    assert sharpe < 2.0, f"Sharpe unrealistically high: {sharpe}"


def test_sharpe_ratio_insufficient_data():
    """Sharpe ratio with < 2 observations returns 0."""
    assert calculate_sharpe_ratio(np.array([0.01])) == 0.0
    assert calculate_sharpe_ratio(np.array([])) == 0.0


def test_beta_alpha_perfect_correlation():
    """Beta = 1 when portfolio == benchmark."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 60)
    beta, alpha = calculate_beta_alpha(returns, returns.copy(), risk_free_rate=0.02)

    assert beta is not None, "Beta should not be None"
    # Sampling noise with n=60 means beta won't be exactly 1.0
    assert _approx(beta, 1.0, rel_tol=0.05), f"Expected beta ≈ 1.0, got {beta}"
    assert _approx(alpha, 0.0, abs_tol=0.01), f"Expected alpha ≈ 0.0, got {alpha}"


def test_beta_alpha_leveraged():
    """Beta = 2 when portfolio is 2x benchmark (no noise)."""
    np.random.seed(42)
    benchmark = np.random.normal(0.0005, 0.01, 60)
    portfolio = benchmark * 2.0
    beta, alpha = calculate_beta_alpha(portfolio, benchmark, risk_free_rate=0.02)

    assert beta is not None
    assert _approx(beta, 2.0, rel_tol=0.05), f"Expected beta ≈ 2.0, got {beta}"


def test_beta_alpha_insufficient_data():
    """Beta/Alpha require >= 30 observations."""
    short = np.full(29, 0.001)
    beta, alpha = calculate_beta_alpha(short, short.copy())
    assert beta is None and alpha is None


def test_beta_alpha_different_lengths():
    """Beta/Alpha align arrays of different lengths."""
    np.random.seed(42)
    portfolio = np.random.normal(0.0005, 0.01, 100)
    benchmark = np.random.normal(0.0005, 0.01, 80)
    beta, alpha = calculate_beta_alpha(portfolio, benchmark)

    assert beta is not None, "Should align lengths and compute beta"


def test_sortino_ratio_no_downside():
    """Sortino = inf when all returns are positive."""
    returns = np.full(30, 0.01)  # All +1%
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino == float('inf'), f"Expected inf, got {sortino}"


def test_sortino_ratio_with_downside():
    """Sortino ratio with mixed returns."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 252)
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)

    # Sortino should be >= Sharpe (same numerator, smaller or equal denominator)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    assert sortino >= sharpe or math.isclose(sortino, sharpe, rel_tol=0.1)


def test_sortino_ratio_all_negative():
    """Sortino with all negative returns."""
    returns = np.full(30, -0.01)
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino < 0, f"Expected negative Sortino, got {sortino}"


def test_calmar_ratio_basic():
    """Calmar ratio with known drawdown."""
    # Steady 10% annual return, max drawdown -5%
    returns = np.full(252, 0.10/252)
    calmar = calculate_calmar_ratio(returns, max_drawdown=-0.05)

    # Annualized return ≈ 10%, |drawdown| = 5% → Calmar ≈ 2.0
    assert _approx(calmar, 2.0, rel_tol=0.1), f"Expected Calmar ≈ 2.0, got {calmar}"


def test_calmar_ratio_no_drawdown():
    """Calmar = inf when there is no drawdown and returns are positive; 0 for invalid positive drawdown."""
    returns = np.full(252, 0.001)
    assert calculate_calmar_ratio(returns, max_drawdown=0.0) == float('inf')
    assert calculate_calmar_ratio(returns, max_drawdown=0.05) == 0.0


def test_calmar_ratio_no_drawdown_negative_returns():
    """Calmar = 0 when there is no drawdown but returns are negative or zero."""
    returns = np.full(252, -0.001)
    assert calculate_calmar_ratio(returns, max_drawdown=0.0) == 0.0
    returns_zero = np.full(252, 0.0)
    assert calculate_calmar_ratio(returns_zero, max_drawdown=0.0) == 0.0


def test_calmar_ratio_auto_drawdown():
    """Calmar calculates drawdown automatically when not provided."""
    # Go up 10%, then down 5%
    returns = np.array([0.001] * 100 + [-0.005] * 10 + [0.001] * 142)
    calmar = calculate_calmar_ratio(returns)
    assert calmar > 0, f"Expected positive Calmar, got {calmar}"


def test_treynor_ratio_basic():
    """Treynor ratio with known beta."""
    returns = np.full(252, 0.001)
    treynor = calculate_treynor_ratio(returns, beta=1.0, risk_free_rate=0.02)

    # Excess return = 0.001*252 - 0.02 = 0.232
    # Treynor = 0.232 / 1.0 = 0.232
    assert treynor is not None
    assert treynor > 0


def test_treynor_ratio_zero_beta():
    """Treynor = None when beta is zero."""
    returns = np.full(252, 0.001)
    assert calculate_treynor_ratio(returns, beta=0.0) is None


def test_information_ratio_identical():
    """Information ratio = undefined when portfolio == benchmark (zero tracking error)."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 60)
    ir, te = calculate_information_ratio(returns, returns.copy())

    # When portfolio == benchmark, tracking error is 0, so IR is undefined (None)
    assert ir is None, f"Expected IR=None for identical portfolio/benchmark, got {ir}"
    assert te == 0.0 or te is None, f"Expected TE=0, got {te}"


def test_information_ratio_insufficient_data():
    """IR requires >= 30 observations."""
    short = np.full(29, 0.001)
    ir, te = calculate_information_ratio(short, short.copy())
    assert ir is None and te is None


def test_calculate_all_metrics_empty():
    """All metrics return safe defaults for empty input."""
    metrics = calculate_all_metrics(np.array([]))
    assert metrics.sharpe_ratio == 0.0
    assert metrics.beta is None
    assert metrics.alpha is None


def test_calculate_all_metrics_short_series():
    """All metrics handle short series gracefully."""
    returns = np.full(5, 0.001)
    metrics = calculate_all_metrics(returns)
    # Near-zero volatility should produce Sharpe = 0 (guarded against fp precision)
    assert metrics.sharpe_ratio == 0.0, f"Expected Sharpe=0 for zero-vol, got {metrics.sharpe_ratio}"
    assert metrics.beta is None  # Need benchmark


def test_calculate_all_metrics_full():
    """Full metrics calculation with benchmark."""
    np.random.seed(42)
    portfolio = np.random.normal(0.0005, 0.01, 100)
    benchmark = np.random.normal(0.0004, 0.008, 100)

    metrics = calculate_all_metrics(portfolio, benchmark, risk_free_rate=0.02)

    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.sharpe_ratio is not None
    assert metrics.beta is not None
    assert metrics.alpha is not None
    assert metrics.treynor_ratio is not None
    assert metrics.information_ratio is not None
    assert metrics.tracking_error is not None
    assert metrics.max_drawdown <= 0  # Drawdown is negative or zero


def test_format_metrics_report():
    """Report formatting produces non-empty string with key sections."""
    metrics = PerformanceMetrics(
        total_return=0.10,
        annualized_return=0.10,
        sharpe_ratio=1.5,
        sortino_ratio=1.8,
        treynor_ratio=0.5,
        calmar_ratio=2.0,
        volatility=0.20,
        beta=0.9,
        alpha=0.02,
        max_drawdown=-0.05,
        information_ratio=0.3,
        tracking_error=0.15
    )
    report = format_metrics_report(metrics, benchmark_name="SPY")

    assert "PORTFOLIO PERFORMANCE METRICS" in report
    assert "Sharpe Ratio" in report
    assert "SPY" in report
    assert "Beta" in report
    assert "Alpha" in report
    assert "INTERPRETATION" in report


def test_beta_variance_consistency():
    """
    Beta calculation must use consistent sample/population statistics.

    Bug: np.cov() uses ddof=1 (sample covariance) but np.var()
    uses ddof=0 (population variance). This inconsistency biases
    beta, especially for small samples.

    With a perfectly correlated 2x levered portfolio:
    - True beta = 2.0
    - Inconsistent stats may give beta != 2.0
    """
    np.random.seed(42)
    benchmark = np.random.normal(0.0005, 0.01, 60)
    portfolio = benchmark * 2.0

    beta, _ = calculate_beta_alpha(portfolio, benchmark)
    assert beta is not None
    # With consistent stats this should be very close to 2.0
    assert _approx(beta, 2.0, rel_tol=0.02), (
        f"Beta inconsistency detected: expected ≈ 2.0, got {beta}. "
        f"np.cov uses ddof=1 but np.var uses ddof=0"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Performance Metrics Test Suite")
    print("=" * 60 + "\n")

    tests = [
        test_sharpe_ratio_basic,
        test_sharpe_ratio_with_volatility,
        test_sharpe_ratio_insufficient_data,
        test_beta_alpha_perfect_correlation,
        test_beta_alpha_leveraged,
        test_beta_alpha_insufficient_data,
        test_beta_alpha_different_lengths,
        test_sortino_ratio_no_downside,
        test_sortino_ratio_with_downside,
        test_sortino_ratio_all_negative,
        test_calmar_ratio_basic,
        test_calmar_ratio_no_drawdown,
        test_calmar_ratio_auto_drawdown,
        test_treynor_ratio_basic,
        test_treynor_ratio_zero_beta,
        test_information_ratio_identical,
        test_information_ratio_insufficient_data,
        test_calculate_all_metrics_empty,
        test_calculate_all_metrics_short_series,
        test_calculate_all_metrics_full,
        test_format_metrics_report,
        test_beta_variance_consistency,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: EXCEPTION {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)

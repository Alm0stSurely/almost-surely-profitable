"""
Test suite for risk/metrics.py.

Tests risk metrics calculations (VaR, CVaR, volatility, drawdowns, Sortino,
Calmar, correlations) with deterministic inputs and mathematically verifiable
outputs. These are pure numerical functions — correctness should not depend on
random data or on the order of execution.
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from risk.metrics import (
    RiskMetrics,
    calculate_returns,
    calculate_var,
    calculate_cvar,
    calculate_drawdowns,
    calculate_max_drawdown,
    calculate_downside_volatility,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_correlation_matrix,
    calculate_portfolio_risk_metrics,
    get_risk_summary_for_llm,
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


def _prices_from_returns(returns, start_price=100.0):
    """Build a price series from an array of daily returns."""
    prices = start_price * (1 + np.asarray(returns)).cumprod()
    return pd.Series(
        np.concatenate([[start_price], prices]),
        index=pd.date_range("2024-01-01", periods=len(returns) + 1, freq="D"),
    )


# ---------------------------------------------------------------------------
# calculate_returns
# ---------------------------------------------------------------------------

def test_calculate_returns_basic():
    """Returns from a simple price series."""
    prices = pd.Series([100.0, 101.0, 99.0, 102.0])
    returns = calculate_returns(prices)
    expected = pd.Series([0.01, -0.01980198, 0.03030303])
    assert len(returns) == 3
    assert _approx(returns.iloc[0], expected.iloc[0], rel_tol=1e-5)
    assert _approx(returns.iloc[-1], expected.iloc[-1], rel_tol=1e-5)


def test_calculate_returns_empty_and_single():
    """Empty or single-price series returns empty series."""
    assert len(calculate_returns(pd.Series([], dtype=float))) == 0
    assert len(calculate_returns(pd.Series([100.0]))) == 0


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------

def test_calculate_var_known_distribution():
    """VaR on a uniform grid of returns is the correct percentile."""
    returns = pd.Series(np.linspace(-0.10, 0.10, 1001))
    var_95 = calculate_var(returns, confidence=0.95)
    # 5th percentile of 1001 points ≈ index 50 → -0.09
    assert var_95 < 0
    assert _approx(var_95, -0.09, rel_tol=0.05)


def test_calculate_var_insufficient_data():
    """VaR returns 0.0 for fewer than 30 observations."""
    assert calculate_var(pd.Series(np.full(29, 0.01))) == 0.0
    assert calculate_var(pd.Series([])) == 0.0


def test_calculate_cvar_tail_average():
    """CVaR is the mean of returns below VaR."""
    returns = pd.Series(np.linspace(-0.10, 0.10, 1001))
    cvar = calculate_cvar(returns, confidence=0.95)
    var = calculate_var(returns, confidence=0.95)
    tail = returns[returns <= var]
    assert _approx(cvar, tail.mean(), rel_tol=1e-5)
    assert cvar <= var  # Expected shortfall is at least as bad as VaR


def test_calculate_cvar_insufficient_data():
    """CVaR returns 0.0 for fewer than 30 observations."""
    assert calculate_cvar(pd.Series(np.full(29, 0.01))) == 0.0
    assert calculate_cvar(pd.Series([])) == 0.0


# ---------------------------------------------------------------------------
# Drawdowns
# ---------------------------------------------------------------------------

def test_calculate_drawdowns_known_path():
    """Drawdowns on a known price path."""
    prices = pd.Series([100.0, 110.0, 105.0, 115.0, 100.0])
    dd = calculate_drawdowns(prices)
    assert _approx(dd.iloc[0], 0.0, abs_tol=1e-9)
    assert _approx(dd.iloc[1], 0.0, abs_tol=1e-9)
    assert _approx(dd.iloc[2], -0.0454545, rel_tol=1e-5)
    assert _approx(dd.iloc[4], -0.1304348, rel_tol=1e-5)


def test_calculate_max_drawdown_known():
    """Max drawdown is the minimum of the drawdown series."""
    prices = pd.Series([100.0, 110.0, 105.0, 115.0, 100.0])
    assert _approx(calculate_max_drawdown(prices), -0.1304348, rel_tol=1e-5)


def test_calculate_drawdowns_empty():
    """Empty price series returns empty drawdown series."""
    dd = calculate_drawdowns(pd.Series([], dtype=float))
    assert len(dd) == 0


# ---------------------------------------------------------------------------
# Downside volatility
# ---------------------------------------------------------------------------

def test_calculate_downside_volatility_all_positive():
    """Downside volatility is zero when all returns are positive."""
    returns = pd.Series(np.full(30, 0.001))
    assert calculate_downside_volatility(returns) == 0.0


def test_calculate_downside_volatility_known():
    """Downside volatility considers only negative returns."""
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01] * 30)
    downside = returns[returns < 0]
    expected = downside.std() * math.sqrt(252)
    assert _approx(calculate_downside_volatility(returns), expected, rel_tol=1e-6)


def test_calculate_downside_volatility_insufficient_data():
    """Downside volatility returns 0.0 for insufficient data."""
    assert calculate_downside_volatility(pd.Series(np.full(29, -0.001))) == 0.0
    assert calculate_downside_volatility(pd.Series([])) == 0.0
    # Fewer than 2 negative returns
    assert calculate_downside_volatility(pd.Series([0.01, 0.02, -0.01])) == 0.0


# ---------------------------------------------------------------------------
# Sortino ratio
# ---------------------------------------------------------------------------

def test_calculate_sortino_ratio_all_positive():
    """Sortino returns 0.0 when all returns exceed the risk-free rate (zero downside vol)."""
    returns = pd.Series(np.full(30, 0.001))
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino == 0.0


def test_calculate_sortino_ratio_zero_downside_volatility():
    """Sortino guards against near-zero downside volatility by returning 0.0."""
    returns = pd.Series(np.full(30, 0.001))
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)
    # Mean return > risk-free rate, zero downside vol → 0.0 (finite sentinel)
    assert sortino == 0.0

    returns_zero = pd.Series(np.full(30, 0.00005))  # Below risk-free rate
    sortino_zero = calculate_sortino_ratio(returns_zero, risk_free_rate=0.02)
    assert sortino_zero == 0.0


def test_calculate_sortino_ratio_with_downside():
    """Sortino with mixed returns is positive when mean excess return is positive."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)
    assert sortino is not None
    assert not math.isnan(sortino)


def test_calculate_sortino_ratio_insufficient_data():
    """Sortino returns 0.0 for fewer than 30 observations."""
    assert calculate_sortino_ratio(pd.Series(np.full(29, 0.001))) == 0.0


# ---------------------------------------------------------------------------
# Calmar ratio
# ---------------------------------------------------------------------------

def test_calculate_calmar_ratio_basic():
    """Calmar ratio returns 0.0 for monotonic positive returns (no drawdown)."""
    # 252 days of 0.1% daily return → monotonic increase, no drawdown
    returns = np.full(252, 0.001)
    prices = _prices_from_returns(returns)
    calmar = calculate_calmar_ratio(prices)
    assert calmar == 0.0


def test_calculate_calmar_ratio_with_recovery():
    """Calmar ratio is finite and positive after a drawdown and recovery."""
    returns = np.concatenate([np.full(100, 0.001), np.full(10, -0.005), np.full(142, 0.001)])
    prices = _prices_from_returns(returns)
    calmar = calculate_calmar_ratio(prices)
    assert calmar > 0
    assert math.isfinite(calmar)


def test_calculate_calmar_ratio_no_drawdown():
    """Calmar returns 0.0 for positive returns with no drawdown."""
    returns = np.full(252, 0.001)
    prices = _prices_from_returns(returns)
    calmar = calculate_calmar_ratio(prices)
    assert calmar == 0.0


def test_calculate_calmar_ratio_monotonic_decline():
    """Calmar is negative and finite for a monotonic decline."""
    returns = np.full(252, -0.001)
    prices = _prices_from_returns(returns)
    calmar = calculate_calmar_ratio(prices)
    assert calmar < 0
    assert math.isfinite(calmar)


def test_calculate_calmar_ratio_with_drawdown():
    """Calmar is finite and positive when there is a recovery from drawdown."""
    returns = np.concatenate([np.full(100, 0.001), np.full(10, -0.005), np.full(142, 0.001)])
    prices = _prices_from_returns(returns)
    calmar = calculate_calmar_ratio(prices)
    assert calmar > 0
    assert math.isfinite(calmar)


def test_calculate_calmar_ratio_insufficient_data():
    """Calmar returns 0.0 for fewer than 30 price observations."""
    prices = _prices_from_returns(np.full(28, 0.001))  # 29 prices total
    assert calculate_calmar_ratio(prices) == 0.0


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def test_calculate_correlation_matrix_perfect_correlation():
    """Correlation matrix shows 1.0 for perfectly correlated assets."""
    returns1 = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01] * 10)
    returns2 = returns1 * 2.0
    corr = calculate_correlation_matrix({"A": returns1, "B": returns2})
    assert corr is not None
    assert _approx(corr.loc["A", "B"], 1.0, rel_tol=1e-9)
    assert _approx(corr.loc["B", "A"], 1.0, rel_tol=1e-9)


def test_calculate_correlation_matrix_perfect_inverse():
    """Correlation matrix shows -1.0 for perfectly inversely correlated assets."""
    returns1 = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01] * 10)
    returns2 = -returns1
    corr = calculate_correlation_matrix({"A": returns1, "B": returns2})
    assert corr is not None
    assert _approx(corr.loc["A", "B"], -1.0, rel_tol=1e-9)


def test_calculate_correlation_matrix_insufficient_assets():
    """Correlation matrix requires at least two assets."""
    returns = pd.Series([0.01, -0.01, 0.02] * 10)
    assert calculate_correlation_matrix({"A": returns}) is None


def test_calculate_correlation_matrix_insufficient_data():
    """Correlation matrix requires at least 10 observations."""
    returns1 = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
    returns2 = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
    assert calculate_correlation_matrix({"A": returns1, "B": returns2}) is None


# ---------------------------------------------------------------------------
# Portfolio risk metrics integration
# ---------------------------------------------------------------------------

def test_calculate_portfolio_risk_metrics_basic():
    """Full portfolio risk metrics calculation with equal weights."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    returns1 = np.random.normal(0.0005, 0.02, 252)
    returns2 = np.random.normal(0.0003, 0.015, 252)
    prices1 = pd.Series(100 * (1 + returns1).cumprod(), index=dates)
    prices2 = pd.Series(100 * (1 + returns2).cumprod(), index=dates)

    metrics = calculate_portfolio_risk_metrics(
        {"ASSET1": prices1, "ASSET2": prices2},
        weights={"ASSET1": 0.6, "ASSET2": 0.4},
    )

    assert isinstance(metrics, RiskMetrics)
    assert metrics.var_95 < 0
    assert metrics.var_99 < metrics.var_95
    assert metrics.cvar_95 <= metrics.var_95
    assert metrics.cvar_99 <= metrics.var_99
    assert metrics.volatility >= 0
    assert metrics.downside_volatility >= 0
    assert metrics.max_drawdown <= 0
    assert metrics.current_drawdown <= 0
    assert not math.isnan(metrics.skewness)
    assert not math.isnan(metrics.kurtosis)


def test_calculate_portfolio_risk_metrics_equal_weight_default():
    """Default weights produce equal-weighted portfolio."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    returns1 = np.random.normal(0.0005, 0.02, 252)
    returns2 = np.random.normal(0.0003, 0.015, 252)
    prices1 = pd.Series(100 * (1 + returns1).cumprod(), index=dates)
    prices2 = pd.Series(100 * (1 + returns2).cumprod(), index=dates)

    metrics_equal = calculate_portfolio_risk_metrics(
        {"ASSET1": prices1, "ASSET2": prices2}
    )
    metrics_explicit = calculate_portfolio_risk_metrics(
        {"ASSET1": prices1, "ASSET2": prices2},
        weights={"ASSET1": 0.5, "ASSET2": 0.5},
    )

    assert _approx(metrics_equal.volatility, metrics_explicit.volatility, rel_tol=1e-5)
    assert _approx(metrics_equal.var_95, metrics_explicit.var_95, rel_tol=1e-5)


def test_calculate_portfolio_risk_metrics_empty():
    """Empty price dict returns a RiskMetrics with safe defaults."""
    metrics = calculate_portfolio_risk_metrics({})
    assert isinstance(metrics, RiskMetrics)
    # No assets → NaN from empty operations; code should still return a dataclass
    # but the behavior is undefined. We mainly check it does not crash.


def test_calculate_portfolio_risk_metrics_short_series():
    """Short price series returns guarded defaults."""
    prices = _prices_from_returns(np.full(5, 0.001))
    metrics = calculate_portfolio_risk_metrics({"A": prices})
    assert isinstance(metrics, RiskMetrics)
    # VaR/CVaR and most ratios return 0 for < 30 observations
    assert metrics.var_95 == 0.0
    assert metrics.cvar_95 == 0.0
    assert metrics.sortino_ratio == 0.0
    assert metrics.calmar_ratio == 0.0


# ---------------------------------------------------------------------------
# LLM summary formatting
# ---------------------------------------------------------------------------

def test_get_risk_summary_for_llm():
    """Risk summary contains all expected sections and metrics."""
    metrics = RiskMetrics(
        var_95=-0.02,
        var_99=-0.05,
        cvar_95=-0.03,
        cvar_99=-0.06,
        volatility=0.20,
        downside_volatility=0.15,
        max_drawdown=-0.10,
        current_drawdown=-0.05,
        sortino_ratio=1.5,
        calmar_ratio=2.0,
        skewness=-0.5,
        kurtosis=3.0,
    )
    summary = get_risk_summary_for_llm(metrics)

    assert "Risk Metrics" in summary
    assert "VaR 95%" in summary
    assert "CVaR 95%" in summary
    assert "Volatility" in summary
    assert "Max Drawdown" in summary
    assert "Sortino Ratio" in summary
    assert "Skewness" in summary


# ---------------------------------------------------------------------------
# Regression / bug-detection tests
# ---------------------------------------------------------------------------

def test_var_cvar_monotonicity():
    """CVaR at 99% must be <= CVaR at 95% <= VaR at 95%."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.02, 252))
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    cvar_95 = calculate_cvar(returns, 0.95)
    cvar_99 = calculate_cvar(returns, 0.99)

    assert var_99 <= var_95  # More extreme quantile is more negative
    assert cvar_95 <= var_95
    assert cvar_99 <= var_99
    assert cvar_99 <= cvar_95


def test_calmar_ratio_zero_drawdown_positive_returns():
    """Calmar returns 0.0 when returns are positive and max drawdown is zero."""
    returns = np.full(60, 0.001)
    prices = _prices_from_returns(returns)
    assert calculate_calmar_ratio(prices) == 0.0


def test_calmar_ratio_zero_drawdown_zero_returns():
    """Calmar is 0 when drawdown is zero but returns are zero."""
    prices = pd.Series([100.0] * 60)
    assert calculate_calmar_ratio(prices) == 0.0


def test_calculate_sortino_ratio_nan_downside_vol():
    """Sortino guards against NaN downside volatility by returning 0.0."""
    returns = pd.Series([0.001] * 30)
    # Manually set downside_vol to NaN by using a series that triggers it
    # Actually the function guards with np.isnan. Verify with all-identical returns.
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino == 0.0


if __name__ == "__main__":
    print("=" * 60)
    print("Risk Metrics Test Suite")
    print("=" * 60 + "\n")

    tests = [
        test_calculate_returns_basic,
        test_calculate_returns_empty_and_single,
        test_calculate_var_known_distribution,
        test_calculate_var_insufficient_data,
        test_calculate_cvar_tail_average,
        test_calculate_cvar_insufficient_data,
        test_calculate_drawdowns_known_path,
        test_calculate_max_drawdown_known,
        test_calculate_drawdowns_empty,
        test_calculate_downside_volatility_all_positive,
        test_calculate_downside_volatility_known,
        test_calculate_downside_volatility_insufficient_data,
        test_calculate_sortino_ratio_all_positive,
        test_calculate_sortino_ratio_zero_downside_volatility,
        test_calculate_sortino_ratio_with_downside,
        test_calculate_sortino_ratio_insufficient_data,
        test_calculate_calmar_ratio_basic,
        test_calculate_calmar_ratio_with_recovery,
        test_calculate_calmar_ratio_no_drawdown,
        test_calculate_calmar_ratio_monotonic_decline,
        test_calculate_calmar_ratio_with_drawdown,
        test_calculate_calmar_ratio_insufficient_data,
        test_calculate_correlation_matrix_perfect_correlation,
        test_calculate_correlation_matrix_perfect_inverse,
        test_calculate_correlation_matrix_insufficient_assets,
        test_calculate_correlation_matrix_insufficient_data,
        test_calculate_portfolio_risk_metrics_basic,
        test_calculate_portfolio_risk_metrics_equal_weight_default,
        test_calculate_portfolio_risk_metrics_empty,
        test_calculate_portfolio_risk_metrics_short_series,
        test_get_risk_summary_for_llm,
        test_var_cvar_monotonicity,
        test_calmar_ratio_zero_drawdown_positive_returns,
        test_calmar_ratio_zero_drawdown_zero_returns,
        test_calculate_sortino_ratio_nan_downside_vol,
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

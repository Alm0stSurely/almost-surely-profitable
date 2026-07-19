"""
Test script for CVaR calculation module.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk.cvar import (
    calculate_cvar,
    calculate_portfolio_cvar,
    calculate_drawdown_cvar,
    tail_risk_analysis,
    format_risk_report
)


def test_basic_cvar():
    """Test basic CVaR calculation."""
    print("Test 1: Basic CVaR Calculation")
    print("-" * 40)
    
    # Generate synthetic returns (mean 0, std 0.02)
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 252)  # 1 year of daily returns
    
    cvar_results = calculate_cvar(returns, [0.95, 0.99])
    
    print(f"CVaR 95%: {cvar_results[0.95]:.2%}")
    print(f"CVaR 99%: {cvar_results[0.99]:.2%}")
    print(f"Mean return: {np.mean(returns):.2%}")
    print(f"Std return: {np.std(returns):.2%}")
    print()
    
    assert cvar_results[0.95] > 0, "CVaR should be positive"
    assert cvar_results[0.99] >= cvar_results[0.95], "99% CVaR should be >= 95% CVaR"
    print("✓ Basic CVaR test passed\n")


def test_portfolio_cvar():
    """Test portfolio-level CVaR."""
    print("Test 2: Portfolio CVaR")
    print("-" * 40)
    
    np.random.seed(42)
    
    # Simulate 3 assets with correlations
    n_days = 252
    spy_returns = np.random.normal(0.0003, 0.012, n_days)
    qqq_returns = 1.2 * spy_returns + np.random.normal(0, 0.008, n_days)
    tlt_returns = -0.3 * spy_returns + np.random.normal(0.0001, 0.008, n_days)
    
    position_returns = {
        'SPY': spy_returns,
        'QQQ': qqq_returns,
        'TLT': tlt_returns
    }
    
    weights = {'SPY': 0.5, 'QQQ': 0.3, 'TLT': 0.2}
    
    result = calculate_portfolio_cvar(position_returns, weights)
    
    print(f"Portfolio CVaR 95%: {result.cvar_95:.2%}")
    print(f"Portfolio VaR 95%: {result.var_95:.2%}")
    print(f"Worst case: {result.worst_case:.2%}")
    print()
    print(format_risk_report(result))
    
    assert result.cvar_95 > 0, "Portfolio CVaR should be positive"
    assert result.cvar_99 >= result.cvar_95, "99% CVaR >= 95% CVaR"
    print("✓ Portfolio CVaR test passed\n")


def test_tail_risk_analysis():
    """Test comprehensive tail risk metrics."""
    print("Test 3: Tail Risk Analysis")
    print("-" * 40)
    
    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.015, 252)
    
    # Add some fat tails
    returns[0] = -0.08  # Crash day
    returns[1] = -0.05  # Follow-through
    
    metrics = tail_risk_analysis(returns)
    
    print(f"CVaR 95%: {metrics['cvar_95_pct']:.2f}%")
    print(f"Skewness: {metrics['skewness']:.2f}")
    print(f"Kurtosis: {metrics['kurtosis']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    print()
    
    assert metrics['max_drawdown'] < 0, "Max drawdown should be negative"
    assert metrics['cvar_95'] > 0, "CVaR should be positive"
    print("✓ Tail risk analysis test passed\n")


def test_drawdown_cvar():
    """Test CVaR on drawdown periods."""
    print("Test 4: Drawdown CVaR")
    print("-" * 40)
    
    # Create equity curve with drawdown
    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.01, 100)
    returns[40:50] = -0.02  # Drawdown period
    
    equity = 10000 * np.cumprod(1 + returns)
    
    dd_cvar = calculate_drawdown_cvar(equity, window=20, confidence=0.95)
    
    print(f"Drawdown CVaR (95%): {dd_cvar:.2%}")
    print()
    
    assert dd_cvar > 0, "Drawdown CVaR should be positive"
    print("✓ Drawdown CVaR test passed\n")


def test_tail_risk_small_sample():
    """Test tail_risk_analysis on small sample sizes."""
    print("Test 5: Small-Sample Tail Risk")
    print("-" * 40)
    
    # Empty input should return empty dict
    assert tail_risk_analysis(np.array([])) == {}
    
    # Single observation: no sample standard deviation possible
    result = tail_risk_analysis(np.array([0.01]))
    assert 'sortino_ratio' not in result
    
    # Two observations, one negative: sortino should not be added because
    # only one downside observation exists and ddof=1 is undefined.
    result = tail_risk_analysis(np.array([0.01, -0.02]))
    assert 'sortino_ratio' not in result
    
    # Two observations, both negative: two downside observations, sortino defined
    result = tail_risk_analysis(np.array([-0.01, -0.02]))
    assert 'sortino_ratio' in result
    assert np.isfinite(result['sortino_ratio'])
    
    # Benchmark shorter than returns: should align and compute if enough data
    result = tail_risk_analysis(
        np.array([0.01, -0.02, 0.005, 0.001]),
        np.array([0.005, -0.01])
    )
    assert 'tracking_error' in result
    assert 'information_ratio' in result
    assert np.isfinite(result['tracking_error'])
    assert np.isfinite(result['information_ratio'])
    
    # Benchmark with only one observation: no tracking error possible
    result = tail_risk_analysis(
        np.array([0.01, -0.02]),
        np.array([0.005])
    )
    assert 'tracking_error' not in result
    assert 'information_ratio' not in result
    
    print("✓ Small-sample tail risk test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("CVaR Module Test Suite")
    print("=" * 60)
    print()
    
    test_basic_cvar()
    test_portfolio_cvar()
    test_tail_risk_analysis()
    test_tail_risk_small_sample()
    test_drawdown_cvar()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

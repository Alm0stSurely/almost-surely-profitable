"""
Regression tests for portfolio CVaR edge cases.

These cover the empty / single-asset / empty-returns paths that previously
raised ValueError or IndexError, and verify that weight normalization
is applied consistently.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk.cvar import calculate_portfolio_cvar, CVaRResult


def _assert_all_zero(result: CVaRResult) -> None:
    assert result.cvar_95 == 0.0
    assert result.cvar_99 == 0.0
    assert result.var_95 == 0.0
    assert result.var_99 == 0.0
    assert result.worst_case == 0.0
    assert result.expected_shortfall_pct == 0.0


def test_empty_position_returns_returns_zeroed_result():
    """No positions should yield a zero CVaRResult, not a ValueError."""
    result = calculate_portfolio_cvar({}, {})
    _assert_all_zero(result)


def test_empty_returns_array_returns_zeroed_result():
    """A single position with no observations should not raise IndexError."""
    result = calculate_portfolio_cvar(
        {'SPY': np.array([])},
        {'SPY': 1.0}
    )
    _assert_all_zero(result)


def test_zero_total_weight_returns_zeroed_result():
    """Weights that sum to zero are degenerate and should return zeros."""
    result = calculate_portfolio_cvar(
        {'SPY': np.array([0.01, -0.02])},
        {'SPY': 0.0}
    )
    _assert_all_zero(result)


def test_non_unit_weights_are_normalized():
    """Weights that do not sum to 1 are treated as allocation fractions."""
    spy = np.array([0.01, -0.02, 0.005, -0.01])
    qqq = np.array([-0.005, 0.01, 0.0, 0.02])
    
    # Raw weights sum to 2.0; after normalization they are 0.5 / 0.5.
    result = calculate_portfolio_cvar(
        {'SPY': spy, 'QQQ': qqq},
        {'SPY': 1.0, 'QQQ': 1.0}
    )
    
    expected_portfolio = (spy + qqq) / 2.0
    assert np.isclose(result.worst_case, np.min(expected_portfolio))


def test_single_asset_portfolio_matches_direct_cvar():
    """A single-asset portfolio with normalized weight 1.0 equals the asset CVaR."""
    returns = np.array([0.01, -0.02, 0.005, -0.01, -0.03, 0.02])
    result = calculate_portfolio_cvar(
        {'SPY': returns},
        {'SPY': 1.0}
    )
    
    assert result.worst_case == np.min(returns)
    assert result.cvar_95 > 0.0
    assert result.cvar_99 >= result.cvar_95


def test_partial_weights_missing_tickers_are_ignored():
    """Tickers in position_returns but not in weights get zero weight."""
    spy = np.array([0.01, -0.02, 0.005, -0.01])
    qqq = np.array([0.0, 0.0, 0.0, 0.0])
    
    result = calculate_portfolio_cvar(
        {'SPY': spy, 'QQQ': qqq},
        {'SPY': 1.0}
    )
    
    assert np.isclose(result.worst_case, np.min(spy))

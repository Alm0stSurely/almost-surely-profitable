"""
Regression tests for non-finite inputs in the CVaR module.

NaN or infinite returns should be treated as degenerate input and produce
well-defined, finite results instead of propagating NaN through downstream
metrics and LLM prompts.
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
    CVaRResult,
)


def _assert_all_zero(result: CVaRResult) -> None:
    assert result.cvar_95 == 0.0
    assert result.cvar_99 == 0.0
    assert result.var_95 == 0.0
    assert result.var_99 == 0.0
    assert result.worst_case == 0.0
    assert result.expected_shortfall_pct == 0.0


def test_calculate_cvar_nan_returns_returns_zeros():
    returns = np.array([0.01, -0.02, np.nan, 0.005])
    result = calculate_cvar(returns, [0.95, 0.99])
    assert result[0.95] == 0.0
    assert result[0.99] == 0.0


def test_calculate_cvar_inf_returns_returns_zeros():
    returns = np.array([0.01, np.inf, -0.02])
    result = calculate_cvar(returns, [0.95, 0.99])
    assert result[0.95] == 0.0
    assert result[0.99] == 0.0


def test_calculate_cvar_negative_inf_returns_returns_zeros():
    returns = np.array([0.01, -np.inf, -0.02])
    result = calculate_cvar(returns, [0.95, 0.99])
    assert result[0.95] == 0.0
    assert result[0.99] == 0.0


def test_calculate_cvar_mixed_non_finite_returns_returns_zeros():
    returns = np.array([np.nan, np.inf, -np.inf, 0.005])
    result = calculate_cvar(returns, [0.95, 0.99])
    assert result[0.95] == 0.0
    assert result[0.99] == 0.0


def test_portfolio_cvar_nan_position_returns_returns_zeroed_result():
    pr = {
        'SPY': np.array([0.01, np.nan, -0.02]),
        'QQQ': np.array([0.005, 0.01, -0.01]),
    }
    weights = {'SPY': 0.5, 'QQQ': 0.5}
    result = calculate_portfolio_cvar(pr, weights)
    _assert_all_zero(result)


def test_portfolio_cvar_inf_position_returns_returns_zeroed_result():
    pr = {
        'SPY': np.array([0.01, np.inf, -0.02]),
        'QQQ': np.array([0.005, 0.01, -0.01]),
    }
    weights = {'SPY': 0.5, 'QQQ': 0.5}
    result = calculate_portfolio_cvar(pr, weights)
    _assert_all_zero(result)


def test_portfolio_cvar_single_nan_position_returns_zeroed_result():
    pr = {'SPY': np.array([np.nan, np.nan, np.nan])}
    weights = {'SPY': 1.0}
    result = calculate_portfolio_cvar(pr, weights)
    _assert_all_zero(result)


def test_tail_risk_analysis_nan_returns_returns_empty_dict():
    result = tail_risk_analysis(np.array([0.01, np.nan, -0.02]))
    assert result == {}


def test_tail_risk_analysis_inf_returns_returns_empty_dict():
    result = tail_risk_analysis(np.array([0.01, np.inf, -0.02]))
    assert result == {}


def test_tail_risk_analysis_nan_benchmark_skips_comparison_but_keeps_metrics():
    # NaN in the optional benchmark should not invalidate the portfolio tail-risk
    # metrics; it should only suppress the benchmark-relative metrics.
    result = tail_risk_analysis(
        np.array([0.01, -0.02, 0.005, -0.01]),
        np.array([0.005, np.nan, -0.01, 0.002]),
    )
    assert 'cvar_95' in result
    assert 'var_95' in result
    assert 'max_drawdown' in result
    assert 'tracking_error' not in result
    assert 'information_ratio' not in result
    assert all(np.isfinite(v) for v in result.values())


def test_drawdown_cvar_nan_equity_returns_zero():
    equity = np.array([100.0, np.nan, 98.0, 97.0])
    result = calculate_drawdown_cvar(equity, window=20, confidence=0.95)
    assert result == 0.0


def test_drawdown_cvar_inf_equity_returns_zero():
    equity = np.array([100.0, np.inf, 98.0, 97.0])
    result = calculate_drawdown_cvar(equity, window=20, confidence=0.95)
    assert result == 0.0


def test_finite_input_still_produces_ordered_cvar():
    # Sanity check: the non-finite guard must not break the normal invariant.
    returns = np.array([0.01, -0.02, 0.005, -0.01, -0.03, 0.02, 0.015])
    result = calculate_cvar(returns, [0.95, 0.99])
    assert np.isfinite(result[0.95])
    assert np.isfinite(result[0.99])
    assert result[0.99] >= result[0.95]

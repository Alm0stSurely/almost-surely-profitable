"""
Benchmark for the CVaR non-finite input guards.

Exercises calculate_cvar, calculate_portfolio_cvar, tail_risk_analysis, and
calculate_drawdown_cvar with NaN and Inf inputs, asserting that the output is
finite and that the full suite remains warning-free under -W error::RuntimeWarning.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from risk.cvar import (
    calculate_cvar,
    calculate_portfolio_cvar,
    calculate_drawdown_cvar,
    tail_risk_analysis,
)


def _assert_zeroed(result):
    assert result.cvar_95 == 0.0
    assert result.cvar_99 == 0.0
    assert result.var_95 == 0.0
    assert result.var_99 == 0.0
    assert result.worst_case == 0.0
    assert result.expected_shortfall_pct == 0.0


def benchmark_cvar_nan():
    returns = np.array([0.01, -0.02, np.nan, 0.005, -0.01])
    result = calculate_cvar(returns, [0.95, 0.99])
    assert result[0.95] == 0.0
    assert result[0.99] == 0.0


def benchmark_cvar_inf():
    returns = np.array([0.01, np.inf, -0.02, 0.005])
    result = calculate_cvar(returns, [0.95, 0.99])
    assert result[0.95] == 0.0
    assert result[0.99] == 0.0


def benchmark_portfolio_cvar_nan():
    pr = {
        'SPY': np.array([0.01, np.nan, -0.02]),
        'QQQ': np.array([0.005, 0.01, -0.01]),
    }
    weights = {'SPY': 0.5, 'QQQ': 0.5}
    result = calculate_portfolio_cvar(pr, weights)
    _assert_zeroed(result)


def benchmark_portfolio_cvar_inf():
    pr = {
        'SPY': np.array([0.01, -np.inf, -0.02]),
        'QQQ': np.array([0.005, 0.01, -0.01]),
    }
    weights = {'SPY': 0.5, 'QQQ': 0.5}
    result = calculate_portfolio_cvar(pr, weights)
    _assert_zeroed(result)


def benchmark_tail_risk_nan():
    result = tail_risk_analysis(np.array([0.01, -0.02, np.nan, 0.005]))
    assert result == {}


def benchmark_drawdown_cvar_nan():
    equity = np.array([100.0, 101.0, np.nan, 97.0, 98.0])
    result = calculate_drawdown_cvar(equity, window=20, confidence=0.95)
    assert result == 0.0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        benchmark_cvar_nan()
        benchmark_cvar_inf()
        benchmark_portfolio_cvar_nan()
        benchmark_portfolio_cvar_inf()
        benchmark_tail_risk_nan()
        benchmark_drawdown_cvar_nan()
    print("All non-finite CVaR benchmarks passed without RuntimeWarning.")

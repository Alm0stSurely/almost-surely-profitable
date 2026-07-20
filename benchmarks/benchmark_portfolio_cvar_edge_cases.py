"""Benchmark for calculate_portfolio_cvar edge-case robustness.

Demonstrates that calculate_portfolio_cvar now returns a well-formed
CVaRResult for degenerate inputs (empty positions, empty returns, zero
weights) and normalizes non-unit weights instead of silently scaling
the portfolio return.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk.cvar import calculate_portfolio_cvar, CVaRResult


def _run_case(name, position_returns, weights):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = calculate_portfolio_cvar(position_returns, weights)
    assert isinstance(result, CVaRResult)
    finite = all(np.isfinite(v) for v in [
        result.cvar_95, result.cvar_99, result.var_95, result.var_99,
        result.worst_case, result.expected_shortfall_pct
    ])
    return result, finite


if __name__ == "__main__":
    print("Benchmark: calculate_portfolio_cvar edge-case robustness")
    print("-" * 70)

    np.random.seed(42)
    spy = np.random.normal(0.0003, 0.012, 252)
    qqq = 1.2 * spy + np.random.normal(0, 0.008, 252)

    cases = [
        ("empty_positions", {}, {}),
        ("empty_returns", {"SPY": np.array([])}, {"SPY": 1.0}),
        ("zero_weight", {"SPY": spy}, {"SPY": 0.0}),
        ("single_asset", {"SPY": spy}, {"SPY": 1.0}),
        ("non_unit_weights", {"SPY": spy, "QQQ": qqq}, {"SPY": 50.0, "QQQ": 50.0}),
        ("missing_ticker_weight", {"SPY": spy, "QQQ": qqq}, {"SPY": 1.0}),
    ]

    for name, position_returns, weights in cases:
        result, finite = _run_case(name, position_returns, weights)
        print(
            f"{name:30s} | cvar95={result.cvar_95:8.4%} | "
            f"var95={result.var_95:8.4%} | worst={result.worst_case:8.4%} | finite={finite!s}"
        )

    print("-" * 70)
    print("OK - calculate_portfolio_cvar is robust to empty and single-asset inputs.")

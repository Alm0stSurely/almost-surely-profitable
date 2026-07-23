"""
Benchmark for the risk/metrics non-finite input/output guards.

Exercises calculate_sortino_ratio, calculate_calmar_ratio, and
calculate_portfolio_risk_metrics with NaN/Inf inputs and with degenerate finite
inputs (zero drawdown, no downside), asserting that every output is finite and
JSON-serializable.

Run under -W error::RuntimeWarning to ensure no silent numerical failures.
"""

import json
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from risk.metrics import (
    RiskMetrics,
    calculate_calmar_ratio,
    calculate_cvar,
    calculate_downside_volatility,
    calculate_max_drawdown,
    calculate_portfolio_risk_metrics,
    calculate_sortino_ratio,
    calculate_var,
    get_risk_summary_for_llm,
)


def _make_prices(n=60, start=100.0, daily_return=0.001):
    returns = np.full(n, daily_return)
    prices = start * (1 + returns).cumprod()
    return pd.Series(np.concatenate([[start], prices]))


def benchmark_sortino_nan():
    returns = pd.Series([0.01, -0.02, np.nan, 0.005] * 15)
    assert calculate_sortino_ratio(returns) == 0.0


def benchmark_sortino_inf():
    returns = pd.Series([0.01, np.inf, -0.02, 0.005] * 15)
    assert calculate_sortino_ratio(returns) == 0.0


def benchmark_sortino_no_downside():
    """All-positive returns used to produce +inf; now the pipeline stays finite."""
    returns = pd.Series(np.full(60, 0.002))
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino == 0.0


def benchmark_calmar_nan():
    prices = _make_prices()
    prices.iloc[10] = np.nan
    assert calculate_calmar_ratio(prices) == 0.0


def benchmark_calmar_inf():
    prices = _make_prices()
    prices.iloc[10] = np.inf
    assert calculate_calmar_ratio(prices) == 0.0


def benchmark_calmar_no_drawdown():
    """A monotonic positive series used to produce +inf; now it stays finite."""
    prices = _make_prices(60, daily_return=0.001)
    calmar = calculate_calmar_ratio(prices)
    assert calmar == 0.0


def benchmark_var_cvar_non_finite():
    returns = pd.Series([0.01, -0.02, np.nan, 0.005] * 15)
    assert calculate_var(returns, 0.95) == 0.0
    assert calculate_cvar(returns, 0.95) == 0.0


def benchmark_downside_volatility_non_finite():
    returns = pd.Series([0.01, -0.02, np.nan, 0.005] * 15)
    assert calculate_downside_volatility(returns) == 0.0


def benchmark_max_drawdown_non_finite():
    prices = _make_prices()
    prices.iloc[10] = np.nan
    assert calculate_max_drawdown(prices) == 0.0


def benchmark_portfolio_risk_metrics_nan_price():
    prices_a = _make_prices()
    prices_b = _make_prices()
    prices_a.iloc[10] = np.nan

    metrics = calculate_portfolio_risk_metrics(
        {"A": prices_a, "B": prices_b},
        weights={"A": 0.5, "B": 0.5},
    )
    assert isinstance(metrics, RiskMetrics)
    for value in metrics.to_dict().values():
        assert math.isfinite(value)


def benchmark_portfolio_risk_metrics_all_nan():
    prices = pd.Series([np.nan] * 60)
    metrics = calculate_portfolio_risk_metrics({"A": prices})
    assert isinstance(metrics, RiskMetrics)
    for value in metrics.to_dict().values():
        assert value == 0.0


def benchmark_risk_summary_json_safe():
    prices_a = _make_prices()
    prices_b = _make_prices(60, daily_return=-0.001)
    prices_a.iloc[10] = np.nan

    metrics = calculate_portfolio_risk_metrics(
        {"A": prices_a, "B": prices_b},
        weights={"A": 0.5, "B": 0.5},
    )
    summary = get_risk_summary_for_llm(metrics)
    assert "inf" not in summary.lower()
    assert "nan" not in summary.lower()

    text = json.dumps(metrics.to_dict(), allow_nan=False)
    assert "Infinity" not in text
    assert "NaN" not in text
    assert json.loads(text) == metrics.to_dict()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        benchmark_sortino_nan()
        benchmark_sortino_inf()
        benchmark_sortino_no_downside()
        benchmark_calmar_nan()
        benchmark_calmar_inf()
        benchmark_calmar_no_drawdown()
        benchmark_var_cvar_non_finite()
        benchmark_downside_volatility_non_finite()
        benchmark_max_drawdown_non_finite()
        benchmark_portfolio_risk_metrics_nan_price()
        benchmark_portfolio_risk_metrics_all_nan()
        benchmark_risk_summary_json_safe()
    print("All risk/metrics non-finite benchmarks passed without RuntimeWarning.")

"""
Regression tests for non-finite input/output handling in risk/metrics.py.

Downstream consumers (daily_run.py, LLM prompts, JSON reports) require every
scalar metric to be finite. NaN/Inf inputs must not propagate.
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
    """Build a deterministic price series."""
    returns = np.full(n, daily_return)
    prices = start * (1 + returns).cumprod()
    return pd.Series(np.concatenate([[start], prices]))


def test_sortino_ratio_nan_returns():
    """NaN returns do not produce NaN/Inf Sortino."""
    returns = pd.Series(np.full(60, 0.001))
    returns.iloc[5] = np.nan
    assert calculate_sortino_ratio(returns) == 0.0


def test_sortino_ratio_inf_returns():
    """Infinite returns produce a finite Sortino."""
    returns = pd.Series(np.full(60, 0.001))
    returns.iloc[5] = np.inf
    assert calculate_sortino_ratio(returns) == 0.0


def test_calmar_ratio_nan_prices():
    """NaN prices do not produce NaN/Inf Calmar."""
    prices = _make_prices()
    prices.iloc[10] = np.nan
    assert calculate_calmar_ratio(prices) == 0.0


def test_calmar_ratio_inf_prices():
    """Infinite prices do not produce NaN/Inf Calmar."""
    prices = _make_prices()
    prices.iloc[10] = np.inf
    assert calculate_calmar_ratio(prices) == 0.0


def test_var_cvar_non_finite_returns():
    """VaR and CVaR return 0.0 for non-finite inputs."""
    returns = pd.Series(np.full(60, 0.001))
    returns.iloc[0] = np.nan
    assert calculate_var(returns, 0.95) == 0.0
    assert calculate_cvar(returns, 0.95) == 0.0


def test_downside_volatility_non_finite_returns():
    """Downside volatility returns 0.0 for non-finite inputs."""
    returns = pd.Series(np.full(60, -0.001))
    returns.iloc[0] = np.nan
    assert calculate_downside_volatility(returns) == 0.0


def test_max_drawdown_non_finite_prices():
    """Max drawdown returns 0.0 for non-finite prices."""
    prices = _make_prices()
    prices.iloc[5] = np.nan
    assert calculate_max_drawdown(prices) == 0.0


def test_portfolio_risk_metrics_all_finite_with_nan_price():
    """A single NaN price does not poison the whole RiskMetrics object."""
    prices_a = _make_prices()
    prices_b = _make_prices()
    prices_a.iloc[10] = np.nan

    metrics = calculate_portfolio_risk_metrics(
        {"A": prices_a, "B": prices_b},
        weights={"A": 0.5, "B": 0.5},
    )

    assert isinstance(metrics, RiskMetrics)
    for field, value in metrics.to_dict().items():
        assert math.isfinite(value), f"{field} is non-finite: {value}"


def test_portfolio_risk_metrics_all_nan_prices():
    """All-NaN prices produce a fully finite sentinel RiskMetrics object."""
    prices = pd.Series([np.nan] * 60)
    metrics = calculate_portfolio_risk_metrics({"A": prices})

    assert isinstance(metrics, RiskMetrics)
    for field, value in metrics.to_dict().items():
        assert math.isfinite(value), f"{field} is non-finite: {value}"
        assert value == 0.0, f"{field} should be 0.0 sentinel, got {value}"


def test_risk_summary_for_llm_is_json_safe():
    """The LLM prompt summary is JSON-serializable with allow_nan=False."""
    prices_a = _make_prices(60, daily_return=0.001)
    prices_b = _make_prices(60, daily_return=-0.001)
    prices_a.iloc[10] = np.nan  # ensure guards are exercised

    metrics = calculate_portfolio_risk_metrics(
        {"A": prices_a, "B": prices_b},
        weights={"A": 0.5, "B": 0.5},
    )
    summary = get_risk_summary_for_llm(metrics)

    # The summary itself is a string; verify it contains no "inf" or "nan" tokens.
    lower_summary = summary.lower()
    assert "inf" not in lower_summary
    assert "nan" not in lower_summary

    # The underlying metrics must be serializable with strict JSON.
    json.dumps(metrics.to_dict(), allow_nan=False)


def test_risk_metrics_no_runtime_warnings():
    """Non-finite inputs do not emit RuntimeWarnings."""
    prices = _make_prices()
    prices.iloc[10] = np.nan
    returns = pd.Series(np.full(60, 0.001))
    returns.iloc[5] = np.nan

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculate_sortino_ratio(returns)
        calculate_calmar_ratio(prices)
        calculate_var(returns, 0.95)
        calculate_cvar(returns, 0.95)
        calculate_downside_volatility(returns)
        calculate_max_drawdown(prices)
        calculate_portfolio_risk_metrics({"A": prices})

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings, "Unexpected RuntimeWarnings: " + "\n".join(
        str(w.message) for w in runtime_warnings
    )

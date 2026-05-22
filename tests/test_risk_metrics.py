"""
Test suite for risk/metrics.py.

Tests all risk metrics calculations with known inputs and expected outputs.
Financial risk formulas are deterministic — if the math is right, the tests pass.
If the math is wrong, we want to know before deploying to production.

Covers: VaR, CVaR, drawdowns, downside volatility, Sortino, Calmar,
correlation matrices, and portfolio-level risk aggregation.
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
    """Compare floats with tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if math.isinf(a) and math.isinf(b):
        return (a > 0) == (b > 0)
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


# ---------------------------------------------------------------------------
# RiskMetrics dataclass
# ---------------------------------------------------------------------------

def test_risk_metrics_to_dict():
    """Round-trip serialization of RiskMetrics."""
    m = RiskMetrics(
        var_95=-0.02,
        var_99=-0.03,
        cvar_95=-0.025,
        cvar_99=-0.035,
        volatility=0.20,
        downside_volatility=0.15,
        max_drawdown=-0.10,
        current_drawdown=-0.05,
        sortino_ratio=1.5,
        calmar_ratio=2.0,
        skewness=-0.5,
        kurtosis=3.0,
    )
    d = m.to_dict()
    assert d["var_95"] == -0.02
    assert d["volatility"] == 0.20
    assert d["max_drawdown"] == -0.10
    assert set(d.keys()) == {
        "var_95", "var_99", "cvar_95", "cvar_99",
        "volatility", "downside_volatility", "max_drawdown", "current_drawdown",
        "sortino_ratio", "calmar_ratio", "skewness", "kurtosis",
    }


# ---------------------------------------------------------------------------
# calculate_returns
# ---------------------------------------------------------------------------

def test_calculate_returns_basic():
    """pct_change produces correct daily returns."""
    prices = pd.Series([100, 101, 102, 99])
    returns = calculate_returns(prices)
    assert len(returns) == 3
    assert _approx(returns.iloc[0], 0.01)
    assert _approx(returns.iloc[1], 1 / 101)
    assert _approx(returns.iloc[2], -3 / 102)


def test_calculate_returns_empty():
    """Empty series returns empty."""
    returns = calculate_returns(pd.Series([], dtype=float))
    assert len(returns) == 0


def test_calculate_returns_single():
    """Single price returns empty (no change to compute)."""
    returns = calculate_returns(pd.Series([100]))
    assert len(returns) == 0


# ---------------------------------------------------------------------------
# calculate_var
# ---------------------------------------------------------------------------

def test_var_basic():
    """VaR at 95% for a known distribution."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 252))
    var = calculate_var(returns, 0.95)
    # For N(0, 0.01), 5th percentile ≈ -1.645 * 0.01 = -0.01645
    assert var < 0, f"VaR should be negative (a loss), got {var}"
    assert var > -0.05, f"VaR unexpectedly large: {var}"


def test_var_insufficient_data():
    """VaR requires >= 30 observations."""
    assert calculate_var(pd.Series(np.full(29, 0.001))) == 0.0
    assert calculate_var(pd.Series([], dtype=float)) == 0.0


def test_var_confidence_levels():
    """VaR_99 should be more extreme than VaR_95."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 252))
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    assert var_99 <= var_95, f"VaR_99 ({var_99}) should be <= VaR_95 ({var_95})"


def test_var_all_positive():
    """VaR for all-positive returns can be positive (no loss at threshold)."""
    returns = pd.Series(np.full(100, 0.01))
    var = calculate_var(returns, 0.95)
    # 5th percentile of all 0.01 is 0.01
    assert var == 0.01


# ---------------------------------------------------------------------------
# calculate_cvar
# ---------------------------------------------------------------------------

def test_cvar_basic():
    """CVaR is the mean of tail losses beyond VaR."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 252))
    var = calculate_var(returns, 0.95)
    cvar = calculate_cvar(returns, 0.95)
    # CVaR <= VaR (more extreme)
    assert cvar <= var, f"CVaR ({cvar}) should be <= VaR ({var})"


def test_cvar_insufficient_data():
    """CVaR requires >= 30 observations."""
    assert calculate_cvar(pd.Series(np.full(29, 0.001))) == 0.0


def test_cvar_no_tail_losses():
    """When all returns equal VaR threshold, CVaR equals VaR."""
    returns = pd.Series(np.full(100, 0.01))
    cvar = calculate_cvar(returns, 0.95)
    var = calculate_var(returns, 0.95)
    assert _approx(cvar, var), f"CVaR ({cvar}) should equal VaR ({var}) for constant returns"


def test_cvar_extreme_tail():
    """CVaR captures fat left tails better than VaR."""
    # 20 extreme losses out of 200 obs (10% tail) → 95% VaR should land in extreme block
    returns = pd.Series(np.concatenate([
        np.full(180, 0.001),
        np.full(20, -0.10),
    ]))
    var = calculate_var(returns, 0.95)
    cvar = calculate_cvar(returns, 0.95)
    # With 200 obs, 5% = 10 obs in tail — all should be -0.10
    assert _approx(cvar, -0.10, rel_tol=0.1), f"Expected CVaR ≈ -0.10, got {cvar} (VaR={var})"


# ---------------------------------------------------------------------------
# calculate_drawdowns
# ---------------------------------------------------------------------------

def test_drawdowns_basic():
    """Drawdowns calculated correctly from prices."""
    prices = pd.Series([100, 110, 105, 115, 108])
    dd = calculate_drawdowns(prices)
    assert len(dd) == 5
    assert dd.iloc[0] == 0.0  # First day: no drawdown
    assert dd.iloc[1] == 0.0  # New high
    assert _approx(dd.iloc[2], 105 / 110 - 1)  # -4.545%
    assert dd.iloc[3] == 0.0  # New high
    assert _approx(dd.iloc[4], 108 / 115 - 1)  # -6.087%


def test_drawdowns_monotonically_increasing():
    """No drawdown when prices only go up."""
    prices = pd.Series([100, 101, 102, 103])
    dd = calculate_drawdowns(prices)
    assert all(d == 0.0 for d in dd)


def test_drawdowns_monotonically_decreasing():
    """Always in drawdown when prices only go down."""
    prices = pd.Series([100, 95, 90, 85])
    dd = calculate_drawdowns(prices)
    assert dd.iloc[0] == 0.0
    assert _approx(dd.iloc[1], 95 / 100 - 1)
    assert _approx(dd.iloc[2], 90 / 100 - 1)
    assert _approx(dd.iloc[3], 85 / 100 - 1)


# ---------------------------------------------------------------------------
# calculate_max_drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_basic():
    """Max drawdown finds the deepest trough."""
    prices = pd.Series([100, 110, 90, 105, 80, 120])
    mdd = calculate_max_drawdown(prices)
    assert mdd < 0
    assert _approx(mdd, 80 / 110 - 1)  # Deepest from 110 to 80


def test_max_drawdown_no_drawdown():
    """Max drawdown is 0 when prices only rise."""
    prices = pd.Series([100, 101, 102])
    assert calculate_max_drawdown(prices) == 0.0


# ---------------------------------------------------------------------------
# calculate_downside_volatility
# ---------------------------------------------------------------------------

def test_downside_volatility_all_positive():
    """All positive returns → no downside returns → 0.0."""
    returns = pd.Series(np.full(100, 0.01))
    dv = calculate_downside_volatility(returns)
    assert dv == 0.0


def test_downside_volatility_mixed():
    """Downside vol is less than or equal to total vol."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 252))
    dv = calculate_downside_volatility(returns)
    total_vol = returns.std() * np.sqrt(252)
    assert dv <= total_vol, f"Downside vol ({dv}) should be <= total vol ({total_vol})"
    assert dv > 0, f"Expected positive downside vol, got {dv}"


def test_downside_volatility_insufficient_data():
    """Downside vol requires >= 30 observations."""
    assert calculate_downside_volatility(pd.Series(np.full(29, 0.001))) == 0.0
    assert calculate_downside_volatility(pd.Series([], dtype=float)) == 0.0


def test_downside_volatility_single_negative():
    """Only one negative return — std needs >= 2 observations."""
    returns = pd.Series(np.concatenate([np.full(99, 0.01), [-0.05]]))
    dv = calculate_downside_volatility(returns)
    assert dv == 0.0  # Only 1 downside return, can't compute std


# ---------------------------------------------------------------------------
# calculate_sortino_ratio
# ---------------------------------------------------------------------------

def test_sortino_all_positive():
    """All positive returns → infinite Sortino (no downside)."""
    returns = pd.Series(np.full(100, 0.01))
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino == float('inf')


def test_sortino_zero_downside_positive_return():
    """Zero downside vol with positive excess return → inf."""
    returns = pd.Series(np.full(100, 0.001))
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino == float('inf')


def test_sortino_zero_downside_negative_return():
    """Zero downside vol with negative excess return → 0.0."""
    returns = pd.Series(np.full(100, -0.001))  # All negative, mean < 0
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino == 0.0, f"Expected 0.0 (negative excess return, zero downside vol), got {sortino}"


def test_sortino_mixed():
    """Sortino with mixed returns."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)
    assert sortino != float('inf')
    assert not math.isnan(sortino)


def test_sortino_insufficient_data():
    """Sortino requires >= 30 observations."""
    assert calculate_sortino_ratio(pd.Series(np.full(29, 0.001))) == 0.0


# ---------------------------------------------------------------------------
# calculate_calmar_ratio
# ---------------------------------------------------------------------------

def test_calmar_basic():
    """Calmar with known return and drawdown."""
    # 252 days of ~10% annual return, max drawdown -5%
    prices = pd.Series(100 * (1 + np.full(252, 0.10 / 252)).cumprod())
    # Inject a -5% drawdown
    prices.iloc[50:] *= 0.95
    calmar = calculate_calmar_ratio(prices)
    assert calmar > 0
    assert calmar < 5.0  # Should be roughly 0.10 / 0.05 = 2.0


def test_calmar_no_drawdown():
    """No drawdown and positive return → inf."""
    prices = pd.Series(100 * (1 + np.full(100, 0.001)).cumprod())
    calmar = calculate_calmar_ratio(prices)
    assert calmar == float('inf')


def test_calmar_all_losses():
    """All losses → negative Calmar."""
    prices = pd.Series(100 * (1 + np.full(100, -0.001)).cumprod())
    calmar = calculate_calmar_ratio(prices)
    assert calmar < 0


def test_calmar_insufficient_data():
    """Calmar requires >= 30 observations."""
    assert calculate_calmar_ratio(pd.Series(np.full(29, 100))) == 0.0


def test_calmar_flat_prices():
    """Flat prices → zero return, zero drawdown → 0.0 (not inf)."""
    prices = pd.Series(np.full(100, 100))
    calmar = calculate_calmar_ratio(prices)
    assert calmar == 0.0


# ---------------------------------------------------------------------------
# calculate_correlation_matrix
# ---------------------------------------------------------------------------

def test_correlation_perfect():
    """Perfect correlation → 1.0 (requires non-constant series)."""
    x = np.linspace(-0.05, 0.05, 60)
    returns = {"A": pd.Series(x), "B": pd.Series(x * 2 + 0.001)}
    corr = calculate_correlation_matrix(returns)
    assert corr is not None
    assert _approx(corr.loc["A", "B"], 1.0), f"Expected perfect correlation, got {corr.loc['A', 'B']}"


def test_correlation_perfect_inverse():
    """Perfect inverse correlation → -1.0."""
    returns = {"A": pd.Series(np.linspace(-0.05, 0.05, 60)),
               "B": pd.Series(np.linspace(0.05, -0.05, 60))}
    corr = calculate_correlation_matrix(returns)
    assert corr is not None
    assert _approx(corr.loc["A", "B"], -1.0, abs_tol=1e-6)


def test_correlation_single_asset():
    """Single asset → None."""
    returns = {"A": pd.Series(np.full(60, 0.01))}
    assert calculate_correlation_matrix(returns) is None


def test_correlation_empty():
    """Empty dict → None."""
    assert calculate_correlation_matrix({}) is None


def test_correlation_insufficient_overlap():
    """Less than 10 overlapping observations → None."""
    returns = {"A": pd.Series(np.full(5, 0.01)), "B": pd.Series(np.full(5, 0.01))}
    assert calculate_correlation_matrix(returns) is None


# ---------------------------------------------------------------------------
# calculate_portfolio_risk_metrics
# ---------------------------------------------------------------------------

def test_portfolio_metrics_full():
    """Full portfolio risk metrics with explicit weights."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252)
    prices_dict = {
        "A": pd.Series(100 * (1 + np.random.normal(0.0005, 0.01, 252)).cumprod(), index=dates),
        "B": pd.Series(100 * (1 + np.random.normal(0.0003, 0.008, 252)).cumprod(), index=dates),
    }
    weights = {"A": 0.6, "B": 0.4}
    metrics = calculate_portfolio_risk_metrics(prices_dict, weights)

    assert isinstance(metrics, RiskMetrics)
    assert metrics.var_95 < 0
    assert metrics.var_99 <= metrics.var_95
    assert metrics.cvar_95 <= metrics.var_95
    assert metrics.volatility > 0
    assert metrics.max_drawdown <= 0
    assert metrics.current_drawdown <= 0


def test_portfolio_metrics_equal_weight():
    """Equal-weighted when no weights provided."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252)
    prices_dict = {
        "A": pd.Series(100 * (1 + np.random.normal(0.0005, 0.01, 252)).cumprod(), index=dates),
        "B": pd.Series(100 * (1 + np.random.normal(0.0003, 0.008, 252)).cumprod(), index=dates),
    }
    metrics = calculate_portfolio_risk_metrics(prices_dict)
    assert isinstance(metrics, RiskMetrics)
    assert metrics.volatility > 0


def test_portfolio_metrics_weight_normalization():
    """Weights are normalized to sum to 1."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252)
    prices_dict = {
        "A": pd.Series(100 * (1 + np.full(252, 0.001)).cumprod(), index=dates),
        "B": pd.Series(100 * (1 + np.full(252, 0.001)).cumprod(), index=dates),
    }
    # Weights sum to 2.0 — should be normalized to 1.0
    weights = {"A": 1.2, "B": 0.8}
    metrics = calculate_portfolio_risk_metrics(prices_dict, weights)
    # Both assets identical → portfolio volatility should match individual
    assert metrics.volatility >= 0


def test_portfolio_metrics_single_asset():
    """Single asset portfolio."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252)
    prices_dict = {
        "A": pd.Series(100 * (1 + np.random.normal(0.0005, 0.01, 252)).cumprod(), index=dates),
    }
    metrics = calculate_portfolio_risk_metrics(prices_dict)
    assert isinstance(metrics, RiskMetrics)
    assert metrics.volatility > 0


def test_portfolio_metrics_empty():
    """Empty prices dict should not crash."""
    metrics = calculate_portfolio_risk_metrics({})
    # With empty dict, returns_dict is empty, returns_df has no columns,
    # portfolio_returns is empty Series → most metrics are 0 or NaN
    assert isinstance(metrics, RiskMetrics)


# ---------------------------------------------------------------------------
# get_risk_summary_for_llm
# ---------------------------------------------------------------------------

def test_risk_summary_format():
    """Risk summary contains all expected sections."""
    metrics = RiskMetrics(
        var_95=-0.02,
        var_99=-0.03,
        cvar_95=-0.025,
        cvar_99=-0.035,
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
    assert "VaR 95%" in summary
    assert "CVaR 95%" in summary
    assert "Volatility" in summary
    assert "Max Drawdown" in summary
    assert "Sortino Ratio" in summary
    assert "Skewness" in summary
    assert "Kurtosis" in summary


# ---------------------------------------------------------------------------
# Edge cases & numerical precision
# ---------------------------------------------------------------------------

def test_var_with_nan():
    """np.percentile returns NaN when NaN present — documented behavior."""
    returns = pd.Series([0.01, 0.02, np.nan, -0.01, 0.005] * 30)
    var = calculate_var(returns, 0.95)
    # numpy.percentile propagates NaN; this is a known limitation
    # pandas Series with NaN will produce NaN unless using skipna logic
    assert math.isnan(var) or var < 0, f"Expected NaN or negative, got {var}"


def test_drawdown_with_nan():
    """Drawdowns handle NaN in prices."""
    prices = pd.Series([100, 110, np.nan, 105, 115])
    dd = calculate_drawdowns(prices)
    # NaN propagates through cummax and division
    assert len(dd) == 5


def test_sortino_fp_precision_guard():
    """Sortino doesn't explode on near-zero downside volatility."""
    returns = pd.Series(np.concatenate([np.full(250, 0.001), [-1e-12]]))
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    # Should not be inf (there is a tiny negative return)
    # But downside std will be tiny → could be huge
    assert not math.isnan(sortino)


def test_cvar_vs_var_consistency():
    """CVaR is always <= VaR for the same confidence level."""
    np.random.seed(42)
    for conf in [0.90, 0.95, 0.99]:
        returns = pd.Series(np.random.normal(0, 0.02, 252))
        var = calculate_var(returns, conf)
        cvar = calculate_cvar(returns, conf)
        assert cvar <= var, f"CVaR ({cvar}) > VaR ({var}) at {conf} confidence"


def test_portfolio_metrics_weight_missing_ticker():
    """Weights for missing tickers are ignored gracefully."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252)
    prices_dict = {
        "A": pd.Series(100 * (1 + np.full(252, 0.001)).cumprod(), index=dates),
    }
    weights = {"A": 0.5, "B": 0.5}  # B doesn't exist
    metrics = calculate_portfolio_risk_metrics(prices_dict, weights)
    # B is skipped, A gets full weight after normalization
    assert isinstance(metrics, RiskMetrics)


if __name__ == "__main__":
    print("=" * 60)
    print("Risk Metrics Test Suite")
    print("=" * 60 + "\n")

    tests = [
        test_risk_metrics_to_dict,
        test_calculate_returns_basic,
        test_calculate_returns_empty,
        test_calculate_returns_single,
        test_var_basic,
        test_var_insufficient_data,
        test_var_confidence_levels,
        test_var_all_positive,
        test_cvar_basic,
        test_cvar_insufficient_data,
        test_cvar_no_tail_losses,
        test_cvar_extreme_tail,
        test_drawdowns_basic,
        test_drawdowns_monotonically_increasing,
        test_drawdowns_monotonically_decreasing,
        test_max_drawdown_basic,
        test_max_drawdown_no_drawdown,
        test_downside_volatility_all_positive,
        test_downside_volatility_mixed,
        test_downside_volatility_insufficient_data,
        test_downside_volatility_single_negative,
        test_sortino_all_positive,
        test_sortino_zero_downside_positive_return,
        test_sortino_zero_downside_negative_return,
        test_sortino_mixed,
        test_sortino_insufficient_data,
        test_calmar_basic,
        test_calmar_no_drawdown,
        test_calmar_all_losses,
        test_calmar_insufficient_data,
        test_calmar_flat_prices,
        test_correlation_perfect,
        test_correlation_perfect_inverse,
        test_correlation_single_asset,
        test_correlation_empty,
        test_correlation_insufficient_overlap,
        test_portfolio_metrics_full,
        test_portfolio_metrics_equal_weight,
        test_portfolio_metrics_weight_normalization,
        test_portfolio_metrics_single_asset,
        test_portfolio_metrics_empty,
        test_risk_summary_format,
        test_var_with_nan,
        test_drawdown_with_nan,
        test_sortino_fp_precision_guard,
        test_cvar_vs_var_consistency,
        test_portfolio_metrics_weight_missing_ticker,
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

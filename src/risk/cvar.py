"""
Conditional Value at Risk (CVaR) calculation module.

CVaR (also known as Expected Shortfall) measures the expected loss
given that a loss exceeds the Value at Risk (VaR) threshold.

Formula: CVaR_α = E[X | X ≤ VaR_α]

Where:
- α is the confidence level (typically 0.95 or 0.99)
- VaR_α is the α-quantile of the loss distribution
- CVaR_α is the mean of losses worse than VaR_α
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CVaRResult:
    """Result of CVaR calculation."""
    cvar_95: float  # CVaR at 95% confidence
    cvar_99: float  # CVaR at 99% confidence
    var_95: float   # VaR at 95% confidence
    var_99: float   # VaR at 99% confidence
    worst_case: float  # Worst historical return
    expected_shortfall_pct: float  # Expected shortfall as percentage


def _has_non_finite(arr: np.ndarray) -> bool:
    """Return True if the array contains any NaN or infinite value."""
    return not np.isfinite(arr).all()


def calculate_cvar(
    returns: np.ndarray,
    confidence_levels: List[float] = [0.95, 0.99]
) -> Dict[float, float]:
    """
    Calculate CVaR (Conditional Value at Risk) for given returns.
    
    Args:
        returns: Array of returns (can be negative for losses)
        confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
    
    Returns:
        Dictionary mapping confidence level to CVaR value
    """
    if len(returns) == 0 or _has_non_finite(returns):
        return {level: 0.0 for level in confidence_levels}
    
    # Convert to losses (positive values are losses)
    losses = -returns
    
    results = {}
    for alpha in confidence_levels:
        # Calculate VaR (quantile)
        var = np.percentile(losses, alpha * 100)
        
        # Calculate CVaR: mean of losses exceeding VaR
        exceedances = losses[losses >= var]
        if len(exceedances) > 0:
            cvar = np.mean(exceedances)
        else:
            cvar = var  # Fallback to VaR if no exceedances
        
        results[alpha] = float(cvar)
    
    return results


def _zero_cvar_result() -> CVaRResult:
    """Return a zeroed CVaRResult for degenerate inputs."""
    return CVaRResult(
        cvar_95=0.0,
        cvar_99=0.0,
        var_95=0.0,
        var_99=0.0,
        worst_case=0.0,
        expected_shortfall_pct=0.0,
    )


def calculate_portfolio_cvar(
    position_returns: Dict[str, np.ndarray],
    weights: Dict[str, float],
    confidence_levels: List[float] = [0.95, 0.99]
) -> CVaRResult:
    """
    Calculate CVaR for a portfolio given individual position returns.
    
    Args:
        position_returns: Dict mapping ticker to array of returns
        weights: Dict mapping ticker to portfolio weight
        confidence_levels: Confidence levels for CVaR calculation
    
    Returns:
        CVaRResult with various risk metrics
    """
    # Guard degenerate inputs so callers always receive a well-formed result.
    if not position_returns:
        return _zero_cvar_result()
    
    # Reject any position return series containing non-finite values so that
    # NaN/Inf cannot propagate into the portfolio aggregation.
    if any(_has_non_finite(r) for r in position_returns.values()):
        return _zero_cvar_result()
    
    # Normalize weights so they represent actual portfolio fractions.
    total_weight = sum(weights.values())
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        return _zero_cvar_result()
    
    normalized_weights = {ticker: weight / total_weight for ticker, weight in weights.items()}
    
    # Align returns (same length)
    min_len = min(len(r) for r in position_returns.values())
    if min_len == 0:
        return _zero_cvar_result()
    
    # Calculate portfolio returns
    portfolio_returns = np.zeros(min_len)
    for ticker, returns in position_returns.items():
        weight = normalized_weights.get(ticker, 0.0)
        if weight == 0.0:
            continue
        portfolio_returns += returns[-min_len:] * weight
    
    # Calculate CVaR at different levels
    cvar_results = calculate_cvar(portfolio_returns, confidence_levels)
    var_results = {level: float(np.percentile(-portfolio_returns, level * 100)) 
                   for level in confidence_levels}
    
    return CVaRResult(
        cvar_95=cvar_results.get(0.95, 0.0),
        cvar_99=cvar_results.get(0.99, 0.0),
        var_95=var_results.get(0.95, 0.0),
        var_99=var_results.get(0.99, 0.0),
        worst_case=float(np.min(portfolio_returns)),
        expected_shortfall_pct=cvar_results.get(0.95, 0.0) * 100
    )


def calculate_drawdown_cvar(
    equity_curve: np.ndarray,
    window: int = 20,
    confidence: float = 0.95
) -> float:
    """
    Calculate CVaR specifically for drawdown periods.
    
    This measures the expected severity of drawdowns given that
    a drawdown exceeds the VaR threshold.
    
    Args:
        equity_curve: Array of portfolio values over time
        window: Rolling window for drawdown calculation
        confidence: Confidence level for CVaR
    
    Returns:
        CVaR of drawdowns
    """
    if len(equity_curve) == 0 or _has_non_finite(equity_curve):
        return 0.0
    
    # Calculate rolling maximum (peak)
    rolling_max = pd.Series(equity_curve).rolling(window=window, min_periods=1).max()
    
    # Calculate drawdowns
    drawdowns = (equity_curve - rolling_max) / rolling_max
    
    # Filter to negative values (actual drawdowns)
    drawdown_returns = drawdowns[drawdowns < 0].values
    
    if len(drawdown_returns) == 0:
        return 0.0
    
    # Calculate CVaR on drawdowns
    cvar_dict = calculate_cvar(drawdown_returns, [confidence])
    return cvar_dict.get(confidence, 0.0)


def tail_risk_analysis(
    returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Comprehensive tail risk analysis.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Optional benchmark returns for comparison
    
    Returns:
        Dictionary with tail risk metrics
    """
    if len(returns) == 0 or _has_non_finite(returns):
        return {}
    
    metrics = {}
    
    # CVaR
    cvar_95 = calculate_cvar(returns, [0.95])[0.95]
    metrics['cvar_95'] = cvar_95
    metrics['cvar_95_pct'] = cvar_95 * 100
    
    # VaR
    var_95 = np.percentile(-returns, 95)
    metrics['var_95'] = var_95
    metrics['var_95_pct'] = var_95 * 100
    
    # Skewness (asymmetry of returns)
    if len(returns) >= 3:
        metrics['skewness'] = float(pd.Series(returns).skew())
    else:
        metrics['skewness'] = 0.0
    
    # Kurtosis (fat tails)
    if len(returns) >= 4:
        metrics['kurtosis'] = float(pd.Series(returns).kurtosis())
    else:
        metrics['kurtosis'] = 0.0
    
    # Maximum drawdown
    cumulative = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - rolling_max) / rolling_max
    metrics['max_drawdown'] = float(np.min(drawdowns))
    
    # Sortino ratio (downside risk adjusted return)
    # Use sample standard deviation (ddof=1) to match performance_metrics.py.
    # A single downside observation cannot estimate a sample std, so we
    # require at least two observations and guard numerical noise.
    downside_returns = returns[returns < 0]
    if len(downside_returns) >= 2:
        downside_std = np.std(downside_returns, ddof=1)
        if abs(downside_std) > 1e-15 and not np.isnan(downside_std):
            metrics['sortino_ratio'] = float(np.mean(returns) / downside_std * np.sqrt(252))
    
    # Compare to benchmark if provided
    if benchmark_returns is not None and len(benchmark_returns) > 0 and len(returns) > 0:
        # Align lengths (same convention as performance_metrics.py) so mixed
        # market calendars do not silently discard the comparison.
        min_len = min(len(returns), len(benchmark_returns))
        aligned_returns = returns[-min_len:]
        aligned_benchmark = benchmark_returns[-min_len:]
        diff = aligned_returns - aligned_benchmark
        
        # Need at least two observations to estimate a sample tracking error.
        if len(diff) >= 2:
            tracking_error = np.std(diff, ddof=1) * np.sqrt(252)
            if abs(tracking_error) > 1e-15 and not np.isnan(tracking_error):
                metrics['tracking_error'] = float(tracking_error)
                mean_active = np.mean(diff) * 252
                metrics['information_ratio'] = float(mean_active / tracking_error)
    
    return metrics


def format_risk_report(result: CVaRResult) -> str:
    """Format CVaR result as a readable report."""
    return f"""
CVaR Risk Report
================
Value at Risk (95%):     {result.var_95:.2%}
Value at Risk (99%):     {result.var_99:.2%}
Expected Shortfall (95%): {result.cvar_95:.2%}
Expected Shortfall (99%): {result.cvar_99:.2%}
Worst Case:              {result.worst_case:.2%}

Interpretation:
- With 95% confidence, losses won't exceed {result.var_95:.2%}
- If losses exceed {result.var_95:.2%}, expect to lose {result.cvar_95:.2%} on average
- In worst historical case: {result.worst_case:.2%}
"""

"""
Portfolio Performance Metrics Module.

Implements key performance metrics for portfolio evaluation:
- Sharpe Ratio: Risk-adjusted return
- Beta: Systematic risk relative to benchmark
- Alpha: Excess return vs benchmark
- Treynor Ratio: Return per unit of systematic risk
- Calmar Ratio: Return per unit of max drawdown
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Complete set of portfolio performance metrics."""
    # Return metrics
    total_return: float
    annualized_return: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    treynor_ratio: Optional[float]  # None if beta calculation fails
    calmar_ratio: float
    
    # Risk metrics
    volatility: float
    beta: Optional[float]  # None if benchmark data unavailable
    alpha: Optional[float]  # Annualized
    
    # Drawdown
    max_drawdown: float
    
    # Information ratio
    information_ratio: Optional[float]
    tracking_error: Optional[float]


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    annualized: bool = True
) -> float:
    """
    Calculate Sharpe Ratio.
    
    Formula: (E[R_p] - R_f) / σ_p
    
    Where:
    - E[R_p] = Expected portfolio return
    - R_f = Risk-free rate
    - σ_p = Portfolio standard deviation
    
    Args:
        returns: Array of daily returns
        risk_free_rate: Annual risk-free rate (default 2%)
        annualized: Whether to annualize the result
    
    Returns:
        Sharpe ratio (higher is better, >1 is good, >2 is very good)
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert annual risk-free to daily
    daily_rf = risk_free_rate / 252
    
    excess_returns = returns - daily_rf
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    sharpe = mean_excess / std_excess
    
    if annualized:
        sharpe = sharpe * np.sqrt(252)
    
    return float(sharpe)


def calculate_beta_alpha(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free_rate: float = 0.02
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate Beta and Alpha relative to a benchmark.
    
    Beta formula: Cov(R_p, R_b) / Var(R_b)
    - Beta = 1: Same risk as market
    - Beta < 1: Less volatile than market
    - Beta > 1: More volatile than market
    
    Alpha formula: R_p - (R_f + β × (R_b - R_f))
    - Positive alpha: Outperformance vs CAPM prediction
    - Negative alpha: Underperformance
    
    Args:
        portfolio_returns: Array of portfolio daily returns
        benchmark_returns: Array of benchmark daily returns
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Tuple of (beta, alpha_annualized) or (None, None) if calculation fails
    """
    if len(portfolio_returns) < 30 or len(benchmark_returns) < 30:
        return None, None
    
    if len(portfolio_returns) != len(benchmark_returns):
        # Align lengths
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]
    
    # Calculate covariance and variance
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    
    if benchmark_variance == 0:
        return None, None
    
    beta = covariance / benchmark_variance
    
    # Calculate alpha (Jensen's alpha)
    daily_rf = risk_free_rate / 252
    portfolio_mean = np.mean(portfolio_returns)
    benchmark_mean = np.mean(benchmark_returns)
    
    alpha_daily = portfolio_mean - (daily_rf + beta * (benchmark_mean - daily_rf))
    alpha_annual = alpha_daily * 252
    
    return float(beta), float(alpha_annual)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    annualized: bool = True
) -> float:
    """
    Calculate Sortino Ratio (using downside deviation only).
    
    Formula: (E[R_p] - R_f) / σ_d
    
    Where σ_d is the standard deviation of negative returns only.
    
    Args:
        returns: Array of daily returns
        risk_free_rate: Annual risk-free rate
        annualized: Whether to annualize
    
    Returns:
        Sortino ratio (higher is better)
    """
    if len(returns) < 2:
        return 0.0
    
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    mean_excess = np.mean(excess_returns)
    
    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf') if mean_excess > 0 else 0.0
    
    downside_std = np.std(downside_returns, ddof=1)
    
    if downside_std == 0:
        return 0.0
    
    sortino = mean_excess / downside_std
    
    if annualized:
        sortino = sortino * np.sqrt(252)
    
    return float(sortino)


def calculate_calmar_ratio(
    returns: np.ndarray,
    max_drawdown: Optional[float] = None
) -> float:
    """
    Calculate Calmar Ratio.
    
    Formula: Annualized Return / |Max Drawdown|
    
    Args:
        returns: Array of daily returns
        max_drawdown: Pre-calculated max drawdown (negative value)
    
    Returns:
        Calmar ratio (higher is better, >0.5 is good, >1 is excellent)
    """
    if len(returns) < 2:
        return 0.0
    
    # Annualized return
    total_return = np.prod(1 + returns) - 1
    n_days = len(returns)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    
    # Max drawdown
    if max_drawdown is None:
        cumulative = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
    
    if max_drawdown >= 0 or max_drawdown == 0:
        return 0.0
    
    calmar = annualized_return / abs(max_drawdown)
    return float(calmar)


def calculate_treynor_ratio(
    returns: np.ndarray,
    beta: float,
    risk_free_rate: float = 0.02
) -> Optional[float]:
    """
    Calculate Treynor Ratio.
    
    Formula: (E[R_p] - R_f) / β
    
    Return per unit of systematic risk.
    
    Args:
        returns: Array of daily returns
        beta: Portfolio beta (must be pre-calculated)
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Treynor ratio or None if beta is invalid
    """
    if len(returns) < 2 or beta is None or beta == 0:
        return None
    
    daily_rf = risk_free_rate / 252
    excess_return = np.mean(returns) - daily_rf
    
    treynor = (excess_return * 252) / beta
    return float(treynor)


def calculate_information_ratio(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate Information Ratio and Tracking Error.
    
    Information Ratio = Active Return / Tracking Error
    
    Args:
        portfolio_returns: Portfolio daily returns
        benchmark_returns: Benchmark daily returns
    
    Returns:
        Tuple of (information_ratio, tracking_error)
    """
    if len(portfolio_returns) < 30 or len(benchmark_returns) < 30:
        return None, None
    
    min_len = min(len(portfolio_returns), len(benchmark_returns))
    portfolio_returns = portfolio_returns[-min_len:]
    benchmark_returns = benchmark_returns[-min_len:]
    
    # Active returns
    active_returns = portfolio_returns - benchmark_returns
    
    # Tracking error (annualized)
    tracking_error = np.std(active_returns, ddof=1) * np.sqrt(252)
    
    if tracking_error == 0:
        return None, None
    
    # Information ratio
    mean_active = np.mean(active_returns) * 252
    information_ratio = mean_active / tracking_error
    
    return float(information_ratio), float(tracking_error)


def calculate_all_metrics(
    returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.02
) -> PerformanceMetrics:
    """
    Calculate all portfolio performance metrics.
    
    Args:
        returns: Array of portfolio daily returns
        benchmark_returns: Optional benchmark returns for relative metrics
        risk_free_rate: Annual risk-free rate
    
    Returns:
        PerformanceMetrics dataclass with all calculated metrics
    """
    if len(returns) < 2:
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            treynor_ratio=None,
            calmar_ratio=0.0,
            volatility=0.0,
            beta=None,
            alpha=None,
            max_drawdown=0.0,
            information_ratio=None,
            tracking_error=None
        )
    
    # Basic return metrics
    total_return = np.prod(1 + returns) - 1
    n_days = len(returns)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    
    # Volatility
    volatility = np.std(returns, ddof=1) * np.sqrt(252)
    
    # Risk-adjusted returns
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    sortino = calculate_sortino_ratio(returns, risk_free_rate)
    
    # Max drawdown
    cumulative = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns)
    
    # Calmar
    calmar = calculate_calmar_ratio(returns, max_drawdown)
    
    # Beta, Alpha, Treynor (require benchmark)
    beta = None
    alpha = None
    treynor = None
    info_ratio = None
    tracking_err = None
    
    if benchmark_returns is not None and len(benchmark_returns) >= 30:
        beta, alpha = calculate_beta_alpha(returns, benchmark_returns, risk_free_rate)
        
        if beta is not None:
            treynor = calculate_treynor_ratio(returns, beta, risk_free_rate)
        
        info_ratio, tracking_err = calculate_information_ratio(returns, benchmark_returns)
    
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        treynor_ratio=treynor,
        calmar_ratio=calmar,
        volatility=volatility,
        beta=beta,
        alpha=alpha,
        max_drawdown=max_drawdown,
        information_ratio=info_ratio,
        tracking_error=tracking_err
    )


def format_metrics_report(metrics: PerformanceMetrics, benchmark_name: str = "Benchmark") -> str:
    """
    Format performance metrics as a readable report.
    
    Args:
        metrics: PerformanceMetrics object
        benchmark_name: Name of the benchmark for display
    
    Returns:
        Formatted report string
    """
    lines = [
        "=" * 50,
        "PORTFOLIO PERFORMANCE METRICS",
        "=" * 50,
        "",
        "RETURN METRICS",
        f"  Total Return:          {metrics.total_return:>8.2%}",
        f"  Annualized Return:     {metrics.annualized_return:>8.2%}",
        "",
        "RISK METRICS",
        f"  Volatility (Ann):      {metrics.volatility:>8.2%}",
        f"  Max Drawdown:          {metrics.max_drawdown:>8.2%}",
        "",
        "RISK-ADJUSTED RETURNS",
        f"  Sharpe Ratio:          {metrics.sharpe_ratio:>8.2f}",
        f"  Sortino Ratio:         {metrics.sortino_ratio:>8.2f}",
        f"  Calmar Ratio:          {metrics.calmar_ratio:>8.2f}",
    ]
    
    if metrics.treynor_ratio is not None:
        lines.append(f"  Treynor Ratio:         {metrics.treynor_ratio:>8.2f}")
    
    lines.extend([
        "",
        f"BENCHMARK RELATIVE ({benchmark_name})",
    ])
    
    if metrics.beta is not None:
        lines.append(f"  Beta:                  {metrics.beta:>8.2f}")
    
    if metrics.alpha is not None:
        lines.append(f"  Alpha (Annual):        {metrics.alpha:>8.2%}")
    
    if metrics.information_ratio is not None:
        lines.append(f"  Information Ratio:     {metrics.information_ratio:>8.2f}")
    
    if metrics.tracking_error is not None:
        lines.append(f"  Tracking Error:        {metrics.tracking_error:>8.2%}")
    
    lines.extend([
        "",
        "=" * 50,
        "INTERPRETATION",
        "=" * 50,
    ])
    
    # Sharpe interpretation
    if metrics.sharpe_ratio < 0.5:
        sharpe_comment = "Poor risk-adjusted return"
    elif metrics.sharpe_ratio < 1.0:
        sharpe_comment = "Acceptable risk-adjusted return"
    elif metrics.sharpe_ratio < 2.0:
        sharpe_comment = "Good risk-adjusted return"
    else:
        sharpe_comment = "Excellent risk-adjusted return"
    
    lines.append(f"• Sharpe: {sharpe_comment}")
    
    # Beta interpretation
    if metrics.beta is not None:
        if metrics.beta < 0.8:
            beta_comment = "Lower risk than market"
        elif metrics.beta < 1.2:
            beta_comment = "Market-like risk"
        else:
            beta_comment = "Higher risk than market"
        lines.append(f"• Beta: {beta_comment}")
    
    # Alpha interpretation
    if metrics.alpha is not None:
        if metrics.alpha > 0.02:
            alpha_comment = "Outperforming benchmark"
        elif metrics.alpha < -0.02:
            alpha_comment = "Underperforming benchmark"
        else:
            alpha_comment = "Tracking benchmark"
        lines.append(f"• Alpha: {alpha_comment}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)

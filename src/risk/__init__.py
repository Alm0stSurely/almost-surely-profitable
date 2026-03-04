"""
Risk management module for almost-surely-profitable.
Calculates CVaR, VaR, and other risk metrics based on Behavioral_RL concepts.
"""

from .metrics import (
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

from .performance_metrics import (
    PerformanceMetrics,
    calculate_sharpe_ratio,
    calculate_beta_alpha,
    calculate_sortino_ratio as calc_sortino_perf,
    calculate_calmar_ratio as calc_calmar_perf,
    calculate_treynor_ratio,
    calculate_information_ratio,
    calculate_all_metrics,
    format_metrics_report,
)

__all__ = [
    "RiskMetrics",
    "PerformanceMetrics",
    "calculate_returns",
    "calculate_var",
    "calculate_cvar",
    "calculate_drawdowns",
    "calculate_max_drawdown",
    "calculate_downside_volatility",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_correlation_matrix",
    "calculate_portfolio_risk_metrics",
    "get_risk_summary_for_llm",
    "calculate_sharpe_ratio",
    "calculate_beta_alpha",
    "calculate_treynor_ratio",
    "calculate_information_ratio",
    "calculate_all_metrics",
    "format_metrics_report",
]

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

__all__ = [
    "RiskMetrics",
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
]

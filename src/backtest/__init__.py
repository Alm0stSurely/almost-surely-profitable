"""
Backtesting framework for the trading agent.
"""

from .backtest import BacktestEngine
from .triple_barrier import (
    BarrierConfig,
    BarrierType,
    TripleBarrierLabel,
    label_events,
    get_events_from_signals,
    analyze_barrier_distribution,
    format_barrier_report,
    calculate_volatility,
    get_barrier_levels
)

__all__ = [
    'BacktestEngine',
    'BarrierConfig',
    'BarrierType',
    'TripleBarrierLabel',
    'label_events',
    'get_events_from_signals',
    'analyze_barrier_distribution',
    'format_barrier_report',
    'calculate_volatility',
    'get_barrier_levels'
]

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
from .cpcv import (
    PurgedKFold,
    CombinatorialPurgedCV,
    apply_purged_cv,
    calculate_purged_cv_score
)
from .meta_labeling import (
    SignalType,
    PrimarySignal,
    MetaLabel,
    MetaLabelingConfig,
    MetaLabeler,
    create_meta_labels_from_triple_barrier,
    apply_meta_labeling
)
from .deflated_sharpe import (
    SharpeMetrics,
    DeflatedSharpeRatio,
    probabilistic_sharpe_ratio,
    minimum_track_record_length
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
    'get_barrier_levels',
    'PurgedKFold',
    'CombinatorialPurgedCV',
    'apply_purged_cv',
    'calculate_purged_cv_score',
    'SignalType',
    'PrimarySignal',
    'MetaLabel',
    'MetaLabelingConfig',
    'MetaLabeler',
    'create_meta_labels_from_triple_barrier',
    'apply_meta_labeling',
    'SharpeMetrics',
    'DeflatedSharpeRatio',
    'probabilistic_sharpe_ratio',
    'minimum_track_record_length'
]

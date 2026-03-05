"""
Triple-Barrier Method for Backtesting.

Implementation of Marcos Lopez de Prada's Triple-Barrier method for labeling
and evaluating trading strategies. More realistic than fixed-time horizons.

A triple barrier consists of:
1. Upper barrier (profit taking level)
2. Lower barrier (stop loss level)
3. Vertical barrier (maximum holding period)

The first barrier touched determines the label.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class BarrierType(Enum):
    """Types of barriers that can be touched first."""
    UPPER = "upper"      # Profit taking
    LOWER = "lower"      # Stop loss
    VERTICAL = "vertical"  # Time expiration


@dataclass
class TripleBarrierLabel:
    """Result of a triple barrier event."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    barrier_type: BarrierType
    return_pct: float
    label: int  # 1 for upper, -1 for lower, 0 for vertical
    holding_periods: int


@dataclass
class BarrierConfig:
    """Configuration for triple barriers."""
    profit_take_std: float = 2.0    # Upper barrier = entry_price * (1 + pt_std * daily_vol)
    stop_loss_std: float = 2.0      # Lower barrier = entry_price * (1 - sl_std * daily_vol)
    max_holding: int = 20           # Maximum bars to hold (vertical barrier)
    trailing_stop: bool = False     # Whether to use trailing stop
    
    @classmethod
    def conservative(cls) -> "BarrierConfig":
        """Conservative config: tight stops, short holding."""
        return cls(profit_take_std=1.5, stop_loss_std=1.0, max_holding=10)
    
    @classmethod
    def aggressive(cls) -> "BarrierConfig":
        """Aggressive config: wide stops, long holding."""
        return cls(profit_take_std=3.0, stop_loss_std=2.5, max_holding=30)
    
    @classmethod
    def symmetric(cls) -> "BarrierConfig":
        """Symmetric config: equal profit/loss levels."""
        return cls(profit_take_std=2.0, stop_loss_std=2.0, max_holding=20)


def calculate_volatility(
    prices: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        prices: Price series
        window: Rolling window size
    
    Returns:
        Rolling volatility series
    """
    returns = prices.pct_change().dropna()
    return returns.rolling(window=window).std()


def get_barrier_levels(
    entry_price: float,
    daily_vol: float,
    config: BarrierConfig
) -> Tuple[float, float]:
    """
    Calculate upper and lower barrier levels.
    
    Args:
        entry_price: Price at entry
        daily_vol: Daily volatility (standard deviation)
        config: Barrier configuration
    
    Returns:
        Tuple of (upper_barrier, lower_barrier)
    """
    upper = entry_price * (1 + config.profit_take_std * daily_vol)
    lower = entry_price * (1 - config.stop_loss_std * daily_vol)
    return upper, lower


def apply_triple_barrier(
    price_series: pd.Series,
    entry_idx: int,
    daily_vol: float,
    config: BarrierConfig = None
) -> Optional[TripleBarrierLabel]:
    """
    Apply triple barrier to a single position.
    
    Args:
        price_series: Price series starting from entry
        entry_idx: Index in price_series where position is entered
        daily_vol: Daily volatility at entry
        config: Barrier configuration
    
    Returns:
        TripleBarrierLabel or None if not enough data
    """
    if config is None:
        config = BarrierConfig.symmetric()
    
    if entry_idx >= len(price_series):
        return None
    
    entry_price = price_series.iloc[entry_idx]
    entry_time = price_series.index[entry_idx]
    
    # Calculate barrier levels
    upper_barrier, lower_barrier = get_barrier_levels(entry_price, daily_vol, config)
    
    # Determine vertical barrier (max holding period)
    vertical_barrier_idx = min(entry_idx + config.max_holding, len(price_series) - 1)
    
    # Walk forward and check which barrier is touched first
    for i in range(entry_idx + 1, vertical_barrier_idx + 1):
        current_price = price_series.iloc[i]
        current_time = price_series.index[i]
        
        # Check upper barrier (profit taking)
        if current_price >= upper_barrier:
            return_pct = (current_price - entry_price) / entry_price
            return TripleBarrierLabel(
                entry_time=entry_time,
                exit_time=current_time,
                barrier_type=BarrierType.UPPER,
                return_pct=return_pct,
                label=1,
                holding_periods=i - entry_idx
            )
        
        # Check lower barrier (stop loss)
        if current_price <= lower_barrier:
            return_pct = (current_price - entry_price) / entry_price
            return TripleBarrierLabel(
                entry_time=entry_time,
                exit_time=current_time,
                barrier_type=BarrierType.LOWER,
                return_pct=return_pct,
                label=-1,
                holding_periods=i - entry_idx
            )
    
    # Vertical barrier touched (time expiration)
    exit_price = price_series.iloc[vertical_barrier_idx]
    exit_time = price_series.index[vertical_barrier_idx]
    return_pct = (exit_price - entry_price) / entry_price
    
    return TripleBarrierLabel(
        entry_time=entry_time,
        exit_time=exit_time,
        barrier_type=BarrierType.VERTICAL,
        return_pct=return_pct,
        label=0,
        holding_periods=vertical_barrier_idx - entry_idx
    )


def label_events(
    prices: pd.Series,
    event_times: List[pd.Timestamp],
    config: BarrierConfig = None,
    volatility_window: int = 20
) -> List[TripleBarrierLabel]:
    """
    Apply triple barrier labeling to multiple events.
    
    Args:
        prices: Price series
        event_times: List of entry times for events
        config: Barrier configuration
        volatility_window: Window for volatility calculation
    
    Returns:
        List of TripleBarrierLabel results
    """
    if config is None:
        config = BarrierConfig.symmetric()
    
    labels = []
    volatility = calculate_volatility(prices, window=volatility_window)
    
    for event_time in event_times:
        # Find entry index
        if event_time not in prices.index:
            continue
        
        entry_idx = prices.index.get_loc(event_time)
        if entry_idx >= len(prices) - 1:
            continue
        
        # Get volatility at entry (use minimum threshold to avoid zero vol)
        daily_vol = volatility.loc[event_time] if event_time in volatility.index else 0.01
        daily_vol = max(daily_vol, 0.005)  # Minimum 0.5% daily vol
        
        # Apply triple barrier
        label = apply_triple_barrier(prices, entry_idx, daily_vol, config)
        if label:
            labels.append(label)
    
    return labels


def get_events_from_signals(
    price_series: pd.Series,
    signals: pd.Series,
    min_hold: int = 1
) -> List[pd.Timestamp]:
    """
    Extract event times from trading signals.
    
    Args:
        price_series: Price series (for index alignment)
        signals: Signal series (1 = buy, -1 = sell, 0 = hold)
        min_hold: Minimum bars between events
    
    Returns:
        List of event timestamps
    """
    events = []
    last_event = -min_hold
    
    for i, (timestamp, signal) in enumerate(signals.items()):
        if signal != 0 and i - last_event >= min_hold:
            events.append(timestamp)
            last_event = i
    
    return events


def analyze_barrier_distribution(
    labels: List[TripleBarrierLabel]
) -> Dict:
    """
    Analyze the distribution of barrier touches.
    
    Args:
        labels: List of triple barrier labels
    
    Returns:
        Dictionary with statistics
    """
    if not labels:
        return {
            'total_events': 0,
            'upper_touches': 0,
            'lower_touches': 0,
            'vertical_touches': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'avg_holding_periods': 0
        }
    
    upper = sum(1 for l in labels if l.barrier_type == BarrierType.UPPER)
    lower = sum(1 for l in labels if l.barrier_type == BarrierType.LOWER)
    vertical = sum(1 for l in labels if l.barrier_type == BarrierType.VERTICAL)
    
    returns = [l.return_pct for l in labels]
    holding_periods = [l.holding_periods for l in labels]
    
    # Win rate: upper touches + positive vertical touches
    wins = upper + sum(1 for l in labels if l.barrier_type == BarrierType.VERTICAL and l.return_pct > 0)
    
    return {
        'total_events': len(labels),
        'upper_touches': upper,
        'lower_touches': lower,
        'vertical_touches': vertical,
        'upper_pct': upper / len(labels) * 100,
        'lower_pct': lower / len(labels) * 100,
        'vertical_pct': vertical / len(labels) * 100,
        'win_rate': wins / len(labels) * 100,
        'avg_return': np.mean(returns) * 100,
        'median_return': np.median(returns) * 100,
        'avg_holding_periods': np.mean(holding_periods),
        'total_return': np.prod([1 + r for r in returns]) - 1
    }


def format_barrier_report(stats: Dict) -> str:
    """Format barrier statistics as a readable report."""
    lines = [
        "=" * 50,
        "TRIPLE BARRIER ANALYSIS",
        "=" * 50,
        "",
        f"Total Events:           {stats['total_events']}",
        "",
        "BARRIER DISTRIBUTION",
        f"  Upper (Profit):       {stats['upper_touches']} ({stats['upper_pct']:.1f}%)",
        f"  Lower (Stop):         {stats['lower_touches']} ({stats['lower_pct']:.1f}%)",
        f"  Vertical (Time):      {stats['vertical_touches']} ({stats['vertical_pct']:.1f}%)",
        "",
        "PERFORMANCE",
        f"  Win Rate:             {stats['win_rate']:.1f}%",
        f"  Avg Return/Event:     {stats['avg_return']:.2f}%",
        f"  Median Return:        {stats['median_return']:.2f}%",
        f"  Avg Holding Period:   {stats['avg_holding_periods']:.1f} bars",
        f"  Total Return:         {stats['total_return']*100:.2f}%",
        "=" * 50
    ]
    return "\n".join(lines)

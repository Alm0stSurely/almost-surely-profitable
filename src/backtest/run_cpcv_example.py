#!/usr/bin/env python3
"""
Example: Using CPCV with Triple-Barrier Labels for Backtesting

This script demonstrates how to combine:
1. Triple-Barrier labeling for realistic trade outcomes
2. Combinatorial Purged Cross-Validation for robust backtesting
3. Meta-labeling for position sizing
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.triple_barrier import TripleBarrierLabeler, get_barrier_config
from backtest.cpcv import CombinatorialPurgedCV, calculate_purged_cv_score
from data.fetch_market_data import fetch_historical_data
from data.indicators import calculate_rsi, calculate_bollinger_bands


def generate_meta_labels(
    data: pd.DataFrame,
    primary_predictions: pd.Series,
    tb_labels: pd.Series,
    min_return: float = 0.001
) -> pd.Series:
    """
    Generate meta-labels for meta-labeling approach.
    
    Meta-labeling uses a secondary model to determine position sizing
    based on the primary model's predictions.
    
    Args:
        data: Price data
        primary_predictions: Primary model predictions (1=long, -1=short, 0=neutral)
        tb_labels: Triple-barrier labels (1=profit, -1=stop, 0=time)
        min_return: Minimum return threshold for positive meta-label
        
    Returns:
        Meta-labels (1=enter position, 0=skip)
    """
    meta_labels = pd.Series(0, index=primary_predictions.index)
    
    # Label as 1 if primary model was correct and return > threshold
    correct_predictions = (primary_predictions == np.sign(tb_labels)) & (tb_labels != 0)
    sufficient_return = np.abs(tb_labels) > min_return
    
    meta_labels[correct_predictions & sufficient_return] = 1
    
    return meta_labels


def calculate_strategy_returns(
    data: pd.DataFrame,
    predictions: pd.Series,
    tb_labels: pd.Series,
    position_size: float = 1.0
) -> pd.Series:
    """
    Calculate strategy returns using triple-barrier outcomes.
    
    Args:
        data: Price data with 'close' column
        predictions: Trading signals (-1, 0, 1)
        tb_labels: Triple-barrier labels for actual outcomes
        position_size: Position sizing (can be from meta-labeling)
        
    Returns:
        Strategy returns series
    """
    # Get forward returns at barrier touch
    returns = pd.Series(0.0, index=predictions.index)
    
    # Only calculate returns where we have predictions
    for idx in predictions[predictions != 0].index:
        if idx in tb_labels.index:
            # Return = prediction direction * triple-barrier outcome
            returns.loc[idx] = predictions.loc[idx] * tb_labels.loc[idx] * position_size
    
    return returns


def run_cpcv_backtest_example():
    """Run a complete CPCV backtest example."""
    
    print("=" * 80)
    print("CPCV + Triple-Barrier Backtesting Example")
    print("=" * 80)
    
    # Fetch data
    print("\n[1/5] Fetching market data...")
    tickers = ["SPY", "QQQ", "GLD"]
    data = fetch_historical_data(tickers, period="2y")
    print(f"  ✓ Fetched data for {len(data)} assets")
    
    # Use SPY as example
    spy = data["SPY"].copy()
    spy['returns'] = spy['close'].pct_change()
    spy['rsi'] = calculate_rsi(spy['close'], period=14)
    spy['bb_upper'], spy['bb_middle'], spy['bb_lower'] = calculate_bollinger_bands(spy['close'])
    spy['bb_position'] = (spy['close'] - spy['bb_lower']) / (spy['bb_upper'] - spy['bb_lower'])
    
    # Drop NaN
    spy = spy.dropna()
    
    # Generate primary model signals (mean reversion)
    print("\n[2/5] Generating primary signals...")
    signals = pd.Series(0, index=spy.index)
    signals[(spy['rsi'] < 30) & (spy['bb_position'] < 0.1)] = 1   # Oversold -> Long
    signals[(spy['rsi'] > 70) & (spy['bb_position'] > 0.9)] = -1  # Overbought -> Short
    
    n_signals = (signals != 0).sum()
    print(f"  ✓ Generated {n_signals} signals ({n_signals/len(signals)*100:.1f}% of bars)")
    
    # Apply triple-barrier labeling
    print("\n[3/5] Applying Triple-Barrier labels...")
    config = get_barrier_config('symmetric')
    labeler = TripleBarrierLabeler(config)
    
    tb_results = labeler.label_dataset(spy, 'close')
    tb_labels = tb_results['label']
    
    print(f"  ✓ Labels: Profit={sum(tb_labels==1)}, Stop={sum(tb_labels==-1)}, Time={sum(tb_labels==0)}")
    
    # Setup CPCV
    print("\n[4/5] Setting up Combinatorial Purged CV...")
    cpcv = CombinatorialPurgedCV(
        n_splits=6,
        n_test_splits=2,
        purge_gap=5,
        embargo_pct=0.02
    )
    print(f"  ✓ Will generate {cpcv.get_n_splits()} backtest paths")
    
    # Run backtest
    print("\n[5/5] Running backtest...")
    
    fold_returns = []
    fold_metrics = []
    
    for fold_idx, (train_idx, test_idx, meta) in enumerate(cpcv.split(spy)):
        # Get train/test data
        train_signals = signals.iloc[train_idx]
        test_signals = signals.iloc[test_idx]
        test_labels = tb_labels.iloc[test_idx]
        
        # Skip if no signals in test
        if (test_signals != 0).sum() == 0:
            continue
        
        # Calculate returns for this fold
        fold_ret = calculate_strategy_returns(
            spy.iloc[test_idx],
            test_signals,
            test_labels
        )
        
        # Store metrics
        if len(fold_ret) > 0:
            fold_returns.append(fold_ret)
            
            total_return = fold_ret.sum()
            sharpe = fold_ret.mean() / fold_ret.std() * np.sqrt(252) if fold_ret.std() > 0 else 0
            win_rate = (fold_ret > 0).sum() / (fold_ret != 0).sum() if (fold_ret != 0).sum() > 0 else 0
            
            fold_metrics.append({
                'fold': fold_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'n_trades': (test_signals != 0).sum(),
                'total_return': total_return,
                'sharpe': sharpe,
                'win_rate': win_rate
            })
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("Backtest Results Summary")
    print("=" * 80)
    
    if fold_metrics:
        metrics_df = pd.DataFrame(fold_metrics)
        
        print(f"\nCompleted {len(fold_metrics)} backtest paths")
        print(f"\nPer-Path Statistics:")
        print(f"  Total Return:  {metrics_df['total_return'].mean()*100:+.2f}% ± {metrics_df['total_return'].std()*100:.2f}%")
        print(f"  Sharpe Ratio:  {metrics_df['sharpe'].mean():.2f} ± {metrics_df['sharpe'].std():.2f}")
        print(f"  Win Rate:      {metrics_df['win_rate'].mean()*100:.1f}% ± {metrics_df['win_rate'].std()*100:.1f}%")
        print(f"  Avg Trades:    {metrics_df['n_trades'].mean():.0f}")
        
        print(f"\nRobustness Metrics:")
        print(f"  Min Return:    {metrics_df['total_return'].min()*100:+.2f}%")
        print(f"  Max Return:    {metrics_df['total_return'].max()*100:+.2f}%")
        print(f"  Paths > 0:     {(metrics_df['total_return'] > 0).sum()}/{len(metrics_df)}")
        
        # Calculate probability of backtest overfitting
        positive_paths = (metrics_df['total_return'] > 0).sum()
        pob = positive_paths / len(metrics_df)
        print(f"  P(Overfitting): {1-pob:.1%}")
    
    print("\n" + "=" * 80)
    print("Key Insights")
    print("=" * 80)
    print("""
1. CPCV generates multiple independent backtest paths, reducing overfitting risk
2. Triple-barrier labels provide realistic trade outcomes (not fixed horizons)
3. The combination tests strategy robustness across different market regimes
4. High variance in path returns indicates regime-dependent performance
5. Meta-labeling can be added to filter low-confidence signals
    """)
    
    return fold_metrics


if __name__ == "__main__":
    metrics = run_cpcv_backtest_example()

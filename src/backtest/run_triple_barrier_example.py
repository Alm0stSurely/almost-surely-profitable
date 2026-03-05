#!/usr/bin/env python3
"""
Example: Using Triple-Barrier Method for Backtesting.

This script demonstrates how to use the triple-barrier method
to evaluate trading strategies with more realistic labels.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetch_market_data import fetch_historical_data
from data.indicators import calculate_all_indicators
from backtest.triple_barrier import (
    BarrierConfig,
    label_events,
    get_events_from_signals,
    analyze_barrier_distribution,
    format_barrier_report
)


def generate_sample_signals(prices: pd.Series) -> pd.Series:
    """
    Generate sample trading signals based on RSI mean reversion.
    
    This is a simple example strategy:
    - Buy when RSI < 30 (oversold)
    - Sell when RSI > 70 (overbought)
    """
    from data.indicators import calculate_rsi
    
    rsi = calculate_rsi(prices, period=14)
    
    signals = pd.Series(0, index=prices.index)
    signals[rsi < 30] = 1   # Buy signal
    signals[rsi > 70] = -1  # Sell signal
    
    return signals


def run_triple_barrier_example():
    """Run a complete example of triple barrier backtesting."""
    
    print("=" * 60)
    print("TRIPLE BARRIER BACKTEST EXAMPLE")
    print("=" * 60)
    print()
    
    # Fetch historical data
    print("Fetching historical data...")
    tickers = ["SPY", "QQQ", "GLD"]
    data = fetch_historical_data(tickers, period="6mo")
    
    if not data:
        print("No data fetched. Check internet connection.")
        return
    
    print(f"✓ Fetched data for {len(data)} assets")
    print()
    
    # Test different barrier configurations
    configs = {
        "Conservative": BarrierConfig.conservative(),
        "Symmetric": BarrierConfig.symmetric(),
        "Aggressive": BarrierConfig.aggressive()
    }
    
    results = {}
    
    for ticker, df in data.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {ticker}")
        print(f"{'='*60}")
        
        prices = df['Close']
        
        # Generate signals
        signals = generate_sample_signals(prices)
        events = get_events_from_signals(prices, signals, min_hold=5)
        
        print(f"Generated {len(events)} trading signals")
        
        if len(events) < 5:
            print("Not enough signals for analysis")
            continue
        
        ticker_results = {}
        
        for config_name, config in configs.items():
            print(f"\n--- {config_name} Config ---")
            print(f"  Profit Take: {config.profit_take_std}σ")
            print(f"  Stop Loss:   {config.stop_loss_std}σ")
            print(f"  Max Hold:    {config.max_holding} bars")
            
            # Apply triple barrier
            labels = label_events(prices, events, config=config)
            
            # Analyze results
            stats = analyze_barrier_distribution(labels)
            ticker_results[config_name] = stats
            
            # Print report
            print(format_barrier_report(stats))
        
        results[ticker] = ticker_results
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("CONFIGURATION COMPARISON SUMMARY")
    print("=" * 60)
    
    for ticker, ticker_results in results.items():
        print(f"\n{ticker}:")
        print(f"{'Config':<15} {'Win Rate':>10} {'Avg Return':>12} {'Lower %':>10}")
        print("-" * 50)
        for config_name, stats in ticker_results.items():
            print(f"{config_name:<15} {stats['win_rate']:>9.1f}% {stats['avg_return']:>10.2f}% {stats['lower_pct']:>9.1f}%")
    
    print("\n" + "=" * 60)
    print("Key Insights:")
    print("=" * 60)
    print("""
1. Conservative config: Higher win rate, but smaller gains per win
2. Aggressive config: More variance, but larger gains when right
3. Lower % shows how often stop-losses are hit
4. Vertical % shows how often time expires before profit/loss

The triple barrier method provides more realistic labels than
fixed-time horizons because it accounts for:
- Early profit taking
- Stop loss exits
- Time-based position limits
    """)


if __name__ == "__main__":
    run_triple_barrier_example()

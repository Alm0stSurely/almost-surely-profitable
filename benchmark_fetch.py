#!/usr/bin/env python3
"""
Benchmark script for fetch_market_data parallelization.
Compares sequential vs parallel fetching performance.
"""

import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.fetch_market_data import fetch_historical_data, fetch_current_prices


def mock_yfinance_with_delay(delay_ms=50):
    """Create a mock yfinance.Ticker that simulates network delay."""
    def create_ticker(ticker):
        time.sleep(delay_ms / 1000.0)
        dates = pd.date_range('2024-01-01', periods=30)
        mock_df = pd.DataFrame({
            'Open': [100 + i for i in range(30)],
            'High': [101 + i for i in range(30)],
            'Low': [99 + i for i in range(30)],
            'Close': [100.5 + i for i in range(30)],
            'Volume': [1000 + i * 100 for i in range(30)]
        }, index=dates)
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_df
        return mock_ticker
    return create_ticker


def benchmark_fetch(tickers, delay_ms=50, max_workers=1):
    """Benchmark fetching with specified worker count."""
    with patch('data.fetch_market_data.yf.Ticker', side_effect=mock_yfinance_with_delay(delay_ms)):
        start = time.perf_counter()
        data = fetch_historical_data(tickers=tickers, period="30d", max_workers=max_workers)
        elapsed = time.perf_counter() - start
        return elapsed, data


def main():
    tickers = ["SPY", "QQQ", "GLD", "TLT", "MC.PA", "TTE.PA", "SAN.PA", "OR.PA",
               "AIR.PA", "SU.PA", "AI.PA", "BNP.PA", "CS.PA", "RMS.PA", "SAF.PA",
               "DSY.PA", "DG.PA", "SGO.PA", "KER.PA", "FEZ", "CAC.PA"]
    
    delay_ms = 50  # Simulate 50ms network latency per request
    
    print("=" * 70)
    print("BENCHMARK: Sequential vs Parallel Market Data Fetching")
    print("=" * 70)
    print(f"Tickers: {len(tickers)}")
    print(f"Simulated network delay: {delay_ms}ms per request")
    print()
    
    # Sequential benchmark
    print("[1/3] Running sequential fetch (max_workers=1)...")
    seq_time, seq_data = benchmark_fetch(tickers, delay_ms, max_workers=1)
    print(f"  Time: {seq_time:.3f}s")
    print(f"  Data points: {sum(len(df) for df in seq_data.values())} rows")
    print()
    
    # Parallel benchmark (4 workers)
    print("[2/3] Running parallel fetch (max_workers=4)...")
    par_time4, par_data4 = benchmark_fetch(tickers, delay_ms, max_workers=4)
    print(f"  Time: {par_time4:.3f}s")
    print(f"  Data points: {sum(len(df) for df in par_data4.values())} rows")
    print()
    
    # Parallel benchmark (8 workers)
    print("[3/3] Running parallel fetch (max_workers=8)...")
    par_time8, par_data8 = benchmark_fetch(tickers, delay_ms, max_workers=8)
    print(f"  Time: {par_time8:.3f}s")
    print(f"  Data points: {sum(len(df) for df in par_data8.values())} rows")
    print()
    
    # Results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Sequential (1 worker):  {seq_time:.3f}s")
    print(f"Parallel (4 workers):   {par_time4:.3f}s  (speedup: {seq_time / par_time4:.2f}x)")
    print(f"Parallel (8 workers):   {par_time8:.3f}s  (speedup: {seq_time / par_time8:.2f}x)")
    print()
    
    # Verify data integrity
    assert set(seq_data.keys()) == set(par_data4.keys()) == set(par_data8.keys()), "Data keys mismatch!"
    for ticker in seq_data:
        assert len(seq_data[ticker]) == len(par_data4[ticker]) == len(par_data8[ticker]), f"Row count mismatch for {ticker}"
    print("✓ Data integrity verified across all configurations")


if __name__ == "__main__":
    main()

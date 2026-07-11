"""
Benchmark: correlation matrix date alignment.

Compares the date-aligned correlation implementation against a naive
positional implementation on a synthetic dataset where two assets trade on
different calendars but are perfectly correlated on shared dates. The
positional approach produces a noisy, wrong correlation; the aligned
approach recovers the true correlation of 1.0.
"""

import time
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.indicators import calculate_correlation_matrix


def naive_correlation_matrix(data_dict, lookback=20):
    """Positional (old) implementation kept here for comparison."""
    returns_df = pd.DataFrame()
    for ticker, df in data_dict.items():
        if 'Close' in df.columns:
            returns_df[ticker] = df['Close'].pct_change()
    if len(returns_df) > lookback:
        returns_df = returns_df.tail(lookback)
    return returns_df.corr()


def make_misaligned_data(seed=42, days=60, eur_days=55):
    np.random.seed(seed)
    dates_spy = pd.date_range(start='2026-05-01', periods=days, freq='B')
    # SPY is fetched at US close (16:00 UTC)
    dates_spy_with_time = dates_spy + pd.Timedelta(hours=16)
    spy_returns = np.random.normal(loc=0.001, scale=0.02, size=len(dates_spy))
    spy_prices = 100 * np.cumprod(1 + spy_returns)
    spy_df = pd.DataFrame({'Close': spy_prices}, index=dates_spy_with_time)

    # EUR is fetched at Euronext close (14:30 UTC) on the same dates.
    # The old implementation aligns by the full timestamp, so SPY 16:00
    # and EUR 14:30 on the same calendar day are treated as different rows.
    # The new implementation normalizes to date-only, so they align correctly.
    dates_eur_with_time = dates_spy + pd.Timedelta(hours=14, minutes=30)
    eur_returns = spy_returns.copy() + np.random.normal(
        loc=0.0, scale=1e-12, size=len(spy_returns)
    )
    eur_prices = 100 * np.cumprod(1 + eur_returns)
    eur_df = pd.DataFrame({'Close': eur_prices}, index=dates_eur_with_time)

    return {'SPY': spy_df, 'EUR': eur_df}


def main():
    data = make_misaligned_data(days=60, eur_days=55)

    t0 = time.perf_counter()
    aligned = calculate_correlation_matrix(data, lookback=20)
    t_aligned = time.perf_counter() - t0

    t0 = time.perf_counter()
    positional = naive_correlation_matrix(data, lookback=20)
    t_positional = time.perf_counter() - t0

    print("=" * 60)
    print("CORRELATION MATRIX DATE ALIGNMENT BENCHMARK")
    print("=" * 60)
    print(f"Assets: SPY ({len(data['SPY'])} rows), EUR ({len(data['EUR'])} rows)")
    print(f"SPY rows: {len(data['SPY'])}, EUR rows: {len(data['EUR'])}, lookback=20")
    print()
    print("Aligned (date-indexed) correlation:")
    print(aligned)
    print(f"SPY-EUR correlation: {aligned.loc['SPY', 'EUR']:.6f}")
    print(f"Time: {t_aligned:.6f}s")
    print()
    print("Naive positional correlation:")
    print(positional)
    print(f"SPY-EUR correlation: {positional.loc['SPY', 'EUR']:.6f}")
    print(f"Time: {t_positional:.6f}s")
    print()

    true_corr = 1.0
    aligned_err = abs(aligned.loc['SPY', 'EUR'] - true_corr)
    positional_val = positional.loc['SPY', 'EUR']

    print(f"Absolute error vs true correlation (1.0):")
    print(f"  Aligned:   {aligned_err:.6e}")
    if pd.isna(positional_val):
        print(f"  Positional: undefined (NaN) — no shared timestamps")
    else:
        positional_err = abs(positional_val - true_corr)
        improvement = positional_err / max(aligned_err, 1e-15)
        print(f"  Positional: {positional_err:.6e}")
        print(f"  Error reduction: {improvement:.1f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()

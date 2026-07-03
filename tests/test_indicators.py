"""
Test suite for Technical Indicators module.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.indicators import (
    calculate_sma,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_volatility,
    calculate_drawdown,
    calculate_returns,
    calculate_correlation_matrix
)


def test_sma_calculation():
    """Test Simple Moving Average calculation."""
    print("Test 1: Simple Moving Average (SMA)")
    print("-" * 40)
    
    # Create test price series
    prices = pd.Series([100, 102, 101, 103, 104, 105, 106, 107, 108, 109])
    
    sma_5 = calculate_sma(prices, window=5)
    
    # Last 5 values: 105, 106, 107, 108, 109
    expected_last = np.mean([105, 106, 107, 108, 109])
    
    assert not pd.isna(sma_5.iloc[-1]), "SMA should not be NaN"
    assert abs(sma_5.iloc[-1] - expected_last) < 1e-6, f"SMA {sma_5.iloc[-1]} != {expected_last}"
    
    print(f"  Price series: {list(prices)}")
    print(f"  SMA(5) last value: {sma_5.iloc[-1]:.2f}")
    print(f"  Expected: {expected_last:.2f}")
    print("✓ SMA calculation test passed\n")


def test_rsi_calculation():
    """Test RSI calculation."""
    print("Test 2: Relative Strength Index (RSI)")
    print("-" * 40)
    
    np.random.seed(42)
    
    # Create series with clear uptrend
    uptrend = pd.Series([100 + i * 2 + np.random.randn() * 0.5 for i in range(20)])
    rsi_up = calculate_rsi(uptrend, period=14)
    
    # Create series with clear downtrend
    downtrend = pd.Series([100 - i * 2 + np.random.randn() * 0.5 for i in range(20)])
    rsi_down = calculate_rsi(downtrend, period=14)
    
    assert not pd.isna(rsi_up.iloc[-1]), "RSI should not be NaN"
    assert 0 <= rsi_up.iloc[-1] <= 100, "RSI should be between 0 and 100"
    assert 0 <= rsi_down.iloc[-1] <= 100, "RSI should be between 0 and 100"
    assert rsi_up.iloc[-1] > rsi_down.iloc[-1], "Uptrend RSI should be higher"
    
    print(f"  Uptrend RSI: {rsi_up.iloc[-1]:.2f}")
    print(f"  Downtrend RSI: {rsi_down.iloc[-1]:.2f}")
    print(f"  RSI range: 0-100 ✓")
    print("✓ RSI calculation test passed\n")


def test_bollinger_bands():
    """Test Bollinger Bands calculation."""
    print("Test 3: Bollinger Bands")
    print("-" * 40)
    
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5))
    
    upper, middle, lower = calculate_bollinger_bands(prices, window=20, num_std=2)
    
    assert not pd.isna(upper.iloc[-1]), "Upper band should not be NaN"
    assert not pd.isna(middle.iloc[-1]), "Middle band should not be NaN"
    assert not pd.isna(lower.iloc[-1]), "Lower band should not be NaN"
    
    assert upper.iloc[-1] > middle.iloc[-1], "Upper band > middle"
    assert middle.iloc[-1] > lower.iloc[-1], "Middle > lower band"
    
    # Bandwidth should be positive
    bandwidth = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
    assert bandwidth > 0, "Bandwidth should be positive"
    
    print(f"  Upper band: {upper.iloc[-1]:.2f}")
    print(f"  Middle band (SMA20): {middle.iloc[-1]:.2f}")
    print(f"  Lower band: {lower.iloc[-1]:.2f}")
    print(f"  Bandwidth: {bandwidth:.2%}")
    print("✓ Bollinger Bands test passed\n")


def test_volatility_calculation():
    """Test volatility calculation."""
    print("Test 4: Volatility (Standard Deviation)")
    print("-" * 40)
    
    np.random.seed(42)
    
    # Low volatility series
    low_vol_prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5))
    low_vol = calculate_volatility(low_vol_prices, window=20)
    
    # High volatility series
    high_vol_prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 2.0))
    high_vol = calculate_volatility(high_vol_prices, window=20)
    
    assert not pd.isna(low_vol.iloc[-1]), "Volatility should not be NaN"
    assert high_vol.iloc[-1] > low_vol.iloc[-1], "High vol series should have higher volatility"
    assert low_vol.iloc[-1] >= 0, "Volatility should be non-negative"
    
    print(f"  Low volatility: {low_vol.iloc[-1]:.2%}")
    print(f"  High volatility: {high_vol.iloc[-1]:.2%}")
    print(f"  High > Low: {high_vol.iloc[-1] > low_vol.iloc[-1]}")
    print("✓ Volatility calculation test passed\n")


def test_drawdown():
    """Test drawdown calculation."""
    print("Test 5: Drawdown")
    print("-" * 40)
    
    # Create equity curve with peak and decline
    equity = pd.Series([100, 105, 110, 108, 105, 100, 95, 98, 102, 105])
    
    dd_series = calculate_drawdown(equity)
    
    # Drawdown should be <= 0
    assert (dd_series <= 0).all(), "Drawdown should be <= 0"
    
    # Max drawdown: from 110 to 95 = -13.64%
    expected_max_dd = (95 - 110) / 110
    assert abs(dd_series.min() - expected_max_dd) < 0.01, f"Max DD {dd_series.min()} != {expected_max_dd}"
    
    print(f"  Equity curve: {list(equity)}")
    print(f"  Drawdown series: {[f'{d:.1%}' for d in dd_series]}")
    print(f"  Max drawdown: {dd_series.min():.2%}")
    print(f"  Expected: {expected_max_dd:.2%}")
    print("✓ Drawdown test passed\n")


def test_returns():
    """Test returns calculation."""
    print("Test 6: Returns Calculation")
    print("-" * 40)
    
    # Create price series
    prices = pd.Series([100, 102, 101, 103, 104])
    
    returns = calculate_returns(prices)
    
    assert 'total_return' in returns
    assert 'daily_returns' in returns
    
    # Total return: (104 - 100) / 100 = 4%
    expected_total = (104 - 100) / 100
    assert abs(returns['total_return'] - expected_total) < 0.01
    
    print(f"  Price series: {list(prices)}")
    print(f"  Total return: {returns['total_return']:.2%}")
    print(f"  Expected: {expected_total:.2%}")
    print("✓ Returns test passed\n")


def test_correlation_matrix():
    """Test correlation matrix calculation."""
    print("Test 7: Correlation Matrix")
    print("-" * 40)
    
    np.random.seed(42)
    n_days = 60
    
    # Create DataFrames with 'Close' column
    market = np.random.randn(n_days)
    spy_df = pd.DataFrame({'Close': 100 + np.cumsum(0.8 * market + 0.2 * np.random.randn(n_days))})
    qqq_df = pd.DataFrame({'Close': 100 + np.cumsum(0.9 * market + 0.1 * np.random.randn(n_days))})
    tlt_df = pd.DataFrame({'Close': 100 + np.cumsum(-0.3 * market + 0.7 * np.random.randn(n_days))})
    
    data_dict = {
        'SPY': spy_df,
        'QQQ': qqq_df,
        'TLT': tlt_df
    }
    
    corr_matrix = calculate_correlation_matrix(data_dict)
    
    assert corr_matrix.shape == (3, 3), "Correlation matrix should be 3x3"
    assert np.allclose(np.diag(corr_matrix), 1.0), "Diagonal should be 1"
    
    # SPY and QQQ should be highly correlated
    spy_qqq_corr = corr_matrix.loc['SPY', 'QQQ']
    assert spy_qqq_corr > 0.5, f"SPY-QQQ correlation {spy_qqq_corr} should be high"
    
    # TLT should have lower/negative correlation
    spy_tlt_corr = corr_matrix.loc['SPY', 'TLT']
    
    print(f"  Correlation matrix:\n{corr_matrix}")
    print(f"  SPY-QQQ correlation: {spy_qqq_corr:.3f}")
    print(f"  SPY-TLT correlation: {spy_tlt_corr:.3f}")
    print("✓ Correlation matrix test passed\n")


def test_correlation_matrix_date_alignment():
    """Test that correlation matrix aligns dates across different trading calendars."""
    print("Test 7b: Correlation Matrix Date Alignment")
    print("-" * 40)
    
    np.random.seed(42)
    
    # SPY trades Mon-Fri: create 30 consecutive business days with varying returns.
    dates_spy = pd.date_range(start='2026-05-01', periods=30, freq='B')
    spy_returns = np.random.normal(loc=0.001, scale=0.01, size=len(dates_spy))
    spy_prices = 100 * np.cumprod(1 + spy_returns)
    spy_df = pd.DataFrame({'Close': spy_prices}, index=dates_spy)
    
    # Euronext asset misses 3 US holidays (e.g., days 5, 12, 18) but is otherwise
    # perfectly correlated with SPY on shared days.
    missing_idx = {5, 12, 18}
    dates_eur = [d for i, d in enumerate(dates_spy) if i not in missing_idx]
    eur_returns = np.array([spy_returns[i] for i in range(len(spy_returns)) if i not in missing_idx])
    eur_prices = 100 * np.cumprod(1 + eur_returns)
    eur_df = pd.DataFrame({'Close': eur_prices}, index=dates_eur)
    
    data_dict = {'SPY': spy_df, 'EUR': eur_df}
    
    corr_matrix = calculate_correlation_matrix(data_dict, lookback=20)
    
    assert corr_matrix.shape == (2, 2), "Correlation matrix should be 2x2"
    spy_eur_corr = corr_matrix.loc['SPY', 'EUR']
    
    # With proper date alignment, the correlation should be exactly 1.0 because
    # the assets moved identically on every shared date. With naive positional
    # alignment, the returns would be misaligned and the correlation would be
    # much lower (or undefined).
    assert abs(spy_eur_corr - 1.0) < 1e-9, f"SPY-EUR correlation should be 1.0, got {spy_eur_corr}"
    
    assert corr_matrix.notna().all().all(), "Correlation matrix should have no NaN values"
    
    print(f"  Dates SPY: {len(dates_spy)}, Dates EUR: {len(dates_eur)}")
    print(f"  SPY-EUR correlation: {spy_eur_corr:.6f}")
    print("✓ Correlation matrix date alignment test passed\n")


def test_nan_handling():
    """Test that NaN Close prices are handled gracefully."""
    print("Test 8: NaN Close Price Handling")
    print("-" * 40)
    
    from data.indicators import calculate_all_indicators, get_latest_indicators
    
    # Create DataFrame with NaN at the end (simulates yfinance pre-market data)
    prices = pd.DataFrame({
        'Close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, np.nan]
    })
    
    df_with_indicators = calculate_all_indicators(prices)
    latest = get_latest_indicators(df_with_indicators)
    
    # Should drop NaN row and use last valid price (105.0)
    assert not pd.isna(latest['price']), "Price should not be NaN after dropping NaN rows"
    assert latest['price'] == 105.0, f"Expected price 105.0, got {latest['price']}"
    assert not pd.isna(latest['rsi_14']), "RSI should not be NaN"
    assert not pd.isna(latest['bb_position']), "Bollinger position should not be NaN"
    assert not pd.isna(latest['drawdown']), "Drawdown should not be NaN"
    
    print(f"  Input: 7 rows with last Close = NaN")
    print(f"  Output price: {latest['price']:.2f} (expected 105.00)")
    print(f"  RSI(14): {latest['rsi_14']:.2f}")
    print(f"  Bollinger position: {latest['bb_position']:.2f}")
    print("✓ NaN handling test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Technical Indicators Test Suite")
    print("=" * 60)
    print()
    
    test_sma_calculation()
    test_rsi_calculation()
    test_bollinger_bands()
    test_volatility_calculation()
    test_drawdown()
    test_returns()
    test_correlation_matrix()
    test_correlation_matrix_date_alignment()
    test_nan_handling()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

"""
Test suite for Market Data Fetching module.
Uses mocks to avoid actual yfinance API calls.
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.fetch_market_data import (
    load_universe,
    get_all_tickers,
    get_ticker_names,
    get_tickers_by_category,
    fetch_historical_data,
    fetch_current_prices,
    UNIVERSE_PATH
)


def test_load_universe():
    """Test loading universe config from JSON."""
    print("Test 1: Load Universe Config")
    print("-" * 40)
    
    # Create temporary universe file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "etf": [{"ticker": "SPY", "name": "S&P 500"}],
            "test": [{"ticker": "TEST", "name": "Test Asset"}]
        }, f)
        temp_path = f.name
    
    # Patch the universe path temporarily
    with patch('data.fetch_market_data.UNIVERSE_PATH', Path(temp_path)):
        universe = load_universe()
        assert "etf" in universe
        assert universe["etf"][0]["ticker"] == "SPY"
        print(f"  Loaded {len(universe)} categories")
        print(f"  ETFs: {[a['ticker'] for a in universe['etf']]}")
        print("✓ Load universe test passed\n")
    
    # Cleanup
    Path(temp_path).unlink()


def test_get_all_tickers():
    """Test extracting all tickers from universe."""
    print("Test 2: Get All Tickers")
    print("-" * 40)
    
    mock_universe = {
        "etf": [{"ticker": "SPY", "name": "S&P 500"}, {"ticker": "QQQ", "name": "Nasdaq"}],
        "commodity": [{"ticker": "GLD", "name": "Gold"}],
        "empty": []
    }
    
    with patch('data.fetch_market_data.load_universe', return_value=mock_universe):
        tickers = get_all_tickers()
        assert "SPY" in tickers
        assert "QQQ" in tickers
        assert "GLD" in tickers
        assert len(tickers) == 3
        print(f"  Tickers: {tickers}")
        print("✓ Get all tickers test passed\n")


def test_get_ticker_names():
    """Test ticker to name mapping."""
    print("Test 3: Get Ticker Names")
    print("-" * 40)
    
    mock_universe = {
        "etf": [{"ticker": "SPY", "name": "S&P 500 ETF"}],
        "euronext": [{"ticker": "MC.PA", "name": "LVMH"}]
    }
    
    with patch('data.fetch_market_data.load_universe', return_value=mock_universe):
        names = get_ticker_names()
        assert names["SPY"] == "S&P 500 ETF"
        assert names["MC.PA"] == "LVMH"
        print(f"  Mappings: {names}")
        print("✓ Get ticker names test passed\n")


def test_get_tickers_by_category():
    """Test filtering tickers by category."""
    print("Test 4: Get Tickers by Category")
    print("-" * 40)
    
    mock_universe = {
        "etf": [{"ticker": "SPY", "name": "S&P 500"}, {"ticker": "QQQ", "name": "Nasdaq"}],
        "commodity": [{"ticker": "GLD", "name": "Gold"}]
    }
    
    with patch('data.fetch_market_data.load_universe', return_value=mock_universe):
        etfs = get_tickers_by_category("etf")
        commodities = get_tickers_by_category("commodity")
        
        assert "SPY" in etfs
        assert "QQQ" in etfs
        assert "GLD" in commodities
        assert len(etfs) == 2
        assert len(commodities) == 1
        print(f"  ETFs: {etfs}")
        print(f"  Commodities: {commodities}")
        print("✓ Get tickers by category test passed\n")


def test_fetch_historical_data_mock():
    """Test historical data fetching with mocked yfinance."""
    print("Test 5: Fetch Historical Data (Mocked)")
    print("-" * 40)
    
    # Create mock DataFrame
    dates = pd.date_range('2024-01-01', periods=5)
    mock_df = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104],
        'High': [101, 102, 103, 104, 105],
        'Low': [99, 100, 101, 102, 103],
        'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
        'Volume': [1000, 1100, 1200, 1300, 1400]
    }, index=dates)
    
    # Mock yfinance.Ticker
    mock_ticker = Mock()
    mock_ticker.history.return_value = mock_df
    
    with patch('data.fetch_market_data.yf.Ticker', return_value=mock_ticker):
        data = fetch_historical_data(tickers=["SPY"], period="5d")
        
        assert "SPY" in data
        assert len(data["SPY"]) == 5
        assert "Close" in data["SPY"].columns
        print(f"  Fetched {len(data)} tickers")
        print(f"  SPY rows: {len(data['SPY'])}")
        print(f"  Last close: ${data['SPY']['Close'].iloc[-1]:.2f}")
        print("✓ Fetch historical data test passed\n")


def test_fetch_current_prices_mock():
    """Test current price fetching with mocked yfinance."""
    print("Test 6: Fetch Current Prices (Mocked)")
    print("-" * 40)
    
    dates = pd.date_range('2024-01-01', periods=1)
    mock_df = pd.DataFrame({
        'Open': [100],
        'High': [105],
        'Low': [99],
        'Close': [104.5],
        'Volume': [1000]
    }, index=dates)
    
    mock_ticker = Mock()
    mock_ticker.history.return_value = mock_df
    
    with patch('data.fetch_market_data.yf.Ticker', return_value=mock_ticker):
        prices = fetch_current_prices(tickers=["SPY", "QQQ"])
        
        assert "SPY" in prices
        assert "QQQ" in prices
        assert prices["SPY"] == 104.5
        print(f"  Prices: {prices}")
        print("✓ Fetch current prices test passed\n")


def test_fetch_with_empty_tickers():
    """Test handling of empty ticker list."""
    print("Test 7: Empty Ticker List Handling")
    print("-" * 40)
    
    data = fetch_historical_data(tickers=[])
    assert data == {}
    print("  Empty list returns empty dict")
    print("✓ Empty ticker list test passed\n")


def test_fetch_with_invalid_ticker():
    """Test handling of invalid ticker."""
    print("Test 8: Invalid Ticker Handling")
    print("-" * 40)
    
    mock_ticker = Mock()
    mock_ticker.history.return_value = pd.DataFrame()  # Empty DataFrame
    
    with patch('data.fetch_market_data.yf.Ticker', return_value=mock_ticker):
        data = fetch_historical_data(tickers=["INVALID"])
        # Should handle gracefully and skip invalid tickers
        print("  Invalid ticker handled gracefully")
        print("✓ Invalid ticker test passed\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running Market Data Fetcher Tests")
    print("=" * 60 + "\n")
    
    test_load_universe()
    test_get_all_tickers()
    test_get_ticker_names()
    test_get_tickers_by_category()
    test_fetch_historical_data_mock()
    test_fetch_current_prices_mock()
    test_fetch_with_empty_tickers()
    test_fetch_with_invalid_ticker()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

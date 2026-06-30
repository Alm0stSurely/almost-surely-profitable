"""
Market data fetching module using yfinance.
Reads asset universe from config/universe.json.
"""

import json
import os
import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve project root (2 levels up from src/data/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UNIVERSE_PATH = PROJECT_ROOT / "config" / "universe.json"


def load_universe() -> Dict:
    """Load asset universe from config/universe.json."""
    if not UNIVERSE_PATH.exists():
        logger.warning(f"Universe config not found at {UNIVERSE_PATH}, using empty universe")
        return {}
    with open(UNIVERSE_PATH) as f:
        return json.load(f)


def get_all_tickers() -> List[str]:
    """Get flat list of all tickers from universe config."""
    universe = load_universe()
    tickers = []
    for category in universe.values():
        for asset in category:
            ticker = asset.get("ticker", "").strip()
            # Filter out empty tickers
            if ticker and ticker != ".PA":
                tickers.append(ticker)
            elif not ticker or ticker == ".PA":
                logger.warning(f"Skipping invalid ticker in category: {category}")
    return tickers


def get_ticker_names() -> Dict[str, str]:
    """Get ticker -> display name mapping from universe config."""
    universe = load_universe()
    names = {}
    for category in universe.values():
        for asset in category:
            names[asset["ticker"]] = asset["name"]
    return names


def get_tickers_by_category(category: str) -> List[str]:
    """Get tickers for a specific category (etf, small_cap, commodity, euronext)."""
    universe = load_universe()
    return [asset["ticker"] for asset in universe.get(category, [])]


# Module-level convenience (lazy-loaded)
ALL_TICKERS = get_all_tickers()


def _fetch_single_ticker(ticker: str, period: str, interval: str, start: Optional[str], end: Optional[str]) -> tuple:
    """Fetch data for a single ticker. Returns (ticker, DataFrame or None)."""
    try:
        logger.info(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        if start and end:
            hist = stock.history(start=start, end=end, interval=interval)
        else:
            hist = stock.history(period=period, interval=interval)

        if hist.empty:
            logger.warning(f"No data returned for {ticker}")
            return ticker, None

        # Normalize timezone-aware index to naive UTC to avoid
        # offset-naive vs offset-aware comparison errors in backtest
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert("UTC").tz_localize(None)

        logger.info(f"  {ticker}: {len(hist)} rows")
        return ticker, hist

    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return ticker, None


def fetch_historical_data(
    tickers: Optional[List[str]] = None,
    period: str = "30d",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_workers: int = 1
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical market data for specified tickers.

    Args:
        tickers: List of ticker symbols (default: all from universe config)
        period: Data period (default: "30d")
        interval: Data interval (default: "1d")
        start: Start date string YYYY-MM-DD (overrides period if provided)
        end: End date string YYYY-MM-DD (overrides period if provided)
        max_workers: Number of parallel threads for I/O-bound fetching.
                     Default 1 (sequential). Set to 4-8 for significant speedup.

    Returns:
        Dict mapping ticker to DataFrame with OHLCV data
    """
    if tickers is None:
        tickers = get_all_tickers()
    
    # Filter out empty or invalid tickers
    tickers = [t.strip() for t in tickers if t and t.strip() and t.strip() != ".PA"]

    results = {}

    if max_workers > 1:
        # Parallel execution with ThreadPoolExecutor
        # yfinance requests are I/O bound, so threads are effective
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(_fetch_single_ticker, ticker, period, interval, start, end): ticker
                for ticker in tickers
            }
            for future in as_completed(future_to_ticker):
                ticker, hist = future.result()
                if hist is not None:
                    results[ticker] = hist
    else:
        # Sequential execution (original behavior)
        for ticker in tickers:
            ticker, hist = _fetch_single_ticker(ticker, period, interval, start, end)
            if hist is not None:
                results[ticker] = hist

    return results


def _fetch_single_price(ticker: str) -> tuple:
    """Fetch current price for a single ticker. Returns (ticker, price or None)."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d", interval="1m")

        if hist.empty:
            hist = stock.history(period="5d")

        if hist.empty:
            # Try 1mo for some indices like ^FCHI
            hist = stock.history(period="1mo")
        
        if hist.empty:
            logger.warning(f"No price data for {ticker} (may be delisted or not available)")
            return ticker, None

        current_price = float(hist["Close"].iloc[-1])
        return ticker, current_price

    except Exception as e:
        logger.warning(f"Could not fetch price for {ticker}: {e}")
        return ticker, None


def fetch_current_prices(
    tickers: Optional[List[str]] = None,
    max_workers: int = 1
) -> Dict[str, Optional[float]]:
    """
    Fetch current/latest prices for specified tickers.

    Args:
        tickers: List of ticker symbols (default: all from universe config)
        max_workers: Number of parallel threads for I/O-bound fetching.
                     Default 1 (sequential). Set to 4-8 for significant speedup.

    Returns:
        Dict mapping ticker to current price (or None if error)
    """
    if tickers is None:
        tickers = get_all_tickers()
    
    # Filter out empty or invalid tickers
    tickers = [t.strip() for t in tickers if t and t.strip() and t.strip() != ".PA"]

    results = {}

    if max_workers > 1:
        # Parallel execution with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(_fetch_single_price, ticker): ticker
                for ticker in tickers
            }
            for future in as_completed(future_to_ticker):
                ticker, price = future.result()
                results[ticker] = price
    else:
        # Sequential execution (original behavior)
        for ticker in tickers:
            ticker, price = _fetch_single_price(ticker)
            results[ticker] = price

    return results


def fetch_ticker_info(ticker: str) -> Dict:
    """
    Fetch general information about a ticker.

    Args:
        ticker: Ticker symbol

    Returns:
        Dict with ticker info (name, sector, currency, etc.)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", info.get("shortName", ticker)),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currency": info.get("currency", "N/A"),
            "market_cap": info.get("marketCap"),
            "country": info.get("country", "N/A"),
        }
    except Exception as e:
        logger.error(f"Error fetching info for {ticker}: {e}")
        return {"name": ticker, "error": str(e)}


if __name__ == "__main__":
    print(f"Universe: {UNIVERSE_PATH}")
    universe = load_universe()
    for cat, assets in universe.items():
        print(f"  {cat}: {len(assets)} assets")
    print(f"Total: {len(get_all_tickers())} tickers")
    print()
    prices = fetch_current_prices(["SPY", "MC.PA"])
    print("Current prices:", prices)

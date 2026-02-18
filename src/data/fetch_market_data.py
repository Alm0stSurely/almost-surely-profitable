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
            tickers.append(asset["ticker"])
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


def fetch_historical_data(
    tickers: Optional[List[str]] = None,
    period: str = "30d",
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical market data for specified tickers.

    Args:
        tickers: List of ticker symbols (default: all from universe config)
        period: Data period (default: "30d")
        interval: Data interval (default: "1d")

    Returns:
        Dict mapping ticker to DataFrame with OHLCV data
    """
    if tickers is None:
        tickers = get_all_tickers()

    results = {}

    for ticker in tickers:
        try:
            logger.info(f"Fetching data for {ticker}...")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)

            if hist.empty:
                logger.warning(f"No data returned for {ticker}")
                continue

            results[ticker] = hist
            logger.info(f"  {ticker}: {len(hist)} rows")

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            continue

    return results


def fetch_current_prices(
    tickers: Optional[List[str]] = None
) -> Dict[str, Optional[float]]:
    """
    Fetch current/latest prices for specified tickers.

    Args:
        tickers: List of ticker symbols (default: all from universe config)

    Returns:
        Dict mapping ticker to current price (or None if error)
    """
    if tickers is None:
        tickers = get_all_tickers()

    results = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d", interval="1m")

            if hist.empty:
                hist = stock.history(period="5d")

            if hist.empty:
                logger.warning(f"No price data for {ticker}")
                results[ticker] = None
                continue

            current_price = hist["Close"].iloc[-1]
            results[ticker] = float(current_price)

        except Exception as e:
            logger.error(f"Error fetching current price for {ticker}: {e}")
            results[ticker] = None

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

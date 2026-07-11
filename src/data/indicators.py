"""
Technical indicators calculation module.
Computes SMA, RSI, Bollinger Bands, volatility, drawdown, returns, and correlations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return prices.rolling(window=window, min_periods=1).mean()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Handle flat prices: when both gain and loss are 0, RSI is neutral (50)
    # Also handle edge case where loss is 0 but gain > 0 (RSI = 100) via natural division
    flat_mask = (gain == 0) & (loss == 0)
    rsi = rsi.where(~flat_mask, 50.0)
    
    return rsi


def calculate_bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Returns:
        (upper_band, middle_band, lower_band)
    """
    middle_band = calculate_sma(prices, window)
    std_dev = prices.rolling(window=window, min_periods=1).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    return upper_band, middle_band, lower_band


def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation of returns).
    """
    returns = prices.pct_change()
    return returns.rolling(window=window, min_periods=1).std() * np.sqrt(252)


def calculate_drawdown(prices: pd.Series) -> pd.Series:
    """
    Calculate drawdown from peak.
    
    Drawdown = (Current Price / Running Max) - 1
    """
    running_max = prices.expanding().max()
    drawdown = (prices / running_max) - 1
    return drawdown


def calculate_returns(prices: pd.Series) -> Dict[str, float]:
    """
    Calculate daily and cumulative returns.
    
    Returns:
        Dict with daily_return, cumulative_return, total_return
    """
    daily_returns = prices.pct_change().dropna()
    
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1 if len(prices) > 1 else 0
    
    return {
        "daily_returns": daily_returns,
        "latest_daily_return": daily_returns.iloc[-1] if len(daily_returns) > 0 else 0,
        "total_return": total_return,
        "cumulative_returns": (1 + daily_returns).cumprod() - 1
    }


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for a single asset.
    
    Args:
        df: DataFrame with 'Close' column
    
    Returns:
        DataFrame with added indicator columns
    """
    # Drop rows with NaN Close prices to avoid indicator corruption
    # This handles cases where yfinance returns a row for today before market close
    df = df.dropna(subset=['Close']).copy()
    
    if df.empty:
        logger.warning("DataFrame is empty after dropping NaN Close prices")
        return df
    
    prices = df['Close']
    
    # Moving averages
    df['SMA_20'] = calculate_sma(prices, 20)
    df['SMA_50'] = calculate_sma(prices, 50)
    df['SMA_200'] = calculate_sma(prices, 200)
    
    # RSI
    df['RSI_14'] = calculate_rsi(prices, 14)
    
    # Bollinger Bands
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(prices, 20, 2.0)
    df['BB_position'] = (prices - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    # Handle flat prices: when bands are equal (std=0), price is at midpoint
    df['BB_position'] = df['BB_position'].fillna(0.5)
    df['BB_position'] = df['BB_position'].replace([np.inf, -np.inf], 0.5)
    
    # Volatility
    df['Volatility_20'] = calculate_volatility(prices, 20)
    
    # Drawdown
    df['Drawdown'] = calculate_drawdown(prices)
    df['Max_Drawdown'] = df['Drawdown'].expanding().min()
    
    # Returns
    df['Daily_Return'] = prices.pct_change()
    
    return df


def calculate_correlation_matrix(
    data_dict: Dict[str, pd.DataFrame],
    lookback: int = 20
) -> pd.DataFrame:
    """
    Calculate correlation matrix between assets based on returns.

    Aligns observations by date before computing correlations so that
    returns from different trading calendars (e.g. US holidays vs Euronext
    holidays) are not compared by position. After alignment, the most recent
    `lookback` common trading days are retained and rows with any missing
    returns are dropped so that every pairwise correlation is computed over
    the same set of dates.

    Args:
        data_dict: Dict mapping ticker to DataFrame with a DatetimeIndex
            and a 'Close' column.
        lookback: Number of common trading days to use for correlation
            (default: 20).

    Returns:
        Correlation matrix DataFrame. Returns an empty DataFrame if there are
        fewer than two assets or fewer than two complete aligned observations.
    """
    returns_series = {}

    for ticker, df in data_dict.items():
        if 'Close' not in df.columns or df.empty:
            continue
        close = df['Close'].dropna()
        if close.empty:
            continue

        returns = close.pct_change()

        # If the index is a DatetimeIndex, normalize it to date-only so that
        # assets fetched at different times of day are still aligned by date.
        if isinstance(returns.index, pd.DatetimeIndex):
            returns.index = returns.index.normalize()

        returns_series[ticker] = returns

    if len(returns_series) < 2:
        return pd.DataFrame()

    # Align all return series on a common date index.
    returns_df = pd.DataFrame(returns_series)

    # Keep only the most recent `lookback` trading days with complete data.
    if len(returns_df) > lookback:
        returns_df = returns_df.tail(lookback)

    # Ensure every pairwise correlation uses the same date observations.
    returns_df = returns_df.dropna()

    if len(returns_df) < 2:
        return pd.DataFrame()

    return returns_df.corr()


def get_latest_indicators(df: pd.DataFrame) -> Dict:
    """
    Extract latest indicator values from a DataFrame.
    
    Returns:
        Dict with current indicator values
    """
    if df.empty:
        return {}
    
    latest = df.iloc[-1]
    
    return {
        "price": float(latest['Close']),
        "sma_20": float(latest.get('SMA_20', 0)),
        "sma_50": float(latest.get('SMA_50', 0)),
        "sma_200": float(latest.get('SMA_200', 0)),
        "rsi_14": float(latest.get('RSI_14', 50)),
        "bb_upper": float(latest.get('BB_upper', 0)),
        "bb_lower": float(latest.get('BB_lower', 0)),
        "bb_position": float(latest.get('BB_position', 0.5)),
        "volatility_annual": float(latest.get('Volatility_20', 0)),
        "drawdown": float(latest.get('Drawdown', 0)),
        "max_drawdown": float(latest.get('Max_Drawdown', 0)),
        "daily_return": float(latest.get('Daily_Return', 0)),
    }


def analyze_market_data(data_dict: Dict[str, pd.DataFrame]) -> Dict:
    """
    Analyze market data for all assets.
    
    Args:
        data_dict: Dict mapping ticker to DataFrame with OHLCV data
    
    Returns:
        Dict with indicators and correlation matrix
    """
    results = {}
    
    # Calculate indicators for each asset
    for ticker, df in data_dict.items():
        try:
            df_with_indicators = calculate_all_indicators(df.copy())
            results[ticker] = {
                "dataframe": df_with_indicators,
                "latest": get_latest_indicators(df_with_indicators),
                "total_return": (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1 if len(df) > 1 else 0,
                "returns": df_with_indicators['Daily_Return'].dropna().tolist()
            }
        except Exception as e:
            logger.error(f"Error calculating indicators for {ticker}: {e}")
            continue
    
    # Calculate correlations
    try:
        correlations = calculate_correlation_matrix(data_dict)
    except Exception as e:
        logger.error(f"Error calculating correlations: {e}")
        correlations = pd.DataFrame()
    
    return {
        "assets": results,
        "correlations": correlations,
        "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }


if __name__ == "__main__":
    # Quick test
    from fetch_market_data import fetch_historical_data
    
    print("Testing indicators...")
    data = fetch_historical_data(["SPY"], period="30d")
    
    if "SPY" in data:
        result = analyze_market_data(data)
        print("\nLatest indicators for SPY:")
        print(result["assets"]["SPY"]["latest"])

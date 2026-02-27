"""
Risk management module.
Calculates risk metrics: CVaR, VaR, volatility, drawdowns, correlations.
Based on Behavioral_RL concepts: prospect theory, CVaR, risk-sensitive decision making.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class RiskMetrics:
    """Container for portfolio risk metrics."""
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    cvar_99: float
    volatility: float  # Annualized volatility
    downside_volatility: float  # Semi-deviation (negative returns only)
    max_drawdown: float
    current_drawdown: float
    sortino_ratio: float
    calmar_ratio: float
    skewness: float
    kurtosis: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "volatility": self.volatility,
            "downside_volatility": self.downside_volatility,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily returns from price series."""
    return prices.pct_change().dropna()


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) using historical method.
    
    VaR is the maximum loss not exceeded with the given confidence level.
    Example: VaR_95 = -2% means there's a 5% chance of losing more than 2%.
    
    Args:
        returns: Series of daily returns
        confidence: Confidence level (default 0.95)
    
    Returns:
        VaR as a negative number (loss)
    """
    if len(returns) < 30:
        return 0.0
    
    return np.percentile(returns, (1 - confidence) * 100)


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    CVaR is the expected loss given that the loss exceeds VaR.
    More sensitive to tail risk than VaR.
    
    Based on prospect theory: investors are more sensitive to tail losses.
    
    Args:
        returns: Series of daily returns
        confidence: Confidence level (default 0.95)
    
    Returns:
        CVaR as a negative number (expected tail loss)
    """
    if len(returns) < 30:
        return 0.0
    
    var = calculate_var(returns, confidence)
    tail_losses = returns[returns <= var]
    
    if len(tail_losses) == 0:
        return var  # Fallback to VaR if no tail losses
    
    return tail_losses.mean()


def calculate_drawdowns(prices: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from prices.
    
    Drawdown at time t = (price_t / max_price_up_to_t) - 1
    
    Args:
        prices: Series of asset prices
    
    Returns:
        Series of drawdown percentages (negative numbers)
    """
    rolling_max = prices.cummax()
    drawdown = (prices / rolling_max) - 1
    return drawdown


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown from price series."""
    return calculate_drawdowns(prices).min()


def calculate_downside_volatility(returns: pd.Series) -> float:
    """
    Calculate downside volatility (semi-deviation).
    
    Only considers returns below a threshold (typically 0 or risk-free rate).
    Used in Sortino ratio.
    
    Args:
        returns: Series of daily returns
    
    Returns:
        Annualized downside volatility
    """
    if len(returns) < 30:
        return 0.0
    
    # Only negative returns
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) < 2:
        return 0.0
    
    # Annualized (252 trading days)
    return downside_returns.std() * np.sqrt(252)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculate Sortino ratio.
    
    Like Sharpe ratio but uses downside volatility instead of total volatility.
    More appropriate for asymmetric return distributions.
    
    Sortino = (Mean Return - Risk Free Rate) / Downside Volatility
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
        Sortino ratio (higher is better)
    """
    if len(returns) < 30:
        return 0.0
    
    # Annualized mean return
    mean_return = returns.mean() * 252
    
    # Downside volatility
    downside_vol = calculate_downside_volatility(returns)
    
    if downside_vol == 0:
        return float('inf') if mean_return > risk_free_rate else 0.0
    
    return (mean_return - risk_free_rate) / downside_vol


def calculate_calmar_ratio(prices: pd.Series) -> float:
    """
    Calculate Calmar ratio.
    
    Calmar = Annualized Return / |Maximum Drawdown|
    
    Measures return per unit of worst-case risk.
    
    Args:
        prices: Series of asset prices
    
    Returns:
        Calmar ratio (higher is better)
    """
    if len(prices) < 30:
        return 0.0
    
    # Annualized return
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    years = len(prices) / 252
    annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # Max drawdown (absolute value)
    max_dd = abs(calculate_max_drawdown(prices))
    
    if max_dd == 0:
        return float('inf') if annualized_return > 0 else 0.0
    
    return annualized_return / max_dd


def calculate_correlation_matrix(
    returns_dict: Dict[str, pd.Series]
) -> Optional[pd.DataFrame]:
    """
    Calculate correlation matrix between assets.
    
    Args:
        returns_dict: Dict mapping ticker to returns Series
    
    Returns:
        Correlation matrix DataFrame or None if insufficient data
    """
    if len(returns_dict) < 2:
        return None
    
    # Align all series to common dates
    df = pd.DataFrame(returns_dict)
    
    if len(df) < 10:
        return None
    
    return df.corr()


def calculate_portfolio_risk_metrics(
    prices_dict: Dict[str, pd.Series],
    weights: Optional[Dict[str, float]] = None
) -> RiskMetrics:
    """
    Calculate comprehensive risk metrics for a portfolio.
    
    Args:
        prices_dict: Dict mapping ticker to price Series
        weights: Optional dict of portfolio weights (sums to 1)
    
    Returns:
        RiskMetrics dataclass with all risk measures
    """
    # Calculate returns for each asset
    returns_dict = {ticker: calculate_returns(prices) for ticker, prices in prices_dict.items()}
    
    # Calculate portfolio returns if weights provided
    if weights and len(weights) > 0:
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Create aligned returns dataframe
        returns_df = pd.DataFrame(returns_dict)
        
        # Portfolio returns (weighted sum)
        portfolio_returns = pd.Series(0.0, index=returns_df.index)
        for ticker, weight in normalized_weights.items():
            if ticker in returns_df.columns:
                portfolio_returns += returns_df[ticker] * weight
    else:
        # Equal weighted if no weights provided
        returns_df = pd.DataFrame(returns_dict)
        portfolio_returns = returns_df.mean(axis=1)
    
    # Calculate metrics
    var_95 = calculate_var(portfolio_returns, 0.95)
    var_99 = calculate_var(portfolio_returns, 0.99)
    cvar_95 = calculate_cvar(portfolio_returns, 0.95)
    cvar_99 = calculate_cvar(portfolio_returns, 0.99)
    
    volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
    downside_vol = calculate_downside_volatility(portfolio_returns)
    
    # Calculate portfolio value series
    portfolio_value = (1 + portfolio_returns).cumprod()
    
    max_dd = calculate_max_drawdown(portfolio_value)
    current_dd = calculate_drawdowns(portfolio_value).iloc[-1] if len(portfolio_value) > 0 else 0.0
    
    sortino = calculate_sortino_ratio(portfolio_returns)
    calmar = calculate_calmar_ratio(portfolio_value)
    
    skewness = portfolio_returns.skew() if len(portfolio_returns) > 3 else 0.0
    kurtosis = portfolio_returns.kurtosis() if len(portfolio_returns) > 3 else 0.0
    
    return RiskMetrics(
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        volatility=volatility,
        downside_volatility=downside_vol,
        max_drawdown=max_dd,
        current_drawdown=current_dd,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        skewness=skewness,
        kurtosis=kurtosis
    )


def get_risk_summary_for_llm(metrics: RiskMetrics) -> str:
    """
    Format risk metrics for inclusion in LLM prompt.
    
    Args:
        metrics: RiskMetrics object
    
    Returns:
        Formatted string for prompt
    """
    return f"""Risk Metrics (Annualized):
- VaR 95%: {metrics.var_95:.2%} (max loss on 5% of days)
- CVaR 95%: {metrics.cvar_95:.2%} (expected loss in tail events)
- Volatility: {metrics.volatility:.2%}
- Downside Volatility: {metrics.downside_volatility:.2%}
- Max Drawdown: {metrics.max_drawdown:.2%}
- Current Drawdown: {metrics.current_drawdown:.2%}
- Sortino Ratio: {metrics.sortino_ratio:.2f}
- Skewness: {metrics.skewness:.2f} (negative = left tail risk)
- Kurtosis: {metrics.kurtosis:.2f} (high = fat tails)"""


if __name__ == "__main__":
    # Test with mock data
    print("Testing risk metrics calculation...")
    
    # Generate mock price data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
    
    # Asset 1: Higher volatility, negative skew
    returns1 = np.random.normal(0.0005, 0.02, 252)
    returns1[returns1 < -0.04] *= 1.5  # Fat left tail
    prices1 = pd.Series(100 * (1 + returns1).cumprod(), index=dates)
    
    # Asset 2: Lower volatility
    returns2 = np.random.normal(0.0003, 0.015, 252)
    prices2 = pd.Series(100 * (1 + returns2).cumprod(), index=dates)
    
    prices_dict = {"ASSET1": prices1, "ASSET2": prices2}
    weights = {"ASSET1": 0.6, "ASSET2": 0.4}
    
    metrics = calculate_portfolio_risk_metrics(prices_dict, weights)
    
    print("\nRisk Metrics:")
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value:.4f}")
    
    print("\nLLM Prompt Format:")
    print(get_risk_summary_for_llm(metrics))

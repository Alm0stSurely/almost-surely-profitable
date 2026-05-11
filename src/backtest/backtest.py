"""
Backtesting framework for the trading agent.
Tests strategy performance on historical data.
"""

import json
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.fetch_market_data import fetch_historical_data
from data.indicators import calculate_all_indicators, get_latest_indicators
from portfolio.portfolio import Portfolio
from llm.trading_agent import TradingAgent
from backtest.backtest_cooldown import BacktestCooldownManager, CooldownConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomStrategy:
    """
    Random baseline strategy for backtesting.
    Generates random buy/sell/hold decisions to establish
    a statistical baseline for strategy comparison.
    """
    
    def __init__(self, seed: Optional[int] = None, max_position_pct: float = 30.0):
        self.rng = np.random.RandomState(seed)
        self.max_position_pct = max_position_pct
        self.actions = ["buy", "sell", "hold"]
        self.weights = [0.25, 0.25, 0.50]  # More holds than trades
    
    def generate_decisions(
        self,
        tickers: List[str],
        portfolio: Portfolio,
        current_prices: Dict[str, float]
    ) -> List[Dict]:
        """Generate random trading decisions for given tickers."""
        decisions = []
        
        for ticker in tickers:
            if ticker not in current_prices:
                continue
            
            action = self.rng.choice(self.actions, p=self.weights)
            
            if action == "hold":
                decisions.append({"ticker": ticker, "action": "hold", "pct": 0})
                continue
            
            if action == "buy":
                # Random allocation between 5% and max_position_pct
                pct = self.rng.uniform(5.0, self.max_position_pct)
                decisions.append({"ticker": ticker, "action": "buy", "pct": round(pct, 1)})
            
            elif action == "sell":
                # Only sell if we have a position
                pos = portfolio.positions.get(ticker)
                if pos is not None and getattr(pos, 'quantity', 0) > 0:
                    # Random partial or full sell
                    pct = self.rng.uniform(25.0, 100.0)
                    decisions.append({"ticker": ticker, "action": "sell", "pct": round(pct, 1)})
        
        return decisions


class BacktestEngine:
    """
    Backtesting engine for paper trading strategies.
    Simulates trading decisions on historical data.
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0,
        tickers: Optional[List[str]] = None,
        rebalance_frequency: str = "daily",  # daily, weekly
        cooldown_config: Optional[CooldownConfig] = None,
        enable_cooldowns: bool = False,
    ):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.initial_capital = initial_capital
        self.tickers = tickers or ["SPY", "QQQ", "GLD"]
        self.rebalance_frequency = rebalance_frequency
        self.enable_cooldowns = enable_cooldowns
        self.cooldown_manager = BacktestCooldownManager(config=cooldown_config) if enable_cooldowns else None
        
        self.portfolio = None
        self.results = []
        
    def fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for backtest period."""
        # Use start/end dates for absolute date range (not relative period)
        start_str = self.start_date.strftime("%Y-%m-%d")
        end_str = self.end_date.strftime("%Y-%m-%d")
        logger.info(f"Fetching data for {len(self.tickers)} tickers, from {start_str} to {end_str}")
        
        data = fetch_historical_data(
            self.tickers,
            start=start_str,
            end=end_str,
            interval="1d"
        )
        
        # Filter to backtest date range using datetime comparison
        filtered_data = {}
        for ticker, df in data.items():
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            if not df.empty:
                filtered_data[ticker] = df
        
        return filtered_data
    
    def calculate_indicators_for_date(
        self,
        data: Dict[str, pd.DataFrame],
        current_date: datetime
    ) -> Dict:
        """Calculate indicators for all assets up to current date."""
        market_data = {}
        
        for ticker, df in data.items():
            # Get data up to current date (normalize to date for daily data)
            mask = df.index.date <= current_date.date()
            hist = df[mask]
            
            if len(hist) < 20:
                continue
            
            # Calculate indicators
            df_with_ind = calculate_all_indicators(hist.copy())
            
            market_data[ticker] = {
                "dataframe": df_with_ind,
                "latest": get_latest_indicators(df_with_ind),
                "total_return": (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1 if len(hist) > 1 else 0
            }
        
        return market_data
    
    def _get_benchmark_returns(self, data: Dict[str, pd.DataFrame], benchmark_ticker: str = "SPY") -> List[float]:
        """Extract benchmark daily returns for beta calculation."""
        if benchmark_ticker not in data:
            return []
        
        df = data[benchmark_ticker]
        # Filter to backtest period using datetime comparison
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
        
        # Vectorized return calculation via numpy
        closes = df['Close'].values
        if len(closes) < 2:
            return []
        returns = np.diff(closes) / closes[:-1]
        return returns.tolist()

    def run_backtest(
        self,
        use_llm: bool = False,
        strategy: str = "buy_and_hold",  # buy_and_hold, equal_weight, llm, random
        benchmark: str = "SPY",
        random_seed: Optional[int] = None
    ) -> Dict:
        """
        Run the backtest simulation.
        
        Args:
            use_llm: Whether to use LLM for decisions (slower)
            strategy: Trading strategy to use
        
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting backtest from {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Strategy: {strategy}, Use LLM: {use_llm}, Cooldowns: {self.enable_cooldowns}")
        
        # Fetch data
        data = self.fetch_historical_data()
        if not data:
            logger.error("No data fetched")
            return {}
        
        # Pre-compute price lookups for fast O(1) date access
        logger.info("Pre-computing price lookups...")
        self._precompute_price_lookups(data)
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            state_file=f"backtest_{strategy}_{self.start_date.strftime('%Y%m%d')}.json",
            data_dir="data/backtest"
        )
        
        # Get trading dates
        trading_dates = self._get_trading_dates(data)
        logger.info(f"Backtesting over {len(trading_dates)} trading days")
        
        # Initialize LLM agent if needed
        agent = TradingAgent() if use_llm else None
        
        # Initialize random strategy if needed
        random_strategy = RandomStrategy(seed=random_seed) if strategy == "random" else None
        
        # Run simulation
        for i, current_date in enumerate(trading_dates):
            if i % 20 == 0:
                logger.info(f"Processing day {i+1}/{len(trading_dates)}: {current_date.date()}")
            
            # Get prices for current date
            current_prices = self._get_prices_for_date(data, current_date)
            
            # Update portfolio with current prices
            self.portfolio.update_prices(current_prices)
            
            # Check if rebalance day
            should_rebalance = self._should_rebalance(i, current_date)
            
            if should_rebalance:
                if strategy == "llm" and use_llm:
                    self._execute_llm_strategy(data, current_date, agent, current_prices)
                elif strategy == "equal_weight":
                    self._execute_equal_weight_strategy(current_date, current_prices)
                elif strategy == "random" and random_strategy:
                    self._execute_random_strategy(current_date, current_prices, random_strategy)
                elif strategy == "buy_and_hold":
                    self._execute_buy_and_hold_strategy(current_date, current_prices)
            
            # Record daily result
            self._record_daily_result(current_date, current_prices)
        
        # Calculate final metrics with benchmark
        benchmark_returns = self._get_benchmark_returns(data, benchmark)
        metrics = self._calculate_metrics(benchmark_returns)
        
        result = {
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "strategy": strategy,
            "initial_capital": self.initial_capital,
            "final_value": self.portfolio.total_value,
            "total_return": metrics["total_return"],
            "annualized_return": metrics["annualized_return"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "calmar_ratio": metrics["calmar_ratio"],
            "omega_ratio": metrics["omega_ratio"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "beta": metrics["beta"],
            "alpha": metrics["alpha"],
            "volatility": metrics["volatility"],
            "num_trades": metrics["num_trades"],
            "equity_curve": metrics["equity_curve"],
            "drawdown_curve": metrics["drawdown_curve"],
            "daily_returns": metrics["daily_returns"],
            "daily_results": self.results
        }
        
        # Add cooldown metrics if enabled
        if self.enable_cooldowns and self.cooldown_manager:
            result["cooldown_metrics"] = self.cooldown_manager.get_metrics()
            result["cooldown_config"] = {
                "min_hold_days": self.cooldown_manager.config.min_hold_days,
                "flip_cooldown_days": self.cooldown_manager.config.flip_cooldown_days,
                "max_trades_per_week": self.cooldown_manager.config.max_trades_per_week,
                "stop_loss_threshold_pct": self.cooldown_manager.config.stop_loss_threshold_pct,
            }
        
        return result
    
    def _get_trading_dates(self, data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Get list of trading dates from data."""
        # Use the first ticker's dates
        first_ticker = list(data.keys())[0]
        dates = pd.to_datetime(data[first_ticker].index).tolist()
        return [d for d in dates if self.start_date <= d <= self.end_date]
    
    def _precompute_price_lookups(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Pre-compute price lookups for O(1) date access during backtest.
        
        The original approach called strftime() on the entire DataFrame index
        for every ticker on every day -- O(n_tickers * n_days * n_rows) overall.
        Pre-computing a nested dict reduces _get_prices_for_date to O(n_tickers).
        """
        self._price_lookups = {}
        for ticker, df in data.items():
            # Build {date_obj: close_price} using vectorized index access
            self._price_lookups[ticker] = {
                idx.date(): float(row['Close'])
                for idx, row in df.iterrows()
            }
    
    def _get_prices_for_date(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime
    ) -> Dict[str, float]:
        """Get closing prices for all tickers on given date."""
        prices = {}
        date_only = date.date() if hasattr(date, 'date') else date
        
        for ticker, lookup in self._price_lookups.items():
            price = lookup.get(date_only)
            if price is not None:
                prices[ticker] = price
        
        return prices
    
    def _should_rebalance(self, day_index: int, current_date: datetime) -> bool:
        """Determine if we should rebalance on this day."""
        if self.rebalance_frequency == "daily":
            return True
        elif self.rebalance_frequency == "weekly":
            return day_index % 5 == 0
        return False
    
    def _execute_llm_strategy(
        self,
        data: Dict[str, pd.DataFrame],
        current_date: datetime,
        agent: TradingAgent,
        current_prices: Dict[str, float]
    ):
        """Execute LLM-based trading strategy."""
        # Calculate indicators
        market_data = self.calculate_indicators_for_date(data, current_date)
        
        # Get portfolio summary
        portfolio_summary = self.portfolio.get_summary()
        
        # Get LLM decision
        decision = agent.get_trading_decision(market_data, portfolio_summary)
        
        # Execute actions with cooldown checks
        for action in decision.get("actions", []):
            ticker = action.get("ticker")
            action_type = action.get("action")
            pct = action.get("pct", 0)
            
            if ticker not in current_prices:
                continue
            
            price = current_prices[ticker]
            
            if action_type == "buy":
                if self.enable_cooldowns and self.cooldown_manager:
                    allowed, reason = self.cooldown_manager.can_buy(ticker, current_date)
                    if not allowed:
                        logger.debug(f"Cooldown blocked BUY {ticker}: {reason}")
                        continue
                    self.cooldown_manager.record_entry(ticker, current_date)
                self.portfolio.buy(ticker, pct, price)
            elif action_type == "sell":
                pos = self.portfolio.positions.get(ticker)
                avg_price = pos.avg_price if pos else 0
                if self.enable_cooldowns and self.cooldown_manager:
                    allowed, reason = self.cooldown_manager.can_sell(ticker, current_date, price, avg_price)
                    if not allowed:
                        logger.debug(f"Cooldown blocked SELL {ticker}: {reason}")
                        continue
                    self.cooldown_manager.record_exit(ticker, current_date)
                self.portfolio.sell(ticker, price, pct=pct if pct > 0 else None)
    
    def _execute_equal_weight_strategy(self, current_date: datetime, current_prices: Dict[str, float]):
        """
        Execute equal weight strategy with periodic rebalancing.
        
        Buys equal weight on first day, then rebalances periodically
        to maintain equal allocation across all tickers.
        """
        n_tickers = len(current_prices)
        if n_tickers == 0:
            return
        
        target_pct_per_ticker = 90.0 / n_tickers
        
        if not self.portfolio.positions:
            # Initial purchase: buy equal weight in all tickers
            for ticker, price in current_prices.items():
                if self.enable_cooldowns and self.cooldown_manager:
                    self.cooldown_manager.record_entry(ticker, current_date)
                self.portfolio.buy(ticker, target_pct_per_ticker, price)
            return
        
        # Rebalance: check if any position deviates significantly from target
        total_value = self.portfolio.total_value
        if total_value <= 0:
            return
        
        # Calculate target dollar amount per ticker
        target_value_per_ticker = total_value * (target_pct_per_ticker / 100)
        
        # First, sell positions that are overweight
        for ticker, pos in list(self.portfolio.positions.items()):
            if ticker not in current_prices:
                continue
            current_value = pos.market_value
            if current_value > target_value_per_ticker * 1.05:  # 5% tolerance
                overweight_pct = ((current_value - target_value_per_ticker) / current_value) * 100
                if self.enable_cooldowns and self.cooldown_manager:
                    allowed, reason = self.cooldown_manager.can_sell(ticker, current_date, current_prices[ticker], pos.avg_price)
                    if not allowed:
                        logger.debug(f"Cooldown blocked rebalance SELL {ticker}: {reason}")
                        continue
                    self.cooldown_manager.record_exit(ticker, current_date)
                self.portfolio.sell(ticker, current_prices[ticker], pct=overweight_pct)
        
        # Then, buy positions that are underweight
        for ticker, price in current_prices.items():
            if ticker in self.portfolio.positions:
                current_value = self.portfolio.positions[ticker].market_value
            else:
                current_value = 0
            
            if current_value < target_value_per_ticker * 0.95:  # 5% tolerance
                underweight_value = target_value_per_ticker - current_value
                # Convert to percentage of available cash
                if self.portfolio.cash > 0:
                    buy_pct = min((underweight_value / self.portfolio.cash) * 100, 100)
                    if buy_pct > 1:  # Only buy if meaningful
                        if self.enable_cooldowns and self.cooldown_manager:
                            allowed, reason = self.cooldown_manager.can_buy(ticker, current_date)
                            if not allowed:
                                logger.debug(f"Cooldown blocked rebalance BUY {ticker}: {reason}")
                                continue
                            self.cooldown_manager.record_entry(ticker, current_date)
                        self.portfolio.buy(ticker, buy_pct, price)
    
    def _execute_buy_and_hold_strategy(self, current_date: datetime, current_prices: Dict[str, float]):
        """Execute pure buy-and-hold strategy: buy equal weight once, never rebalance."""
        # Only execute once at start
        if self.portfolio.positions:
            return
        
        # Buy equal weight in all available tickers
        n_tickers = len(current_prices)
        if n_tickers == 0:
            return
        
        pct_per_ticker = 90.0 / n_tickers  # 90% invested, 10% cash buffer
        
        for ticker, price in current_prices.items():
            if self.enable_cooldowns and self.cooldown_manager:
                self.cooldown_manager.record_entry(ticker, current_date)
            self.portfolio.buy(ticker, pct_per_ticker, price)
    
    def _execute_random_strategy(
        self,
        current_date: datetime,
        current_prices: Dict[str, float],
        random_strategy: RandomStrategy
    ):
        """Execute random baseline trading strategy."""
        decisions = random_strategy.generate_decisions(
            tickers=list(current_prices.keys()),
            portfolio=self.portfolio,
            current_prices=current_prices
        )
        
        for action in decisions:
            ticker = action.get("ticker")
            action_type = action.get("action")
            pct = action.get("pct", 0)
            
            if ticker not in current_prices:
                continue
            
            price = current_prices[ticker]
            
            if action_type == "buy":
                if self.enable_cooldowns and self.cooldown_manager:
                    allowed, reason = self.cooldown_manager.can_buy(ticker, current_date)
                    if not allowed:
                        logger.debug(f"Cooldown blocked BUY {ticker}: {reason}")
                        continue
                    self.cooldown_manager.record_entry(ticker, current_date)
                self.portfolio.buy(ticker, pct, price)
            elif action_type == "sell":
                pos = self.portfolio.positions.get(ticker)
                avg_price = pos.avg_price if pos else 0
                if self.enable_cooldowns and self.cooldown_manager:
                    allowed, reason = self.cooldown_manager.can_sell(ticker, current_date, price, avg_price)
                    if not allowed:
                        logger.debug(f"Cooldown blocked SELL {ticker}: {reason}")
                        continue
                    self.cooldown_manager.record_exit(ticker, current_date)
                self.portfolio.sell(ticker, price, pct=pct if pct > 0 else None)
    
    def _record_daily_result(self, date: datetime, prices: Dict[str, float]):
        """Record daily portfolio state."""
        summary = self.portfolio.get_summary()
        
        self.results.append({
            "date": date.strftime("%Y-%m-%d"),
            "total_value": summary["total_value"],
            "cash": summary["cash"],
            "positions_value": summary["positions_value"],
            "total_return_pct": summary["total_return_pct"],
            "num_positions": summary["num_positions"]
        })
    
    def _calculate_metrics(self, benchmark_returns: Optional[List[float]] = None) -> Dict:
        """Calculate performance metrics."""
        if not self.results:
            return {}
        
        values = [r["total_value"] for r in self.results]
        returns = [(values[i] - values[i-1]) / values[i-1] 
                   for i in range(1, len(values))]
        
        # Total return
        total_return = (values[-1] / self.initial_capital) - 1
        
        # Annualized return
        days = len(self.results)
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) if returns else 0
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = ((annualized_return - risk_free_rate) / volatility 
                       if volatility > 0 else 0)
        
        # Max drawdown
        peak = self.initial_capital
        max_drawdown = 0
        drawdowns = []
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calmar ratio (annualized return / max drawdown)
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Omega ratio (sum of gains / sum of losses)
        gains = sum([r for r in returns if r > 0])
        losses = sum([abs(r) for r in returns if r < 0])
        omega_ratio = gains / losses if losses > 0 else float('inf')
        
        # Win rate (profitable days / total days)
        profitable_days = sum([1 for r in returns if r > 0])
        win_rate = profitable_days / len(returns) if returns else 0
        
        # Profit factor (gross profit / gross loss)
        gross_profit = sum([r for r in returns if r > 0])
        gross_loss = sum([abs(r) for r in returns if r < 0])
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sortino ratio (downside deviation only)
        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
        sortino_ratio = ((annualized_return - risk_free_rate) / downside_std 
                        if downside_std > 0 else 0)
        
        # Beta vs benchmark (if provided)
        beta = 0
        alpha = 0
        if benchmark_returns and len(benchmark_returns) == len(returns):
            covariance = np.cov(returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            alpha = annualized_return - (risk_free_rate + beta * (np.mean(benchmark_returns) * 252 - risk_free_rate))
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "omega_ratio": omega_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "num_trades": len(self.portfolio.trades),
            "beta": beta,
            "alpha": alpha,
            "equity_curve": values,
            "drawdown_curve": drawdowns,
            "daily_returns": returns
        }


def print_backtest_report(result: Dict, strategy_name: str):
    """Print formatted backtest report."""
    if not result:
        logger.error(f"No results for {strategy_name}")
        return
    
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS: {strategy_name.upper()}")
    print(f"{'='*70}")
    print(f"Period: {result['start_date']} to {result['end_date']}")
    print(f"Initial Capital: €{result['initial_capital']:,.2f}")
    print(f"Final Value: €{result['final_value']:,.2f}")
    print()
    print("RETURNS:")
    print(f"  Total Return:        {result['total_return']*100:>8.2f}%")
    print(f"  Annualized Return:   {result['annualized_return']*100:>8.2f}%")
    print()
    print("RISK METRICS:")
    print(f"  Volatility:          {result['volatility']*100:>8.2f}%")
    print(f"  Max Drawdown:        {result['max_drawdown']*100:>8.2f}%")
    print(f"  Sharpe Ratio:        {result['sharpe_ratio']:>8.2f}")
    print(f"  Sortino Ratio:       {result['sortino_ratio']:>8.2f}")
    print(f"  Calmar Ratio:        {result['calmar_ratio']:>8.2f}")
    print(f"  Omega Ratio:         {result['omega_ratio']:>8.2f}")
    print()
    print("TRADE STATISTICS:")
    print(f"  Number of Trades:    {result['num_trades']:>8}")
    print(f"  Win Rate:            {result['win_rate']*100:>8.2f}%")
    print(f"  Profit Factor:       {result['profit_factor']:>8.2f}")
    print()
    print("BENCHMARK RELATIVE:")
    print(f"  Beta (vs SPY):       {result['beta']:>8.3f}")
    print(f"  Alpha (vs SPY):      {result['alpha']*100:>8.2f}%")
    print(f"{'='*70}\n")


def run_comparison_backtest(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    tickers: Optional[List[str]] = None,
    include_llm: bool = False
) -> Dict:
    """
    Run backtest comparing multiple strategies.
    
    Args:
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        tickers: List of tickers to trade
        include_llm: Whether to include LLM strategy (slower, requires API key)
    
    Returns:
        Comparison results
    """
    tickers = tickers or ["SPY", "QQQ", "GLD"]
    
    strategies = ["buy_and_hold", "equal_weight", "random"]
    if include_llm:
        strategies.append("llm")
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running backtest: {strategy}")
        logger.info(f"{'='*60}")
        
        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            tickers=tickers,
            rebalance_frequency="daily"
        )
        
        use_llm = (strategy == "llm")
        kwargs = {"use_llm": use_llm, "strategy": strategy}
        if strategy == "random":
            kwargs["random_seed"] = 42
        result = engine.run_backtest(**kwargs)
        results[strategy] = result
        
        print_backtest_report(result, strategy)
    
    return results


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Run backtest for trading strategies")
    parser.add_argument("--start", default="2024-06-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--tickers", default="SPY,QQQ,GLD", help="Comma-separated tickers")
    parser.add_argument("--llm", action="store_true", help="Include LLM strategy (slower)")
    parser.add_argument("--output", default="results/backtest_comparison.json", help="Output file")
    
    args = parser.parse_args()
    
    tickers = args.tickers.split(",")
    
    print(f"\n{'#'*70}")
    print(f"# BACKTEST SESSION")
    print(f"# Period: {args.start} to {args.end}")
    print(f"# Universe: {tickers}")
    print(f"# LLM Strategy: {'Yes' if args.llm else 'No'}")
    print(f"{'#'*70}\n")
    
    # Run comparison backtest
    results = run_comparison_backtest(
        start_date=args.start,
        end_date=args.end,
        tickers=tickers,
        include_llm=args.llm
    )
    
    # Save results
    output_file = args.output
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    print(f"\nBacktest complete. Run 'python -m src.backtest.visualize' to generate charts.")

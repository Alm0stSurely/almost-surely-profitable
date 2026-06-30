"""
Live Equal-Weight Benchmark Tracker

Tracks an equal-weight portfolio in real-time alongside the LLM-driven strategy.
Provides a fair baseline for comparison.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class LiveEqualWeightBenchmark:
    """
    Live equal-weight benchmark for real-time strategy comparison.
    
    Holds equal weight in all available universe tickers.
    Rebalances daily (or on a configured frequency) to maintain equal weight.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        data_dir: str = "data",
        target_cash_buffer_pct: float = 10.0  # Keep 10% cash, invest 90%
    ):
        self.initial_capital = initial_capital
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.state_file = self.data_dir / "equalweight_benchmark_state.json"
        self.target_cash_buffer_pct = target_cash_buffer_pct
        
        self.shares: Dict[str, float] = {}
        self.cash: float = initial_capital
        self.start_date: Optional[str] = None
        self.last_rebalanced: Optional[str] = None
        
        self._load_state()
    
    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            self.shares = state.get("shares", {})
            self.cash = state.get("cash", self.initial_capital)
            self.start_date = state.get("start_date")
            self.last_rebalanced = state.get("last_rebalanced")
        except Exception:
            self.shares = {}
            self.cash = self.initial_capital
            self.start_date = None
            self.last_rebalanced = None
    
    def save_state(self) -> None:
        state = {
            "shares": self.shares,
            "cash": self.cash,
            "start_date": self.start_date,
            "last_rebalanced": self.last_rebalanced,
            "initial_capital": self.initial_capital,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def get_value(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate current benchmark value given current prices."""
        positions_value = 0.0
        position_details = {}
        
        for ticker, shares in self.shares.items():
            price = current_prices.get(ticker)
            if price is not None:
                value = shares * price
                positions_value += value
                position_details[ticker] = {
                    "shares": shares,
                    "price": price,
                    "value": value,
                }
        
        total_value = self.cash + positions_value
        total_return_pct = ((total_value - self.initial_capital) / self.initial_capital) * 100
        
        return {
            "total_value": total_value,
            "cash": self.cash,
            "positions_value": positions_value,
            "total_return_pct": total_return_pct,
            "num_positions": len(self.shares),
            "position_details": position_details,
        }
    
    def rebalance(self, current_prices: Dict[str, float]) -> Dict:
        """
        Rebalance to equal weight across all available tickers.
        
        Returns the updated benchmark value.
        """
        available_tickers = [t for t in current_prices.keys() if current_prices[t] > 0]
        n_tickers = len(available_tickers)
        
        if n_tickers == 0:
            return self.get_value(current_prices)
        
        # First time initialization
        if not self.start_date:
            self.start_date = datetime.now().strftime("%Y-%m-%d")
        
        # Calculate total value
        total_value = self.cash
        for ticker, shares in self.shares.items():
            price = current_prices.get(ticker)
            if price is not None:
                total_value += shares * price
        
        # Target allocation: invest (100 - cash_buffer)% equally across all tickers
        investable_pct = 1.0 - (self.target_cash_buffer_pct / 100)
        target_value_per_ticker = (total_value * investable_pct) / n_tickers
        
        # Rebalance: sell all positions first, then buy equal weight
        self.cash = total_value
        self.shares = {}
        
        for ticker in available_tickers:
            price = current_prices[ticker]
            if price > 0:
                shares = target_value_per_ticker / price
                cost = shares * price
                self.shares[ticker] = shares
                self.cash -= cost
        
        self.last_rebalanced = datetime.now().isoformat()
        self.save_state()
        
        return self.get_value(current_prices)
    
    def update(self, current_prices: Dict[str, float]) -> Dict:
        """Update benchmark value without rebalancing (just mark-to-market)."""
        return self.get_value(current_prices)
    
    def get_daily_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get a summary for daily reporting."""
        value = self.get_value(current_prices)
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_value": value["total_value"],
            "cash": value["cash"],
            "positions_value": value["positions_value"],
            "total_return_pct": value["total_return_pct"],
            "num_positions": value["num_positions"],
            "start_date": self.start_date,
            "last_rebalanced": self.last_rebalanced,
        }

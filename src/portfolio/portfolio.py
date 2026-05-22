"""
Paper portfolio management module.
Handles positions, cash, order execution, P&L calculation, and state persistence.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Position:
    """Represents a single position in the portfolio."""
    ticker: str
    quantity: float
    avg_price: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis of the position."""
        return self.quantity * self.avg_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100


@dataclass
class Trade:
    """Represents a single trade execution."""
    timestamp: str
    ticker: str
    action: str  # 'buy' or 'sell'
    quantity: float
    price: float
    total_value: float
    fees: float = 0.0
    realized_pnl: float = 0.0


class Portfolio:
    """
    Paper portfolio manager.
    
    Initial capital: 10,000 EUR
    Handles buy/sell orders, tracks P&L, persists state.
    """
    
    INITIAL_CAPITAL = 10000.0
    CURRENCY = "EUR"
    
    def __init__(
        self,
        state_file: str = "portfolio_state.json",
        trades_file: str = "trades_history.json",
        data_dir: str = "data"
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.state_file = self.data_dir / state_file
        self.trades_file = self.data_dir / trades_file
        
        self.cash: float = self.INITIAL_CAPITAL
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.total_realized_pnl: float = 0.0
        
        # Load existing state if available
        self.load_state()
    
    def load_state(self) -> None:
        """Load portfolio state from JSON file."""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.cash = state.get('cash', self.INITIAL_CAPITAL)
            self.total_realized_pnl = state.get('total_realized_pnl', 0.0)
            
            # Restore positions
            positions_data = state.get('positions', {})
            self.positions = {}
            for ticker, pos_data in positions_data.items():
                self.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=pos_data['quantity'],
                    avg_price=pos_data['avg_price'],
                    current_price=pos_data.get('current_price', 0.0)
                )
            
            print(f"✓ Loaded portfolio state: {len(self.positions)} positions, €{self.cash:.2f} cash")
            
        except Exception as e:
            print(f"Error loading state: {e}. Starting fresh.")
            self.cash = self.INITIAL_CAPITAL
            self.positions = {}
    
    def save_state(self) -> None:
        """Save portfolio state to JSON file."""
        state = {
            'cash': self.cash,
            'total_realized_pnl': self.total_realized_pnl,
            'positions': {
                ticker: {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price
                }
                for ticker, pos in self.positions.items()
            },
            'last_updated': datetime.now().isoformat(),
            'total_value': self.total_value
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def save_trade(self, trade: Trade) -> None:
        """Append trade to history file and in-memory list."""
        self.trades.append(trade)
        
        trades = []
        if self.trades_file.exists():
            try:
                with open(self.trades_file, 'r') as f:
                    trades = json.load(f)
            except:
                trades = []
        
        trades.append(asdict(trade))
        
        with open(self.trades_file, 'w') as f:
            json.dump(trades, f, indent=2)
    
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.total_realized_pnl + self.total_unrealized_pnl
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all positions."""
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price
    
    def buy(self, ticker: str, pct_of_cash: float, current_price: float) -> bool:
        """
        Execute a buy order.
        
        Args:
            ticker: Asset to buy
            pct_of_cash: Percentage of available cash to use (0-100)
            current_price: Current market price
        
        Returns:
            True if order executed successfully
        """
        if pct_of_cash <= 0 or pct_of_cash > 100:
            print(f"Invalid percentage: {pct_of_cash}")
            return False
        
        if current_price <= 0:
            print(f"Invalid price: {current_price}")
            return False
        
        cash_to_use = self.cash * (pct_of_cash / 100)
        quantity = cash_to_use / current_price
        
        if quantity < 0.001:
            print(f"Order too small: {quantity} shares")
            return False
        
        total_cost = quantity * current_price
        
        if total_cost > self.cash:
            print(f"Insufficient cash: €{self.cash:.2f} < €{total_cost:.2f}")
            return False
        
        # Execute trade
        if ticker in self.positions:
            # Update existing position
            pos = self.positions[ticker]
            total_quantity = pos.quantity + quantity
            new_avg_price = ((pos.quantity * pos.avg_price) + total_cost) / total_quantity
            pos.quantity = total_quantity
            pos.avg_price = new_avg_price
            pos.current_price = current_price
        else:
            # Create new position
            self.positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity,
                avg_price=current_price,
                current_price=current_price
            )
        
        self.cash -= total_cost
        
        # Record trade
        trade = Trade(
            timestamp=datetime.now().isoformat(),
            ticker=ticker,
            action='buy',
            quantity=quantity,
            price=current_price,
            total_value=total_cost
        )
        self.save_trade(trade)
        
        self.save_state()
        
        print(f"✓ BUY {ticker}: {quantity:.4f} @ €{current_price:.2f} = €{total_cost:.2f}")
        return True
    
    def sell(self, ticker: str, current_price: float, pct: Optional[float] = None) -> bool:
        """
        Execute a sell order.
        
        Args:
            ticker: Asset to sell
            current_price: Current market price
            pct: Percentage of position to sell (0-100). If None or 100, sells entire position.
        
        Returns:
            True if order executed successfully
        """
        if ticker not in self.positions:
            print(f"No position in {ticker}")
            return False
        
        pos = self.positions[ticker]
        
        # Determine sell percentage
        sell_pct = 100.0 if pct is None else pct
        if sell_pct <= 0 or sell_pct > 100:
            print(f"Invalid sell percentage: {sell_pct}")
            return False
        
        # Calculate quantity to sell
        quantity_to_sell = pos.quantity * (sell_pct / 100)
        if quantity_to_sell < 0.001:
            print(f"Sell quantity too small: {quantity_to_sell}")
            return False
        
        # Ensure we don't sell more than we have (floating point safety)
        quantity_to_sell = min(quantity_to_sell, pos.quantity)
        
        sale_value = quantity_to_sell * current_price
        # Realized PnL proportional to quantity sold (avg_price unchanged)
        cost_basis_sold = quantity_to_sell * pos.avg_price
        realized_pnl = sale_value - cost_basis_sold
        
        # Execute trade
        self.cash += sale_value
        self.total_realized_pnl += realized_pnl
        
        # Record trade
        trade = Trade(
            timestamp=datetime.now().isoformat(),
            ticker=ticker,
            action='sell',
            quantity=quantity_to_sell,
            price=current_price,
            total_value=sale_value,
            realized_pnl=realized_pnl
        )
        self.save_trade(trade)
        
        if sell_pct >= 99.99:
            print(f"✓ SELL {ticker}: {quantity_to_sell:.4f} @ €{current_price:.2f} = €{sale_value:.2f} (P&L: €{realized_pnl:+.2f})")
            # Remove position
            del self.positions[ticker]
        else:
            print(f"✓ SELL {ticker}: {quantity_to_sell:.4f} ({sell_pct:.1f}%) @ €{current_price:.2f} = €{sale_value:.2f} (P&L: €{realized_pnl:+.2f})")
            # Reduce position
            pos.quantity -= quantity_to_sell
            pos.current_price = current_price
        
        self.save_state()
        
        return True
    
    def get_summary(self) -> Dict:
        """Get portfolio summary."""
        return {
            'cash': self.cash,
            'positions_value': sum(pos.market_value for pos in self.positions.values()),
            'total_value': self.total_value,
            'total_return_pct': ((self.total_value / self.INITIAL_CAPITAL) - 1) * 100,
            'total_realized_pnl': self.total_realized_pnl,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'total_pnl': self.total_pnl,
            'num_positions': len(self.positions),
            'positions': [
                {
                    'ticker': pos.ticker,
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct
                }
                for pos in self.positions.values()
            ]
        }
    
    def print_summary(self) -> None:
        """Print portfolio summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PORTFOLIO SUMMARY")
        print("="*60)
        print(f"Cash:           €{summary['cash']:>12.2f}")
        print(f"Positions:      €{summary['positions_value']:>12.2f}")
        print(f"Total Value:    €{summary['total_value']:>12.2f}")
        print(f"Total Return:   {summary['total_return_pct']:>11.2f}%")
        print(f"Realized P&L:   €{summary['total_realized_pnl']:>+12.2f}")
        print(f"Unrealized P&L: €{summary['total_unrealized_pnl']:>+12.2f}")
        print("-"*60)
        
        if summary['positions']:
            print("\nPOSITIONS:")
            for pos in summary['positions']:
                print(f"  {pos['ticker']:<8} {pos['quantity']:>10.4f} @ €{pos['current_price']:.2f} = €{pos['market_value']:>10.2f} ({pos['unrealized_pnl_pct']:+.2f}%)")
        
        print("="*60)


if __name__ == "__main__":
    # Quick test
    print("Testing Portfolio...")
    
    portfolio = Portfolio(data_dir="/tmp/test_portfolio")
    
    # Simulate some trades
    portfolio.buy("SPY", 50, 400.0)
    portfolio.buy("MC.PA", 30, 800.0)
    
    # Update prices and print
    portfolio.update_prices({"SPY": 410.0, "MC.PA": 780.0})
    portfolio.print_summary()
    
    # Save state
    portfolio.save_state()
    print(f"\n✓ State saved to {portfolio.state_file}")

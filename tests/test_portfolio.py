"""
Test suite for Portfolio module.
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from portfolio.portfolio import Portfolio, Position


def test_position_creation():
    """Test Position object creation."""
    print("Test 1: Position Creation")
    print("-" * 40)
    
    pos = Position(
        ticker="SPY",
        quantity=10.0,
        avg_price=400.0,
        current_price=410.0
    )
    
    assert pos.ticker == "SPY"
    assert pos.quantity == 10.0
    assert pos.avg_price == 400.0
    assert pos.current_price == 410.0
    
    market_value = pos.market_value
    expected = 10.0 * 410.0
    assert market_value == expected, f"Market value {market_value} != {expected}"
    
    pnl = pos.unrealized_pnl
    expected_pnl = 10.0 * (410.0 - 400.0)
    assert pnl == expected_pnl, f"PnL {pnl} != {expected_pnl}"
    
    pnl_pct = pos.unrealized_pnl_pct
    expected_pct = ((410.0 - 400.0) / 400.0) * 100  # Module returns percentage
    assert abs(pnl_pct - expected_pct) < 1e-6, f"PnL% {pnl_pct} != {expected_pct}"
    
    print(f"  Ticker: {pos.ticker}")
    print(f"  Market Value: ${market_value:.2f}")
    print(f"  Unrealized PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
    print("✓ Position creation test passed\n")


def test_portfolio_initialization():
    """Test Portfolio initialization."""
    print("Test 2: Portfolio Initialization")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        portfolio = Portfolio(data_dir=tmpdir)
        
        assert portfolio.INITIAL_CAPITAL == 10000.0
        assert portfolio.cash == 10000.0
        assert len(portfolio.positions) == 0
        assert portfolio.total_value == 10000.0
        
        print(f"  Initial Capital: ${portfolio.INITIAL_CAPITAL:.2f}")
        print(f"  Cash: ${portfolio.cash:.2f}")
        print(f"  Positions: {len(portfolio.positions)}")
        print(f"  Total Value: ${portfolio.total_value:.2f}")
        print("✓ Portfolio initialization test passed\n")


def test_buy_order():
    """Test buy order execution."""
    print("Test 3: Buy Order Execution")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        portfolio = Portfolio(data_dir=tmpdir)
        
        # Use 40% of cash to buy SPY at $400
        result = portfolio.buy("SPY", 40.0, 400.0)
        
        assert result is True, "Buy order should succeed"
        assert "SPY" in portfolio.positions
        # 40% of 10000 = 4000, at $400 = 10 shares
        assert abs(portfolio.positions["SPY"].quantity - 10.0) < 0.01
        assert portfolio.positions["SPY"].avg_price == 400.0
        assert abs(portfolio.cash - 6000.0) < 0.01  # 60% remaining
        
        print(f"  Used 40% cash to buy SPY @ $400.00")
        print(f"  Quantity: {portfolio.positions['SPY'].quantity:.4f}")
        print(f"  Cash remaining: ${portfolio.cash:.2f}")
        print(f"  Position value: ${portfolio.positions['SPY'].market_value:.2f}")
        print("✓ Buy order test passed\n")


def test_sell_order():
    """Test sell order execution."""
    print("Test 4: Sell Order Execution")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        portfolio = Portfolio(data_dir=tmpdir)
        
        # Buy using 40% cash, then sell entire position
        portfolio.buy("SPY", 40.0, 400.0)  # 10 shares
        initial_cash = portfolio.cash
        
        result = portfolio.sell("SPY", 410.0)  # Sell entire position
        
        assert result is True, "Sell order should succeed"
        assert "SPY" not in portfolio.positions  # Position removed after sell
        expected_cash = initial_cash + (10.0 * 410.0)
        assert abs(portfolio.cash - expected_cash) < 0.01
        
        # Realized PnL should be tracked
        expected_pnl = 10.0 * (410.0 - 400.0)
        assert abs(portfolio.total_realized_pnl - expected_pnl) < 0.01
        
        print(f"  Sold entire SPY position @ $410.00")
        print(f"  Realized PnL: ${portfolio.total_realized_pnl:.2f}")
        print(f"  Cash after sale: ${portfolio.cash:.2f}")
        print("✓ Sell order test passed\n")


def test_partial_sell_order():
    """Test partial sell order execution."""
    print("Test 4b: Partial Sell Order Execution")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        portfolio = Portfolio(data_dir=tmpdir)
        
        # Buy using 50% cash = 12.5 shares @ $400
        portfolio.buy("SPY", 50.0, 400.0)
        initial_cash = portfolio.cash
        
        # Sell 40% of position = 5 shares @ $410
        result = portfolio.sell("SPY", 410.0, pct=40.0)
        
        assert result is True, "Partial sell order should succeed"
        assert "SPY" in portfolio.positions, "Position should remain after partial sell"
        
        # Remaining quantity: 12.5 * 0.6 = 7.5 shares
        assert abs(portfolio.positions["SPY"].quantity - 7.5) < 0.01
        # avg_price should remain unchanged
        assert abs(portfolio.positions["SPY"].avg_price - 400.0) < 0.01
        
        # Cash should increase by 5 * 410 = 2050
        expected_cash = initial_cash + (5.0 * 410.0)
        assert abs(portfolio.cash - expected_cash) < 0.01
        
        # Realized PnL: 5 * (410 - 400) = 50
        expected_pnl = 5.0 * (410.0 - 400.0)
        assert abs(portfolio.total_realized_pnl - expected_pnl) < 0.01
        
        print(f"  Sold 40% of SPY position @ $410.00")
        print(f"  Remaining quantity: {portfolio.positions['SPY'].quantity:.4f}")
        print(f"  Realized PnL: ${portfolio.total_realized_pnl:.2f}")
        print(f"  Cash after sale: ${portfolio.cash:.2f}")
        print("✓ Partial sell order test passed\n")


def test_insufficient_funds():
    """Test buy order with invalid parameters."""
    print("Test 5: Invalid Order Protection")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        portfolio = Portfolio(data_dir=tmpdir)
        
        # Try invalid percentage (>100%)
        result = portfolio.buy("SPY", 150.0, 400.0)
        
        assert result is False, "Buy order should fail"
        assert "SPY" not in portfolio.positions
        assert portfolio.cash == 10000.0  # Cash unchanged
        
        print(f"  Attempted to buy with 150% of cash")
        print(f"  Order rejected (invalid percentage)")
        print(f"  Cash unchanged: ${portfolio.cash:.2f}")
        print("✓ Invalid order test passed\n")


def test_position_update():
    """Test updating position prices."""
    print("Test 6: Position Price Update")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        portfolio = Portfolio(data_dir=tmpdir)
        portfolio.buy("SPY", 40.0, 400.0)  # 10 shares
        
        initial_value = portfolio.positions["SPY"].market_value
        
        # Update price
        portfolio.update_prices({"SPY": 420.0})
        
        assert portfolio.positions["SPY"].current_price == 420.0
        expected_value = portfolio.positions["SPY"].quantity * 420.0
        assert abs(portfolio.positions["SPY"].market_value - expected_value) < 0.01
        
        print(f"  Initial position value: ${initial_value:.2f}")
        print(f"  Updated price: $420.00")
        print(f"  New position value: ${portfolio.positions['SPY'].market_value:.2f}")
        print(f"  Unrealized PnL: ${portfolio.positions['SPY'].unrealized_pnl:.2f}")
        print("✓ Position update test passed\n")


def test_portfolio_persistence():
    """Test saving and loading portfolio state."""
    print("Test 7: Portfolio Persistence")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and modify portfolio
        portfolio1 = Portfolio(data_dir=tmpdir)
        portfolio1.buy("SPY", 40.0, 400.0)  # 40% of 10000 = 4000, at 400 = 10 shares
        # After SPY buy: cash = 6000, so 17.5% of remaining = 1050, at 350 = 3 shares
        portfolio1.buy("QQQ", 17.5, 350.0)  # 17.5% of remaining cash = 3 shares
        portfolio1.save_state()
        
        # Load in new instance
        portfolio2 = Portfolio(data_dir=tmpdir)
        
        assert "SPY" in portfolio2.positions
        assert "QQQ" in portfolio2.positions
        assert abs(portfolio2.positions["SPY"].quantity - 10.0) < 0.01
        assert abs(portfolio2.positions["QQQ"].quantity - 3.0) < 0.01  # 3 shares, not 5
        
        print(f"  Saved portfolio with SPY and QQQ positions")
        print(f"  Loaded portfolio: {len(portfolio2.positions)} positions")
        print(f"  SPY qty: {portfolio2.positions['SPY'].quantity:.4f}")
        print(f"  QQQ qty: {portfolio2.positions['QQQ'].quantity:.4f}")
        print("✓ Portfolio persistence test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Portfolio Module Test Suite")
    print("=" * 60)
    print()
    
    test_position_creation()
    test_portfolio_initialization()
    test_buy_order()
    test_sell_order()
    test_partial_sell_order()
    test_insufficient_funds()
    test_position_update()
    test_portfolio_persistence()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

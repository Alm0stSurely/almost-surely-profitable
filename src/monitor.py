#!/repos/almost-surely-profitable/.venv/bin/python3
"""
Intraday monitoring script.
Checks for significant price movements and alerts if thresholds are breached.
Called every 2 hours during market hours (8h-20h UTC) by external cron.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from data.fetch_market_data import fetch_current_prices
from portfolio.portfolio import Portfolio


# Alert thresholds
THRESHOLDS = {
    'position_movement_pct': 2.0,  # Alert if portfolio position moves > 2%
    'index_movement_pct': 3.0,     # Alert if index moves > 3%
    'portfolio_drawdown_pct': 1.5  # Alert if portfolio draws down > 1.5%
}

# Indices to monitor
INDICES = ["SPY", "^FCHI"]  # S&P 500 and CAC 40


def load_previous_close(portfolio: Portfolio) -> Dict[str, float]:
    """
    Load previous closing prices from portfolio state or market data.
    """
    state_file = Path("data/market_state.json")
    
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            return state.get('previous_close', {})
        except:
            pass
    
    # If no saved state, use average price of positions as reference
    return {ticker: pos.avg_price for ticker, pos in portfolio.positions.items()}


def save_market_state(prices: Dict[str, float]):
    """Save current prices as reference for next check."""
    state_file = Path("data/market_state.json")
    state_file.parent.mkdir(exist_ok=True)
    
    state = {
        'timestamp': datetime.now().isoformat(),
        'previous_close': prices
    }
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def check_movements(
    current_prices: Dict[str, float],
    reference_prices: Dict[str, float],
    portfolio: Portfolio
) -> List[Dict]:
    """
    Check for significant price movements.
    
    Returns:
        List of alerts
    """
    alerts = []
    
    # Check portfolio positions
    for ticker, position in portfolio.positions.items():
        current_price = current_prices.get(ticker)
        if not current_price:
            continue
        
        # Use position average price as reference
        reference_price = position.avg_price
        
        if reference_price > 0:
            movement_pct = ((current_price - reference_price) / reference_price) * 100
            
            if abs(movement_pct) >= THRESHOLDS['position_movement_pct']:
                alerts.append({
                    'type': 'position_movement',
                    'ticker': ticker,
                    'severity': 'high' if abs(movement_pct) > 5 else 'medium',
                    'current_price': current_price,
                    'reference_price': reference_price,
                    'movement_pct': movement_pct,
                    'position_size': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl
                })
    
    # Check indices
    for index in INDICES:
        current_price = current_prices.get(index)
        reference_price = reference_prices.get(index)
        
        if current_price and reference_price:
            movement_pct = ((current_price - reference_price) / reference_price) * 100
            
            if abs(movement_pct) >= THRESHOLDS['index_movement_pct']:
                alerts.append({
                    'type': 'index_movement',
                    'ticker': index,
                    'severity': 'high',
                    'current_price': current_price,
                    'reference_price': reference_price,
                    'movement_pct': movement_pct
                })
    
    # Check portfolio drawdown
    if portfolio.positions:
        total_cost = sum(pos.cost_basis for pos in portfolio.positions.values())
        total_current = sum(
            current_prices.get(ticker, pos.current_price) * pos.quantity
            for ticker, pos in portfolio.positions.items()
        )
        
        if total_cost > 0:
            drawdown_pct = ((total_current - total_cost) / total_cost) * 100
            
            if drawdown_pct <= -THRESHOLDS['portfolio_drawdown_pct']:
                alerts.append({
                    'type': 'portfolio_drawdown',
                    'ticker': 'PORTFOLIO',
                    'severity': 'critical',
                    'current_value': total_current,
                    'cost_basis': total_cost,
                    'drawdown_pct': drawdown_pct
                })
    
    return alerts


def run_monitor():
    """Run the monitoring check."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting intraday monitor...")
    
    # Load portfolio
    portfolio = Portfolio(data_dir="data")
    
    # Get tickers to monitor (positions + indices)
    tickers_to_monitor = list(portfolio.positions.keys()) + INDICES
    
    if not tickers_to_monitor:
        print("No positions to monitor.")
        return [], 0
    
    # Fetch current prices
    print(f"Fetching prices for {len(tickers_to_monitor)} tickers...")
    current_prices = fetch_current_prices(tickers_to_monitor)
    
    # Load reference prices
    reference_prices = load_previous_close(portfolio)
    
    # Check for movements
    alerts = check_movements(current_prices, reference_prices, portfolio)
    
    # Save current prices for next check
    save_market_state(current_prices)
    
    # Output results
    if alerts:
        print(f"\n⚠️  {len(alerts)} ALERT(S) DETECTED:\n")
        
        for alert in alerts:
            print(f"Type: {alert['type'].upper()}")
            print(f"Ticker: {alert['ticker']}")
            print(f"Severity: {alert['severity'].upper()}")
            
            if alert['type'] == 'position_movement':
                print(f"Movement: {alert['movement_pct']:+.2f}%")
                print(f"Price: €{alert['current_price']:.2f} (ref: €{alert['reference_price']:.2f})")
                print(f"Position P&L: €{alert['unrealized_pnl']:+.2f}")
            elif alert['type'] == 'index_movement':
                print(f"Movement: {alert['movement_pct']:+.2f}%")
                print(f"Price: {alert['current_price']:.2f}")
            elif alert['type'] == 'portfolio_drawdown':
                print(f"Drawdown: {alert['drawdown_pct']:.2f}%")
                print(f"Value: €{alert['current_value']:.2f} (cost: €{alert['cost_basis']:.2f})")
            
            print("-" * 50)
        
        # Output JSON for external processing
        output = {
            'timestamp': datetime.now().isoformat(),
            'alert_count': len(alerts),
            'alerts': alerts,
            'portfolio_value': portfolio.total_value
        }
        print("\nJSON_OUTPUT:")
        print(json.dumps(output, indent=2))
        
        return alerts, 1  # Exit code 1 = alert triggered
    else:
        print("✓ No significant movements detected.")
        print(f"Portfolio Value: €{portfolio.total_value:.2f}")
        
        return [], 0  # Exit code 0 = normal


if __name__ == "__main__":
    try:
        alerts, exit_code = run_monitor()
        sys.exit(exit_code)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)  # Exit code 2 = error

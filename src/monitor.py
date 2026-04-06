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


# Load thresholds from config
CONFIG_PATH = Path(__file__).parent.parent / "config" / "monitor.json"
UNIVERSE_PATH = Path(__file__).parent.parent / "config" / "universe.json"

def load_monitor_config() -> dict:
    """Load monitor configuration from config/monitor.json."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load monitor config: {e}")
    
    # Default config
    return {
        'alert_thresholds': {
            'position_movement_pct': 2.0,
            'index_movement_pct': 3.0,
            'portfolio_drawdown_pct': 1.5,
            'bollinger_breakout_std': 2.0
        },
        'indices': ["SPY", "^FCHI"],
        'check_stop_losses': True,
        'stop_loss_threshold_pct': 5.0,
        'check_bollinger': True
    }

def load_universe() -> dict:
    """Load asset universe from config/universe.json."""
    if UNIVERSE_PATH.exists():
        try:
            with open(UNIVERSE_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load universe: {e}")
    return {}

# Initialize config
_monitor_config = load_monitor_config()
THRESHOLDS = _monitor_config.get('alert_thresholds', {})
INDICES = _monitor_config.get('indices', ["SPY", "^FCHI"])
CHECK_STOP_LOSSES = _monitor_config.get('check_stop_losses', True)
STOP_LOSS_THRESHOLD = _monitor_config.get('stop_loss_threshold_pct', 5.0)
CHECK_BOLLINGER = _monitor_config.get('check_bollinger', True)


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


def check_stop_losses(
    current_prices: Dict[str, float],
    portfolio: Portfolio
) -> List[Dict]:
    """Check if any positions have hit their stop-loss threshold."""
    alerts = []
    
    if not CHECK_STOP_LOSSES:
        return alerts
    
    for ticker, position in portfolio.positions.items():
        current_price = current_prices.get(ticker)
        if not current_price or position.avg_price <= 0:
            continue
        
        drawdown_pct = ((current_price - position.avg_price) / position.avg_price) * 100
        
        if drawdown_pct <= -STOP_LOSS_THRESHOLD:
            alerts.append({
                'type': 'stop_loss_triggered',
                'ticker': ticker,
                'severity': 'critical',
                'current_price': current_price,
                'entry_price': position.avg_price,
                'drawdown_pct': drawdown_pct,
                'stop_threshold': STOP_LOSS_THRESHOLD,
                'action_required': 'SELL'
            })
    
    return alerts


def check_bollinger_breakouts(
    current_prices: Dict[str, float],
    portfolio: Portfolio
) -> List[Dict]:
    """Check for Bollinger Band breakouts."""
    alerts = []
    
    if not CHECK_BOLLINGER:
        return alerts
    
    # Import indicators here to avoid circular imports
    try:
        from data.indicators import calculate_bollinger_bands
        import pandas as pd
        from data.fetch_market_data import fetch_market_data
    except ImportError:
        return alerts
    
    for ticker in portfolio.positions.keys():
        current_price = current_prices.get(ticker)
        if not current_price:
            continue
        
        try:
            # Fetch recent data for Bollinger calculation
            df = fetch_market_data(ticker, period='20d')
            if df is None or len(df) < 20:
                continue
            
            upper, middle, lower = calculate_bollinger_bands(df['Close'])
            
            if upper is None or lower is None:
                continue
            
            latest_upper = upper.iloc[-1]
            latest_lower = lower.iloc[-1]
            
            # Check for breakout
            if current_price > latest_upper:
                alerts.append({
                    'type': 'bollinger_breakout',
                    'ticker': ticker,
                    'severity': 'medium',
                    'current_price': current_price,
                    'bollinger_upper': float(latest_upper),
                    'direction': 'upper',
                    'interpretation': 'Overbought - potential mean reversion'
                })
            elif current_price < latest_lower:
                alerts.append({
                    'type': 'bollinger_breakout',
                    'ticker': ticker,
                    'severity': 'medium',
                    'current_price': current_price,
                    'bollinger_lower': float(latest_lower),
                    'direction': 'lower',
                    'interpretation': 'Oversold - potential bounce'
                })
        except Exception as e:
            # Silently skip if calculation fails
            continue
    
    return alerts


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
    
    # Check stop-losses first (highest priority)
    stop_alerts = check_stop_losses(current_prices, portfolio)
    alerts.extend(stop_alerts)
    
    # Check portfolio positions for significant movements
    for ticker, position in portfolio.positions.items():
        current_price = current_prices.get(ticker)
        if not current_price:
            continue
        
        # Use position average price as reference
        reference_price = position.avg_price
        
        if reference_price > 0:
            movement_pct = ((current_price - reference_price) / reference_price) * 100
            
            # Skip if stop-loss already triggered (avoid duplicate alerts)
            if any(a['ticker'] == ticker and a['type'] == 'stop_loss_triggered' for a in stop_alerts):
                continue
            
            if abs(movement_pct) >= THRESHOLDS.get('position_movement_pct', 2.0):
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
            
            threshold = THRESHOLDS.get('portfolio_drawdown_pct', 1.5)
            if drawdown_pct <= -threshold:
                alerts.append({
                    'type': 'portfolio_drawdown',
                    'ticker': 'PORTFOLIO',
                    'severity': 'critical',
                    'current_value': total_current,
                    'cost_basis': total_cost,
                    'drawdown_pct': drawdown_pct
                })
    
    # Check Bollinger Band breakouts
    bollinger_alerts = check_bollinger_breakouts(current_prices, portfolio)
    alerts.extend(bollinger_alerts)
    
    return alerts


# Wrapper functions for backward compatibility and testing
def check_position_movements(positions: Dict, previous_close: Dict, current_prices: Dict, threshold_pct: float = 2.0) -> List[Dict]:
    """
    Check for significant position movements (wrapper for check_movements).
    
    Args:
        positions: Dict of position data
        previous_close: Previous closing prices
        current_prices: Current market prices
        threshold_pct: Alert threshold percentage
        
    Returns:
        List of movement alerts
    """
    from portfolio.portfolio import Portfolio, Position
    
    # Create a mock portfolio from positions dict
    portfolio = Portfolio(data_dir="data")
    for ticker, pos_data in positions.items():
        portfolio.positions[ticker] = Position(
            ticker=ticker,
            quantity=pos_data.get('quantity', 0),
            avg_price=pos_data.get('avg_price', 0),
            current_price=pos_data.get('current_price', pos_data.get('avg_price', 0))
        )
    
    all_alerts = check_movements(current_prices, previous_close, portfolio)
    
    # Filter only position movement alerts
    return [a for a in all_alerts if a['type'] == 'position_movement']


def check_portfolio_drawdown(portfolio_value: float, cost_basis: float, threshold_pct: float = 1.5) -> Dict:
    """
    Check if portfolio drawdown exceeds threshold.
    
    Args:
        portfolio_value: Current portfolio value
        cost_basis: Total cost basis
        threshold_pct: Drawdown threshold percentage
        
    Returns:
        Alert dict if drawdown exceeds threshold, None otherwise
    """
    if cost_basis <= 0:
        return None
    
    drawdown_pct = ((portfolio_value - cost_basis) / cost_basis) * 100
    
    if drawdown_pct <= -threshold_pct:
        return {
            'type': 'portfolio_drawdown',
            'ticker': 'PORTFOLIO',
            'severity': 'critical',
            'current_value': portfolio_value,
            'cost_basis': cost_basis,
            'drawdown_pct': drawdown_pct
        }
    
    return None


def check_index_movements(current_prices: Dict, reference_prices: Dict, indices: List[str], threshold_pct: float = 3.0) -> List[Dict]:
    """
    Check for significant index movements.
    
    Args:
        current_prices: Current market prices
        reference_prices: Reference prices (previous close)
        indices: List of index tickers to check
        threshold_pct: Alert threshold percentage
        
    Returns:
        List of index movement alerts
    """
    alerts = []
    
    for index in indices:
        current_price = current_prices.get(index)
        reference_price = reference_prices.get(index)
        
        if current_price and reference_price and reference_price > 0:
            movement_pct = ((current_price - reference_price) / reference_price) * 100
            
            if abs(movement_pct) >= threshold_pct:
                alerts.append({
                    'type': 'index_movement',
                    'ticker': index,
                    'severity': 'high',
                    'current_price': current_price,
                    'reference_price': reference_price,
                    'movement_pct': movement_pct
                })
    
    return alerts


def generate_alerts(alerts_or_current_prices, reference_prices=None, portfolio=None) -> Dict:
    """
    Generate alert summary from alerts list or by checking movements.
    
    Can be called in two ways:
    1. generate_alerts(alerts_list) - Format alerts for output (used by tests)
    2. generate_alerts(current_prices, reference_prices, portfolio) - Generate from data
    
    Args:
        alerts_or_current_prices: Either a list of alerts or current prices dict
        reference_prices: Reference prices (required for mode 2)
        portfolio: Portfolio object (required for mode 2)
        
    Returns:
        Dict with 'alert_count' and 'alerts' keys
    """
    # Mode 1: Called with pre-generated alerts list (from tests)
    if isinstance(alerts_or_current_prices, list):
        alerts = alerts_or_current_prices
    # Mode 2: Called with price data - generate alerts
    elif reference_prices is not None and portfolio is not None:
        alerts = check_movements(alerts_or_current_prices, reference_prices, portfolio)
    else:
        alerts = []
    
    return {
        'alert_count': len(alerts),
        'alerts': alerts,
        'timestamp': datetime.now().isoformat()
    }


def run_monitor():
    """Run the monitoring check."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting intraday monitor...")
    
    # Load portfolio
    portfolio = Portfolio(data_dir="data")
    
    # Get tickers to monitor (positions + indices)
    tickers_to_monitor = list(portfolio.positions.keys()) + INDICES
    
    # Filter out empty or invalid tickers
    tickers_to_monitor = [t.strip() for t in tickers_to_monitor if t and t.strip() and t.strip() != ".PA"]
    
    # Remove duplicates while preserving order
    seen = set()
    tickers_to_monitor = [t for t in tickers_to_monitor if not (t in seen or seen.add(t))]
    
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
            elif alert['type'] == 'stop_loss_triggered':
                print(f"🚨 STOP-LOSS TRIGGERED 🚨")
                print(f"Drawdown: {alert['drawdown_pct']:+.2f}% (threshold: -{alert['stop_threshold']}%)")
                print(f"Price: €{alert['current_price']:.2f} (entry: €{alert['entry_price']:.2f})")
                print(f"ACTION REQUIRED: {alert['action_required']}")
            elif alert['type'] == 'bollinger_breakout':
                print(f"Direction: {alert['direction'].upper()} breakout")
                print(f"Price: €{alert['current_price']:.2f}")
                if alert['direction'] == 'upper':
                    print(f"Bollinger Upper: €{alert['bollinger_upper']:.2f}")
                else:
                    print(f"Bollinger Lower: €{alert['bollinger_lower']:.2f}")
                print(f"Interpretation: {alert['interpretation']}")
            
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

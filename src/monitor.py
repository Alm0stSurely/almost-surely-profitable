#!/repos/almost-surely-profitable/.venv/bin/python3
"""
Intraday monitoring script with alert deduplication.
Checks for significant price movements and alerts if thresholds are breached.
Called every 2 hours during market hours (8h-20h UTC) by external cron.

Alert Deduplication Logic:
- First alert for a ticker: triggers immediately
- Subsequent alerts for same ticker: suppressed for COOLDOWN_HOURS (default 6h)
- Escalation alert: triggers if movement exceeds previous by ESCALATION_THRESHOLD (default 2%)
- Session reset: all alerts reset at market open (08:00 UTC)
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from data.fetch_market_data import fetch_current_prices
from portfolio.portfolio import Portfolio


# Load thresholds from config
CONFIG_PATH = Path(__file__).parent.parent / "config" / "monitor.json"
UNIVERSE_PATH = Path(__file__).parent.parent / "config" / "universe.json"
ALERT_HISTORY_PATH = Path(__file__).parent.parent / "data" / "alert_history.json"

# Alert deduplication settings
COOLDOWN_HOURS = 6  # Don't re-alert same ticker within 6 hours
ESCALATION_THRESHOLD_PCT = 2.0  # Re-alert if movement exceeds previous by 2%


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
        'check_bollinger': True,
        'cooldown_hours': COOLDOWN_HOURS,
        'escalation_threshold_pct': ESCALATION_THRESHOLD_PCT
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


def load_alert_history() -> Dict:
    """Load alert history for deduplication."""
    if ALERT_HISTORY_PATH.exists():
        try:
            with open(ALERT_HISTORY_PATH, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        'alerts': [],
        'last_reset': datetime.now().isoformat()
    }


def save_alert_history(history: Dict):
    """Save alert history."""
    ALERT_HISTORY_PATH.parent.mkdir(exist_ok=True)
    with open(ALERT_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)


def should_alert(ticker: str, movement_pct: float, alert_type: str, history: Dict) -> Tuple[bool, str]:
    """
    Determine if we should alert for this ticker.
    
    Returns:
        (should_alert: bool, reason: str)
    """
    now = datetime.now()
    
    # Check if market just opened (reset history at 08:00 UTC)
    if now.hour == 8 and now.minute < 30:
        # Clear history at market open
        history['alerts'] = []
        history['last_reset'] = now.isoformat()
        return True, "New trading session"
    
    # Find previous alert for this ticker
    previous_alerts = [
        a for a in history.get('alerts', [])
        if a.get('ticker') == ticker and a.get('type') == alert_type
    ]
    
    if not previous_alerts:
        return True, "First alert for this ticker"
    
    # Get most recent alert
    last_alert = max(previous_alerts, key=lambda x: x.get('timestamp', ''))
    last_time = datetime.fromisoformat(last_alert['timestamp'])
    last_movement = last_alert.get('movement_pct', 0)
    
    # Check cooldown period
    cooldown = timedelta(hours=COOLDOWN_HOURS)
    if now - last_time < cooldown:
        # Within cooldown - check for escalation
        movement_diff = abs(movement_pct) - abs(last_movement)
        if movement_diff >= ESCALATION_THRESHOLD_PCT:
            return True, f"Escalation: movement increased by {movement_diff:.2f}%"
        else:
            return False, f"Within cooldown ({COOLDOWN_HOURS}h), no escalation"
    
    # Cooldown expired
    return True, "Cooldown expired"


def record_alert(ticker: str, movement_pct: float, alert_type: str, severity: str, history: Dict):
    """Record an alert in history."""
    history['alerts'].append({
        'ticker': ticker,
        'type': alert_type,
        'movement_pct': movement_pct,
        'severity': severity,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 100 alerts to prevent file bloat
    history['alerts'] = history['alerts'][-100:]


# Initialize config
_monitor_config = load_monitor_config()
THRESHOLDS = _monitor_config.get('alert_thresholds', {})
INDICES = _monitor_config.get('indices', ["SPY", "^FCHI"])
CHECK_STOP_LOSSES = _monitor_config.get('check_stop_losses', True)
STOP_LOSS_THRESHOLD = _monitor_config.get('stop_loss_threshold_pct', 5.0)
CHECK_BOLLINGER = _monitor_config.get('check_bollinger', True)


def load_previous_close(portfolio: Portfolio) -> Dict[str, float]:
    """Load previous closing prices from portfolio state or market data."""
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
    history = load_alert_history()
    
    if not CHECK_STOP_LOSSES:
        return alerts
    
    for ticker, position in portfolio.positions.items():
        current_price = current_prices.get(ticker)
        if not current_price or position.avg_price <= 0:
            continue
        
        drawdown_pct = ((current_price - position.avg_price) / position.avg_price) * 100
        
        if drawdown_pct <= -STOP_LOSS_THRESHOLD:
            should_alert_flag, reason = should_alert(
                ticker, drawdown_pct, 'stop_loss_triggered', history
            )
            
            if should_alert_flag:
                alerts.append({
                    'type': 'stop_loss_triggered',
                    'ticker': ticker,
                    'severity': 'critical',
                    'current_price': current_price,
                    'entry_price': position.avg_price,
                    'drawdown_pct': drawdown_pct,
                    'stop_threshold': STOP_LOSS_THRESHOLD,
                    'action_required': 'SELL',
                    'alert_reason': reason
                })
                record_alert(ticker, drawdown_pct, 'stop_loss_triggered', 'critical', history)
    
    save_alert_history(history)
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
    
    history = load_alert_history()
    
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
                movement_pct = ((current_price - latest_upper) / latest_upper) * 100
                should_alert_flag, reason = should_alert(
                    ticker, movement_pct, 'bollinger_breakout_upper', history
                )
                
                if should_alert_flag:
                    alerts.append({
                        'type': 'bollinger_breakout',
                        'ticker': ticker,
                        'severity': 'medium',
                        'current_price': current_price,
                        'bollinger_upper': float(latest_upper),
                        'direction': 'upper',
                        'interpretation': 'Overbought - potential mean reversion',
                        'alert_reason': reason
                    })
                    record_alert(ticker, movement_pct, 'bollinger_breakout_upper', 'medium', history)
                    
            elif current_price < latest_lower:
                movement_pct = ((current_price - latest_lower) / latest_lower) * 100
                should_alert_flag, reason = should_alert(
                    ticker, movement_pct, 'bollinger_breakout_lower', history
                )
                
                if should_alert_flag:
                    alerts.append({
                        'type': 'bollinger_breakout',
                        'ticker': ticker,
                        'severity': 'medium',
                        'current_price': current_price,
                        'bollinger_lower': float(latest_lower),
                        'direction': 'lower',
                        'interpretation': 'Oversold - potential bounce',
                        'alert_reason': reason
                    })
                    record_alert(ticker, movement_pct, 'bollinger_breakout_lower', 'medium', history)
                    
        except Exception as e:
            # Silently skip if calculation fails
            continue
    
    save_alert_history(history)
    return alerts


def check_movements(
    current_prices: Dict[str, float],
    reference_prices: Dict[str, float],
    portfolio: Portfolio
) -> List[Dict]:
    """Check for significant price movements with deduplication."""
    alerts = []
    history = load_alert_history()
    
    # Check stop-losses first (highest priority)
    stop_alerts = check_stop_losses(current_prices, portfolio)
    alerts.extend(stop_alerts)
    
    # Track which tickers already have stop-loss alerts
    stop_loss_tickers = {a['ticker'] for a in stop_alerts}
    
    # Check portfolio positions for significant movements
    for ticker, position in portfolio.positions.items():
        current_price = current_prices.get(ticker)
        if not current_price:
            continue
        
        # Skip if stop-loss already triggered
        if ticker in stop_loss_tickers:
            continue
        
        # Use position average price as reference
        reference_price = position.avg_price
        
        if reference_price > 0:
            movement_pct = ((current_price - reference_price) / reference_price) * 100
            
            if abs(movement_pct) >= THRESHOLDS.get('position_movement_pct', 2.0):
                should_alert_flag, reason = should_alert(
                    ticker, movement_pct, 'position_movement', history
                )
                
                if should_alert_flag:
                    alerts.append({
                        'type': 'position_movement',
                        'ticker': ticker,
                        'severity': 'high' if abs(movement_pct) > 5 else 'medium',
                        'current_price': current_price,
                        'reference_price': reference_price,
                        'movement_pct': movement_pct,
                        'position_size': position.market_value,
                        'unrealized_pnl': position.unrealized_pnl,
                        'alert_reason': reason
                    })
                    record_alert(ticker, movement_pct, 'position_movement', 
                               'high' if abs(movement_pct) > 5 else 'medium', history)
    
    # Check indices
    for index in INDICES:
        current_price = current_prices.get(index)
        reference_price = reference_prices.get(index)
        
        if current_price and reference_price:
            movement_pct = ((current_price - reference_price) / reference_price) * 100
            
            if abs(movement_pct) >= THRESHOLDS['index_movement_pct']:
                should_alert_flag, reason = should_alert(
                    index, movement_pct, 'index_movement', history
                )
                
                if should_alert_flag:
                    alerts.append({
                        'type': 'index_movement',
                        'ticker': index,
                        'severity': 'high',
                        'current_price': current_price,
                        'reference_price': reference_price,
                        'movement_pct': movement_pct,
                        'alert_reason': reason
                    })
                    record_alert(index, movement_pct, 'index_movement', 'high', history)
    
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
                should_alert_flag, reason = should_alert(
                    'PORTFOLIO', drawdown_pct, 'portfolio_drawdown', history
                )
                
                if should_alert_flag:
                    alerts.append({
                        'type': 'portfolio_drawdown',
                        'ticker': 'PORTFOLIO',
                        'severity': 'critical',
                        'current_value': total_current,
                        'cost_basis': total_cost,
                        'drawdown_pct': drawdown_pct,
                        'alert_reason': reason
                    })
                    record_alert('PORTFOLIO', drawdown_pct, 'portfolio_drawdown', 'critical', history)
    
    # Check Bollinger Band breakouts
    bollinger_alerts = check_bollinger_breakouts(current_prices, portfolio)
    alerts.extend(bollinger_alerts)
    
    save_alert_history(history)
    return alerts


# Wrapper functions for backward compatibility
def check_position_movements(positions: Dict, previous_close: Dict, current_prices: Dict, threshold_pct: float = 2.0) -> List[Dict]:
    """Check for significant position movements (wrapper for check_movements)."""
    from portfolio.portfolio import Portfolio, Position
    
    portfolio = Portfolio(data_dir="data")
    for ticker, pos_data in positions.items():
        portfolio.positions[ticker] = Position(
            ticker=ticker,
            quantity=pos_data.get('quantity', 0),
            avg_price=pos_data.get('avg_price', 0),
            current_price=pos_data.get('current_price', pos_data.get('avg_price', 0))
        )
    
    all_alerts = check_movements(current_prices, previous_close, portfolio)
    return [a for a in all_alerts if a['type'] == 'position_movement']


def check_portfolio_drawdown(portfolio_value: float, cost_basis: float, threshold_pct: float = 1.5) -> Dict:
    """Check if portfolio drawdown exceeds threshold."""
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
    """Check for significant index movements."""
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
    """Generate alert summary from alerts list or by checking movements."""
    if isinstance(alerts_or_current_prices, list):
        alerts = alerts_or_current_prices
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
    """Run the monitoring check with deduplication."""
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
    current_prices = fetch_current_prices(tickers_to_monitor, max_workers=4)
    
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
            print(f"Reason: {alert.get('alert_reason', 'N/A')}")
            
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
                print(f"Drawdown: {alert['drawdown_pct']:.2f}% (threshold: -{alert['stop_threshold']}%)")
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
        
        return alerts, 1
    else:
        print("✓ No significant movements detected.")
        print(f"Portfolio Value: €{portfolio.total_value:.2f}")
        
        # Show suppressed alerts count
        history = load_alert_history()
        recent_suppressed = len([
            a for a in history.get('alerts', [])
            if datetime.now() - datetime.fromisoformat(a['timestamp']) < timedelta(hours=COOLDOWN_HOURS)
        ])
        if recent_suppressed > 0:
            print(f"(Alert history: {recent_suppressed} tickers in cooldown period)")
        
        return [], 0


if __name__ == "__main__":
    try:
        alerts, exit_code = run_monitor()
        sys.exit(exit_code)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

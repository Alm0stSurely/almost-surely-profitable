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
import logging
import math
import sys
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from data.fetch_market_data import fetch_current_prices, fetch_current_quotes
from portfolio.portfolio import Portfolio
from utils import load_json_dict_or_quarantine

logger = logging.getLogger(__name__)


# Resolve paths relative to the repository root so the script is safe to run
# from any working directory (e.g. via cron from the parent workspace).
REPO_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = REPO_ROOT / "data"


def _is_finite_number(value) -> bool:
    """Return True if value is a real finite scalar (int, float, or numpy/pandas convertible)."""
    if value is None or isinstance(value, bool) or isinstance(value, str):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


# Load thresholds from config
CONFIG_PATH = REPO_ROOT / "config" / "monitor.json"
UNIVERSE_PATH = REPO_ROOT / "config" / "universe.json"
ALERT_HISTORY_PATH = DATA_DIR / "alert_history.json"
MARKET_STATE_PATH = DATA_DIR / "market_state.json"

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
            'bollinger_breakout_std': 2.0,
            'bollinger_breakout_min_pct': 1.0
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
    """Load alert history for deduplication.

    The loaded dict flows into ``save_alert_history`` within the same monitor
    run (load -> record -> save). A corrupt file must therefore be quarantined,
    not silently replaced by a fresh fallback: the save would otherwise
    overwrite the corrupt file and destroy the evidence — the same data-loss
    class as the trade ledger (PR #60).

    Returns a fresh ``{'alerts': [], 'last_reset': <now>}`` when the file is
    missing, corrupt, or not a dict. A dict without an ``'alerts'`` list has
    no alert records to preserve; it starts fresh too, but the failure is
    logged so the subsequent overwrite is loud, not silent.
    """
    history = load_json_dict_or_quarantine(ALERT_HISTORY_PATH, context="alert history")
    if not isinstance(history.get('alerts'), list):
        if ALERT_HISTORY_PATH.exists():
            logger.error(
                "Alert history %s has no 'alerts' list (%r); starting fresh "
                "history.",
                ALERT_HISTORY_PATH, history.get('alerts'),
            )
        return {
            'alerts': [],
            'last_reset': datetime.now().isoformat()
        }
    return history


def save_alert_history(history: Dict):
    """Save alert history.

    Uses ``allow_nan=False`` so any non-finite values that slip through are
    caught immediately rather than written as non-standard JSON (``NaN``,
    ``Infinity``) that downstream consumers cannot parse.
    """
    ALERT_HISTORY_PATH.parent.mkdir(exist_ok=True)
    with open(ALERT_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2, allow_nan=False)


def should_alert(ticker: str, movement_pct: float, alert_type: str, history: Dict) -> Tuple[bool, str]:
    """
    Determine if we should alert for this ticker.
    
    Returns:
        (should_alert: bool, reason: str)
    """
    now = datetime.now()

    # Check if market just opened (reset history at 08:00 UTC)
    if now.hour == 8 and now.minute < 30:
        # Session reset — idempotent per calendar day, not per call. A
        # per-call reset wiped every alert recorded earlier in the same run:
        # check_movements shares one history dict across all alert classes
        # and saves once at the end, so each successive should_alert call in
        # the 08:00-08:30 window cleared the records of the previous ones and
        # the final save persisted only the last alert of the run (observed
        # 2026-11-05 08:06: six alerts fired, one persisted — MC.PA
        # Bollinger, the last recorded).
        last_reset_raw = history.get('last_reset', '')
        try:
            already_reset_today = (
                bool(last_reset_raw)
                and datetime.fromisoformat(last_reset_raw).date() == now.date()
            )
        except (ValueError, TypeError):
            already_reset_today = False
        if not already_reset_today:
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
    if not _is_finite_number(movement_pct):
        print(
            f"Skipping alert record for {ticker} ({alert_type}): "
            f"non-finite movement_pct {movement_pct!r}"
        )
        return

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
BOLLINGER_MIN_BREAKOUT_PCT = THRESHOLDS.get('bollinger_breakout_min_pct', 1.0)


def _breakout_margin(current_price: float, band: float) -> float:
    """Return the absolute percentage distance from a Bollinger band.

    A value of 1.0 means the price is 1% beyond the band. The result is
    always non-negative so it can be compared against the same threshold for
    both upper and lower breakouts.
    """
    if band == 0:
        return 0.0
    return abs((current_price - band) / band) * 100


def _is_significant_breakout(current_price: float, band: float, min_pct: float) -> bool:
    """Return True when the price has moved at least ``min_pct`` beyond a band."""
    return _breakout_margin(current_price, band) >= min_pct


def load_previous_close(portfolio: Portfolio) -> Dict[str, float]:
    """Load previous closing prices from the persisted market state.

    ``run_monitor`` unconditionally re-saves the market state after the checks,
    so a corrupt state file must be quarantined rather than silently bypassed:
    the save would otherwise overwrite the corrupt file and destroy the
    evidence — the same data-loss class as the trade ledger (PR #60).

    On corruption the reference map is empty; ``check_movements`` already
    falls back to ``position.avg_price`` per ticker, so movement semantics are
    unchanged. (The ``portfolio`` parameter is kept for API compatibility.)
    """
    state = load_json_dict_or_quarantine(MARKET_STATE_PATH, context="market state")
    previous_close = state.get('previous_close', {})
    if not isinstance(previous_close, dict):
        logger.error(
            "Market state %s has a non-dict 'previous_close' (%r); ignoring "
            "saved references for this run.",
            MARKET_STATE_PATH, previous_close,
        )
        return {}
    # Only return finite prices; non-finite references break movement math.
    return {
        ticker: price for ticker, price in previous_close.items()
        if _is_finite_number(price)
    }


def save_market_state(prices: Dict[str, float]):
    """Save current prices as reference for next check.

    Non-finite prices are filtered out so the saved JSON remains standard and
    cannot corrupt the next monitor run.
    """
    MARKET_STATE_PATH.parent.mkdir(exist_ok=True)

    finite_prices = {
        ticker: price for ticker, price in prices.items()
        if _is_finite_number(price)
    }

    state = {
        'timestamp': datetime.now().isoformat(),
        'previous_close': finite_prices
    }

    with open(MARKET_STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, allow_nan=False)


# --- Feed freshness guard -------------------------------------------------
#
# The classifier lives in ``data.feed_freshness`` and is shared with the
# nightly pipeline (daily_run), which forces a HOLD when a held position's
# feed is stale — closing the 2026-11-13 gap "le guard protège le monitor,
# pas le pipeline". Production incident 2026-11-05 and the full session
# arithmetic rationale are documented there. The names are re-exported
# here so existing callers and tests keep importing from ``monitor``.

from data.feed_freshness import (  # noqa: F401
    _expected_session_date,
    _market_of,
    _session_date_from_bar,
    is_feed_stale,
    split_fresh_and_stale,
)


# ---------------------------------------------------------------------------


def check_stop_losses(
    current_prices: Dict[str, float],
    portfolio: Portfolio,
    history: Dict = None,
) -> List[Dict]:
    """Check if any positions have hit their stop-loss threshold.

    When ``history`` is provided, records are appended to that shared dict
    and persistence is the caller's responsibility (``check_movements``
    loads/saves once per run). Standalone callers omit it and this function
    owns the load/save cycle.
    """
    alerts = []
    owns_history = history is None
    if owns_history:
        history = load_alert_history()

    if not CHECK_STOP_LOSSES:
        return alerts
    
    for ticker, position in portfolio.positions.items():
        current_price = current_prices.get(ticker)
        if not _is_finite_number(current_price) or position.avg_price <= 0:
            continue

        drawdown_pct = ((current_price - position.avg_price) / position.avg_price) * 100

        if not _is_finite_number(drawdown_pct):
            print(
                f"Skipping stop-loss check for {ticker}: non-finite drawdown_pct {drawdown_pct!r}"
            )
            continue

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

    if owns_history:
        save_alert_history(history)
    return alerts


def check_bollinger_breakouts(
    current_prices: Dict[str, float],
    portfolio: Portfolio,
    history: Dict = None,
) -> List[Dict]:
    """Check for Bollinger Band breakouts.

    Bands are computed on the window EXCLUDING the most recent bar
    ("excl.-barre" basis). The latest daily bar returned by the data feed is
    the forming bar of the current session: its close tracks the live price,
    so including it measures the price against a band inflated by itself and
    dilutes the breakout margin exactly when the move is largest (observed
    2026-10-29 AI.PA, 2026-11-02 DBA, 2026-11-04 AI.PA/IJR). The manual
    excl.-barre basis used by the research sessions is the governing one.

    When ``history`` is provided, records are appended to that shared dict
    and persistence is the caller's responsibility (``check_movements``
    loads/saves once per run). Standalone callers omit it and this function
    owns the load/save cycle.
    """
    alerts = []

    if not CHECK_BOLLINGER:
        return alerts

    owns_history = history is None
    if owns_history:
        history = load_alert_history()

    # Import indicators here to avoid circular imports
    try:
        from data.indicators import calculate_bollinger_bands
        import pandas as pd
        from data.fetch_market_data import fetch_historical_data
    except ImportError:
        return alerts

    for ticker in portfolio.positions.keys():
        current_price = current_prices.get(ticker)
        if not _is_finite_number(current_price):
            continue

        try:
            # Fetch recent data for Bollinger calculation. The 3-month window
            # leaves >= 20 completed bars after the forming bar is excluded.
            df = fetch_historical_data([ticker], period='3mo')
            if not df or ticker not in df or len(df[ticker]) < 21:
                continue

            # Exclude the forming bar (last row): its close tracks the live
            # price and would inflate the band with the very price being
            # tested (self-inclusion dilution).
            closes = df[ticker]['Close'].iloc[:-1]
            upper, middle, lower = calculate_bollinger_bands(closes)

            if upper is None or lower is None:
                continue

            latest_upper = upper.iloc[-1]
            latest_lower = lower.iloc[-1]

            if not (_is_finite_number(latest_upper) and _is_finite_number(latest_lower)):
                print(
                    f"Skipping Bollinger check for {ticker}: non-finite bands "
                    f"upper={latest_upper!r}, lower={latest_lower!r}"
                )
                continue

            # Check for breakout, requiring a minimum margin beyond the band to
            # avoid firing on micro-pierces (e.g. price 0.05% above the band).
            if current_price > latest_upper:
                breakout_margin_pct = _breakout_margin(current_price, latest_upper)
                if not _is_finite_number(breakout_margin_pct):
                    print(
                        f"Skipping Bollinger upper breakout for {ticker}: "
                        f"non-finite margin {breakout_margin_pct!r}"
                    )
                    continue
                if not _is_significant_breakout(current_price, latest_upper, BOLLINGER_MIN_BREAKOUT_PCT):
                    continue
                should_alert_flag, reason = should_alert(
                    ticker, breakout_margin_pct, 'bollinger_breakout_upper', history
                )

                if should_alert_flag:
                    alerts.append({
                        'type': 'bollinger_breakout',
                        'ticker': ticker,
                        'severity': 'medium',
                        'current_price': current_price,
                        'bollinger_upper': float(latest_upper),
                        'direction': 'upper',
                        'breakout_margin_pct': breakout_margin_pct,
                        'interpretation': 'Overbought - potential mean reversion',
                        'alert_reason': reason
                    })
                    record_alert(ticker, breakout_margin_pct, 'bollinger_breakout_upper', 'medium', history)

            elif current_price < latest_lower:
                breakout_margin_pct = _breakout_margin(current_price, latest_lower)
                if not _is_finite_number(breakout_margin_pct):
                    print(
                        f"Skipping Bollinger lower breakout for {ticker}: "
                        f"non-finite margin {breakout_margin_pct!r}"
                    )
                    continue
                if not _is_significant_breakout(current_price, latest_lower, BOLLINGER_MIN_BREAKOUT_PCT):
                    continue
                should_alert_flag, reason = should_alert(
                    ticker, breakout_margin_pct, 'bollinger_breakout_lower', history
                )

                if should_alert_flag:
                    alerts.append({
                        'type': 'bollinger_breakout',
                        'ticker': ticker,
                        'severity': 'medium',
                        'current_price': current_price,
                        'bollinger_lower': float(latest_lower),
                        'direction': 'lower',
                        'breakout_margin_pct': breakout_margin_pct,
                        'interpretation': 'Oversold - potential bounce',
                        'alert_reason': reason
                    })
                    record_alert(ticker, breakout_margin_pct, 'bollinger_breakout_lower', 'medium', history)

        except Exception as e:
            # Loud skip (read-fallback class): a crashing indicator calc
            # otherwise disables this alert class for the ticker with no
            # trace — the failure path must not be quieter than the state
            # it replaces.
            print(f"Warning: Bollinger check failed for {ticker}: {type(e).__name__}: {e}")
            continue

    if owns_history:
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
    
    # Check stop-losses first (highest priority). The shared history dict is
    # passed down so every alert class lands in one in-memory state; the
    # single save at the end of this function persists them all. (Three
    # independent load/save cycles previously raced: each nested save was
    # overwritten by the outer check_movements save, so only
    # position_movement alerts ever reached alert_history.json — observed
    # 2026-10-30 through 2026-11-04.)
    stop_alerts = check_stop_losses(current_prices, portfolio, history)
    alerts.extend(stop_alerts)
    
    # Track which tickers already have stop-loss alerts
    stop_loss_tickers = {a['ticker'] for a in stop_alerts}
    
    # Check portfolio positions for significant movements
    for ticker, position in portfolio.positions.items():
        current_price = current_prices.get(ticker)
        if not _is_finite_number(current_price):
            continue

        # Skip if stop-loss already triggered
        if ticker in stop_loss_tickers:
            continue

        # Use previous close as reference for intraday movement detection;
        # fall back to avg_price if previous close unavailable
        reference_price = reference_prices.get(ticker, position.avg_price)

        if not _is_finite_number(reference_price) or reference_price <= 0:
            continue

        movement_pct = ((current_price - reference_price) / reference_price) * 100

        if not _is_finite_number(movement_pct):
            print(
                f"Skipping position movement check for {ticker}: "
                f"non-finite movement_pct {movement_pct!r}"
            )
            continue

        if abs(movement_pct) >= THRESHOLDS.get('position_movement_pct', 2.0):
            should_alert_flag, reason = should_alert(
                ticker, movement_pct, 'position_movement', history
            )

            if should_alert_flag:
                position_size = position.market_value
                unrealized_pnl = position.unrealized_pnl
                if not (_is_finite_number(position_size) and _is_finite_number(unrealized_pnl)):
                    print(
                        f"Skipping position movement alert for {ticker}: "
                        f"non-finite position_size={position_size!r} or "
                        f"unrealized_pnl={unrealized_pnl!r}"
                    )
                    continue

                alerts.append({
                    'type': 'position_movement',
                    'ticker': ticker,
                    'severity': 'high' if abs(movement_pct) > 5 else 'medium',
                    'current_price': current_price,
                    'reference_price': reference_price,
                    'movement_pct': movement_pct,
                    'position_size': position_size,
                    'unrealized_pnl': unrealized_pnl,
                    'alert_reason': reason
                })
                record_alert(ticker, movement_pct, 'position_movement',
                           'high' if abs(movement_pct) > 5 else 'medium', history)
    
    # Check indices
    for index in INDICES:
        current_price = current_prices.get(index)
        reference_price = reference_prices.get(index)

        if not (_is_finite_number(current_price) and _is_finite_number(reference_price)):
            continue
        if reference_price <= 0:
            continue

        movement_pct = ((current_price - reference_price) / reference_price) * 100

        if not _is_finite_number(movement_pct):
            print(
                f"Skipping index movement check for {index}: "
                f"non-finite movement_pct {movement_pct!r}"
            )
            continue

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

        if not (_is_finite_number(total_cost) and _is_finite_number(total_current)):
            print(
                f"Skipping portfolio drawdown check: non-finite "
                f"total_cost={total_cost!r} or total_current={total_current!r}"
            )
        elif total_cost > 0:
            drawdown_pct = ((total_current - total_cost) / total_cost) * 100

            if not _is_finite_number(drawdown_pct):
                print(
                    f"Skipping portfolio drawdown check: "
                    f"non-finite drawdown_pct {drawdown_pct!r}"
                )
            else:
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
    
    # Check Bollinger Band breakouts (shared history — see stop-loss note)
    bollinger_alerts = check_bollinger_breakouts(current_prices, portfolio, history)
    alerts.extend(bollinger_alerts)

    save_alert_history(history)
    return alerts


# Wrapper functions for backward compatibility
def check_position_movements(positions: Dict, previous_close: Dict, current_prices: Dict, threshold_pct: float = 2.0) -> List[Dict]:
    """Check for significant position movements (wrapper for check_movements)."""
    from portfolio.portfolio import Portfolio, Position
    
    portfolio = Portfolio(data_dir=str(DATA_DIR))
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
    if not (_is_finite_number(portfolio_value) and _is_finite_number(cost_basis)):
        return None
    if cost_basis <= 0:
        return None

    drawdown_pct = ((portfolio_value - cost_basis) / cost_basis) * 100

    if not _is_finite_number(drawdown_pct):
        return None

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

        if not (_is_finite_number(current_price) and _is_finite_number(reference_price)):
            continue
        if reference_price <= 0:
            continue

        movement_pct = ((current_price - reference_price) / reference_price) * 100

        if not _is_finite_number(movement_pct):
            continue

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
    portfolio = Portfolio(data_dir=str(DATA_DIR))
    
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
    
    # Fetch current quotes with last-bar timestamps (freshness guard below)
    print(f"Fetching prices for {len(tickers_to_monitor)} tickers...")
    quotes = fetch_current_quotes(tickers_to_monitor, max_workers=4)

    # Load reference prices BEFORE the freshness split: a stale quote must
    # not overwrite the reference it cannot vouch for (incident 2026-11-05 —
    # stale closes persisted as previous_close poisoned the next run).
    reference_prices = load_previous_close(portfolio)

    # Feed freshness guard: a price is only as trustworthy as the bar behind
    # it. Stale tickers are excluded from alerts (movement, Bollinger,
    # stop-loss) and keep their previous reference for the next run.
    current_prices, stale_tickers = split_fresh_and_stale(quotes, datetime.now())
    if stale_tickers:
        print(
            f"\n⚠️  STALE FEED — {len(stale_tickers)} ticker(s) excluded from "
            f"alerts and reference update: {sorted(stale_tickers)}"
        )

    # Check for movements
    alerts = check_movements(current_prices, reference_prices, portfolio)

    # Save current prices for next check: fresh prices win, stale tickers
    # keep their previous reference instead of being dropped or overwritten.
    prices_to_save = {**reference_prices, **current_prices}
    save_market_state(prices_to_save)
    
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
            'portfolio_value': portfolio.total_value if _is_finite_number(portfolio.total_value) else None
        }
        print("\nJSON_OUTPUT:")
        try:
            print(json.dumps(output, indent=2, allow_nan=False))
        except ValueError as e:
            print(f"Alert output contains non-finite values: {e}")
            # Fallback: emit an empty but valid JSON summary rather than crashing.
            print(json.dumps({
                'timestamp': output['timestamp'],
                'alert_count': len(alerts),
                'alerts': [],
                'portfolio_value': output['portfolio_value'],
                'error': 'Non-finite values were filtered from alert output'
            }, indent=2, allow_nan=False))

        return alerts, 1
    else:
        print("✓ No significant movements detected.")
        if _is_finite_number(portfolio.total_value):
            print(f"Portfolio Value: €{portfolio.total_value:.2f}")
        else:
            print(f"Portfolio Value: {portfolio.total_value!r} (non-finite)")

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

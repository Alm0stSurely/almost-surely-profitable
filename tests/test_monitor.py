"""
Test suite for Intraday Monitor module.
Tests alert generation and threshold detection.
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monitor import (
    load_monitor_config,
    check_movements,
    check_position_movements,
    check_portfolio_drawdown,
    check_index_movements,
    check_stop_losses,
    check_bollinger_breakouts,
    generate_alerts,
    THRESHOLDS
)


def test_load_monitor_config_default():
    """Test loading default monitor configuration."""
    print("Test 1: Load Default Monitor Config")
    print("-" * 40)
    
    with patch('monitor.CONFIG_PATH', Path('/nonexistent/config.json')):
        config = load_monitor_config()
        
        assert 'alert_thresholds' in config
        assert 'position_movement_pct' in config['alert_thresholds']
        assert 'portfolio_drawdown_pct' in config['alert_thresholds']
        assert config['alert_thresholds']['position_movement_pct'] == 2.0
        
        print(f"  Default thresholds: {config['alert_thresholds']}")
        print("✓ Load default config test passed\n")


def test_load_monitor_config_custom():
    """Test loading custom monitor configuration."""
    print("Test 2: Load Custom Monitor Config")
    print("-" * 40)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            'alert_thresholds': {
                'position_movement_pct': 3.5,
                'portfolio_drawdown_pct': 2.0
            },
            'indices': ['SPY', 'QQQ']
        }, f)
        temp_path = f.name
    
    with patch('monitor.CONFIG_PATH', Path(temp_path)):
        config = load_monitor_config()
        assert config['alert_thresholds']['position_movement_pct'] == 3.5
        assert config['indices'] == ['SPY', 'QQQ']
        print(f"  Custom thresholds loaded")
        print("✓ Load custom config test passed\n")
    
    Path(temp_path).unlink()


def test_check_position_movements_normal():
    """Test position movement check with normal changes."""
    print("Test 3: Check Position Movements - Normal")
    print("-" * 40)
    
    positions = {
        "SPY": {"quantity": 10, "avg_price": 400, "current_price": 402},  # +0.5%
        "TLT": {"quantity": 5, "avg_price": 100, "current_price": 99}     # -1%
    }
    
    previous_close = {"SPY": 400, "TLT": 100}
    current_prices = {"SPY": 402, "TLT": 99}
    
    alerts = check_position_movements(positions, previous_close, current_prices, threshold_pct=2.0)
    
    # No alerts expected - movements below 2% threshold
    assert len(alerts) == 0
    print("  No alerts generated (movements below threshold)")
    print("✓ Normal movements test passed\n")


def test_check_position_movements_alert():
    """Test position movement check with significant changes."""
    print("Test 4: Check Position Movements - Alert Triggered")
    print("-" * 40)
    
    positions = {
        "RMS.PA": {"quantity": 1, "avg_price": 1669.50, "current_price": 1669.50},
    }
    
    previous_close = {"RMS.PA": 1669.50}
    current_prices = {"RMS.PA": 1626.50}  # -2.58% drop
    
    with patch('monitor.load_alert_history', return_value={'alerts': [], 'last_reset': datetime.now().isoformat()}):
        alerts = check_position_movements(positions, previous_close, current_prices, threshold_pct=2.0)
    
    assert len(alerts) > 0
    assert alerts[0]['type'] == 'position_movement'
    assert alerts[0]['ticker'] == 'RMS.PA'
    assert 'movement_pct' in alerts[0]
    
    print(f"  Alert generated: {alerts[0]['type']}")
    print(f"  Movement: {alerts[0]['movement_pct']:.2f}%")
    print("✓ Alert triggered test passed\n")


def test_check_position_movements_no_false_positive():
    """Test that POSITION_MOVEMENT uses previous close, not avg_price, as reference.
    
    This guards against false positives when the market is flat intraday
    but the position has unrealized P&L since entry.
    """
    print("Test 4b: Check Position Movements - No False Positive")
    print("-" * 40)
    
    positions = {
        "AI.PA": {"quantity": 10, "avg_price": 150.0, "current_price": 180.0},  # +20% since entry
    }
    
    # Market is flat: previous_close == current_price
    previous_close = {"AI.PA": 180.0}
    current_prices = {"AI.PA": 180.0}
    
    with patch('monitor.load_alert_history', return_value={'alerts': [], 'last_reset': datetime.now().isoformat()}):
        alerts = check_position_movements(positions, previous_close, current_prices, threshold_pct=2.0)
    
    # No alert expected: intraday movement is 0%, even though unrealized P&L is +20%
    assert len(alerts) == 0
    print("  No false positive: flat intraday price does not alert")
    print("✓ False-positive guard test passed\n")


def test_check_portfolio_drawdown_normal():
    """Test portfolio drawdown check with normal value."""
    print("Test 5: Check Portfolio Drawdown - Normal")
    print("-" * 40)
    
    portfolio_value = 9500
    cost_basis = 9400  # Currently in profit
    
    alert = check_portfolio_drawdown(portfolio_value, cost_basis, threshold_pct=1.5)
    
    assert alert is None  # No drawdown, no alert
    print("  No drawdown alert (portfolio in profit)")
    print("✓ Normal drawdown test passed\n")


def test_check_portfolio_drawdown_alert():
    """Test portfolio drawdown check with significant drawdown."""
    print("Test 6: Check Portfolio Drawdown - Alert Triggered")
    print("-" * 40)
    
    portfolio_value = 1381
    cost_basis = 1403  # ~1.6% drawdown
    
    alert = check_portfolio_drawdown(portfolio_value, cost_basis, threshold_pct=1.5)
    
    assert alert is not None
    assert alert['type'] == 'portfolio_drawdown'
    assert 'drawdown_pct' in alert
    
    print(f"  Alert: {alert['type']}")
    print(f"  Drawdown: {alert['drawdown_pct']:.2f}%")
    print("✓ Drawdown alert test passed\n")


def test_check_index_movements():
    """Test index movement check."""
    print("Test 7: Check Index Movements")
    print("-" * 40)
    
    current_prices = {
        "SPY": 650,      # From 630 = +3.17%
        "^FCHI": 7800    # From 8000 = -2.5%
    }
    previous_close = {
        "SPY": 630,
        "^FCHI": 8000
    }
    indices = ["SPY", "^FCHI"]
    
    alerts = check_index_movements(current_prices, previous_close, indices, threshold_pct=3.0)
    
    # SPY moved +3.17%, should trigger alert (above 3% threshold)
    spy_alert = [a for a in alerts if a['ticker'] == 'SPY']
    assert len(spy_alert) > 0
    
    print(f"  Index alerts: {len(alerts)}")
    print(f"  SPY movement: +3.17% (alert triggered)")
    print("✓ Index movements test passed\n")


def test_generate_alerts_empty():
    """Test alert generation with no alerts."""
    print("Test 8: Generate Alerts - Empty")
    print("-" * 40)
    
    alerts = []
    output = generate_alerts(alerts)
    
    assert output is not None
    assert 'alert_count' in output
    assert output['alert_count'] == 0
    
    print(f"  Alert count: {output['alert_count']}")
    print("✓ Empty alerts test passed\n")


def test_generate_alerts_with_data():
    """Test alert generation with actual alerts."""
    print("Test 9: Generate Alerts - With Data")
    print("-" * 40)
    
    alerts = [
        {
            'type': 'position_movement',
            'ticker': 'RMS.PA',
            'severity': 'medium',
            'movement_pct': -2.58
        },
        {
            'type': 'portfolio_drawdown',
            'ticker': 'PORTFOLIO',
            'severity': 'critical',
            'drawdown_pct': -1.6
        }
    ]
    
    output = generate_alerts(alerts)
    
    assert output['alert_count'] == 2
    assert len(output['alerts']) == 2
    
    print(f"  Alert count: {output['alert_count']}")
    print(f"  Severities: {[a['severity'] for a in output['alerts']]}")
    print("✓ Alerts with data test passed\n")


def test_thresholds_loaded():
    """Test that thresholds are loaded from config."""
    print("Test 10: Thresholds Loaded")
    print("-" * 40)
    
    assert THRESHOLDS is not None
    assert 'position_movement_pct' in THRESHOLDS
    assert 'portfolio_drawdown_pct' in THRESHOLDS
    
    print(f"  Position movement threshold: {THRESHOLDS['position_movement_pct']}%")
    print(f"  Drawdown threshold: {THRESHOLDS['portfolio_drawdown_pct']}%")
    print("✓ Thresholds loaded test passed\n")


def test_check_position_with_missing_price():
    """Test position check when current price is missing."""
    print("Test 11: Position Check - Missing Price")
    print("-" * 40)
    
    positions = {"SPY": {"quantity": 10, "avg_price": 400}}
    previous_close = {"SPY": 400}
    current_prices = {}  # Missing SPY price
    
    alerts = check_position_movements(positions, previous_close, current_prices, threshold_pct=2.0)
    
    # Should handle gracefully - no crash
    print("  Missing price handled gracefully")
    print("✓ Missing price test passed\n")


def test_check_bollinger_breakouts_upper():
    """Test Bollinger Band upper breakout detection."""
    print("Test 12: Bollinger Breakout - Upper")
    print("-" * 40)
    
    import pandas as pd
    import numpy as np
    
    # Create mock historical data with a breakout
    dates = pd.date_range(end=datetime.now(), periods=20, freq='D')
    prices = np.linspace(100, 120, 20)  # Trending up to trigger upper breakout
    
    mock_df = pd.DataFrame({
        'Open': prices,
        'High': prices + 1,
        'Low': prices - 1,
        'Close': prices,
        'Volume': [1000] * 20
    }, index=dates)
    
    with patch('monitor.load_alert_history', return_value={'alerts': [], 'last_reset': datetime.now().isoformat()}), \
         patch('monitor.save_alert_history'), \
         patch('data.fetch_market_data.fetch_historical_data', return_value={'SPY': mock_df}), \
         patch('monitor.CHECK_BOLLINGER', True):
        
        from portfolio.portfolio import Portfolio
        portfolio = Portfolio(data_dir="/tmp/test_bollinger")
        portfolio.positions = {}
        
        # Create a mock position
        from portfolio.portfolio import Position
        portfolio.positions['SPY'] = Position(ticker='SPY', quantity=10, avg_price=100, current_price=125)
        
        current_prices = {'SPY': 125}  # Well above typical upper band
        alerts = check_bollinger_breakouts(current_prices, portfolio)
        
        # Should detect breakout without crashing
        print(f"  Bollinger check completed: {len(alerts)} alerts")
        print("✓ Bollinger upper breakout test passed\n")


def test_check_bollinger_breakouts_no_data():
    """Test Bollinger Band check when no data is available."""
    print("Test 13: Bollinger Breakout - No Data")
    print("-" * 40)
    
    with patch('monitor.load_alert_history', return_value={'alerts': [], 'last_reset': datetime.now().isoformat()}), \
         patch('monitor.save_alert_history'), \
         patch('data.fetch_market_data.fetch_historical_data', return_value={}), \
         patch('monitor.CHECK_BOLLINGER', True):
        
        from portfolio.portfolio import Portfolio
        portfolio = Portfolio(data_dir="/tmp/test_bollinger2")
        portfolio.positions = {}
        
        from portfolio.portfolio import Position
        portfolio.positions['SPY'] = Position(ticker='SPY', quantity=10, avg_price=100, current_price=100)
        
        current_prices = {'SPY': 100}
        alerts = check_bollinger_breakouts(current_prices, portfolio)
        
        assert len(alerts) == 0  # No data = no alerts
        print("  No alerts generated (no historical data)")
        print("✓ Bollinger no data test passed\n")


# --- New tests added 2026-07-10: Bollinger breakout minimum margin threshold
import monitor
import pandas as pd
import pytest

from portfolio.portfolio import Portfolio, Position


def _make_portfolio(tmp_path, ticker="DBA", quantity=10, avg_price=26.0, current_price=26.5):
    """Build a minimal portfolio with a single position for monitor tests."""
    portfolio = Portfolio(data_dir=str(tmp_path / "data"))
    portfolio.positions[ticker] = Position(
        ticker=ticker,
        quantity=quantity,
        avg_price=avg_price,
        current_price=current_price,
    )
    return portfolio


def _make_bollinger_series(band_value, length=30):
    """Return a constant pandas Series representing a flat Bollinger band."""
    return pd.Series([float(band_value)] * length)


class TestBollingerBreakoutMargin:
    """Bollinger breakouts must exceed a minimum margin before alerting."""

    @pytest.fixture(autouse=True)
    def isolate_history(self, tmp_path, monkeypatch):
        """Redirect alert history to a temp file and keep bollinger checks on."""
        history_path = tmp_path / "data" / "alert_history.json"
        history_path.parent.mkdir(exist_ok=True)
        monkeypatch.setattr(monitor, "ALERT_HISTORY_PATH", history_path)
        monkeypatch.setattr(monitor, "CHECK_BOLLINGER", True)

    @pytest.fixture
    def mock_bollinger(self):
        """Patch network/data dependencies so tests run offline."""
        with patch("data.fetch_market_data.fetch_historical_data") as mock_fetch, patch(
            "data.indicators.calculate_bollinger_bands"
        ) as mock_bands:

            def setup(upper, lower):
                mock_fetch.return_value = {"DBA": pd.DataFrame({"Close": [100.0] * 30})}
                middle = (upper + lower) / 2.0
                mock_bands.return_value = (
                    _make_bollinger_series(upper),
                    _make_bollinger_series(middle),
                    _make_bollinger_series(lower),
                )

            yield setup

    def test_marginal_upper_breakout_is_suppressed(self, tmp_path, mock_bollinger):
        """A 0.05% upper pierce must not create an alert."""
        mock_bollinger(upper=100.0, lower=90.0)
        portfolio = _make_portfolio(tmp_path, ticker="DBA")
        alerts = check_bollinger_breakouts({"DBA": 100.05}, portfolio)
        assert alerts == []

    def test_clear_upper_breakout_is_alerted(self, tmp_path, mock_bollinger):
        """A 1.5% upper breakout must trigger an alert with the margin recorded."""
        mock_bollinger(upper=100.0, lower=90.0)
        portfolio = _make_portfolio(tmp_path, ticker="DBA")
        alerts = check_bollinger_breakouts({"DBA": 101.5}, portfolio)
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["ticker"] == "DBA"
        assert alert["direction"] == "upper"
        assert alert["breakout_margin_pct"] == pytest.approx(1.5, abs=0.01)
        assert alert["alert_reason"] == "First alert for this ticker"

    def test_exact_threshold_upper_breakout_is_alerted(self, tmp_path, mock_bollinger):
        """A breakout exactly at the configured threshold (1.0%) must alert."""
        mock_bollinger(upper=100.0, lower=90.0)
        portfolio = _make_portfolio(tmp_path, ticker="DBA")
        alerts = check_bollinger_breakouts({"DBA": 101.0}, portfolio)
        assert len(alerts) == 1

    def test_lower_breakout_is_alerted(self, tmp_path, mock_bollinger):
        """A 1.5% lower breakout must trigger an alert with the margin recorded."""
        mock_bollinger(upper=110.0, lower=100.0)
        portfolio = _make_portfolio(tmp_path, ticker="DBA")
        alerts = check_bollinger_breakouts({"DBA": 98.5}, portfolio)
        assert len(alerts) == 1
        assert alerts[0]["direction"] == "lower"
        assert alerts[0]["breakout_margin_pct"] == pytest.approx(1.5, abs=0.01)

    def test_marginal_lower_breakout_is_suppressed(self, tmp_path, mock_bollinger):
        """A 0.05% lower pierce must not create an alert."""
        mock_bollinger(upper=110.0, lower=100.0)
        portfolio = _make_portfolio(tmp_path, ticker="DBA")
        alerts = check_bollinger_breakouts({"DBA": 99.95}, portfolio)
        assert alerts == []

    def test_bollinger_disabled_returns_empty(self, tmp_path, monkeypatch):
        """When bollinger checks are disabled, no alerts are produced."""
        monkeypatch.setattr(monitor, "CHECK_BOLLINGER", False)
        portfolio = _make_portfolio(tmp_path)
        alerts = check_bollinger_breakouts({"DBA": 200.0}, portfolio)
        assert alerts == []

    def test_config_threshold_is_loaded(self):
        """The threshold loaded from config/monitor.json must be 1.0%."""
        assert monitor.BOLLINGER_MIN_BREAKOUT_PCT == pytest.approx(1.0, abs=0.01)


class TestBreakoutMarginHelpers:
    """Direct unit tests for the margin helper functions."""

    def test_breakout_margin_zero_band(self):
        assert monitor._breakout_margin(100.0, 0.0) == 0.0

    def test_breakout_margin_upper(self):
        assert monitor._breakout_margin(101.5, 100.0) == pytest.approx(1.5, abs=1e-9)

    def test_breakout_margin_lower(self):
        assert monitor._breakout_margin(98.5, 100.0) == pytest.approx(1.5, abs=1e-9)

    def test_is_significant_breakout_exact(self):
        assert monitor._is_significant_breakout(101.0, 100.0, 1.0) is True

    def test_is_significant_breakout_below(self):
        assert monitor._is_significant_breakout(100.5, 100.0, 1.0) is False

    def test_is_significant_breakout_zero_band(self):
        assert monitor._is_significant_breakout(100.0, 0.0, 1.0) is False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running Intraday Monitor Tests")
    print("=" * 60 + "\n")
    
    test_load_monitor_config_default()
    test_load_monitor_config_custom()
    test_check_position_movements_normal()
    test_check_position_movements_alert()
    test_check_portfolio_drawdown_normal()
    test_check_portfolio_drawdown_alert()
    test_check_index_movements()
    test_generate_alerts_empty()
    test_generate_alerts_with_data()
    test_thresholds_loaded()
    test_check_position_with_missing_price()
    test_check_bollinger_breakouts_upper()
    test_check_bollinger_breakouts_no_data()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

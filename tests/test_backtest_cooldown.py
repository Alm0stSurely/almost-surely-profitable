"""
Tests for backtest cooldown manager integration.

Validates that the BacktestCooldownManager correctly enforces
guardrails within simulated backtest runs.
"""

import sys
import pytest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest.backtest_cooldown import BacktestCooldownManager, CooldownConfig


class TestBacktestCooldownManager:
    """Test suite for BacktestCooldownManager."""

    def test_can_buy_fresh_ticker(self):
        """Buying a ticker with no prior exit should be allowed."""
        mgr = BacktestCooldownManager()
        allowed, reason = mgr.can_buy("AAPL", datetime(2024, 1, 15))
        assert allowed is True
        assert "No cooldown" in reason

    def test_can_sell_without_entry_record(self):
        """Selling without an entry record should be blocked."""
        mgr = BacktestCooldownManager()
        allowed, reason = mgr.can_sell("AAPL", datetime(2024, 1, 15), 100.0, 90.0)
        assert allowed is False
        assert "No entry record" in reason

    def test_min_hold_period_blocks_sell(self):
        """Selling before min_hold_days should be blocked."""
        mgr = BacktestCooldownManager(config=CooldownConfig(min_hold_days=5))
        entry_date = datetime(2024, 1, 1)
        mgr.record_entry("AAPL", entry_date)

        # Try to sell after 2 days
        allowed, reason = mgr.can_sell("AAPL", datetime(2024, 1, 3), 100.0, 90.0)
        assert allowed is False
        assert "Minimum hold period" in reason

    def test_min_hold_period_allows_sell_after(self):
        """Selling after min_hold_days should be allowed."""
        mgr = BacktestCooldownManager(config=CooldownConfig(min_hold_days=5))
        entry_date = datetime(2024, 1, 1)
        mgr.record_entry("AAPL", entry_date)

        # Try to sell after 6 days
        allowed, reason = mgr.can_sell("AAPL", datetime(2024, 1, 7), 100.0, 90.0)
        assert allowed is True
        assert "Hold period satisfied" in reason

    def test_stop_loss_override_allows_early_sell(self):
        """Stop-loss override should allow selling before min_hold_days."""
        mgr = BacktestCooldownManager(
            config=CooldownConfig(min_hold_days=5, stop_loss_threshold_pct=5.0)
        )
        entry_date = datetime(2024, 1, 1)
        mgr.record_entry("AAPL", entry_date)

        # Price dropped 10% from avg_price
        allowed, reason = mgr.can_sell("AAPL", datetime(2024, 1, 2), 90.0, 100.0)
        assert allowed is True
        assert "Stop-loss override" in reason

    def test_stop_loss_override_respects_threshold(self):
        """Stop-loss override should NOT trigger for small drawdowns."""
        mgr = BacktestCooldownManager(
            config=CooldownConfig(min_hold_days=5, stop_loss_threshold_pct=5.0)
        )
        entry_date = datetime(2024, 1, 1)
        mgr.record_entry("AAPL", entry_date)

        # Price dropped only 2% from avg_price
        allowed, reason = mgr.can_sell("AAPL", datetime(2024, 1, 2), 98.0, 100.0)
        assert allowed is False
        assert "Minimum hold period" in reason

    def test_flip_cooldown_blocks_reentry(self):
        """Buying a ticker within flip_cooldown_days of exit should be blocked."""
        mgr = BacktestCooldownManager(config=CooldownConfig(flip_cooldown_days=10))
        mgr.record_exit("AAPL", datetime(2024, 1, 1))

        # Try to buy after 3 days
        allowed, reason = mgr.can_buy("AAPL", datetime(2024, 1, 4))
        assert allowed is False
        assert "Flip cooldown" in reason

    def test_flip_cooldown_allows_reentry_after(self):
        """Buying a ticker after flip_cooldown_days should be allowed."""
        mgr = BacktestCooldownManager(config=CooldownConfig(flip_cooldown_days=10))
        mgr.record_exit("AAPL", datetime(2024, 1, 1))

        # Try to buy after 12 days
        allowed, reason = mgr.can_buy("AAPL", datetime(2024, 1, 13))
        assert allowed is True

    def test_weekly_trade_cap_blocks_buy(self):
        """Buying when weekly trade cap is reached should be blocked."""
        mgr = BacktestCooldownManager(config=CooldownConfig(max_trades_per_week=2))
        mgr.record_entry("X", datetime(2024, 1, 15))
        mgr.record_entry("Y", datetime(2024, 1, 16))

        allowed, reason = mgr.can_buy("AAPL", datetime(2024, 1, 17))
        assert allowed is False
        assert "Weekly trade cap" in reason

    def test_weekly_trade_cap_blocks_sell(self):
        """Selling when weekly trade cap is reached should be blocked."""
        mgr = BacktestCooldownManager(config=CooldownConfig(max_trades_per_week=2))
        mgr.record_entry("AAPL", datetime(2024, 1, 10))
        mgr.record_entry("X", datetime(2024, 1, 15))
        mgr.record_entry("Y", datetime(2024, 1, 16))

        allowed, reason = mgr.can_sell("AAPL", datetime(2024, 1, 17), 100.0, 90.0)
        assert allowed is False
        assert "Weekly trade cap" in reason

    def test_weekly_trade_cap_resets_after_7_days(self):
        """Weekly trade cap should reset after 7 days."""
        mgr = BacktestCooldownManager(config=CooldownConfig(max_trades_per_week=2))
        mgr.record_entry("X", datetime(2024, 1, 1))
        mgr.record_entry("Y", datetime(2024, 1, 2))

        # After 8 days, cap should reset
        allowed, reason = mgr.can_buy("AAPL", datetime(2024, 1, 10))
        assert allowed is True

    def test_get_status_structure(self):
        """get_status should return expected structure."""
        mgr = BacktestCooldownManager()
        mgr.record_entry("AAPL", datetime(2024, 1, 1))
        mgr.record_exit("TSLA", datetime(2024, 1, 5))

        status = mgr.get_status(datetime(2024, 1, 10))
        assert "active_entries" in status
        assert "recent_exits" in status
        assert "trades_this_week" in status
        assert "weekly_cap" in status
        assert "config" in status
        assert "AAPL" in status["active_entries"]
        assert "TSLA" in status["recent_exits"]

    def test_get_metrics(self):
        """get_metrics should report aggregate statistics."""
        mgr = BacktestCooldownManager(config=CooldownConfig(max_trades_per_week=1))
        mgr.record_entry("AAPL", datetime(2024, 1, 1))

        # This should be blocked by weekly cap (entry counts as 1 trade)
        mgr.can_buy("TSLA", datetime(2024, 1, 1))

        metrics = mgr.get_metrics()
        assert metrics["blocked_buys"] == 1
        assert metrics["total_blocked"] == 1
        assert metrics["trade_attempts"] == 1
        assert metrics["block_rate"] == 1.0

    def test_stop_loss_override_counted_in_metrics(self):
        """Stop-loss overrides should be counted separately."""
        mgr = BacktestCooldownManager(
            config=CooldownConfig(min_hold_days=5, stop_loss_threshold_pct=5.0)
        )
        mgr.record_entry("AAPL", datetime(2024, 1, 1))
        mgr.can_sell("AAPL", datetime(2024, 1, 2), 90.0, 100.0)

        metrics = mgr.get_metrics()
        assert metrics["stop_loss_overrides"] == 1

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = CooldownConfig(
            min_hold_days=10,
            flip_cooldown_days=20,
            max_trades_per_week=5,
            stop_loss_threshold_pct=10.0
        )
        mgr = BacktestCooldownManager(config=config)
        assert mgr.config.min_hold_days == 10
        assert mgr.config.flip_cooldown_days == 20
        assert mgr.config.max_trades_per_week == 5
        assert mgr.config.stop_loss_threshold_pct == 10.0

    def test_multiple_tickers_independent(self):
        """Cooldowns should be tracked independently per ticker."""
        mgr = BacktestCooldownManager(config=CooldownConfig(flip_cooldown_days=5))
        mgr.record_exit("AAPL", datetime(2024, 1, 1))

        # AAPL should still be in cooldown on Jan 3
        allowed, _ = mgr.can_buy("AAPL", datetime(2024, 1, 3))
        assert allowed is False

        # TSLA should be allowed on Jan 3 (no prior exit, and only 1 trade in weekly cap)
        allowed, _ = mgr.can_buy("TSLA", datetime(2024, 1, 3))
        assert allowed is True


class TestCooldownConfig:
    """Test suite for CooldownConfig dataclass."""

    def test_default_values(self):
        """Default config should have expected values."""
        config = CooldownConfig()
        assert config.min_hold_days == 5
        assert config.flip_cooldown_days == 10
        assert config.max_trades_per_week == 2
        assert config.allow_stop_loss_override is True
        assert config.stop_loss_threshold_pct == 5.0

    def test_asdict(self):
        """asdict should serialize all fields."""
        config = CooldownConfig(min_hold_days=3)
        d = __import__('dataclasses').asdict(config)
        assert d["min_hold_days"] == 3
        assert d["flip_cooldown_days"] == 10

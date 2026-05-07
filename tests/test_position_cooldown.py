"""Tests for position cooldown guardrails."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk.position_cooldown import PositionCooldownManager, CooldownConfig


class TestPositionCooldownManager:
    """Test suite for PositionCooldownManager."""

    def setup_method(self):
        self.mgr = PositionCooldownManager(
            data_dir="/tmp/test_cooldown_manager",
            config=CooldownConfig(
                min_hold_days=5,
                flip_cooldown_days=10,
                max_trades_per_week=2,
                allow_stop_loss_override=True,
                stop_loss_threshold_pct=5.0,
            ),
        )

    def teardown_method(self):
        import shutil
        shutil.rmtree("/tmp/test_cooldown_manager", ignore_errors=True)

    def test_can_buy_fresh_ticker(self):
        ok, reason = self.mgr.can_buy("SPY")
        assert ok is True
        assert "No cooldown" in reason

    def test_can_sell_without_entry_record(self):
        ok, reason = self.mgr.can_sell("SPY", 400.0, 400.0)
        assert ok is False
        assert "No entry record" in reason

    def test_min_hold_period_blocks_sell(self):
        self.mgr.record_entry("SPY")
        ok, reason = self.mgr.can_sell("SPY", 400.0, 400.0)
        assert ok is False
        assert "Minimum hold period" in reason

    def test_stop_loss_override_allows_early_sell(self):
        self.mgr.record_entry("SPY")
        # 10% drawdown should trigger stop-loss override
        ok, reason = self.mgr.can_sell("SPY", 360.0, 400.0)
        assert ok is True
        assert "Stop-loss override" in reason

    def test_stop_loss_override_respects_threshold(self):
        self.mgr.record_entry("SPY")
        # 2% drawdown is below 5% threshold
        ok, reason = self.mgr.can_sell("SPY", 392.0, 400.0)
        assert ok is False
        assert "Minimum hold period" in reason

    def test_flip_cooldown_blocks_reentry(self):
        self.mgr.config.max_trades_per_week = 10  # avoid weekly cap
        self.mgr.record_entry("SPY")
        self.mgr.record_exit("SPY")
        ok, reason = self.mgr.can_buy("SPY")
        assert ok is False
        assert "Flip cooldown" in reason

    def test_weekly_trade_cap_blocks_buy(self):
        self.mgr.record_entry("A")
        self.mgr.record_entry("B")
        self.mgr.record_entry("C")
        ok, reason = self.mgr.can_buy("D")
        assert ok is False
        assert "Weekly trade cap" in reason

    def test_weekly_trade_cap_blocks_sell(self):
        self.mgr.record_entry("A")
        self.mgr.record_exit("A")
        self.mgr.record_entry("B")
        self.mgr.record_exit("B")
        self.mgr.record_entry("C")
        self.mgr.record_exit("C")
        ok, reason = self.mgr.can_sell("C", 100.0, 100.0)
        assert ok is False
        assert "Weekly trade cap" in reason

    def test_get_status_structure(self):
        self.mgr.record_entry("SPY")
        status = self.mgr.get_status()
        assert "active_entries" in status
        assert "recent_exits" in status
        assert "trades_this_week" in status
        assert status["trades_this_week"] == 1
        assert "SPY" in status["active_entries"]

    def test_save_and_load_state(self):
        self.mgr.record_entry("SPY")
        self.mgr.save_state()

        mgr2 = PositionCooldownManager(
            data_dir="/tmp/test_cooldown_manager",
            config=CooldownConfig(min_hold_days=5),
        )
        assert "SPY" in mgr2.entries

    def test_persistence_creates_file(self):
        self.mgr.save_state()
        assert (Path("/tmp/test_cooldown_manager") / "position_cooldowns.json").exists()

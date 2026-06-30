"""
Tests for PositionCooldownManager integration into daily_run.py.

Validates that cooldown guardrails are correctly enforced during
the daily trading pipeline.
"""

import sys
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daily_run import backpopulate_cooldown_entries
from risk.position_cooldown import PositionCooldownManager, CooldownConfig


class TestBackpopulateCooldownEntries:
    """Test suite for backpopulate_cooldown_entries helper."""

    def setup_method(self):
        self.test_dir = Path("/tmp/test_backpopulate")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir()

    def teardown_method(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def create_mock_portfolio(self, positions, trades=None):
        """Create a mock portfolio with given positions and trades."""
        portfolio = Mock()
        portfolio.positions = {
            ticker: Mock(avg_price=100.0, quantity=1.0, current_price=100.0)
            for ticker in positions
        }
        trades_file = self.test_dir / "trades_history.json"
        if trades is not None:
            with open(trades_file, 'w') as f:
                json.dump(trades, f)
        portfolio.trades_file = trades_file
        return portfolio

    def test_backpopulate_from_trade_history(self):
        """Should populate entries from most recent buy per ticker."""
        entry_time = datetime(2026, 5, 7, 10, 0, 0)
        trades = [
            {
                'timestamp': '2026-05-01T10:00:00',
                'ticker': 'SPY',
                'action': 'buy',
                'price': 400.0,
                'quantity': 1.0,
                'total_value': 400.0
            },
            {
                'timestamp': entry_time.isoformat(),
                'ticker': 'SPY',
                'action': 'buy',
                'price': 410.0,
                'quantity': 1.0,
                'total_value': 410.0
            },
        ]
        portfolio = self.create_mock_portfolio(['SPY'], trades)
        mgr = PositionCooldownManager(data_dir=str(self.test_dir))

        backpopulate_cooldown_entries(mgr, portfolio)

        assert 'SPY' in mgr.entries
        assert mgr.entries['SPY'] == entry_time

    def test_backpopulate_uses_most_recent_buy(self):
        """Should use the most recent buy, not the first."""
        old_time = datetime(2026, 5, 1, 10, 0, 0)
        new_time = datetime(2026, 5, 10, 10, 0, 0)
        trades = [
            {
                'timestamp': old_time.isoformat(),
                'ticker': 'SPY',
                'action': 'buy',
                'price': 400.0,
                'quantity': 1.0,
                'total_value': 400.0
            },
            {
                'timestamp': new_time.isoformat(),
                'ticker': 'SPY',
                'action': 'buy',
                'price': 410.0,
                'quantity': 1.0,
                'total_value': 410.0
            },
        ]
        portfolio = self.create_mock_portfolio(['SPY'], trades)
        mgr = PositionCooldownManager(data_dir=str(self.test_dir))

        backpopulate_cooldown_entries(mgr, portfolio)

        assert mgr.entries['SPY'] == new_time

    def test_backpopulate_fallback_when_no_trades(self):
        """Should use current time when no trade history exists."""
        portfolio = self.create_mock_portfolio(['SPY'], trades=[])
        mgr = PositionCooldownManager(data_dir=str(self.test_dir))

        before = datetime.now()
        backpopulate_cooldown_entries(mgr, portfolio)
        after = datetime.now()

        assert 'SPY' in mgr.entries
        assert before <= mgr.entries['SPY'] <= after

    def test_backpopulate_no_file_fallback(self):
        """Should handle missing trades file gracefully."""
        portfolio = self.create_mock_portfolio(['SPY'])
        # Don't create trades file
        portfolio.trades_file = self.test_dir / "nonexistent.json"
        mgr = PositionCooldownManager(data_dir=str(self.test_dir))

        # Should not raise
        backpopulate_cooldown_entries(mgr, portfolio)
        assert 'SPY' not in mgr.entries

    def test_backpopulate_multiple_tickers(self):
        """Should handle multiple positions independently."""
        spy_time = datetime(2026, 5, 7, 10, 0, 0)
        tlt_time = datetime(2026, 3, 23, 10, 0, 0)
        trades = [
            {
                'timestamp': spy_time.isoformat(),
                'ticker': 'SPY',
                'action': 'buy',
                'price': 400.0,
                'quantity': 1.0,
                'total_value': 400.0
            },
            {
                'timestamp': tlt_time.isoformat(),
                'ticker': 'TLT',
                'action': 'buy',
                'price': 85.0,
                'quantity': 1.0,
                'total_value': 85.0
            },
        ]
        portfolio = self.create_mock_portfolio(['SPY', 'TLT'], trades)
        mgr = PositionCooldownManager(data_dir=str(self.test_dir))

        backpopulate_cooldown_entries(mgr, portfolio)

        assert mgr.entries['SPY'] == spy_time
        assert mgr.entries['TLT'] == tlt_time

    def test_backpopulate_ignores_sell_trades(self):
        """Should only use buy trades for entry records."""
        buy_time = datetime(2026, 5, 7, 10, 0, 0)
        sell_time = datetime(2026, 5, 8, 10, 0, 0)
        trades = [
            {
                'timestamp': buy_time.isoformat(),
                'ticker': 'SPY',
                'action': 'buy',
                'price': 400.0,
                'quantity': 1.0,
                'total_value': 400.0
            },
            {
                'timestamp': sell_time.isoformat(),
                'ticker': 'SPY',
                'action': 'sell',
                'price': 410.0,
                'quantity': 1.0,
                'total_value': 410.0
            },
        ]
        portfolio = self.create_mock_portfolio(['SPY'], trades)
        mgr = PositionCooldownManager(data_dir=str(self.test_dir))

        backpopulate_cooldown_entries(mgr, portfolio)

        assert mgr.entries['SPY'] == buy_time

    def test_backpopulate_corrupt_file(self):
        """Should handle corrupt trades file gracefully."""
        trades_file = self.test_dir / "trades_history.json"
        with open(trades_file, 'w') as f:
            f.write("not valid json")

        portfolio = self.create_mock_portfolio(['SPY'])
        portfolio.trades_file = trades_file
        mgr = PositionCooldownManager(data_dir=str(self.test_dir))

        # Should not raise
        backpopulate_cooldown_entries(mgr, portfolio)
        assert 'SPY' not in mgr.entries


class TestCooldownExecutionLoop:
    """
    Test suite for the cooldown enforcement in the execution loop.

    These tests validate the logic that would run inside daily_run.py's
    order execution loop by simulating the cooldown checks directly.
    """

    def setup_method(self):
        self.test_dir = Path("/tmp/test_exec_loop")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir()

    def teardown_method(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_buy_blocked_by_flip_cooldown(self):
        """Buying a ticker within flip cooldown should be blocked."""
        mgr = PositionCooldownManager(
            data_dir=str(self.test_dir),
            config=CooldownConfig(flip_cooldown_days=10)
        )
        mgr.record_exit("SPY")

        can_buy, reason = mgr.can_buy("SPY")
        assert can_buy is False
        assert "Flip cooldown" in reason

    def test_sell_blocked_by_min_hold(self):
        """Selling before min hold period should be blocked."""
        mgr = PositionCooldownManager(
            data_dir=str(self.test_dir),
            config=CooldownConfig(min_hold_days=5)
        )
        mgr.record_entry("SPY")

        can_sell, reason = mgr.can_sell("SPY", 400.0, 400.0)
        assert can_sell is False
        assert "Minimum hold period" in reason

    def test_sell_allowed_after_hold_period(self):
        """Selling after min hold period should be allowed."""
        mgr = PositionCooldownManager(
            data_dir=str(self.test_dir),
            config=CooldownConfig(min_hold_days=5)
        )
        # Manually set entry to 6 days ago
        mgr.entries["SPY"] = datetime.now() - timedelta(days=6)

        can_sell, reason = mgr.can_sell("SPY", 400.0, 400.0)
        assert can_sell is True
        assert "Hold period satisfied" in reason

    def test_weekly_cap_blocks_additional_trades(self):
        """Weekly trade cap should block trades beyond limit."""
        mgr = PositionCooldownManager(
            data_dir=str(self.test_dir),
            config=CooldownConfig(
                max_trades_per_week=2,
                max_trades_high_vol=2,
                max_trades_normal_vol=2,
                max_trades_low_vol=2,
            )
        )
        mgr.record_entry("A")
        mgr.record_entry("B")

        can_buy, reason = mgr.can_buy("C")
        assert can_buy is False
        assert "Weekly trade cap" in reason

    def test_stop_loss_override_allows_early_exit(self):
        """Stop-loss override should allow selling before min hold."""
        mgr = PositionCooldownManager(
            data_dir=str(self.test_dir),
            config=CooldownConfig(min_hold_days=5, stop_loss_threshold_pct=5.0)
        )
        mgr.record_entry("SPY")

        can_sell, reason = mgr.can_sell("SPY", 90.0, 100.0)
        assert can_sell is True
        assert "Stop-loss override" in reason

    def test_trade_records_persist(self):
        """Entry/exit records should persist to disk."""
        mgr = PositionCooldownManager(data_dir=str(self.test_dir))
        mgr.record_entry("SPY")
        mgr.save_state()

        # Create new manager instance
        mgr2 = PositionCooldownManager(data_dir=str(self.test_dir))
        assert "SPY" in mgr2.entries

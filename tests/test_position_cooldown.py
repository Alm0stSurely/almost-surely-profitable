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
        # Fail-open on exit: a bookkeeping gap must not block the exit door.
        ok, reason = self.mgr.can_sell("SPY", 400.0, 400.0)
        assert ok is True
        assert "No entry record" in reason
        assert "exit allowed" in reason

    def test_missing_entry_record_does_not_block_stop_loss_exit(self):
        # A position bleeding past the stop-loss threshold with lost
        # bookkeeping must remain exitable through the guarded path.
        ok, reason = self.mgr.can_sell("SPY", 340.0, 400.0)
        assert ok is True
        assert "No entry record" in reason

    def test_missing_entry_record_still_respects_weekly_cap(self):
        # The weekly trade cap is verifiable bookkeeping and restrains
        # frequency, not risk exits of unrecorded positions — but it is
        # evaluated before the entry-record check and still applies.
        self.mgr.record_entry("A")
        self.mgr.record_exit("A")
        self.mgr.record_entry("B")
        self.mgr.record_exit("B")
        ok, reason = self.mgr.can_sell("SPY", 400.0, 400.0)
        assert ok is False
        assert "Weekly trade cap" in reason

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

    def test_trades_from_previous_week_not_counted(self):
        """A trade from the previous ISO week should not count toward the current week's cap."""
        from datetime import datetime
        from unittest.mock import patch

        # Two trades in the current week, one from the previous Friday
        self.mgr.weekly_trades = [
            datetime(2026, 6, 26, 21, 0, 0),  # Friday previous week
            datetime(2026, 6, 30, 21, 0, 0),  # Tuesday current week
            datetime(2026, 7, 1, 21, 0, 0),  # Wednesday current week
        ]

        class _FakeDatetime:
            @classmethod
            def now(cls):
                return datetime(2026, 7, 3, 21, 0, 0)  # Friday current week

            @classmethod
            def fromisoformat(cls, s):
                return datetime.fromisoformat(s)

            # timedelta is used elsewhere in the module, but it's imported directly
            # so we don't need to expose it here.

        with patch("risk.position_cooldown.datetime", _FakeDatetime):
            status = self.mgr.get_status()
            # The previous Friday trade should be filtered out, leaving 2 current-week trades
            assert status["trades_this_week"] == 2

    def test_weekly_cap_resets_after_week_boundary(self):
        """Once the cap is hit, new trades should be allowed again after the week boundary."""
        from datetime import datetime
        from unittest.mock import patch

        # Hit the cap last week
        self.mgr.weekly_trades = [
            datetime(2026, 6, 24, 21, 0, 0),  # Wednesday
            datetime(2026, 6, 25, 21, 0, 0),  # Thursday
        ]

        class _FakeDatetime:
            @classmethod
            def now(cls):
                return datetime(2026, 6, 29, 21, 0, 0)  # Monday next week

            @classmethod
            def fromisoformat(cls, s):
                return datetime.fromisoformat(s)

        with patch("risk.position_cooldown.datetime", _FakeDatetime):
            # After the week boundary, no trades should count against the current cap
            ok, reason = self.mgr.can_buy("SPY")
            assert ok is True
            assert "No cooldown" in reason

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


class TestReconcileEntries:
    """reconcile_entries() drops entry records for no-longer-held tickers."""

    def setup_method(self):
        self.mgr = PositionCooldownManager(
            data_dir="/tmp/test_cooldown_reconcile",
            config=CooldownConfig(min_hold_days=5, flip_cooldown_days=10),
        )

    def teardown_method(self):
        import shutil
        shutil.rmtree("/tmp/test_cooldown_reconcile", ignore_errors=True)

    def test_prunes_stale_entry_and_keeps_held(self):
        self.mgr.entries["SPY"] = datetime.now() - timedelta(days=20)
        self.mgr.entries["QQQ"] = datetime.now() - timedelta(days=90)
        self.mgr.entries["TTE.PA"] = datetime.now() - timedelta(days=79)

        pruned = self.mgr.reconcile_entries(["SPY", "GLD"])

        assert pruned == ["QQQ", "TTE.PA"]
        assert "SPY" in self.mgr.entries
        assert "QQQ" not in self.mgr.entries
        assert "TTE.PA" not in self.mgr.entries

    def test_reconcile_does_not_arm_flip_cooldown(self):
        """Pruning must not synthesize exits — re-entry must stay allowed.

        Writing an exit timestamp for a pruned ticker would arm the flip
        cooldown and block legitimate re-entry for flip_cooldown_days.
        """
        self.mgr.entries["QQQ"] = datetime.now() - timedelta(days=90)

        self.mgr.reconcile_entries([])

        assert "QQQ" not in self.mgr.exits
        can_buy, reason = self.mgr.can_buy("QQQ")
        assert can_buy is True
        assert "Flip cooldown" not in reason

    def test_reconcile_empty_held_prunes_everything(self):
        self.mgr.entries["SPY"] = datetime.now()
        self.mgr.entries["TLT"] = datetime.now()

        pruned = self.mgr.reconcile_entries([])

        assert sorted(pruned) == ["SPY", "TLT"]
        assert self.mgr.entries == {}

    def test_reconcile_is_idempotent(self):
        self.mgr.entries["SPY"] = datetime.now()

        first = self.mgr.reconcile_entries(["SPY"])
        second = self.mgr.reconcile_entries(["SPY"])

        assert first == []
        assert second == []
        assert "SPY" in self.mgr.entries

    def test_reconcile_returns_sorted_list(self):
        self.mgr.entries["ZZZ"] = datetime.now()
        self.mgr.entries["AAA"] = datetime.now()
        self.mgr.entries["MMM"] = datetime.now()

        pruned = self.mgr.reconcile_entries([])

        assert pruned == ["AAA", "MMM", "ZZZ"]

    def test_reconcile_pruned_state_persists(self):
        """After save_state, a fresh manager must not resurrect pruned entries."""
        self.mgr.entries["SPY"] = datetime.now()
        self.mgr.entries["QQQ"] = datetime.now()
        self.mgr.reconcile_entries(["SPY"])
        self.mgr.save_state()

        mgr2 = PositionCooldownManager(
            data_dir="/tmp/test_cooldown_reconcile",
            config=CooldownConfig(min_hold_days=5),
        )
        assert "SPY" in mgr2.entries
        assert "QQQ" not in mgr2.entries


class TestStrictJsonPersistenceBoundary:
    """The cooldown state file is float-free (ISO date strings only); the
    strict serializer flag is a static contract that it stays that way."""

    def setup_method(self):
        self.mgr = PositionCooldownManager(
            data_dir="/tmp/test_cooldown_manager",
            config=CooldownConfig(min_hold_days=5),
        )

    def teardown_method(self):
        import shutil
        shutil.rmtree("/tmp/test_cooldown_manager", ignore_errors=True)

    def _parse_strict(self, text):
        import json

        def _reject(token):
            raise ValueError(f"Non-standard JSON token: {token}")

        return json.loads(text, parse_constant=_reject)

    def test_state_file_uses_strict_json_tokens(self):
        self.mgr.record_entry("SPY")
        self.mgr.record_exit("QQQ")
        self.mgr.save_state()

        raw = (Path("/tmp/test_cooldown_manager") / "position_cooldowns.json").read_text()
        state = self._parse_strict(raw)
        assert "SPY" in state["entries"]
        assert "QQQ" in state["exits"]

    def test_round_trip_unchanged_with_strict_flag(self):
        self.mgr.record_entry("MC.PA")
        self.mgr.save_state()

        mgr2 = PositionCooldownManager(
            data_dir="/tmp/test_cooldown_manager",
            config=CooldownConfig(min_hold_days=5),
        )
        assert "MC.PA" in mgr2.entries

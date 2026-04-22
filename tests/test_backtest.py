"""
Tests for the backtesting framework.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
from backtest.backtest import RandomStrategy
from portfolio.portfolio import Portfolio


class TestRandomStrategy:
    """Tests for the RandomStrategy baseline."""

    def test_init_with_seed(self):
        """Test that RandomStrategy initializes with a seed."""
        rs = RandomStrategy(seed=42)
        assert rs.rng is not None
        assert rs.max_position_pct == 30.0

    def test_init_custom_max_position(self):
        """Test custom max position percentage."""
        rs = RandomStrategy(seed=42, max_position_pct=20.0)
        assert rs.max_position_pct == 20.0

    def test_generate_decisions_respects_weights(self):
        """Test that decisions respect action probabilities."""
        rs = RandomStrategy(seed=42)
        portfolio = Portfolio(state_file="test_random.json", data_dir="data/test")

        prices = {"SPY": 100.0, "QQQ": 200.0, "GLD": 150.0}

        # Run many times to check distribution
        all_decisions = []
        for _ in range(100):
            decisions = rs.generate_decisions(["SPY", "QQQ", "GLD"], portfolio, prices)
            all_decisions.extend(decisions)

        # Should have some holds, possibly some buys (no sells since no positions)
        actions = [d["action"] for d in all_decisions]
        assert "hold" in actions
        # With 100 iterations * 3 tickers = 300 decisions, we expect some buys
        buy_count = actions.count("buy")
        assert buy_count > 0, "Expected some buy decisions over 300 samples"
        assert actions.count("sell") == 0, "Should not sell without positions"

    def test_generate_decisions_sell_only_with_positions(self):
        """Test that sell only happens when we have positions."""
        rs = RandomStrategy(seed=42)
        portfolio = Portfolio(state_file="test_random.json", data_dir="data/test")
        portfolio.buy("SPY", 50.0, 100.0)

        prices = {"SPY": 100.0}

        # Run many times to trigger potential sells
        sell_found = False
        for _ in range(200):
            decisions = rs.generate_decisions(["SPY"], portfolio, prices)
            for d in decisions:
                if d["action"] == "sell":
                    sell_found = True
                    assert d["pct"] >= 25.0 and d["pct"] <= 100.0
                    break
            if sell_found:
                break

        # With enough iterations, we should see at least one sell
        # (25% probability * 200 = expected 50 sells)
        assert sell_found, "Expected at least one sell decision with positions"

    def test_generate_decisions_buy_range(self):
        """Test that buy pct is within expected range."""
        rs = RandomStrategy(seed=42, max_position_pct=30.0)
        portfolio = Portfolio(state_file="test_random.json", data_dir="data/test")
        prices = {"SPY": 100.0}

        for _ in range(100):
            decisions = rs.generate_decisions(["SPY"], portfolio, prices)
            for d in decisions:
                if d["action"] == "buy":
                    assert 5.0 <= d["pct"] <= 30.0

    def test_reproducibility_with_same_seed(self):
        """Test that same seed produces identical decisions."""
        rs1 = RandomStrategy(seed=123)
        rs2 = RandomStrategy(seed=123)
        portfolio = Portfolio(state_file="test_random.json", data_dir="data/test")
        prices = {"SPY": 100.0, "QQQ": 200.0}

        for _ in range(50):
            d1 = rs1.generate_decisions(["SPY", "QQQ"], portfolio, prices)
            d2 = rs2.generate_decisions(["SPY", "QQQ"], portfolio, prices)
            assert len(d1) == len(d2)
            for a, b in zip(d1, d2):
                assert a["ticker"] == b["ticker"]
                assert a["action"] == b["action"]
                assert a["pct"] == b["pct"]

    def test_different_seeds_produce_different_decisions(self):
        """Test that different seeds produce different sequences."""
        rs1 = RandomStrategy(seed=1)
        rs2 = RandomStrategy(seed=999)
        portfolio = Portfolio(state_file="test_random.json", data_dir="data/test")
        prices = {"SPY": 100.0, "QQQ": 200.0, "GLD": 150.0}

        d1 = rs1.generate_decisions(["SPY", "QQQ", "GLD"], portfolio, prices)
        d2 = rs2.generate_decisions(["SPY", "QQQ", "GLD"], portfolio, prices)

        # Different seeds should likely produce different first decisions
        # (probability of collision is low but non-zero; check action counts)
        actions1 = [d["action"] for d in d1]
        actions2 = [d["action"] for d in d2]
        # They could be the same by chance, but over many runs they'd diverge
        # This is a weak test but sufficient for seed differentiation
        assert actions1 != actions2 or d1[0]["pct"] != d2[0]["pct"], \
            "Expected different decisions with different seeds"

    def test_skips_tickers_without_prices(self):
        """Test that tickers without prices are skipped."""
        rs = RandomStrategy(seed=42)
        portfolio = Portfolio(state_file="test_random.json", data_dir="data/test")
        prices = {"SPY": 100.0}  # QQQ not in prices

        # Run multiple times to ensure SPY shows up (it might be skipped
        # on individual draws if action is sell with no position)
        all_tickers = set()
        for _ in range(20):
            decisions = rs.generate_decisions(["SPY", "QQQ"], portfolio, prices)
            all_tickers.update(d["ticker"] for d in decisions)

        assert "SPY" in all_tickers
        assert "QQQ" not in all_tickers

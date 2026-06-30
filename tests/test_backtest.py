"""
Tests for the backtesting framework.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import pandas as pd
from backtest.backtest import BacktestEngine, RandomStrategy, print_backtest_report, run_comparison_backtest
from portfolio.portfolio import Portfolio


class TestRandomStrategy:
    """Tests for the RandomStrategy baseline."""

    def _make_portfolio(self, tmp_path):
        """Create an isolated Portfolio using a temporary directory."""
        return Portfolio(state_file="test_random.json", data_dir=str(tmp_path))

    def test_init_with_seed(self, tmp_path):
        """Test that RandomStrategy initializes with a seed."""
        rs = RandomStrategy(seed=42)
        assert rs.rng is not None
        assert rs.max_position_pct == 30.0

    def test_init_custom_max_position(self, tmp_path):
        """Test custom max position percentage."""
        rs = RandomStrategy(seed=42, max_position_pct=20.0)
        assert rs.max_position_pct == 20.0

    def test_generate_decisions_respects_weights(self, tmp_path):
        """Test that decisions respect action probabilities."""
        rs = RandomStrategy(seed=42)
        portfolio = self._make_portfolio(tmp_path)

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

    def test_generate_decisions_sell_only_with_positions(self, tmp_path):
        """Test that sell only happens when we have positions."""
        rs = RandomStrategy(seed=42)
        portfolio = self._make_portfolio(tmp_path)
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

    def test_generate_decisions_buy_range(self, tmp_path):
        """Test that buy pct is within expected range."""
        rs = RandomStrategy(seed=42, max_position_pct=30.0)
        portfolio = self._make_portfolio(tmp_path)
        prices = {"SPY": 100.0}

        for _ in range(100):
            decisions = rs.generate_decisions(["SPY"], portfolio, prices)
            for d in decisions:
                if d["action"] == "buy":
                    assert 5.0 <= d["pct"] <= 30.0

    def test_reproducibility_with_same_seed(self, tmp_path):
        """Test that same seed produces identical decisions."""
        rs1 = RandomStrategy(seed=123)
        rs2 = RandomStrategy(seed=123)
        portfolio = self._make_portfolio(tmp_path)
        prices = {"SPY": 100.0, "QQQ": 200.0}

        for _ in range(50):
            d1 = rs1.generate_decisions(["SPY", "QQQ"], portfolio, prices)
            d2 = rs2.generate_decisions(["SPY", "QQQ"], portfolio, prices)
            assert len(d1) == len(d2)
            for a, b in zip(d1, d2):
                assert a["ticker"] == b["ticker"]
                assert a["action"] == b["action"]
                assert a["pct"] == b["pct"]

    def test_different_seeds_produce_different_decisions(self, tmp_path):
        """Test that different seeds produce different sequences."""
        rs1 = RandomStrategy(seed=1)
        rs2 = RandomStrategy(seed=999)
        portfolio = self._make_portfolio(tmp_path)
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

    def test_skips_tickers_without_prices(self, tmp_path):
        """Test that tickers without prices are skipped."""
        rs = RandomStrategy(seed=42)
        portfolio = self._make_portfolio(tmp_path)
        prices = {"SPY": 100.0}  # QQQ not in prices

        # Run multiple times to ensure SPY shows up (it might be skipped
        # on individual draws if action is sell with no position)
        all_tickers = set()
        for _ in range(20):
            decisions = rs.generate_decisions(["SPY", "QQQ"], portfolio, prices)
            all_tickers.update(d["ticker"] for d in decisions)

        assert "SPY" in all_tickers
        assert "QQQ" not in all_tickers


class TestBacktestEngine:
    """Tests for the BacktestEngine — the core simulation engine."""

    # ── Helpers ──────────────────────────────────────────────────────────

    def _make_engine(self, start="2024-01-01", end="2024-01-31", **kwargs):
        return BacktestEngine(
            start_date=start,
            end_date=end,
            tickers=["SPY", "QQQ"],
            **kwargs
        )

    def _make_portfolio(self, tmp_path, cash=10000.0):
        p = Portfolio(
            state_file="bt_test.json",
            trades_file="bt_trades.json",
            data_dir=str(tmp_path)
        )
        p.cash = cash
        p.positions = {}
        p.trades = []
        p.total_realized_pnl = 0.0
        return p

    def _make_mock_data(self, tickers, dates, base_price=100.0):
        """Build a dict of DataFrames with Open/High/Low/Close."""
        data = {}
        for ticker in tickers:
            prices = [base_price + i * 0.5 for i in range(len(dates))]
            data[ticker] = pd.DataFrame({
                "Open": prices,
                "High": [p + 0.5 for p in prices],
                "Low": [p - 0.5 for p in prices],
                "Close": prices,
            }, index=pd.to_datetime(dates))
        return data

    # ── Init / Config ─────────────────────────────────────────────────────

    def test_init_defaults(self):
        engine = self._make_engine()
        assert engine.initial_capital == 10000.0
        assert engine.rebalance_frequency == "daily"
        assert engine.tickers == ["SPY", "QQQ"]
        assert engine.enable_cooldowns is False

    def test_init_custom_config(self, tmp_path):
        engine = self._make_engine(
            rebalance_frequency="weekly",
            enable_cooldowns=True,
            initial_capital=5000.0
        )
        assert engine.rebalance_frequency == "weekly"
        assert engine.enable_cooldowns is True
        assert engine.initial_capital == 5000.0
        # portfolio is not set until run_backtest, so no crash
        assert engine.portfolio is None

    # ── _calculate_metrics ─────────────────────────────────────────────────

    def _setup_metrics_engine(self, engine, tmp_path):
        """Attach a minimal portfolio so _calculate_metrics can access trades."""
        engine.portfolio = self._make_portfolio(tmp_path, cash=10000.0)

    def test_calculate_metrics_empty_results(self, tmp_path):
        engine = self._make_engine()
        self._setup_metrics_engine(engine, tmp_path)
        engine.results = []
        metrics = engine._calculate_metrics()
        assert metrics == {}

    def test_calculate_metrics_single_day(self, tmp_path):
        engine = self._make_engine()
        self._setup_metrics_engine(engine, tmp_path)
        engine.initial_capital = 10000.0
        engine.results = [{"total_value": 10100.0}]
        metrics = engine._calculate_metrics()
        assert metrics["total_return"] == pytest.approx(0.01)
        assert metrics["annualized_return"] > 0
        assert metrics["volatility"] == 0  # no returns with single day
        assert metrics["sharpe_ratio"] == 0
        assert metrics["num_trades"] == 0

    def test_calculate_metrics_basic(self, tmp_path):
        engine = self._make_engine()
        self._setup_metrics_engine(engine, tmp_path)
        engine.initial_capital = 10000.0
        # 10 days: small positive drift
        values = [10000.0 + i * 10 for i in range(10)]
        engine.results = [{"total_value": v} for v in values]
        metrics = engine._calculate_metrics()
        assert metrics["total_return"] == (values[-1] / 10000.0) - 1
        assert metrics["volatility"] > 0
        assert metrics["sharpe_ratio"] > 0
        assert metrics["max_drawdown"] == 0  # monotonic increase
        assert metrics["win_rate"] == 1.0  # every day profitable
        assert metrics["profit_factor"] == float("inf")  # no losses
        assert metrics["omega_ratio"] == float("inf")
        assert metrics["calmar_ratio"] >= 0
        assert metrics["annualized_return"] > 0
        assert len(metrics["equity_curve"]) == 10
        assert len(metrics["daily_returns"]) == 9
        assert len(metrics["drawdown_curve"]) == 10

    def test_calculate_metrics_with_losses(self, tmp_path):
        engine = self._make_engine()
        self._setup_metrics_engine(engine, tmp_path)
        engine.initial_capital = 10000.0
        # Alternating up/down
        values = [10000.0, 10100.0, 10050.0, 10150.0, 10000.0]
        engine.results = [{"total_value": v} for v in values]
        metrics = engine._calculate_metrics()
        assert metrics["total_return"] == 0.0
        assert metrics["win_rate"] == 0.5  # 2 up, 2 down out of 4 returns
        assert metrics["profit_factor"] > 0
        assert metrics["profit_factor"] != float("inf")
        assert metrics["max_drawdown"] > 0
        # With risk-free rate=0.02 and flat total return, sortino can be negative
        assert metrics["sortino_ratio"] is not None

    def test_calculate_metrics_zero_volatility(self, tmp_path):
        engine = self._make_engine()
        self._setup_metrics_engine(engine, tmp_path)
        engine.initial_capital = 10000.0
        # Flat — no returns, constant value
        engine.results = [{"total_value": 10000.0} for _ in range(10)]
        metrics = engine._calculate_metrics()
        assert metrics["total_return"] == 0.0
        assert metrics["volatility"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["sortino_ratio"] == 0.0
        assert metrics["calmar_ratio"] == 0.0  # max_drawdown = 0
        assert metrics["win_rate"] == 0.0  # no profitable returns
        # profit_factor undefined (no losses) → inf
        assert metrics["profit_factor"] == float("inf")

    def test_calculate_metrics_with_benchmark(self, tmp_path):
        engine = self._make_engine()
        self._setup_metrics_engine(engine, tmp_path)
        engine.initial_capital = 10000.0
        values = [10000.0 + i * 10 for i in range(10)]
        engine.results = [{"total_value": v} for v in values]
        # 9 daily returns — use varying benchmark so covariance is non-zero
        benchmark_returns = [0.001, 0.002, 0.0015, 0.0005, 0.002, 0.001, 0.003, 0.001, 0.002]
        metrics = engine._calculate_metrics(benchmark_returns)
        assert metrics["beta"] is not None
        assert metrics["alpha"] is not None
        # beta is a real number (can be zero, positive, or negative)
        assert np.isfinite(metrics["beta"])

    def test_calculate_metrics_benchmark_mismatch_length(self, tmp_path):
        engine = self._make_engine()
        self._setup_metrics_engine(engine, tmp_path)
        engine.initial_capital = 10000.0
        values = [10000.0 + i * 10 for i in range(10)]
        engine.results = [{"total_value": v} for v in values]
        benchmark_returns = [0.001] * 5  # wrong length
        metrics = engine._calculate_metrics(benchmark_returns)
        assert metrics["beta"] == 0
        assert metrics["alpha"] == 0

    def test_calculate_metrics_benchmark_empty(self, tmp_path):
        engine = self._make_engine()
        self._setup_metrics_engine(engine, tmp_path)
        engine.initial_capital = 10000.0
        values = [10000.0 + i * 10 for i in range(10)]
        engine.results = [{"total_value": v} for v in values]
        metrics = engine._calculate_metrics([])
        assert metrics["beta"] == 0
        assert metrics["alpha"] == 0

    def test_calculate_metrics_negative_total_return(self, tmp_path):
        engine = self._make_engine()
        self._setup_metrics_engine(engine, tmp_path)
        engine.initial_capital = 10000.0
        values = [10000.0 - i * 20 for i in range(10)]
        engine.results = [{"total_value": v} for v in values]
        metrics = engine._calculate_metrics()
        assert metrics["total_return"] < 0
        assert metrics["annualized_return"] < 0
        assert metrics["sharpe_ratio"] < 0

    # ── _should_rebalance ──────────────────────────────────────────────────

    def test_should_rebalance_daily(self):
        engine = self._make_engine(rebalance_frequency="daily")
        for i in range(10):
            assert engine._should_rebalance(i, datetime(2024, 1, 1 + i)) is True

    def test_should_rebalance_weekly(self):
        engine = self._make_engine(rebalance_frequency="weekly")
        assert engine._should_rebalance(0, datetime(2024, 1, 1)) is True
        assert engine._should_rebalance(1, datetime(2024, 1, 2)) is False
        assert engine._should_rebalance(4, datetime(2024, 1, 5)) is False
        assert engine._should_rebalance(5, datetime(2024, 1, 8)) is True
        assert engine._should_rebalance(10, datetime(2024, 1, 15)) is True

    def test_should_rebalance_unknown_frequency(self):
        engine = self._make_engine(rebalance_frequency="monthly")
        assert engine._should_rebalance(0, datetime(2024, 1, 1)) is False

    # ── _get_trading_dates ─────────────────────────────────────────────────

    def test_get_trading_dates(self):
        engine = self._make_engine(start="2024-01-01", end="2024-01-05")
        dates = pd.date_range("2024-01-01", "2024-01-05")
        data = {"SPY": pd.DataFrame({"Close": [100.0] * 5}, index=dates)}
        trading_dates = engine._get_trading_dates(data)
        assert len(trading_dates) == 5
        assert trading_dates[0] == datetime(2024, 1, 1)
        assert trading_dates[-1] == datetime(2024, 1, 5)

    def test_get_trading_dates_filters_outside_range(self):
        engine = self._make_engine(start="2024-01-02", end="2024-01-04")
        dates = pd.date_range("2024-01-01", "2024-01-05")
        data = {"SPY": pd.DataFrame({"Close": [100.0] * 5}, index=dates)}
        trading_dates = engine._get_trading_dates(data)
        assert len(trading_dates) == 3
        assert trading_dates[0] == datetime(2024, 1, 2)
        assert trading_dates[-1] == datetime(2024, 1, 4)

    # ── _precompute_price_lookups / _get_prices_for_date ──────────────────

    def test_precompute_and_get_prices(self):
        engine = self._make_engine()
        dates = pd.date_range("2024-01-01", "2024-01-03")
        data = {
            "SPY": pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=dates),
            "QQQ": pd.DataFrame({"Close": [200.0, 201.0, 202.0]}, index=dates),
        }
        engine._precompute_price_lookups(data)
        assert len(engine._price_lookups) == 2
        assert engine._price_lookups["SPY"][datetime(2024, 1, 2).date()] == 101.0
        assert engine._price_lookups["QQQ"][datetime(2024, 1, 3).date()] == 202.0

        prices = engine._get_prices_for_date(data, datetime(2024, 1, 2))
        assert prices["SPY"] == 101.0
        assert prices["QQQ"] == 201.0
        assert len(prices) == 2

    def test_get_prices_missing_date(self):
        engine = self._make_engine()
        dates = pd.date_range("2024-01-01", "2024-01-03")
        data = {"SPY": pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=dates)}
        engine._precompute_price_lookups(data)
        prices = engine._get_prices_for_date(data, datetime(2024, 1, 10))
        assert "SPY" not in prices
        assert len(prices) == 0

    # ── _get_benchmark_returns ─────────────────────────────────────────────

    def test_get_benchmark_returns(self):
        engine = self._make_engine(start="2024-01-01", end="2024-01-05")
        dates = pd.date_range("2024-01-01", "2024-01-05")
        closes = [100.0, 101.0, 102.0, 101.0, 103.0]
        data = {"SPY": pd.DataFrame({"Close": closes}, index=dates)}
        returns = engine._get_benchmark_returns(data, "SPY")
        assert len(returns) == 4
        expected = [1.0 / 100.0, 1.0 / 101.0, -1.0 / 102.0, 2.0 / 101.0]
        for r, e in zip(returns, expected):
            assert abs(r - e) < 1e-12

    def test_get_benchmark_returns_missing_ticker(self):
        engine = self._make_engine(start="2024-01-01", end="2024-01-05")
        dates = pd.date_range("2024-01-01", "2024-01-05")
        data = {"QQQ": pd.DataFrame({"Close": [100.0] * 5}, index=dates)}
        returns = engine._get_benchmark_returns(data, "SPY")
        assert returns == []

    def test_get_benchmark_returns_single_close(self):
        engine = self._make_engine(start="2024-01-01", end="2024-01-01")
        dates = pd.date_range("2024-01-01", "2024-01-01")
        data = {"SPY": pd.DataFrame({"Close": [100.0]}, index=dates)}
        returns = engine._get_benchmark_returns(data, "SPY")
        assert returns == []

    # ── Strategy: buy_and_hold ─────────────────────────────────────────────

    def test_buy_and_hold_initial_purchase(self, tmp_path):
        engine = self._make_engine()
        engine.portfolio = self._make_portfolio(tmp_path, cash=10000.0)
        prices = {"SPY": 100.0, "QQQ": 200.0}
        engine._execute_buy_and_hold_strategy(datetime(2024, 1, 1), prices)
        assert len(engine.portfolio.positions) == 2
        assert engine.portfolio.positions["SPY"].quantity > 0
        assert engine.portfolio.positions["QQQ"].quantity > 0
        # 90% invested, 10% cash buffer
        invested = 10000.0 - engine.portfolio.cash
        assert invested > 0
        assert invested <= 9000.0 + 1e-9

    def test_buy_and_hold_no_rebalance(self, tmp_path):
        engine = self._make_engine()
        engine.portfolio = self._make_portfolio(tmp_path, cash=10000.0)
        prices = {"SPY": 100.0, "QQQ": 200.0}
        engine._execute_buy_and_hold_strategy(datetime(2024, 1, 1), prices)
        initial_cash = engine.portfolio.cash
        initial_spy_qty = engine.portfolio.positions["SPY"].quantity
        # Call again — should not trade
        engine._execute_buy_and_hold_strategy(datetime(2024, 1, 2), prices)
        assert engine.portfolio.cash == initial_cash
        assert engine.portfolio.positions["SPY"].quantity == initial_spy_qty

    def test_buy_and_hold_empty_prices(self, tmp_path):
        engine = self._make_engine()
        engine.portfolio = self._make_portfolio(tmp_path, cash=10000.0)
        engine._execute_buy_and_hold_strategy(datetime(2024, 1, 1), {})
        assert len(engine.portfolio.positions) == 0
        assert engine.portfolio.cash == 10000.0

    # ── Strategy: equal_weight ───────────────────────────────────────────

    def test_equal_weight_initial_purchase(self, tmp_path):
        engine = self._make_engine()
        engine.portfolio = self._make_portfolio(tmp_path, cash=10000.0)
        prices = {"SPY": 100.0, "QQQ": 200.0, "GLD": 150.0}
        engine._execute_equal_weight_strategy(datetime(2024, 1, 1), prices)
        assert len(engine.portfolio.positions) == 3
        # 90% / 3 = 30% per ticker
        for ticker in prices:
            assert engine.portfolio.positions[ticker].quantity > 0

    def test_equal_weight_rebalance_sell_overweight(self, tmp_path):
        engine = self._make_engine()
        engine.portfolio = self._make_portfolio(tmp_path, cash=10000.0)
        prices = {"SPY": 100.0, "QQQ": 200.0}
        engine._execute_equal_weight_strategy(datetime(2024, 1, 1), prices)
        # Update prices to reflect market movement before rebalancing
        new_prices = {"SPY": 200.0, "QQQ": 200.0}
        engine.portfolio.update_prices(new_prices)
        engine._execute_equal_weight_strategy(datetime(2024, 1, 2), new_prices)
        # SPY doubled → now overweight. Should have been partially sold.
        spy_value = engine.portfolio.positions["SPY"].market_value
        total = engine.portfolio.total_value
        spy_pct = spy_value / total
        assert spy_pct < 0.55  # should be closer to 45% after rebalance

    def test_equal_weight_no_rebalance_within_tolerance(self, tmp_path):
        engine = self._make_engine()
        engine.portfolio = self._make_portfolio(tmp_path, cash=10000.0)
        prices = {"SPY": 100.0, "QQQ": 200.0}
        engine._execute_equal_weight_strategy(datetime(2024, 1, 1), prices)
        initial_trades = len(engine.portfolio.trades)
        # Tiny price change — within 5% tolerance, no rebalance
        new_prices = {"SPY": 101.0, "QQQ": 200.0}
        engine.portfolio.update_prices(new_prices)
        engine._execute_equal_weight_strategy(datetime(2024, 1, 2), new_prices)
        # No new trades (rebalance only when >5% deviation)
        assert len(engine.portfolio.trades) == initial_trades

    def test_equal_weight_empty_prices(self, tmp_path):
        engine = self._make_engine()
        engine.portfolio = self._make_portfolio(tmp_path, cash=10000.0)
        engine._execute_equal_weight_strategy(datetime(2024, 1, 1), {})
        assert len(engine.portfolio.positions) == 0

    # ── Integration: run_backtest with buy_and_hold ────────────────────────

    @patch("backtest.backtest.Portfolio")
    @patch("backtest.backtest.fetch_historical_data")
    def test_run_backtest_buy_and_hold(self, mock_fetch, mock_Portfolio, tmp_path):
        # Force Portfolio to use our temp directory so stale state files
        # in data/backtest don't pollute the test.
        def _make_portfolio(*args, **kwargs):
            kwargs["data_dir"] = str(tmp_path)
            return Portfolio(*args, **kwargs)
        mock_Portfolio.side_effect = _make_portfolio

        engine = self._make_engine(start="2024-01-01", end="2024-01-05")
        dates = pd.date_range("2024-01-01", "2024-01-05")
        mock_data = {
            "SPY": pd.DataFrame({
                "Open": [100.0] * 5,
                "High": [101.0] * 5,
                "Low": [99.0] * 5,
                "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
            }, index=dates),
            "QQQ": pd.DataFrame({
                "Open": [200.0] * 5,
                "High": [201.0] * 5,
                "Low": [199.0] * 5,
                "Close": [200.0, 201.0, 202.0, 203.0, 204.0],
            }, index=dates),
        }
        mock_fetch.return_value = mock_data
        result = engine.run_backtest(strategy="buy_and_hold")
        assert result["strategy"] == "buy_and_hold"
        assert result["num_trades"] > 0
        assert result["total_return"] > 0
        assert len(result["daily_results"]) == 5
        assert len(result["equity_curve"]) == 5
        assert result["final_value"] > result["initial_capital"]

    @patch("backtest.backtest.fetch_historical_data")
    def test_run_backtest_no_data(self, mock_fetch, tmp_path):
        engine = self._make_engine(start="2024-01-01", end="2024-01-05")
        mock_fetch.return_value = {}
        result = engine.run_backtest(strategy="buy_and_hold")
        assert result == {}

    # ── print_backtest_report ──────────────────────────────────────────────

    def test_print_backtest_report_empty(self, caplog):
        with patch("backtest.backtest.logger") as mock_logger:
            print_backtest_report({}, "test")
            mock_logger.error.assert_called_once()

    def test_print_backtest_report_basic(self, capsys):
        result = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-05",
            "initial_capital": 10000.0,
            "final_value": 10500.0,
            "total_return": 0.05,
            "annualized_return": 5.0,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.5,
            "max_drawdown": 0.01,
            "calmar_ratio": 500.0,
            "omega_ratio": 2.0,
            "win_rate": 0.75,
            "profit_factor": 3.0,
            "beta": 0.8,
            "alpha": 0.02,
            "volatility": 0.15,
            "num_trades": 10,
            "equity_curve": [10000.0, 10200.0, 10500.0],
            "drawdown_curve": [0.0, 0.0, 0.0],
            "daily_returns": [0.02, 0.0294],
        }
        print_backtest_report(result, "buy_and_hold")
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out
        assert "BUY_AND_HOLD" in captured.out
        assert "5.00%" in captured.out

    # ── Edge case: _record_daily_result ───────────────────────────────────

    def test_record_daily_result(self, tmp_path):
        engine = self._make_engine()
        engine.portfolio = self._make_portfolio(tmp_path, cash=5000.0)
        engine.portfolio.buy("SPY", 50.0, 100.0)
        engine._record_daily_result(datetime(2024, 1, 1), {"SPY": 100.0})
        assert len(engine.results) == 1
        assert engine.results[0]["date"] == "2024-01-01"
        assert engine.results[0]["total_value"] > 0
        assert engine.results[0]["num_positions"] == 1

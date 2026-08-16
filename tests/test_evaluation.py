"""
Test suite for evaluation.py.

Tests the comprehensive trading system evaluation module including
portfolio data loading, performance trend calculation, and report generation.
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import pandas as pd

from evaluation import (
    load_portfolio_data,
    load_recent_results,
    calculate_performance_trends,
    generate_comprehensive_report,
    _get_benchmark_return,
)


class TestLoadPortfolioData:
    """Tests for load_portfolio_data function."""

    def test_load_valid_portfolio(self, tmp_path, monkeypatch):
        """Load a valid portfolio state JSON."""
        monkeypatch.chdir(tmp_path)
        portfolio = {
            "cash": 7500.0,
            "total_value": 9500.0,
            "total_realized_pnl": -500.0,
            "positions": {"SPY": {"quantity": 10, "avg_price": 100.0, "current_price": 95.0}},
            "last_updated": "2026-05-01T12:00:00"
        }
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        with open(data_dir / "portfolio_state.json", "w") as f:
            json.dump(portfolio, f)

        result = load_portfolio_data()
        assert result == portfolio
        assert result["total_value"] == 9500.0

    def test_load_missing_file_returns_none(self, tmp_path, monkeypatch):
        """Return None when portfolio state file does not exist."""
        monkeypatch.chdir(tmp_path)
        result = load_portfolio_data()
        assert result is None

    def test_load_malformed_json_raises_error(self, tmp_path, monkeypatch):
        """Malformed JSON should raise json.JSONDecodeError."""
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        with open(data_dir / "portfolio_state.json", "w") as f:
            f.write("not valid json")

        with pytest.raises(json.JSONDecodeError):
            load_portfolio_data()


class TestLoadRecentResults:
    """Tests for load_recent_results function."""

    def test_load_recent_results_basic(self, tmp_path, monkeypatch):
        """Load recent daily result files."""
        monkeypatch.chdir(tmp_path)
        results_dir = tmp_path / "results" / "daily"
        results_dir.mkdir(parents=True)

        for i in range(5):
            date_str = f"2026-01-{i+1:02d}"
            with open(results_dir / f"{date_str}.json", "w") as f:
                json.dump({"date": date_str, "portfolio_after": {"total_value": 10000 + i * 100}}, f)

        results = load_recent_results(days=3)
        assert len(results) == 3
        # Should be sorted, so last 3
        assert results[0]["date"] == "2026-01-03"
        assert results[-1]["date"] == "2026-01-05"

    def test_load_recent_results_empty_dir(self, tmp_path, monkeypatch):
        """Return empty list when no results exist."""
        monkeypatch.chdir(tmp_path)
        results = load_recent_results(days=30)
        assert results == []

    def test_load_recent_results_missing_dir(self, tmp_path, monkeypatch):
        """Return empty list when results directory does not exist."""
        monkeypatch.chdir(tmp_path)
        results = load_recent_results(days=30)
        assert results == []

    def test_load_recent_results_skips_malformed(self, tmp_path, monkeypatch):
        """Skip malformed JSON files silently."""
        monkeypatch.chdir(tmp_path)
        results_dir = tmp_path / "results" / "daily"
        results_dir.mkdir(parents=True)

        with open(results_dir / "2026-01-01.json", "w") as f:
            json.dump({"date": "2026-01-01"}, f)
        with open(results_dir / "2026-01-02.json", "w") as f:
            f.write("bad json")
        with open(results_dir / "2026-01-03.json", "w") as f:
            json.dump({"date": "2026-01-03"}, f)

        results = load_recent_results(days=30)
        assert len(results) == 2
        dates = [r["date"] for r in results]
        assert "2026-01-01" in dates
        assert "2026-01-03" in dates

    def test_load_recent_results_respects_days_limit(self, tmp_path, monkeypatch):
        """Respect the days parameter limit."""
        monkeypatch.chdir(tmp_path)
        results_dir = tmp_path / "results" / "daily"
        results_dir.mkdir(parents=True)

        for i in range(10):
            date_str = f"2026-01-{i+1:02d}"
            with open(results_dir / f"{date_str}.json", "w") as f:
                json.dump({"date": date_str}, f)

        results = load_recent_results(days=5)
        assert len(results) == 5

    def test_load_recent_results_sorting(self, tmp_path, monkeypatch):
        """Results should be sorted chronologically."""
        monkeypatch.chdir(tmp_path)
        results_dir = tmp_path / "results" / "daily"
        results_dir.mkdir(parents=True)

        # Create out of order
        for date_str in ["2026-01-05", "2026-01-01", "2026-01-03"]:
            with open(results_dir / f"{date_str}.json", "w") as f:
                json.dump({"date": date_str}, f)

        results = load_recent_results(days=30)
        dates = [r["date"] for r in results]
        assert dates == ["2026-01-01", "2026-01-03", "2026-01-05"]


class TestCalculatePerformanceTrends:
    """Tests for calculate_performance_trends function."""

    def test_empty_results(self):
        """Empty results should return empty dict."""
        trends = calculate_performance_trends([])
        assert trends == {}

    def test_single_result(self):
        """Single result has portfolio values but no daily returns."""
        results = [
            {"portfolio_after": {"total_value": 10000, "cash": 5000, "num_positions": 2}}
        ]
        trends = calculate_performance_trends(results)
        assert trends["portfolio_values"] == [10000]
        assert trends["cash_levels"] == [5000]
        assert trends["position_counts"] == [2]
        assert trends["daily_returns"] == []

    def test_multiple_results_daily_returns(self):
        """Multiple results should compute daily returns correctly."""
        results = [
            {"portfolio_after": {"total_value": 10000, "cash": 5000, "num_positions": 2}},
            {"portfolio_after": {"total_value": 10100, "cash": 4900, "num_positions": 3}},
            {"portfolio_after": {"total_value": 9900, "cash": 4800, "num_positions": 3}},
        ]
        trends = calculate_performance_trends(results)
        assert trends["portfolio_values"] == [10000, 10100, 9900]
        assert len(trends["daily_returns"]) == 2
        assert pytest.approx(trends["daily_returns"][0], rel=1e-6) == 0.01
        assert pytest.approx(trends["daily_returns"][1], rel=1e-6) == -0.01980198

    def test_missing_portfolio_after_key(self):
        """Results without portfolio_after should be skipped gracefully."""
        results = [
            {"portfolio_after": {"total_value": 10000, "cash": 5000, "num_positions": 2}},
            {"date": "2026-01-02"},  # missing portfolio_after
            {"portfolio_after": {"total_value": 10200, "cash": 4800, "num_positions": 3}},
        ]
        trends = calculate_performance_trends(results)
        assert trends["portfolio_values"] == [10000, 10200]
        assert len(trends["daily_returns"]) == 1
        assert pytest.approx(trends["daily_returns"][0], rel=1e-6) == 0.02

    def test_zero_previous_value_no_division_error(self):
        """Zero previous value should not cause division by zero."""
        results = [
            {"portfolio_after": {"total_value": 0, "cash": 0, "num_positions": 0}},
            {"portfolio_after": {"total_value": 100, "cash": 50, "num_positions": 1}},
        ]
        trends = calculate_performance_trends(results)
        assert trends["portfolio_values"] == [0, 100]
        # Zero previous value should skip return calculation
        assert trends["daily_returns"] == []

    def test_negative_portfolio_value(self):
        """Negative portfolio values are handled (edge case)."""
        results = [
            {"portfolio_after": {"total_value": 1000, "cash": 500, "num_positions": 1}},
            {"portfolio_after": {"total_value": -500, "cash": 0, "num_positions": 0}},
        ]
        trends = calculate_performance_trends(results)
        assert len(trends["daily_returns"]) == 1
        assert pytest.approx(trends["daily_returns"][0], rel=1e-6) == -1.5


class TestGenerateComprehensiveReport:
    """Integration tests for generate_comprehensive_report."""

    def test_report_with_no_data(self, tmp_path, monkeypatch, capsys):
        """Report should handle missing data gracefully."""
        monkeypatch.chdir(tmp_path)
        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "COMPREHENSIVE TRADING SYSTEM EVALUATION" in captured.out
        assert "Evaluation complete" in captured.out

    def test_report_with_portfolio(self, tmp_path, monkeypatch, capsys):
        """Report should display portfolio status when available."""
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        portfolio = {
            "cash": 7500.0,
            "total_value": 9500.0,
            "total_realized_pnl": -500.0,
            "positions": {}
        }
        with open(data_dir / "portfolio_state.json", "w") as f:
            json.dump(portfolio, f)

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "PORTFOLIO STATUS" in captured.out
        assert "Total Value: €9,500.00" in captured.out
        assert "Cash: €7,500.00" in captured.out

    def test_report_with_trends(self, tmp_path, monkeypatch, capsys):
        """Report should display performance trends when data exists."""
        monkeypatch.chdir(tmp_path)
        results_dir = tmp_path / "results" / "daily"
        results_dir.mkdir(parents=True)

        for i in range(5):
            date_str = f"2026-01-{i+1:02d}"
            with open(results_dir / f"{date_str}.json", "w") as f:
                json.dump({
                    "date": date_str,
                    "portfolio_after": {
                        "total_value": 10000 + i * 100,
                        "cash": 5000,
                        "num_positions": 2
                    }
                }, f)

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "PERFORMANCE TRENDS" in captured.out
        assert "Period Return:" in captured.out

    @patch("evaluation.DecisionAnalyzer")
    def test_report_with_decisions(self, mock_analyzer_class, tmp_path, monkeypatch, capsys):
        """Report should display LLM decision quality when analyzer has data."""
        monkeypatch.chdir(tmp_path)

        mock_analyzer = Mock()
        mock_analyzer.load_decisions.return_value = [
            {"date": "2026-01-01", "trades": [{"ticker": "SPY"}]},
            {"date": "2026-01-02", "trades": []},
        ]
        mock_analyzer.analyze_outcomes.return_value = {
            "win_rate": 0.5,
            "buy_accuracy": 0.6,
            "sell_accuracy": 0.4,
        }
        mock_analyzer_class.return_value = mock_analyzer

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "LLM DECISION QUALITY" in captured.out
        assert "Win Rate: 50.0%" in captured.out
        assert "Buy Accuracy: 60.0%" in captured.out
        assert "Sell Accuracy: 40.0%" in captured.out

    @patch("evaluation.DecisionAnalyzer")
    def test_report_no_decisions(self, mock_analyzer_class, tmp_path, monkeypatch, capsys):
        """Report should handle no decision data gracefully."""
        monkeypatch.chdir(tmp_path)

        mock_analyzer = Mock()
        mock_analyzer.load_decisions.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "LLM DECISION QUALITY" in captured.out
        assert "No decision data available yet" in captured.out

    @patch("data.fetch_market_data.fetch_current_prices")
    def test_data_feed_operational(self, mock_fetch, tmp_path, monkeypatch, capsys):
        """Report should show data feed as operational when prices fetch succeeds."""
        monkeypatch.chdir(tmp_path)
        mock_fetch.return_value = {"SPY": 450.0}

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "Data feed: Operational" in captured.out

    @patch("data.fetch_market_data.fetch_current_prices")
    def test_data_feed_error(self, mock_fetch, tmp_path, monkeypatch, capsys):
        """Report should show data feed error when fetch fails."""
        monkeypatch.chdir(tmp_path)
        mock_fetch.side_effect = Exception("API timeout")

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "Data feed: Error" in captured.out

    def test_report_saves_to_file(self, tmp_path, monkeypatch):
        """main() should save a copy to results/analysis directory."""
        monkeypatch.chdir(tmp_path)
        from evaluation import main
        main()

        analysis_dir = tmp_path / "results" / "analysis"
        assert analysis_dir.exists()
        files = list(analysis_dir.glob("comprehensive_evaluation_*.txt"))
        assert len(files) == 1
        content = files[0].read_text()
        assert "COMPREHENSIVE TRADING SYSTEM EVALUATION" in content

    def test_report_with_risk_metrics(self, tmp_path, monkeypatch, capsys):
        """Report should display risk metrics when returns are available."""
        monkeypatch.chdir(tmp_path)
        results_dir = tmp_path / "results" / "daily"
        results_dir.mkdir(parents=True)

        values = [10000, 10200, 10100, 10400, 10300]
        for i, val in enumerate(values):
            date_str = f"2026-01-{i+1:02d}"
            with open(results_dir / f"{date_str}.json", "w") as f:
                json.dump({
                    "date": date_str,
                    "portfolio_after": {
                        "total_value": val,
                        "cash": 5000,
                        "num_positions": 2
                    }
                }, f)

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "RISK ASSESSMENT" in captured.out
        assert "VaR (95%)" in captured.out
        assert "CVaR (95%)" in captured.out

    def test_report_total_return_calculation(self, tmp_path, monkeypatch, capsys):
        """Report should calculate total return vs initial 10,000 EUR."""
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        portfolio = {
            "cash": 5000.0,
            "total_value": 9500.0,
            "total_realized_pnl": -500.0,
            "positions": {}
        }
        with open(data_dir / "portfolio_state.json", "w") as f:
            json.dump(portfolio, f)

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "Total Return:" in captured.out
        assert "-5.00%" in captured.out

    def test_report_system_health_files_present(self, tmp_path, monkeypatch, capsys):
        """Report should confirm core files are present."""
        monkeypatch.chdir(tmp_path)
        # Create required files
        (tmp_path / "src" / "data").mkdir(parents=True)
        (tmp_path / "src" / "portfolio").mkdir(parents=True)
        (tmp_path / "src" / "llm").mkdir(parents=True)
        (tmp_path / "config").mkdir(parents=True)

        open(tmp_path / "src" / "data" / "fetch_market_data.py", "w").close()
        open(tmp_path / "src" / "portfolio" / "portfolio.py", "w").close()
        open(tmp_path / "src" / "llm" / "trading_agent.py", "w").close()
        open(tmp_path / "config" / "universe.json", "w").close()

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "Core files: Present" in captured.out

    def test_report_system_health_missing_files(self, tmp_path, monkeypatch, capsys):
        """Report should warn about missing core files."""
        monkeypatch.chdir(tmp_path)
        # Don't create any required files

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "Missing files:" in captured.out

class TestNonFiniteGuards:
    """Regression tests for non-finite and zero portfolio values."""

    def test_zero_total_value_does_not_raise(self, tmp_path, monkeypatch, capsys):
        """Zero total value should not trigger a ZeroDivisionError."""
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        portfolio = {
            "cash": 0.0,
            "total_value": 0.0,
            "total_realized_pnl": 0.0,
            "positions": {},
        }
        with open(data_dir / "portfolio_state.json", "w") as f:
            json.dump(portfolio, f)

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "Cash: €0.00 (—)" in captured.out
        assert "Total Return: -100.00%" in captured.out

    def test_nan_total_value_shows_na(self, tmp_path, monkeypatch, capsys):
        """Non-finite total value should not produce NaN output."""
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        portfolio = {
            "cash": 1000.0,
            "total_value": float("nan"),
            "total_realized_pnl": 0.0,
            "positions": {},
        }
        with open(data_dir / "portfolio_state.json", "w") as f:
            json.dump(portfolio, f)

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "Cash: €1,000.00 (—)" in captured.out
        assert "Total Return: n/a" in captured.out
        assert "nan%" not in captured.out

    def test_nan_daily_returns_skip_risk_metrics(self, tmp_path, monkeypatch, capsys):
        """Non-finite daily returns should skip risk/volatility metrics."""
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        portfolio = {
            "cash": 5000.0,
            "total_value": 10000.0,
            "total_realized_pnl": 0.0,
            "positions": {},
        }
        with open(data_dir / "portfolio_state.json", "w") as f:
            json.dump(portfolio, f)

        results_dir = tmp_path / "results" / "daily"
        results_dir.mkdir(parents=True)
        values = [10000, float("nan"), 10100]
        for i, val in enumerate(values):
            date_str = f"2026-01-{i+1:02d}"
            with open(results_dir / f"{date_str}.json", "w") as f:
                json.dump({
                    "date": date_str,
                    "portfolio_after": {
                        "total_value": val,
                        "cash": 5000,
                        "num_positions": 0,
                    }
                }, f)

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "PERFORMANCE TRENDS" in captured.out
        assert "Period Return:" in captured.out
        assert "VaR (95%)" not in captured.out
        assert "CVaR (95%)" not in captured.out
        assert "nan%" not in captured.out

    def test_infinite_total_return_skips_benchmark_alpha(self, tmp_path, monkeypatch, capsys):
        """If total return is non-finite, the benchmark comparison should be skipped."""
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        portfolio = {
            "cash": 0.0,
            "total_value": float("inf"),
            "total_realized_pnl": 0.0,
            "positions": {},
        }
        with open(data_dir / "portfolio_state.json", "w") as f:
            json.dump(portfolio, f)

        results_dir = tmp_path / "results" / "daily"
        results_dir.mkdir(parents=True)
        with open(results_dir / "2026-01-01.json", "w") as f:
            json.dump({"date": "2026-01-01"}, f)

        monkeypatch.setattr(
            "evaluation.fetch_historical_data",
            lambda *a, **k: {"SPY": pd.DataFrame({"Close": [100.0, 105.0]})},
        )

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "Total Return: n/a" in captured.out
        assert "vs Buy & Hold" not in captured.out

    def test_first_portfolio_value_zero_skips_period_return(self, tmp_path, monkeypatch, capsys):
        """If the first portfolio value is zero, the period return block should be skipped."""
        monkeypatch.chdir(tmp_path)
        results_dir = tmp_path / "results" / "daily"
        results_dir.mkdir(parents=True)
        values = [0, 100, 200]
        for i, val in enumerate(values):
            date_str = f"2026-01-{i+1:02d}"
            with open(results_dir / f"{date_str}.json", "w") as f:
                json.dump({
                    "date": date_str,
                    "portfolio_after": {
                        "total_value": val,
                        "cash": 0,
                        "num_positions": 0,
                    }
                }, f)

        generate_comprehensive_report()
        captured = capsys.readouterr()
        assert "PERFORMANCE TRENDS" in captured.out
        assert "Period Return:" not in captured.out
        assert "Highest Value:" not in captured.out


class TestEvaluationConsistency:
    def test_trade_counts_use_analyzed_outcomes(self, capsys):
        """Ensure trade counts match the analyzable trades from DecisionAnalyzer."""
        fake_decisions = [
            {
                "date": f"2026-07-{i:02d}",
                "actions": [],
                "trades": [],
                "reasoning": "",
                "portfolio_before": {},
                "portfolio_after": {},
            }
            for i in range(1, 11)
        ]
        fake_outcomes = {
            "buy_count": 1,
            "sell_count": 0,
            "win_rate": 0.0,
            "buy_accuracy": 0.0,
            "sell_accuracy": 0.0,
        }

        with patch("evaluation.load_portfolio_data", return_value=None), \
             patch("evaluation.load_recent_results", return_value=[]), \
             patch("data.fetch_market_data.fetch_current_prices", return_value={"SPY": 750.0}), \
             patch("evaluation.DecisionAnalyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.load_decisions.return_value = fake_decisions
            instance.analyze_outcomes.return_value = fake_outcomes

            generate_comprehensive_report()

            captured = capsys.readouterr()
            assert "Total Trades: 1" in captured.out
            assert "Avg Trades/Day: 0.1" in captured.out
            assert "Win Rate: 0.0%" in captured.out


class TestBenchmarkReturn:
    """Tests for the SPY buy-and-hold benchmark helper."""

    def test_get_benchmark_return_computes_spy_return(self, monkeypatch):
        """_get_benchmark_return should compute (end/start - 1) for SPY closes."""
        fake_df = pd.DataFrame({"Close": [100.0, 105.0]})
        fake_data = {"SPY": fake_df}

        def fake_fetch(tickers, start, end):
            assert tickers == ["SPY"]
            return fake_data

        monkeypatch.setattr("evaluation.fetch_historical_data", fake_fetch)
        result = _get_benchmark_return("2026-01-01", "2026-01-02")
        assert result == pytest.approx(0.05)

    def test_get_benchmark_return_none_when_data_empty(self, monkeypatch):
        """_get_benchmark_return should return None when the fetched data is empty."""
        monkeypatch.setattr(
            "evaluation.fetch_historical_data",
            lambda *a, **k: {"SPY": pd.DataFrame()},
        )
        assert _get_benchmark_return("2026-01-01", "2026-01-02") is None


class TestBenchmarkSummaryConsistency:
    """Tests that the benchmark comparison uses the same horizon as the portfolio."""

    def test_summary_uses_full_history_for_benchmark(self, monkeypatch, capsys):
        """The benchmark comparison must use the earliest valid result, not the 30-day window."""
        # Portfolio state: down 5% since the 10 000 EUR inception
        monkeypatch.setattr(
            "evaluation.load_portfolio_data",
            lambda: {
                "total_value": 9500.0,
                "cash": 2500.0,
                "total_realized_pnl": -200.0,
                "positions": {"SPY": {"quantity": 1, "avg_price": 100, "current_price": 95}},
            },
        )

        # 30-day window starts at 2026-07-01
        recent_results = [{"date": "2026-07-01"}, {"date": "2026-07-28"}]
        monkeypatch.setattr(
            "evaluation.load_valid_daily_results_limited",
            lambda *a, **k: recent_results,
        )

        # Full history starts at 2026-02-17
        all_results = [{"date": "2026-02-17"}, {"date": "2026-07-28"}]
        monkeypatch.setattr(
            "evaluation.load_valid_daily_results",
            lambda *a, **k: all_results,
        )

        # SPY buy-and-hold return over the full period: +10%
        def fake_fetch(tickers, start, end):
            assert start == "2026-02-17"
            assert end == "2026-07-28"
            return {"SPY": pd.DataFrame({"Close": [100.0, 110.0]})}

        monkeypatch.setattr("evaluation.fetch_historical_data", fake_fetch)
        # Mock fetch_current_prices used locally inside generate_comprehensive_report
        monkeypatch.setattr(
            "data.fetch_market_data.fetch_current_prices",
            lambda *a, **k: {"SPY": 95.0},
        )

        # Mock DecisionAnalyzer to avoid loading real decision history
        mock_analyzer = MagicMock()
        mock_analyzer.load_decisions.return_value = []
        mock_analyzer.analyze_outcomes.return_value = {}
        monkeypatch.setattr("evaluation.DecisionAnalyzer", lambda: mock_analyzer)

        generate_comprehensive_report()
        captured = capsys.readouterr()

        assert "Total Return: -5.00%" in captured.out
        assert "vs Buy & Hold (SPY) since 2026-02-17: -15.00%" in captured.out
        # Make sure we do NOT compare against the 30-day window start
        assert "since 2026-07-01" not in captured.out

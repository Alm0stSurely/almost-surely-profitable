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

from evaluation import (
    load_portfolio_data,
    load_recent_results,
    calculate_performance_trends,
    generate_comprehensive_report,
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

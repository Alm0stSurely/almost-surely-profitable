"""
Test suite for backtest/visualize.py.

Tests visualization helper functions including JSON loading,
summary table formatting, and matplotlib availability checks.
Plotting functions are mocked to avoid image comparison dependencies.
"""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from backtest.visualize import (
    check_matplotlib,
    load_backtest_results,
    print_summary_table,
)


class TestCheckMatplotlib:
    """Tests for matplotlib availability check."""

    def test_matplotlib_available(self):
        """Should not raise when matplotlib is installed."""
        # In test environment, matplotlib is typically available
        # If not, this will raise ImportError which is valid behavior
        try:
            check_matplotlib()
        except ImportError:
            pytest.skip("matplotlib not installed in test environment")

    @patch("backtest.visualize.HAS_MATPLOTLIB", False)
    def test_matplotlib_not_available(self):
        """Should raise ImportError when matplotlib is unavailable."""
        with pytest.raises(ImportError, match="matplotlib required"):
            check_matplotlib()


class TestLoadBacktestResults:
    """Tests for loading backtest results from JSON."""

    @pytest.fixture
    def mock_backtest_file(self):
        """Create a temporary JSON file with mock backtest results."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {
                "buy_and_hold": {
                    "total_return": 0.15,
                    "annualized_return": 0.18,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -0.10,
                    "num_trades": 1,
                    "win_rate": 1.0,
                    "daily_results": [
                        {"date": "2026-01-01", "total_value": 10000},
                        {"date": "2026-01-02", "total_value": 10100}
                    ],
                    "drawdown_curve": [0.0, -0.01]
                },
                "llm": {
                    "total_return": 0.20,
                    "annualized_return": 0.24,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": -0.08,
                    "num_trades": 10,
                    "win_rate": 0.6,
                    "daily_results": [
                        {"date": "2026-01-01", "total_value": 10000},
                        {"date": "2026-01-02", "total_value": 10200}
                    ],
                    "drawdown_curve": [0.0, -0.005]
                }
            }
            json.dump(data, f)
            f.flush()
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    def test_load_valid_file(self, mock_backtest_file):
        """Should load and parse a valid backtest JSON file."""
        result = load_backtest_results(mock_backtest_file)
        assert "buy_and_hold" in result
        assert "llm" in result
        assert result["buy_and_hold"]["total_return"] == 0.15
        assert result["llm"]["sharpe_ratio"] == 1.5

    def test_load_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_backtest_results("/nonexistent/path/results.json")

    def test_load_invalid_json(self):
        """Should raise JSONDecodeError for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            path = f.name
        try:
            with pytest.raises(json.JSONDecodeError):
                load_backtest_results(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_empty_json(self):
        """Should load empty dict for empty JSON object."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            path = f.name
        try:
            result = load_backtest_results(path)
            assert result == {}
        finally:
            Path(path).unlink(missing_ok=True)


class TestPrintSummaryTable:
    """Tests for print_summary_table output formatting."""

    @pytest.fixture
    def mock_results(self):
        """Mock backtest results with two strategies."""
        return {
            "buy_and_hold": {
                "total_return": 0.15,
                "annualized_return": 0.18,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.10,
                "num_trades": 1,
                "win_rate": 1.0,
                "daily_results": []
            },
            "llm": {
                "total_return": 0.20,
                "annualized_return": 0.24,
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.08,
                "num_trades": 10,
                "win_rate": 0.6,
                "daily_results": []
            },
            "empty_strategy": None
        }

    def test_prints_header(self, mock_results, capsys):
        """Should print a formatted header row."""
        print_summary_table(mock_results)
        captured = capsys.readouterr()
        assert "Strategy" in captured.out
        assert "Return" in captured.out
        assert "Sharpe" in captured.out
        assert "Max DD" in captured.out

    def test_prints_strategy_names(self, mock_results, capsys):
        """Should print strategy names in title case."""
        print_summary_table(mock_results)
        captured = capsys.readouterr()
        assert "Buy And Hold" in captured.out
        assert "Llm" in captured.out

    def test_prints_numeric_values(self, mock_results, capsys):
        """Should print formatted numeric values."""
        print_summary_table(mock_results)
        captured = capsys.readouterr()
        assert "15.00%" in captured.out
        assert "20.00%" in captured.out
        assert "1.20" in captured.out
        assert "1.50" in captured.out
        assert "-10.00%" in captured.out
        assert "-8.00%" in captured.out

    def test_skips_none_results(self, mock_results, capsys):
        """Should skip strategies with None results."""
        print_summary_table(mock_results)
        captured = capsys.readouterr()
        # empty_strategy is None, so it should not appear
        lines = [l for l in captured.out.split("\n") if "Empty Strategy" in l]
        assert len(lines) == 0

    def test_empty_dict(self, capsys):
        """Should handle empty results dict gracefully."""
        print_summary_table({})
        captured = capsys.readouterr()
        # Should print header but no data rows
        assert "Strategy" in captured.out
        assert captured.out.count("\n") >= 2


class TestPlottingFunctionsMocked:
    """Tests for plotting functions with mocked matplotlib."""

    @pytest.fixture
    def mock_results(self):
        """Mock backtest results for plotting tests."""
        return {
            "buy_and_hold": {
                "total_return": 0.15,
                "annualized_return": 0.18,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.10,
                "num_trades": 1,
                "win_rate": 1.0,
                "daily_results": [
                    {"date": "2026-01-01", "total_value": 10000},
                    {"date": "2026-01-02", "total_value": 10100},
                    {"date": "2026-01-03", "total_value": 10050}
                ],
                "drawdown_curve": [0.0, -0.01, -0.015]
            },
            "llm": {
                "total_return": 0.20,
                "annualized_return": 0.24,
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.08,
                "num_trades": 10,
                "win_rate": 0.6,
                "daily_results": [
                    {"date": "2026-01-01", "total_value": 10000},
                    {"date": "2026-01-02", "total_value": 10200},
                    {"date": "2026-01-03", "total_value": 10150}
                ],
                "drawdown_curve": [0.0, -0.005, -0.01]
            }
        }

    @patch("backtest.visualize.plt")
    @patch("backtest.visualize.HAS_MATPLOTLIB", True)
    def test_plot_equity_curves(self, mock_plt, mock_results, tmp_path):
        """plot_equity_curves should call matplotlib plotting methods."""
        from backtest.visualize import plot_equity_curves
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        output = str(tmp_path / "equity.png")
        plot_equity_curves(mock_results, output)
        mock_plt.subplots.assert_called_once()
        mock_plt.savefig.assert_called_once()

    @patch("backtest.visualize.plt")
    @patch("backtest.visualize.HAS_MATPLOTLIB", True)
    def test_plot_metrics_comparison(self, mock_plt, mock_results, tmp_path):
        """plot_metrics_comparison should call matplotlib plotting methods."""
        from backtest.visualize import plot_metrics_comparison
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        output = str(tmp_path / "metrics.png")
        plot_metrics_comparison(mock_results, output)
        mock_plt.subplots.assert_called_once()
        mock_plt.savefig.assert_called_once()

    @patch("backtest.visualize.plt")
    @patch("backtest.visualize.HAS_MATPLOTLIB", True)
    def test_plot_backtest_results(self, mock_plt, tmp_path):
        """plot_backtest_results should call matplotlib plotting methods."""
        from backtest.visualize import plot_backtest_results
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        result = {
            "strategy": "llm",
            "start_date": "2026-01-01",
            "end_date": "2026-01-03",
            "initial_capital": 10000,
            "daily_results": [
                {"date": "2026-01-01", "total_value": 10000},
                {"date": "2026-01-02", "total_value": 10200},
                {"date": "2026-01-03", "total_value": 10150}
            ]
        }
        output = str(tmp_path / "backtest.png")
        plot_backtest_results(result, output)
        mock_plt.subplots.assert_called_once()
        mock_plt.savefig.assert_called_once()

    @patch("backtest.visualize.plt")
    @patch("backtest.visualize.HAS_MATPLOTLIB", True)
    def test_plot_backtest_results_empty(self, mock_plt, tmp_path, capsys):
        """plot_backtest_results with empty daily_results should print message."""
        from backtest.visualize import plot_backtest_results
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        result = {
            "strategy": "llm",
            "start_date": "2026-01-01",
            "end_date": "2026-01-01",
            "initial_capital": 10000,
            "daily_results": []
        }
        output = str(tmp_path / "backtest.png")
        plot_backtest_results(result, output)
        captured = capsys.readouterr()
        assert "No daily results to plot" in captured.out
        # Should return early without calling savefig
        mock_plt.savefig.assert_not_called()

    @patch("backtest.visualize.plt")
    @patch("backtest.visualize.HAS_MATPLOTLIB", True)
    def test_plot_equity_curves_skips_empty_strategies(self, mock_plt, tmp_path):
        """Should skip strategies with empty or missing daily_results."""
        from backtest.visualize import plot_equity_curves
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        results = {
            "valid": {
                "daily_results": [
                    {"date": "2026-01-01", "total_value": 10000}
                ],
                "drawdown_curve": [0.0]
            },
            "empty": {
                "daily_results": [],
                "drawdown_curve": []
            },
            "missing": {}
        }
        output = str(tmp_path / "equity.png")
        plot_equity_curves(results, output)
        # Should complete without error despite empty/missing entries
        mock_plt.savefig.assert_called_once()

    @patch("backtest.visualize.HAS_MATPLOTLIB", False)
    def test_plot_equity_curves_no_matplotlib(self, tmp_path):
        """Should raise ImportError when matplotlib is unavailable."""
        from backtest.visualize import plot_equity_curves
        with pytest.raises(ImportError, match="matplotlib required"):
            plot_equity_curves({}, str(tmp_path / "equity.png"))

    @patch("backtest.visualize.HAS_MATPLOTLIB", False)
    def test_plot_metrics_comparison_no_matplotlib(self, tmp_path):
        """Should raise ImportError when matplotlib is unavailable."""
        from backtest.visualize import plot_metrics_comparison
        with pytest.raises(ImportError, match="matplotlib required"):
            plot_metrics_comparison({}, str(tmp_path / "metrics.png"))

    @patch("backtest.visualize.HAS_MATPLOTLIB", False)
    def test_plot_backtest_results_no_matplotlib(self, tmp_path):
        """Should raise ImportError when matplotlib is unavailable."""
        from backtest.visualize import plot_backtest_results
        with pytest.raises(ImportError, match="matplotlib required"):
            plot_backtest_results({}, str(tmp_path / "backtest.png"))

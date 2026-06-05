"""
Test suite for weekly_report.py.

Tests the weekly report generator including return calculations,
benchmark fetching, and report generation logic.
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np

from weekly_report import calculate_weekly_returns, fetch_benchmark_returns


class TestCalculateWeeklyReturns:
    """Tests for calculate_weekly_returns function."""

    def test_empty_list(self):
        """Empty list should return empty array."""
        result = calculate_weekly_returns([])
        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_single_day(self):
        """Single day has no previous day to compare."""
        week_results = [
            {"portfolio_after": {"total_value": 10000.0}}
        ]
        result = calculate_weekly_returns(week_results)
        assert len(result) == 0

    def test_two_days_positive_return(self):
        """Two days with positive return."""
        week_results = [
            {"portfolio_after": {"total_value": 10000.0}},
            {"portfolio_after": {"total_value": 10100.0}}
        ]
        result = calculate_weekly_returns(week_results)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.01, abs=1e-6)

    def test_two_days_negative_return(self):
        """Two days with negative return."""
        week_results = [
            {"portfolio_after": {"total_value": 10000.0}},
            {"portfolio_after": {"total_value": 9900.0}}
        ]
        result = calculate_weekly_returns(week_results)
        assert len(result) == 1
        assert result[0] == pytest.approx(-0.01, abs=1e-6)

    def test_five_days_mixed(self):
        """Five trading days with mixed returns."""
        week_results = [
            {"portfolio_after": {"total_value": 10000.0}},
            {"portfolio_after": {"total_value": 10100.0}},
            {"portfolio_after": {"total_value": 10050.0}},
            {"portfolio_after": {"total_value": 10200.0}},
            {"portfolio_after": {"total_value": 10150.0}}
        ]
        result = calculate_weekly_returns(week_results)
        assert len(result) == 4
        expected = [0.01, -0.004950495, 0.014925373, -0.004901961]
        for i, exp in enumerate(expected):
            assert result[i] == pytest.approx(exp, abs=1e-6)

    def test_zero_previous_value(self):
        """Zero previous value should be skipped to avoid division by zero."""
        week_results = [
            {"portfolio_after": {"total_value": 0.0}},
            {"portfolio_after": {"total_value": 10000.0}}
        ]
        result = calculate_weekly_returns(week_results)
        assert len(result) == 0

    def test_missing_portfolio_after(self):
        """Missing portfolio_after key should be handled gracefully."""
        week_results = [
            {"date": "2026-01-01"},
            {"portfolio_after": {"total_value": 10000.0}}
        ]
        result = calculate_weekly_returns(week_results)
        assert len(result) == 0

    def test_missing_total_value(self):
        """Missing total_value should be treated as 0."""
        week_results = [
            {"portfolio_after": {"positions": []}},
            {"portfolio_after": {"total_value": 10000.0}}
        ]
        result = calculate_weekly_returns(week_results)
        assert len(result) == 0


class TestFetchBenchmarkReturns:
    """Tests for fetch_benchmark_returns function."""

    @patch("weekly_report.fetch_historical_data")
    def test_successful_fetch(self, mock_fetch):
        """Successful benchmark data fetch."""
        mock_fetch.return_value = {
            "SPY": {
                "history": {
                    "close": [400.0, 402.0, 401.0, 405.0]
                }
            }
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10", benchmark="SPY")
        assert result is not None
        assert len(result) == 3
        expected = [0.005, -0.002487562, 0.009975062]
        for i, exp in enumerate(expected):
            assert result[i] == pytest.approx(exp, abs=1e-6)

    @patch("weekly_report.fetch_historical_data")
    def test_insufficient_data(self, mock_fetch):
        """Only one close price — cannot compute returns."""
        mock_fetch.return_value = {
            "SPY": {
                "history": {
                    "close": [400.0]
                }
            }
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10")
        assert result is None

    @patch("weekly_report.fetch_historical_data")
    def test_empty_history(self, mock_fetch):
        """Empty history should return None."""
        mock_fetch.return_value = {
            "SPY": {
                "history": {
                    "close": []
                }
            }
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10")
        assert result is None

    @patch("weekly_report.fetch_historical_data")
    def test_ticker_not_found(self, mock_fetch):
        """Requested ticker not in returned data."""
        mock_fetch.return_value = {
            "QQQ": {
                "history": {
                    "close": [400.0, 405.0]
                }
            }
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10", benchmark="SPY")
        assert result is None

    @patch("weekly_report.fetch_historical_data")
    def test_missing_history_key(self, mock_fetch):
        """Data exists but 'history' key is missing."""
        mock_fetch.return_value = {
            "SPY": {
                "prices": [400.0, 405.0]
            }
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10")
        assert result is None

    @patch("weekly_report.fetch_historical_data")
    def test_fetch_exception(self, mock_fetch):
        """Exception during fetch should return None, not crash."""
        mock_fetch.side_effect = Exception("Network error")
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10")
        assert result is None

    @patch("weekly_report.fetch_historical_data")
    def test_default_benchmark_spy(self, mock_fetch):
        """Default benchmark should be SPY."""
        mock_fetch.return_value = {
            "SPY": {
                "history": {
                    "close": [400.0, 405.0]
                }
            }
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10")
        assert result is not None
        mock_fetch.assert_called_once()
        args = mock_fetch.call_args[0][0]
        assert args == ["SPY"]


class TestWeeklyReportImports:
    """Tests that weekly_report.py imports correctly and has expected structure."""

    def test_module_imports(self):
        """Module should import without errors."""
        import weekly_report
        assert hasattr(weekly_report, "calculate_weekly_returns")
        assert hasattr(weekly_report, "fetch_benchmark_returns")
        assert hasattr(weekly_report, "generate_weekly_report")

    def test_calculate_weekly_returns_is_callable(self):
        """Function should be callable."""
        from weekly_report import calculate_weekly_returns
        assert callable(calculate_weekly_returns)

    def test_fetch_benchmark_returns_is_callable(self):
        """Function should be callable."""
        from weekly_report import fetch_benchmark_returns
        assert callable(fetch_benchmark_returns)

"""
Test suite for weekly_report.py.

Tests the weekly report generator including return calculations,
benchmark fetching, and report generation logic.
"""

import json
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from weekly_report import (
    _safe_benchmark_alpha,
    _safe_pct_str,
    _safe_position_field,
    _safe_positive_scalar,
    _safe_value_str,
    _safe_weekly_return,
    calculate_weekly_returns,
    fetch_benchmark_returns,
    generate_weekly_report,
)


class TestCalculateWeeklyReturns:
    """Tests for calculate_weekly_returns function."""

    def test_empty_list(self):
        """Empty list should return empty dates and an empty array."""
        dates, result = calculate_weekly_returns([])
        assert len(result) == 0
        assert len(dates) == 0
        assert isinstance(result, np.ndarray)

    def test_single_day(self):
        """Single day has no previous day to compare."""
        week_results = [
            {"portfolio_after": {"total_value": 10000.0}}
        ]
        dates, result = calculate_weekly_returns(week_results)
        assert len(result) == 0
        assert len(dates) == 0

    def test_two_days_positive_return(self):
        """Two days with positive return."""
        week_results = [
            {"portfolio_after": {"total_value": 10000.0}},
            {"portfolio_after": {"total_value": 10100.0}}
        ]
        dates, result = calculate_weekly_returns(week_results)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.01, abs=1e-6)
        assert len(dates) == 1

    def test_two_days_negative_return(self):
        """Two days with negative return."""
        week_results = [
            {"portfolio_after": {"total_value": 10000.0}},
            {"portfolio_after": {"total_value": 9900.0}}
        ]
        dates, result = calculate_weekly_returns(week_results)
        assert len(result) == 1
        assert result[0] == pytest.approx(-0.01, abs=1e-6)
        assert len(dates) == 1

    def test_five_days_mixed(self):
        """Five trading days with mixed returns."""
        week_results = [
            {"portfolio_after": {"total_value": 10000.0}},
            {"portfolio_after": {"total_value": 10100.0}},
            {"portfolio_after": {"total_value": 10050.0}},
            {"portfolio_after": {"total_value": 10200.0}},
            {"portfolio_after": {"total_value": 10150.0}}
        ]
        dates, result = calculate_weekly_returns(week_results)
        assert len(result) == 4
        assert len(dates) == 4
        expected = [0.01, -0.004950495, 0.014925373, -0.004901961]
        for i, exp in enumerate(expected):
            assert result[i] == pytest.approx(exp, abs=1e-6)

    def test_zero_previous_value(self):
        """Zero previous value should be skipped to avoid division by zero."""
        week_results = [
            {"portfolio_after": {"total_value": 0.0}},
            {"portfolio_after": {"total_value": 10000.0}}
        ]
        dates, result = calculate_weekly_returns(week_results)
        assert len(result) == 0
        assert len(dates) == 0

    def test_missing_portfolio_after(self):
        """Missing portfolio_after key should be handled gracefully."""
        week_results = [
            {"date": "2026-01-01"},
            {"portfolio_after": {"total_value": 10000.0}}
        ]
        dates, result = calculate_weekly_returns(week_results)
        assert len(result) == 0
        assert len(dates) == 0

    def test_missing_total_value(self):
        """Missing total_value should be treated as 0."""
        week_results = [
            {"portfolio_after": {"positions": []}},
            {"portfolio_after": {"total_value": 10000.0}}
        ]
        dates, result = calculate_weekly_returns(week_results)
        assert len(result) == 0
        assert len(dates) == 0


class TestFetchBenchmarkReturns:
    """Tests for fetch_benchmark_returns function."""

    @patch("weekly_report.fetch_historical_data")
    def test_successful_fetch(self, mock_fetch):
        """Successful benchmark data fetch returns returns and cumulative return."""
        mock_fetch.return_value = {
            "SPY": pd.DataFrame({"Close": [400.0, 402.0, 401.0, 405.0]}),
            "CAC.PA": pd.DataFrame({"Close": [7000.0, 7100.0, 7050.0, 7200.0]})
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10", benchmarks=["SPY"])
        assert result is not None
        assert "SPY" in result
        assert len(result["SPY"]["returns"]) == 3
        expected = [0.005, -0.002487562, 0.009975062]
        for i, exp in enumerate(expected):
            assert result["SPY"]["returns"][i] == pytest.approx(exp, abs=1e-6)
        assert result["SPY"]["cumulative_return"] == pytest.approx(0.0125, abs=1e-6)

    @patch("weekly_report.fetch_historical_data")
    def test_cumulative_return_for_mismatched_calendars(self, mock_fetch):
        """Cumulative comparison works even when markets have different holidays."""
        mock_fetch.return_value = {
            "SPY": pd.DataFrame({"Close": [400.0, 402.0, 401.0, 405.0]}),
            "CAC.PA": pd.DataFrame({"Close": [7000.0, 7100.0, 7050.0, 7150.0, 7200.0]})
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10")
        assert result is not None
        assert "SPY" in result
        assert "CAC.PA" in result
        # Daily returns differ in length but cumulative returns are comparable.
        assert len(result["SPY"]["returns"]) == 3
        assert len(result["CAC.PA"]["returns"]) == 4
        assert result["SPY"]["cumulative_return"] == pytest.approx(0.0125, abs=1e-6)
        assert result["CAC.PA"]["cumulative_return"] == pytest.approx(0.028571, abs=1e-5)

    @patch("weekly_report.fetch_historical_data")
    def test_inclusive_end_date(self, mock_fetch):
        """The end date is inclusive, so the next day is passed to yfinance."""
        mock_fetch.return_value = {
            "SPY": pd.DataFrame({"Close": [400.0, 405.0]})
        }
        fetch_benchmark_returns("2026-01-01", "2026-01-02", benchmarks=["SPY"])
        mock_fetch.assert_called_once()
        _, kwargs = mock_fetch.call_args
        assert kwargs["start"] == "2026-01-01"
        assert kwargs["end"] == "2026-01-03"

    @patch("weekly_report.fetch_historical_data")
    def test_insufficient_data(self, mock_fetch):
        """Only one close price — cannot compute returns."""
        mock_fetch.return_value = {
            "SPY": pd.DataFrame({"Close": [400.0]}),
            "CAC.PA": pd.DataFrame({"Close": [7000.0]})
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10")
        assert result is None

    @patch("weekly_report.fetch_historical_data")
    def test_empty_history(self, mock_fetch):
        """Empty history should return None."""
        mock_fetch.return_value = {
            "SPY": pd.DataFrame({"Close": []}),
            "CAC.PA": pd.DataFrame({"Close": []})
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10")
        assert result is None

    @patch("weekly_report.fetch_historical_data")
    def test_ticker_not_found(self, mock_fetch):
        """Requested ticker not in returned data."""
        mock_fetch.return_value = {
            "QQQ": pd.DataFrame({"Close": [400.0, 405.0]})
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10", benchmarks=["SPY"])
        assert result is None

    @patch("weekly_report.fetch_historical_data")
    def test_missing_history_key(self, mock_fetch):
        """Data exists but 'Close' column is missing."""
        mock_fetch.return_value = {
            "SPY": pd.DataFrame({"Open": [400.0, 405.0]}),
            "CAC.PA": pd.DataFrame({"Open": [7000.0, 7100.0]})
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
    def test_default_benchmarks(self, mock_fetch):
        """Default benchmarks should include SPY and CAC.PA."""
        mock_fetch.return_value = {
            "SPY": pd.DataFrame({"Close": [400.0, 405.0]}),
            "CAC.PA": pd.DataFrame({"Close": [7000.0, 7100.0]})
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10")
        assert result is not None
        mock_fetch.assert_called_once()
        args = mock_fetch.call_args[0][0]
        assert "SPY" in args
        assert "CAC.PA" in args
        assert "SPY" in result
        assert "CAC.PA" in result

    @patch("weekly_report.fetch_historical_data")
    def test_no_runtime_warning(self, mock_fetch):
        """A normal fetch should not emit RuntimeWarnings."""
        mock_fetch.return_value = {
            "SPY": pd.DataFrame({"Close": [400.0, 402.0, 405.0]})
        }
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            result = fetch_benchmark_returns("2026-01-01", "2026-01-10", benchmarks=["SPY"])
        runtime_warnings = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert not runtime_warnings
        assert result is not None


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


class TestNonFiniteGuards:
    """Regression tests for non-finite value handling in weekly_report.py."""

    def test_calculate_weekly_returns_skips_nan_previous_value(self):
        """A NaN previous total_value should not produce a RuntimeWarning."""
        week_results = [
            {"portfolio_after": {"total_value": float("nan")}},
            {"portfolio_after": {"total_value": 10100.0}},
        ]
        dates, returns = calculate_weekly_returns(week_results)
        assert len(returns) == 0
        assert len(dates) == 0

    def test_calculate_weekly_returns_skips_nan_current_value(self):
        """A NaN current total_value breaks adjacent return pairs."""
        week_results = [
            {"portfolio_after": {"total_value": 10000.0}},
            {"portfolio_after": {"total_value": float("nan")}},
            {"portfolio_after": {"total_value": 10200.0}},
        ]
        dates, returns = calculate_weekly_returns(week_results)
        # Both pairs (10000->NaN and NaN->10200) must be discarded.
        assert len(returns) == 0
        assert len(dates) == 0

    def test_calculate_weekly_returns_skips_inf_previous_value(self):
        """An infinite previous total_value should not poison the return array."""
        week_results = [
            {"portfolio_after": {"total_value": float("inf")}},
            {"portfolio_after": {"total_value": 10100.0}},
        ]
        dates, returns = calculate_weekly_returns(week_results)
        assert len(returns) == 0

    def test_calculate_weekly_returns_skips_non_finite_return(self):
        """A computed return that is not finite should be dropped."""
        week_results = [
            {"portfolio_after": {"total_value": 10000.0}},
            {"portfolio_after": {"total_value": float("inf")}},
        ]
        dates, returns = calculate_weekly_returns(week_results)
        assert len(returns) == 0

    @patch("weekly_report.fetch_historical_data")
    def test_fetch_benchmark_returns_drops_nan_closes(self, mock_fetch):
        """NaN close prices should be filtered before return calculation."""
        mock_fetch.return_value = {
            "SPY": pd.DataFrame({"Close": [400.0, float("nan"), 405.0]}),
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10", benchmarks=["SPY"])
        assert result is not None
        assert "SPY" in result
        assert len(result["SPY"]["returns"]) == 1
        assert result["SPY"]["cumulative_return"] == pytest.approx(0.0125, abs=1e-6)

    @patch("weekly_report.fetch_historical_data")
    def test_fetch_benchmark_returns_drops_inf_closes(self, mock_fetch):
        """Infinite close prices should be filtered before return calculation."""
        mock_fetch.return_value = {
            "SPY": pd.DataFrame({"Close": [400.0, float("inf"), 405.0]}),
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10", benchmarks=["SPY"])
        assert result is not None
        assert "SPY" in result
        assert len(result["SPY"]["returns"]) == 1
        assert result["SPY"]["cumulative_return"] == pytest.approx(0.0125, abs=1e-6)

    @patch("weekly_report.fetch_historical_data")
    def test_fetch_benchmark_returns_returns_none_when_all_closes_non_finite(self, mock_fetch):
        """All non-finite closes should make the benchmark unavailable."""
        mock_fetch.return_value = {
            "SPY": pd.DataFrame({"Close": [float("nan"), float("nan")]}),
        }
        result = fetch_benchmark_returns("2026-01-01", "2026-01-10", benchmarks=["SPY"])
        assert result is None

    def test_safe_positive_scalar_rejects_non_finite(self):
        assert _safe_positive_scalar(float("nan")) == 0.0
        assert _safe_positive_scalar(float("inf")) == 0.0
        assert _safe_positive_scalar(-100.0) == 0.0
        assert _safe_positive_scalar(0.0) == 0.0
        assert _safe_positive_scalar(None) == 0.0
        assert _safe_positive_scalar("100") == 0.0

    def test_safe_positive_scalar_accepts_valid_with_default_override(self):
        assert _safe_positive_scalar(100.0, default=None) == 100.0

    def test_safe_weekly_return_with_valid_values(self):
        assert _safe_weekly_return(10000.0, 10100.0) == pytest.approx(1.0, abs=1e-6)

    def test_safe_weekly_return_with_nan_start(self):
        assert _safe_weekly_return(float("nan"), 10100.0) is None

    def test_safe_weekly_return_with_nan_end(self):
        assert _safe_weekly_return(10000.0, float("nan")) is None

    def test_safe_benchmark_alpha_with_valid_inputs(self):
        alpha = _safe_benchmark_alpha(1.0, 0.005)
        assert alpha == pytest.approx(0.005, abs=1e-6)

    def test_safe_benchmark_alpha_with_nan_benchmark(self):
        assert _safe_benchmark_alpha(1.0, float("nan")) is None

    def test_safe_benchmark_alpha_with_none_weekly_return(self):
        assert _safe_benchmark_alpha(None, 0.005) is None

    def test_no_runtime_warning_with_non_finite_inputs(self):
        """Non-finite inputs should not emit RuntimeWarnings under the guards."""
        week_results = [
            {"portfolio_after": {"total_value": float("nan")}},
            {"portfolio_after": {"total_value": 10100.0}},
        ]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            calculate_weekly_returns(week_results)
        runtime_warnings = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert not runtime_warnings


class TestFormatGuards:
    """Tests for the safe formatting helpers added to weekly_report.py."""

    def test_safe_value_str_handles_finite_and_non_finite(self):
        assert _safe_value_str(1234.5) == "€1234.50"
        assert _safe_value_str(1234.5, symbol="$") == "$1234.50"
        assert _safe_value_str(float("nan")) == "n/a"
        assert _safe_value_str(float("inf")) == "n/a"
        assert _safe_value_str(None) == "n/a"
        assert _safe_value_str("not a number") == "n/a"

    def test_safe_pct_str_handles_finite_and_non_finite(self):
        assert _safe_pct_str(5.123) == "+5.12%"
        assert _safe_pct_str(-5.123) == "-5.12%"
        assert _safe_pct_str(5.123, signed=False) == "5.12%"
        assert _safe_pct_str(float("nan")) == "n/a"
        assert _safe_pct_str(float("inf")) == "n/a"
        assert _safe_pct_str(None) == "n/a"

    def test_safe_position_field_handles_finite_and_non_finite(self):
        assert _safe_position_field(12.345) == "12.35"
        assert _safe_position_field(12.345, fmt=".4f") == "12.3450"
        assert _safe_position_field(float("nan")) == "n/a"
        assert _safe_position_field(None) == "n/a"


class TestGenerateWeeklyReportFormatting:
    """Integration tests for generate_weekly_report with mocked dependencies."""

    @staticmethod
    def _make_summary(**overrides):
        summary = {
            "cash": 2418.67,
            "positions_value": 7573.19,
            "total_value": 9991.86,
            "total_return_pct": -0.0814,
            "total_realized_pnl": -150.0,
            "total_unrealized_pnl": -8.14,
            "total_pnl": -158.14,
            "num_positions": 1,
            "positions": [
                {
                    "ticker": "SPY",
                    "quantity": 10.0,
                    "avg_price": 450.0,
                    "current_price": 500.0,
                    "market_value": 5000.0,
                    "unrealized_pnl": 500.0,
                    "unrealized_pnl_pct": 11.11,
                }
            ],
        }
        summary.update(overrides)
        return summary

    @staticmethod
    def _patch_external_calls(tmp_path, monkeypatch, summary):
        """Patch every network/file dependency of generate_weekly_report."""
        fixed = SimpleNamespace(
            now=lambda: datetime(2026, 8, 28, 10, 0, 0),
            strptime=datetime.strptime,
        )
        monkeypatch.setattr("weekly_report.datetime", fixed)

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        daily_dir = results_dir / "daily"
        daily_dir.mkdir()
        reports_dir = results_dir / "reports"

        monkeypatch.setattr("weekly_report.DATA_DIR", data_dir)
        monkeypatch.setattr("weekly_report.RESULTS_DIR", results_dir)
        monkeypatch.setattr("weekly_report.DAILY_RESULTS_DIR", daily_dir)
        monkeypatch.setattr("weekly_report.REPORTS_DIR", reports_dir)

        mock_portfolio = MagicMock()
        mock_portfolio.positions = {}
        mock_portfolio.get_summary.return_value = summary

        mock_rg = MagicMock()
        mock_rg.load_daily_results.return_value = [
            {
                "date": "2026-08-24",
                "portfolio_before": {"total_value": 10000.0},
                "portfolio_after": {"total_value": 10000.0},
                "executed_trades": [],
            },
            {
                "date": "2026-08-25",
                "portfolio_before": {"total_value": 10000.0},
                "portfolio_after": {"total_value": 9991.86},
                "executed_trades": [
                    {
                        "action": "buy",
                        "ticker": "SPY",
                        "price": 500.0,
                        "status": "executed",
                    }
                ],
            },
        ]

        monkeypatch.setattr(
            "weekly_report.ReportGenerator",
            MagicMock(return_value=mock_rg),
        )

        patches = {
            "Portfolio": MagicMock(return_value=mock_portfolio),
            "fetch_current_prices": MagicMock(return_value={}),
            "fetch_historical_data": MagicMock(return_value={}),
            "fetch_benchmark_returns": MagicMock(return_value=None),
            "calculate_all_metrics": MagicMock(return_value=MagicMock(
                sharpe_ratio=1.0,
                sortino_ratio=1.0,
                max_drawdown=-0.05,
                volatility=0.1,
                beta=None,
                alpha=None,
                information_ratio=None,
            )),
            "calculate_portfolio_cvar": MagicMock(return_value=MagicMock(
                cvar_95=-0.02,
                var_95=-0.01,
            )),
            "tail_risk_analysis": MagicMock(return_value={"skewness": 0.0, "kurtosis": 3.0}),
        }
        return patches

    def test_generate_weekly_report_renders_finite_values(self, capsys, tmp_path, monkeypatch):
        summary = self._make_summary()
        patches = self._patch_external_calls(tmp_path, monkeypatch, summary)

        with patch.multiple("weekly_report", **patches):
            generate_weekly_report()

        captured = capsys.readouterr().out
        assert "Cash: €2418.67" in captured
        assert "Total Value: €9991.86" in captured
        assert "Total Return: -0.08%" in captured
        assert "SPY: 10.00 shares @ €500.00" in captured
        assert "Value: €5000.00 | P&L: +11.11%" in captured
        assert "BUY SPY @ €500.00" in captured

    def test_generate_weekly_report_renders_non_finite_values_as_na(self, capsys, tmp_path, monkeypatch):
        summary = self._make_summary(
            cash=float("nan"),
            total_value=float("inf"),
            total_return_pct=float("nan"),
            positions_value=float("-inf"),
            total_realized_pnl=float("nan"),
            total_unrealized_pnl=float("nan"),
            positions=[
                {
                    "ticker": "SPY",
                    "quantity": float("nan"),
                    "avg_price": 450.0,
                    "current_price": float("inf"),
                    "market_value": float("nan"),
                    "unrealized_pnl": float("nan"),
                    "unrealized_pnl_pct": float("nan"),
                }
            ],
        )
        patches = self._patch_external_calls(tmp_path, monkeypatch, summary)
        rg_mock = patches["Portfolio"].return_value.get_summary.return_value = summary
        # Also exercise a non-finite trade price.
        weekly_report = sys.modules["weekly_report"]
        weekly_report.ReportGenerator.return_value.load_daily_results.return_value = [
            {
                "date": "2026-08-25",
                "portfolio_before": {"total_value": 10000.0},
                "portfolio_after": {"total_value": 10000.0},
                "executed_trades": [
                    {
                        "action": "buy",
                        "ticker": "SPY",
                        "price": float("nan"),
                        "status": "executed",
                    }
                ],
            },
        ]

        with patch.multiple("weekly_report", **patches):
            generate_weekly_report()

        captured = capsys.readouterr().out
        # No raw nan/inf tokens should leak into the printed report.
        assert "nan" not in captured.lower()
        assert "inf" not in captured.lower()
        assert "Cash: n/a" in captured
        assert "Total Value: n/a" in captured
        assert "Total Return: n/a" in captured
        assert "SPY: n/a shares @ n/a" in captured
        assert "Value: n/a | P&L: n/a" in captured
        assert "BUY SPY @ n/a" in captured

        # Verify the markdown file also contains safe placeholders.
        report_file = tmp_path / "results" / f"weekly-2026-W35.md"
        assert report_file.exists()
        markdown = report_file.read_text()
        assert "| Cash | n/a |" in markdown
        assert "| **Total Value** | **n/a** |" in markdown
        assert "| Total Return | n/a |" in markdown
        assert "| SPY | n/a | n/a | n/a | n/a | n/a |" in markdown

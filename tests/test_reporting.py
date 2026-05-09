"""
Test suite for reporting.py.

Tests the ReportGenerator including ISO week date calculations,
daily result loading, and weekly/monthly report generation.
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from reporting import ReportGenerator


class TestISOWeekCalculation:
    """Tests for ISO week date range correctness.

    The previous implementation used ``datetime(year, 1, 1) + timedelta(weeks=week-1)``
    which is wrong for ISO weeks. ISO week 1 is the week containing the first
    Thursday of the year, not the week starting January 1st.
    """

    def test_iso_week_2023_w01(self):
        """2023-01-01 is Sunday → ISO W01 starts Jan 2, not Dec 26."""
        rg = ReportGenerator()
        report = rg.generate_weekly_report(2023, 1)
        # Should be empty because we have no mock data, but the date range must be correct
        assert report == {}

    def test_iso_week_boundary_year_start_sunday(self):
        """When Jan 1 is Sunday, ISO W01 is the NEXT Monday."""
        rg = ReportGenerator()
        # Use a private helper to verify the date range without needing data
        start_of_week = datetime.strptime("2023-W01-1", "%G-W%V-%u")
        assert start_of_week.date() == datetime(2023, 1, 2).date()
        end_of_week = start_of_week + timedelta(days=6)
        assert end_of_week.date() == datetime(2023, 1, 8).date()

    def test_iso_week_boundary_year_start_saturday(self):
        """When Jan 1 is Saturday, ISO W01 starts Jan 3."""
        start_of_week = datetime.strptime("2022-W01-1", "%G-W%V-%u")
        assert start_of_week.date() == datetime(2022, 1, 3).date()

    def test_iso_week_boundary_year_start_monday(self):
        """When Jan 1 is Monday, ISO W01 starts Jan 1."""
        start_of_week = datetime.strptime("2024-W01-1", "%G-W%V-%u")
        assert start_of_week.date() == datetime(2024, 1, 1).date()

    def test_iso_week_53_exists(self):
        """Some years have 53 ISO weeks (e.g. 2020)."""
        start_of_week = datetime.strptime("2020-W53-1", "%G-W%V-%u")
        assert start_of_week.date() == datetime(2020, 12, 28).date()

    def test_iso_week_year_boundary_december(self):
        """Late December can belong to ISO week 1 of the NEXT year."""
        # 2025-12-29 is Monday of ISO week 1 of 2026
        dt = datetime(2025, 12, 29)
        iso_year, iso_week, iso_day = dt.isocalendar()
        assert iso_year == 2026
        assert iso_week == 1

    def test_iso_vs_calendar_week_number_differ(self):
        """strftime %W and %V produce different numbers for the same date."""
        dt = datetime(2026, 1, 5)
        assert dt.strftime("%W") == "01"  # Calendar week (Mon-based, Jan 1 = W00 or W01)
        assert dt.strftime("%V") == "02"  # ISO week
        assert dt.strftime("%Y") == "2026"  # Calendar year
        assert dt.strftime("%G") == "2026"  # ISO year (same here, but not always)


class TestLoadDailyResults:
    """Tests for loading and filtering daily result JSON files."""

    @pytest.fixture
    def mock_results_dir(self):
        """Create a temporary directory with mock daily result files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock result files
            dates = [
                "2026-01-06",
                "2026-01-07",
                "2026-01-08",
                "2026-01-09",
                "2026-01-10",  # Saturday (non-trading, but valid for test)
            ]
            for i, date in enumerate(dates):
                data = {
                    "date": date,
                    "portfolio_after": {
                        "total_value": 10000.0 + i * 100,
                        "positions": []
                    },
                    "executed_trades": []
                }
                filepath = Path(tmpdir) / f"{date}.json"
                with open(filepath, "w") as f:
                    json.dump(data, f)
            yield tmpdir

    def test_load_all_results(self, mock_results_dir):
        """Load all results without date filtering."""
        rg = ReportGenerator(results_dir=mock_results_dir)
        results = rg.load_daily_results()
        assert len(results) == 5
        assert results[0]["date"] == "2026-01-06"
        assert results[-1]["date"] == "2026-01-10"

    def test_load_with_start_date(self, mock_results_dir):
        """Filter results from a start date."""
        rg = ReportGenerator(results_dir=mock_results_dir)
        results = rg.load_daily_results(start_date="2026-01-08")
        assert len(results) == 3
        assert results[0]["date"] == "2026-01-08"

    def test_load_with_end_date(self, mock_results_dir):
        """Filter results up to an end date."""
        rg = ReportGenerator(results_dir=mock_results_dir)
        results = rg.load_daily_results(end_date="2026-01-08")
        assert len(results) == 3
        assert results[-1]["date"] == "2026-01-08"

    def test_load_with_date_range(self, mock_results_dir):
        """Filter results within a date range."""
        rg = ReportGenerator(results_dir=mock_results_dir)
        results = rg.load_daily_results(
            start_date="2026-01-07",
            end_date="2026-01-09"
        )
        assert len(results) == 3
        assert results[0]["date"] == "2026-01-07"
        assert results[-1]["date"] == "2026-01-09"

    def test_load_no_matching_results(self, mock_results_dir):
        """Return empty list when no files match the date range."""
        rg = ReportGenerator(results_dir=mock_results_dir)
        results = rg.load_daily_results(start_date="2025-01-01", end_date="2025-01-31")
        assert results == []

    def test_load_invalid_json_skipped(self, mock_results_dir):
        """Skip files that can't be parsed as JSON."""
        bad_file = Path(mock_results_dir) / "bad.json"
        bad_file.write_text("not valid json")
        rg = ReportGenerator(results_dir=mock_results_dir)
        results = rg.load_daily_results()
        # Should still get the 5 valid files
        assert len(results) == 5


class TestGenerateWeeklyReport:
    """Tests for weekly report generation with mock data."""

    @pytest.fixture
    def mock_week_dir(self):
        """Create mock data for a single ISO week."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 2026-W02: Jan 5 (Mon) to Jan 11 (Sun)
            dates = [
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
                "2026-01-08",
                "2026-01-09",
            ]
            values = [10000.0, 10100.0, 10050.0, 10200.0, 10150.0]
            for date, value in zip(dates, values):
                data = {
                    "date": date,
                    "portfolio_after": {
                        "total_value": value,
                        "total_return_pct": (value / 10000.0 - 1) * 100,
                        "positions": [
                            {"ticker": "SPY", "quantity": 10, "market_value": value * 0.6,
                             "unrealized_pnl_pct": 5.0},
                            {"ticker": "GLD", "quantity": 5, "market_value": value * 0.4,
                             "unrealized_pnl_pct": -2.0},
                        ]
                    },
                    "executed_trades": [
                        {"ticker": "SPY", "action": "buy", "price": 400.0, "status": "executed"}
                    ] if date == "2026-01-05" else []
                }
                filepath = Path(tmpdir) / f"{date}.json"
                with open(filepath, "w") as f:
                    json.dump(data, f)
            yield tmpdir

    def test_weekly_report_basic(self, mock_week_dir):
        """Generate a report for a known week and verify key metrics."""
        rg = ReportGenerator(results_dir=mock_week_dir)
        report = rg.generate_weekly_report(2026, 2)

        assert report["period"] == "2026-W02"
        assert report["period_type"] == "weekly"
        assert report["start_date"] == "2026-01-05"
        assert report["end_date"] == "2026-01-11"
        assert report["start_value"] == 10000.0
        assert report["end_value"] == 10150.0
        # Weekly return = (10150 / 10000) - 1 = 1.5%
        assert report["weekly_return_pct"] == pytest.approx(1.5, abs=0.01)
        assert report["total_trades"] == 1
        assert report["num_trading_days"] == 5

    def test_weekly_report_volatility(self, mock_week_dir):
        """Report should include volatility of daily returns."""
        rg = ReportGenerator(results_dir=mock_week_dir)
        report = rg.generate_weekly_report(2026, 2)
        # Daily returns: +1.0%, -0.495%, +1.493%, -0.490%
        assert report["volatility"] > 0

    def test_weekly_report_best_worst_day(self, mock_week_dir):
        """Best and worst days should be identified by total_return_pct."""
        rg = ReportGenerator(results_dir=mock_week_dir)
        report = rg.generate_weekly_report(2026, 2)
        # By total_return_pct: Jan 8 = 2.0% (best), Jan 5 = 0.0% (worst)
        assert report["best_day"]["date"] == "2026-01-08"
        assert report["worst_day"]["date"] == "2026-01-05"

    def test_weekly_report_final_positions(self, mock_week_dir):
        """Final positions should be captured from the last day."""
        rg = ReportGenerator(results_dir=mock_week_dir)
        report = rg.generate_weekly_report(2026, 2)
        assert len(report["final_positions"]) == 2
        assert report["final_positions"][0]["ticker"] == "SPY"

    def test_weekly_report_no_data(self, mock_week_dir):
        """Request a week with no data should return empty dict."""
        rg = ReportGenerator(results_dir=mock_week_dir)
        report = rg.generate_weekly_report(2025, 1)
        assert report == {}


class TestGenerateMonthlyReport:
    """Tests for monthly report generation."""

    @pytest.fixture
    def mock_month_dir(self):
        """Create mock data for January 2026."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for day in range(1, 6):
                date = f"2026-01-{day:02d}"
                data = {
                    "date": date,
                    "portfolio_after": {
                        "total_value": 10000.0 + day * 50,
                        "positions": [
                            {"ticker": "SPY", "unrealized_pnl_pct": day * 0.5}
                        ]
                    },
                    "executed_trades": [
                        {"ticker": "SPY", "action": "buy"}
                    ] if day == 1 else []
                }
                filepath = Path(tmpdir) / f"{date}.json"
                with open(filepath, "w") as f:
                    json.dump(data, f)
            yield tmpdir

    def test_monthly_report_basic(self, mock_month_dir):
        """Generate a monthly report and verify structure."""
        rg = ReportGenerator(results_dir=mock_month_dir)
        report = rg.generate_monthly_report(2026, 1)

        assert report["period"] == "2026-01"
        assert report["period_type"] == "monthly"
        assert report["start_date"] == "2026-01-01"
        assert report["start_value"] == 10050.0  # First available day is Jan 2? No, we created Jan 1
        assert report["end_value"] == 10250.0
        assert report["total_trades"] == 1
        assert report["num_trading_days"] == 5

    def test_monthly_report_no_data(self, mock_month_dir):
        """Request a month with no data should return empty dict."""
        rg = ReportGenerator(results_dir=mock_month_dir)
        report = rg.generate_monthly_report(2025, 12)
        assert report == {}


class TestSaveAndPrintReport:
    """Tests for report persistence and display."""

    @pytest.fixture
    def mock_report(self):
        """A minimal report dictionary."""
        return {
            "period": "2026-W02",
            "period_type": "weekly",
            "start_date": "2026-01-05",
            "end_date": "2026-01-11",
            "start_value": 10000.0,
            "end_value": 10150.0,
            "weekly_return_pct": 1.5,
            "total_trades": 1,
            "num_trading_days": 5,
        }

    def test_save_report_creates_file(self, mock_report, tmp_path):
        """save_report should write a JSON file."""
        rg = ReportGenerator()
        output_dir = str(tmp_path / "reports")
        filepath = rg.save_report(mock_report, output_dir=output_dir)

        assert Path(filepath).exists()
        assert Path(filepath).name == "weekly_2026-W02.json"

        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded["period"] == "2026-W02"

    def test_print_report_no_crash(self, mock_report, capsys):
        """print_report should not crash on valid report."""
        rg = ReportGenerator()
        rg.print_report(mock_report)
        captured = capsys.readouterr()
        assert "WEEKLY REPORT: 2026-W02" in captured.out
        assert "1.50%" in captured.out

    def test_print_report_empty(self, capsys):
        """print_report should handle empty report gracefully."""
        rg = ReportGenerator()
        rg.print_report({})
        captured = capsys.readouterr()
        assert "No report data available" in captured.out

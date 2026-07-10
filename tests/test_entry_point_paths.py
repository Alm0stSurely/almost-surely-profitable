"""
Tests for entry-point script path resolution.

Verifies that daily_run.py, weekly_report.py and monitor.py resolve their
data/ and results/ paths relative to the repository root, so they can be run
from any working directory without silently creating files elsewhere.
"""

import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import daily_run
import weekly_report
import monitor


class TestPathResolutionConstants:
    """Path constants in entry-point scripts must be absolute and repo-rooted."""

    def test_daily_run_repo_root_is_absolute(self):
        assert daily_run.REPO_ROOT.is_absolute()
        assert (daily_run.REPO_ROOT / "src" / "daily_run.py").exists()

    def test_daily_run_data_dir_is_under_repo_root(self):
        assert daily_run.DATA_DIR.is_absolute()
        assert daily_run.DATA_DIR == daily_run.REPO_ROOT / "data"

    def test_daily_run_results_dir_is_under_repo_root(self):
        assert daily_run.DAILY_RESULTS_DIR.is_absolute()
        assert daily_run.DAILY_RESULTS_DIR == daily_run.REPO_ROOT / "results" / "daily"

    def test_weekly_report_paths_are_absolute(self):
        assert weekly_report.REPO_ROOT.is_absolute()
        assert weekly_report.DATA_DIR.is_absolute()
        assert weekly_report.DAILY_RESULTS_DIR.is_absolute()
        assert weekly_report.RESULTS_DIR.is_absolute()
        assert weekly_report.REPORTS_DIR.is_absolute()

    def test_weekly_report_paths_under_repo_root(self):
        assert weekly_report.DATA_DIR == weekly_report.REPO_ROOT / "data"
        assert weekly_report.DAILY_RESULTS_DIR == weekly_report.REPO_ROOT / "results" / "daily"
        assert weekly_report.RESULTS_DIR == weekly_report.REPO_ROOT / "results"
        assert weekly_report.REPORTS_DIR == weekly_report.REPO_ROOT / "results" / "reports"

    def test_monitor_paths_are_absolute(self):
        assert monitor.REPO_ROOT.is_absolute()
        assert monitor.DATA_DIR.is_absolute()
        assert monitor.CONFIG_PATH.is_absolute()
        assert monitor.UNIVERSE_PATH.is_absolute()
        assert monitor.ALERT_HISTORY_PATH.is_absolute()
        assert monitor.MARKET_STATE_PATH.is_absolute()

    def test_monitor_paths_under_repo_root(self):
        assert monitor.DATA_DIR == monitor.REPO_ROOT / "data"
        assert monitor.CONFIG_PATH == monitor.REPO_ROOT / "config" / "monitor.json"
        assert monitor.UNIVERSE_PATH == monitor.REPO_ROOT / "config" / "universe.json"
        assert monitor.ALERT_HISTORY_PATH == monitor.REPO_ROOT / "data" / "alert_history.json"
        assert monitor.MARKET_STATE_PATH == monitor.REPO_ROOT / "data" / "market_state.json"


class TestSetupDirectories:
    """setup_directories() must create dirs inside the repo, not cwd."""

    def test_setup_directories_creates_under_repo_root(self, tmp_path):
        # Temporarily override path constants so we don't touch the real repo.
        original_data_dir = daily_run.DATA_DIR
        original_results_dir = daily_run.DAILY_RESULTS_DIR

        test_data_dir = tmp_path / "data"
        test_results_dir = tmp_path / "results" / "daily"

        daily_run.DATA_DIR = test_data_dir
        daily_run.DAILY_RESULTS_DIR = test_results_dir

        try:
            daily_run.setup_directories()
            assert test_data_dir.exists()
            assert test_results_dir.exists()
        finally:
            daily_run.DATA_DIR = original_data_dir
            daily_run.DAILY_RESULTS_DIR = original_results_dir


class TestMonitorMarketStatePaths:
    """Monitor load/save market state must use repo-rooted paths."""

    def test_save_and_load_market_state_use_absolute_path(self, tmp_path):
        from portfolio.portfolio import Portfolio

        # Create a temporary portfolio so load_previous_close has a fallback.
        portfolio = Portfolio(data_dir=str(tmp_path / "data"))

        original_state_path = monitor.MARKET_STATE_PATH
        test_state_path = tmp_path / "data" / "market_state.json"
        monitor.MARKET_STATE_PATH = test_state_path

        try:
            prices = {"SPY": 400.0, "TLT": 100.0}
            monitor.save_market_state(prices)

            assert test_state_path.exists()
            loaded = monitor.load_previous_close(portfolio)
            assert loaded["SPY"] == 400.0
            assert loaded["TLT"] == 100.0
        finally:
            monitor.MARKET_STATE_PATH = original_state_path

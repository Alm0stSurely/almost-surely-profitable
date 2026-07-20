"""Tests for the daily result validation utilities."""
import json
import tempfile
from pathlib import Path

import pytest

from utils import is_valid_daily_result, load_valid_daily_results, load_valid_daily_results_limited


@pytest.fixture
def sample_results(tmp_path):
    """Create a temporary results directory with valid, dry-run and test files."""
    results_dir = tmp_path / "daily"
    results_dir.mkdir()

    valid = {
        "date": "2026-07-20",
        "dry_run": False,
        "market_summary": {"assets_analyzed": 32},
        "decision": {"reasoning": "Normal market analysis."},
        "portfolio_after": {"cash": 2623.93, "total_value": 9716.20},
    }

    dry_run = {
        "date": "2026-07-19",
        "dry_run": True,
        "market_summary": {"assets_analyzed": 32},
        "decision": {"reasoning": "Normal market analysis."},
        "portfolio_after": {"cash": 2623.93, "total_value": 9716.20},
    }

    test_artifact = {
        "date": "2026-07-18",
        "dry_run": False,
        "market_summary": {"assets_analyzed": 1},
        "decision": {"reasoning": "HOLD for test"},
        "portfolio_after": {"cash": 10000.0, "total_value": 10000.0},
    }

    for name, data in [
        ("2026-07-20.json", valid),
        ("2026-07-19_dry_run.json", dry_run),
        ("2026-07-18.json", test_artifact),
    ]:
        with open(results_dir / name, "w") as f:
            json.dump(data, f)

    return results_dir


def test_is_valid_daily_result_accepts_valid():
    data = {
        "date": "2026-07-20",
        "dry_run": False,
        "market_summary": {"assets_analyzed": 32},
        "decision": {"reasoning": "Normal market analysis."},
    }
    assert is_valid_daily_result(data) is True


def test_is_valid_daily_result_rejects_dry_run():
    data = {
        "date": "2026-07-20",
        "dry_run": True,
        "market_summary": {"assets_analyzed": 32},
        "decision": {"reasoning": "Normal market analysis."},
    }
    assert is_valid_daily_result(data) is False


def test_is_valid_daily_result_rejects_test_reasoning():
    data = {
        "date": "2026-07-20",
        "dry_run": False,
        "market_summary": {"assets_analyzed": 32},
        "decision": {"reasoning": "HOLD for test"},
    }
    assert is_valid_daily_result(data) is False


def test_is_valid_daily_result_rejects_too_few_assets():
    data = {
        "date": "2026-07-20",
        "dry_run": False,
        "market_summary": {"assets_analyzed": 1},
        "decision": {"reasoning": "Normal market analysis."},
    }
    assert is_valid_daily_result(data) is False


def test_is_valid_daily_result_missing_market_summary():
    data = {
        "date": "2026-07-20",
        "dry_run": False,
        "decision": {"reasoning": "Normal market analysis."},
    }
    assert is_valid_daily_result(data) is True


def test_load_valid_daily_results(sample_results):
    results = load_valid_daily_results(str(sample_results))
    assert len(results) == 1
    assert results[0]["date"] == "2026-07-20"


def test_load_valid_daily_results_limited(sample_results):
    # Add a second valid file for the limit test
    second = {
        "date": "2026-07-21",
        "dry_run": False,
        "market_summary": {"assets_analyzed": 32},
        "decision": {"reasoning": "Normal market analysis."},
    }
    with open(sample_results / "2026-07-21.json", "w") as f:
        json.dump(second, f)

    results = load_valid_daily_results_limited(str(sample_results), days=1)
    assert len(results) == 1
    assert results[0]["date"] == "2026-07-21"


def test_load_valid_daily_results_returns_sorted(sample_results):
    # Add an out-of-order valid file
    for date, fname in [("2026-07-15", "2026-07-15.json")]:
        data = {
            "date": date,
            "dry_run": False,
            "market_summary": {"assets_analyzed": 32},
            "decision": {"reasoning": "Normal market analysis."},
        }
        with open(sample_results / fname, "w") as f:
            json.dump(data, f)

    results = load_valid_daily_results(str(sample_results))
    dates = [r["date"] for r in results]
    assert dates == sorted(dates)

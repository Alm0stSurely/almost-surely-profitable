"""Tests for the daily result validation utilities."""
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import (
    dump_json_safe,
    is_valid_daily_result,
    load_valid_daily_results,
    load_valid_daily_results_limited,
    safe_format_float,
    safe_format_pct,
    sanitize_for_json,
)


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


class TestSanitizeForJson:
    """Tests for the recursive JSON sanitization helper."""

    def test_finite_values_preserved(self):
        data = {
            "int": 42,
            "float": 3.14,
            "string": "ok",
            "bool": True,
            "none": None,
            "list": [1, 2.5, "x"],
        }
        assert sanitize_for_json(data) == data

    def test_nan_replaced_with_none(self):
        assert sanitize_for_json({"x": float("nan")})["x"] is None

    def test_infinity_replaced_with_none(self):
        assert sanitize_for_json({"x": float("inf")})["x"] is None
        assert sanitize_for_json({"x": float("-inf")})["x"] is None

    def test_nested_structures_sanitized(self):
        data = {
            "outer": [
                {"inner": float("nan")},
                [1.0, float("inf"), 3.0],
            ]
        }
        result = sanitize_for_json(data)
        assert result["outer"][0]["inner"] is None
        assert result["outer"][1] == [1.0, None, 3.0]

    def test_numpy_scalar_non_finite_sanitized(self):
        numpy = pytest.importorskip("numpy")
        assert sanitize_for_json({"x": numpy.nan})["x"] is None
        assert sanitize_for_json({"x": numpy.inf})["x"] is None


class TestSafeFormat:
    """Tests for safe percentage and float formatters."""

    def test_safe_format_pct_positive(self):
        assert safe_format_pct(0.1234) == "12.34%"

    def test_safe_format_pct_signed(self):
        assert safe_format_pct(0.1234, sign=True) == "+12.34%"
        assert safe_format_pct(-0.05, sign=True) == "-5.00%"

    def test_safe_format_pct_non_finite(self):
        assert safe_format_pct(float("nan")) == "n/a"
        assert safe_format_pct(float("inf")) == "n/a"
        assert safe_format_pct(float("-inf")) == "n/a"
        assert safe_format_pct(None) == "n/a"

    def test_safe_format_pct_custom_fallback(self):
        assert safe_format_pct(float("nan"), fallback="--") == "--"

    def test_safe_format_float_signed(self):
        assert safe_format_float(1.234, sign=True) == "+1.23"

    def test_safe_format_float_non_finite(self):
        assert safe_format_float(float("nan")) == "n/a"


class TestDumpJsonSafe:
    """Tests for the JSON-safe serialization wrapper."""

    def test_writes_valid_json_for_non_finite_input(self, tmp_path):
        data = {"good": 1.0, "bad": float("nan"), "worse": float("inf")}
        path = tmp_path / "out.json"
        with open(path, "w") as f:
            dump_json_safe(data, f)

        loaded = json.loads(path.read_text())
        assert loaded["good"] == 1.0
        assert loaded["bad"] is None
        assert loaded["worse"] is None

    def test_default_callback_still_used(self, tmp_path):
        data = {"date": object()}  # not JSON serializable without default
        path = tmp_path / "out.json"
        with open(path, "w") as f:
            dump_json_safe(data, f, default=str)

        loaded = json.loads(path.read_text())
        assert loaded["date"].startswith("<object object at")

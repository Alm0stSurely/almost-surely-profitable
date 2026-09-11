"""Tests for the daily result validation utilities."""
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import is_valid_daily_result, load_valid_daily_results, load_valid_daily_results_limited, sanitize_for_json, dump_json_safe
from utils.formatting import _fmt_finite, _fmt_pct


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


class TestSharedFormattingHelpers:
    """Tests for utils.formatting — the canonical finite-safe formatter pair.

    Both helpers were extracted from backtest.formatting so that every
    console/LLM formatter validates before scaling instead of growing
    per-module copies of the same convention.
    """

    def test_fmt_finite_formats_finite_float(self):
        assert _fmt_finite(3.14159, ".2f") == "3.14"
        assert _fmt_finite(-0.5, ">6.2f") == " -0.50"

    def test_fmt_finite_formats_int_and_np_integer(self):
        assert _fmt_finite(42, ">3") == " 42"
        numpy = pytest.importorskip("numpy")
        assert _fmt_finite(numpy.int64(7), ">3") == "  7"

    def test_fmt_finite_rejects_non_finite(self):
        assert _fmt_finite(float("nan"), ".2f") == "n/a"
        assert _fmt_finite(float("inf"), ".2f") == "n/a"
        assert _fmt_finite(float("-inf"), ".2f") == "n/a"

    def test_fmt_finite_rejects_non_numeric(self):
        assert _fmt_finite(None, ".2f") == "n/a"
        assert _fmt_finite("3.14", ".2f") == "n/a"
        assert _fmt_finite([1.0], ".2f") == "n/a"

    def test_fmt_finite_rejects_bool(self):
        # bool subclasses int; truthy flags must not render as 1/0
        assert _fmt_finite(True, ">3") == "n/a"
        assert _fmt_finite(False, ">3") == "n/a"

    def test_fmt_pct_scales_finite_fraction(self):
        assert _fmt_pct(0.0521, ">6.2f") == "  5.21"

    def test_fmt_pct_validates_before_scaling(self):
        # None must not crash on * 100; non-finite must not be multiplied first
        assert _fmt_pct(None, ">6.2f") == "n/a"
        assert _fmt_pct(float("nan"), ">6.2f") == "n/a"
        assert _fmt_pct(float("inf"), ">6.2f") == "n/a"

    def test_fmt_pct_rejects_bool(self):
        assert _fmt_pct(True, ">6.2f") == "n/a"

    def test_backtest_modules_reexport_shared_helpers(self):
        # PR #45/#50 regression tests and benchmark imports must keep working
        import backtest.backtest as bt
        import backtest.formatting as btf

        assert btf._fmt_finite is _fmt_finite
        assert btf._fmt_pct is _fmt_pct
        assert bt._fmt_finite is _fmt_finite
        assert bt._fmt_pct is _fmt_pct

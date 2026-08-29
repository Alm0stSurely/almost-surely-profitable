"""Tests for keyword_trends.py formatting guards."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.keyword_trends import (
    _safe_pct_str,
    _safe_signed_str,
    format_report,
    linear_slope,
    rolling_average,
)
from utils import _is_finite_number


class TestSafeFormatHelpers:
    def test_safe_pct_str_finite(self):
        assert _safe_pct_str(12.34) == "   12.3%"

    def test_safe_pct_str_nan(self):
        assert _safe_pct_str(float("nan")) == "     n/a"

    def test_safe_pct_str_inf(self):
        assert _safe_pct_str(float("inf")) == "     n/a"

    def test_safe_pct_str_none(self):
        assert _safe_pct_str(None) == "     n/a"

    def test_safe_signed_str_finite_positive(self):
        assert _safe_signed_str(0.42) == "    +0.42"

    def test_safe_signed_str_finite_negative(self):
        assert _safe_signed_str(-1.23) == "    -1.23"

    def test_safe_signed_str_nan(self):
        assert _safe_signed_str(float("nan")) == "      n/a"

    def test_safe_signed_str_inf(self):
        assert _safe_signed_str(float("inf")) == "      n/a"


class TestRollingAverage:
    def test_window_with_nan_returns_nan(self):
        values = [10.0, 20.0, float("nan"), 40.0]
        out = rolling_average(values, window=4)
        assert any(v != v for v in out[2:])
        assert all(_is_finite_number(v) for v in out[:2])

    def test_all_finite_returns_expected(self):
        values = [10.0, 20.0, 30.0]
        out = rolling_average(values, window=2)
        assert out == [10.0, 15.0, 25.0]


class TestLinearSlope:
    def test_non_finite_input_returns_zero(self):
        assert linear_slope([1.0, float("nan"), 3.0]) == 0.0
        assert linear_slope([1.0, float("inf"), 3.0]) == 0.0

    def test_short_series_returns_zero(self):
        assert linear_slope([5.0]) == 0.0

    def test_rising_trend_positive_slope(self):
        slope = linear_slope([0.0, 1.0, 2.0, 3.0])
        assert slope > 0.9


class TestFormatReportGuards:
    def test_non_finite_rolling_average_renders_na(self):
        weekly_rates = {
            "2026-W01": {"_n": 1, "loss aversion": 10.0},
            "2026-W02": {"_n": 1, "loss aversion": float("nan")},
            "2026-W03": {"_n": 1, "loss aversion": 30.0},
            "2026-W04": {"_n": 1, "loss aversion": 40.0},
        }
        report = format_report(weekly_rates, highlight_concepts=["loss aversion"], window=4)
        # The rolling average for the last week spans a window containing NaN.
        lines = report.splitlines()
        summary_line = [l for l in lines if l.strip().startswith("loss aversion")][0]
        assert "     n/a" in summary_line

    def test_non_finite_latest_renders_na(self):
        weekly_rates = {
            "2026-W01": {"_n": 1, "loss aversion": float("nan")},
        }
        report = format_report(weekly_rates, highlight_concepts=["loss aversion"], window=4)
        assert "     n/a" in report

    def test_finite_values_render_normally(self):
        weekly_rates = {
            "2026-W01": {"_n": 2, "loss aversion": 50.0},
            "2026-W02": {"_n": 2, "loss aversion": 50.0},
            "2026-W03": {"_n": 2, "loss aversion": 50.0},
            "2026-W04": {"_n": 2, "loss aversion": 50.0},
        }
        report = format_report(weekly_rates, highlight_concepts=["loss aversion"], window=4)
        assert "50.0%" in report
        assert "nan%" not in report
        assert "inf%" not in report

    def test_non_finite_slope_renders_direction_na(self):
        weekly_rates = {
            "2026-W01": {"_n": 1, "loss aversion": 0.0},
            "2026-W02": {"_n": 1, "loss aversion": float("nan")},
        }
        report = format_report(weekly_rates, highlight_concepts=["loss aversion"], window=4)
        assert "       n/a" in report

    def test_empty_weekly_rates(self):
        assert format_report({}) == "No weekly data available."
